import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st
import numpy as np

# 環境変数のロード
load_dotenv()

client = OpenAI()

# Pineconeの初期化
pc = Pinecone(api_key=os.environ.get("PINECONE_IOT_API_KEY"))

# Pineconeインデックスの設定
index_name = 'iot-api'
pinecone_index = pc.Index(index_name)

# PineconeVectorStoreの設定
vector_store = PineconeVectorStore(pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# クエリエンジンの設定
query_engine = index.as_query_engine()

def get_openai_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ユーザー入力に関連するインデックスを検索する関数
def query_index(user_query):
    query_embedding = get_openai_embedding(user_query)
    context = query_engine.query(user_query)
    context = context.response

    doc_embedding = get_openai_embedding(context)
    similarity = cosine_similarity(query_embedding, doc_embedding)

    combined_query = f"""
        You are an API-specific AI assistant, Use the following pieces of context to answer the question at the end. Keep the answer as concise as possible. Answer in Japanese.
        Context: {context}
        Question: {user_query}
        Answer:
   """

    chatgpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an API-specific AI assistant, Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, can I help with anything else, don't try to make up an answer. Keep the answer as concise as possible. Answer in Japanese."},
            {"role": "user", "content": combined_query}
        ]
    )
    return chatgpt_response.choices[0].message.content, context, similarity

# streamlitでGUI表示
st.title("API検索アシスタント")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.write("関連度: ", message['similarity'])
            st.write(message["content"])
            with st.expander("詳細"):
                st.write(message["expandar_content"])
        else:
            st.write(message["content"])

if prompt := st.chat_input("検索したい内容を入力してください:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner('検索中...'):
        response, relevant_info, similarity = query_index(prompt)

    with st.chat_message("assistant"):
        st.write("関連度: ", similarity)
        st.write(response)
        if response != "関連情報が見つかりませんでした":
            with st.expander("詳細"):
                st.write(relevant_info)

    st.session_state.messages.append({"role": "assistant", "content": response, "expandar_content": relevant_info, "similarity": similarity})
