# llama_indexでRetrievalをしたい
# ただしdocをindexに格納後，検索する関数しか見つからない
import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st

# 環境変数のロード
load_dotenv()

client = OpenAI()

# Pineconeの初期化
pc = Pinecone(api_key=os.environ.get("PINECONE_IOT_API_KEY"))

# Pineconeインデックスの設定
index_name = 'iot-api'
pinecone_index = pc.Index(index_name)

# PineconeVectorStoreの設定
# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
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


def query_llama(user_query):
    # Pineconeインデックスにクエリを送信
    result = query_engine.query(user_query)

    return result

def query_pinecone_index(embedding, top_k=5):
    # Pineconeインデックスにクエリを送信
    result = query_engine.query(
        query=embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )

    return result['matches']

# ユーザー入力に関連するインデックスを検索する関数
def query_index(user_query):
    # Pineconeインデックスにクエリを送信
    vector_db_response = query_llama(user_query)

    relevant_info = "\n".join([f"ID: {item['id']}, Score: {item['score']}, Content: {item['metadata']}" for item in vector_db_response if item['score'] >= 0.5])

    if not relevant_info:
        return "関連情報が見つかりませんでした", "no data"

    combined_query = f"""
    Answer the following question based on the provided information: {user_query}
    Relevant information: {vector_db_response}
    """

    chatgpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are ChatBot answering questions about SwitchBot API."},
            {"role": "user", "content": combined_query}
        ]
    )

    return chatgpt_response.choices[0].message.content, vector_db_response

st.title("関連するインデックスの抽出")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            with st.expander("詳細"):
                st.write(message["expandar_content"])

if prompt := st.chat_input("検索したい内容を入力してください:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner('検索中...'):
        response, relevant_info = query_index(prompt)

    with st.chat_message("assistant"):
        st.write(response)
        with st.expander("詳細"):
            st.write(relevant_info)

    st.session_state.messages.append({"role": "assistant", "content": response, "expandar_content": relevant_info})
