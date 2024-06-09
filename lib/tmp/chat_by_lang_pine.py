import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# 環境変数のロード
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Pineconeクライアントの初期化
pc = Pinecone(api_key=os.environ.get('PINECONE_LLAMA_API_KEY'))
index_name = "switchbot"
pinecone_index = pc.Index(index_name)
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)

def get_openai_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def similarity_search_query_with_score(user_query):
    """クエリに関連した文章(関連度:0.8以上)と元のクエリをChatGPTに与える"""
    similarities = []
    context = ""
    try:
        for doc, score in vectorstore.similarity_search_with_score(query=user_query):
            print("--------------doc--------------")
            print(doc)
            if score >= 0.8:
                if 'page_content' in doc:
                    context += doc.page_content + "\n"
                    similarities.append(score)
                else:
                    print("Found document with no `page_content` key. Skipping.")
    except Exception as e:
        print(f"Error during similarity search: {e}")

    average_similarity = np.mean(similarities) if similarities else 0.0

    if context == "":
        return "関連情報が見つかりませんでした", "", average_similarity

    combined_query = f"""
        You are an API-specific AI assistant, Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, can I help with anything else, don't try to make up an answer. Keep the answer as concise as possible. Answer in Japanese.
        Context: {context}
        Question: {user_query}
        Answer:
    """

    chatgpt_response = client.chat_completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "You are ChatBot answering questions about SwitchBot API. Relevant information must be followed."},
                  {"role": "user", "content": combined_query}]
    )
    return chatgpt_response.choices[0].message.content, context, average_similarity

# StreamlitでGUI化
st.title("関連するインデックスの抽出")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
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
    # ユーザの入力をチャット履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ユーザの入力を表示
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner('検索中...'):
        response, relevant_info, similarity = similarity_search_query_with_score(prompt)

    # ChatBotの返答を表示
    with st.chat_message("assistant"):
        st.write("関連度: ", similarity)
        st.write(response)
        if response != "関連情報が見つかりませんでした":
            with st.expander("詳細"):
                st.write(relevant_info)

    # ユーザの入力に関連する情報とChatBotの返答をチャット履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response, "expandar_content": relevant_info, "similarity": similarity})
