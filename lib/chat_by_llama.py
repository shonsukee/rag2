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
index_name = 'switchbot-llama'
pinecone_index = pc.Index(index_name)

# PineconeVectorStoreの設定
vector_store = PineconeVectorStore(pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

retriever = index.as_retriever(search_kwargs={"k": 5})

# ユーザー入力に関連するインデックスを検索する関数
def query_index(user_query):
    context = ""
    similarities = []
    i = 0
    context_nodes = retriever.retrieve(user_query)
    for node in context_nodes:
        if node.score >= 0.75:
            i += 1
            context += f"""
                Context number {i} (score: {node.score}):
                {node.text}
            """
            similarities.append(node.score)

    similarity = np.mean(similarities)
    combined_query = f"""
        ### Instruction
        You are an API-specific AI assistant, use the following pieces of context to answer the requirement at the end. If you don't know the answer, just say that you don't know, can I help with anything else, don't try to make up an answer.

        ### Context
        {context}

        ### Input Data
        {user_query}

        ### Output Indicator
        Follow the contextual information. Make all modifications in the function except for imports. Keep the answer as concise as possible. Output only code.
    """

    chatgpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are ChatBot answering questions about SwitchBot API. Relevant information must be followed."},
            {"role": "user", "content": combined_query}
        ]
    )
    return chatgpt_response.choices[0].message.content, context, similarity

def main():
    # streamlitでGUI表示
    st.title("API search assistant")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.write("score: ", message['similarity'])
                st.write(message["content"])
                with st.expander("detail"):
                    st.write(message["expandar_content"])
            else:
                st.write(message["content"])

    if prompt := st.chat_input("Please enter what you want to search for:"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner('searching...'):
            response, relevant_info, similarity = query_index(prompt)

        with st.chat_message("assistant"):
            st.write("score: ", similarity)
            st.write(response)
            if response != "Relevant information not found.":
                with st.expander("detail"):
                    st.write(relevant_info)

        st.session_state.messages.append({"role": "assistant", "content": response, "expandar_content": relevant_info, "similarity": similarity})

if __name__ == "__main__":
    main()