import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st
import numpy as np
from store_llama import insert_query_response_to_db
from repo import save_doc_to_text, mov_saved_doc

API_KEY_PROMPT="API name to be modified"

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

def click_button(idx):
    st.session_state.messages[idx]["flag"] = True
    st.rerun()

def add_history(prompt, response, key):
    # 履歴をテキストに保存する
    prompt_lists = prompt.split('## ')
    api_lists = [item.split('\n') for item in prompt_lists]

    api_list = list(filter(lambda x: x[0]==API_KEY_PROMPT, api_lists))
    api_name = ""
    if api_list:
        api_name = api_list[0][1].split()[0].lower()

    if api_name == "":
        return

    save_doc_to_text(query=prompt, response=response, api_name=api_name)

    # 履歴をインデックス化する
    input_dir = f"../data/history/{api_name}"
    index_name = "revision-history"
    insert_query_response_to_db(index_name=index_name, input_dir=input_dir)

    # 履歴のテキストファイルを移動
    mov_saved_doc(source_path=input_dir, api_name=api_name)

    # ボタンの無効化
    click_button(key)


def main():
    # streamlitでGUI表示
    st.title("API search assistant")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    prompt = ""
    # 過去の結果を表示
    for i, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                prompt = message["content"]
                st.write(message["content"])

            elif message["role"] == "assistant":
                st.write("score: ", message['similarity'])
                st.write(message["content"])

                if not message["flag"]:
                    if st.button('👍', key=f"button_{i}"):
                        with st.spinner('inserting...'):
                            add_history(prompt=prompt, response=message["content"], key=i)
                else:
                    st.write("Added data.")

                with st.expander("detail"):
                    st.write(message["expandar_content"])

    if 'button' not in st.session_state:
        st.session_state.button = False

    # 現在の回答を表示
    if prompt := st.chat_input("Please enter what you want to search for:"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner('searching...'):
            response, relevant_info, similarity = query_index(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response, "expandar_content": relevant_info, "similarity": similarity, "flag": False})
        with st.chat_message("assistant"):
            st.write("score: ", similarity)
            st.write(response)

            key_cnt = len(st.session_state["messages"]) - 1
            if not st.session_state.messages[-1]["flag"]:
                if st.button('👍', key=f"button_{key_cnt}"):
                    with st.spinner('inserting...'):
                        add_history(prompt=prompt, response=response, key=key_cnt)
            if response != "Relevant information not found.":
                with st.expander("detail"):
                    st.write(relevant_info)

if __name__ == "__main__":
    main()
