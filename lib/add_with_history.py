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
SOURCE_INDEX_NAME="switchbot-llama-v1"
HISTORY_INDEX_NAME="revision-history"

# 環境変数のロード
load_dotenv()

client = OpenAI()

# Pineconeの初期化
pc = Pinecone(api_key=os.environ.get("PINECONE_IOT_API_KEY"))

# Pineconeインデックスの設定
pinecone_index = pc.Index(SOURCE_INDEX_NAME)

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

    combined_query = \
        f"""### Instruction
        You are an API-specific AI assistant, use the following pieces of context to answer the requirement at the end. If you don't know the answer, just say that you don't know, can I help with anything else, don't try to make up an answer.

        ### Context
        {context}

        ### Input Data
        {user_query}

        ### Output Indicator
        First, you read the code and the context corresponding to the previous specification in turn and identify any new information or changes. Follow the contextual information when making modifications. Make all modifications in the function except for imports. Keep the answer as concise as possible. Output only code.
    """

    chatgpt_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are ChatBot answering questions about SwitchBot API. Relevant information must be followed."},
            {"role": "user", "content": combined_query}
        ]
    )
    response = chatgpt_response.choices[0].message.content

    # 回答の評価
    ## コード入れ替え
    h_queries = user_query.split('```\n')
    h_queries[1] = '```\n' + response + '\n```\n'
    h_query = ""
    for q in h_queries:
        h_query += q

    ## 関連度抽出
    h_pinecone_index = pc.Index(SOURCE_INDEX_NAME)
    h_vector_store = PineconeVectorStore(h_pinecone_index)
    h_index = VectorStoreIndex.from_vector_store(vector_store=h_vector_store)
    h_retriever = h_index.as_retriever(search_kwargs={"k": 5})
    h_context_nodes = h_retriever.retrieve(h_query)
    similarities = []
    for node in h_context_nodes:
        if node.score >= 0.1:
            similarities.append(node.score)

    ## 履歴DBへ格納
    if similarity < np.mean(similarities):
        add_history_to_db(user_query, response)
    else:
        print("--------格納しませんでした--------")

    return response, context, str(similarity) + " : " + str(np.mean(similarities))

def add_history_to_db(prompt, response):
    # 履歴をテキストに保存する
    prompt_lists = prompt.split('## ')
    api_lists = [item.split('\n') for item in prompt_lists]

    api_list = list(filter(lambda x: x[0]==API_KEY_PROMPT, api_lists))
    api_name = ""
    if api_list:
        api_name = api_list[0][1].split()[0].lower()

    if api_name == "":
        print("--------API名が見つかりません--------")
        return

    save_doc_to_text(query=prompt, response=response, api_name=api_name)

    # 履歴をインデックス化する
    input_dir = f"../history/{api_name}"
    insert_query_response_to_db(index_name=HISTORY_INDEX_NAME, input_dir=input_dir)

    # 履歴のテキストファイルを移動
    mov_saved_doc(source_path=input_dir, api_name=api_name)
    print("--------格納しました--------")

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