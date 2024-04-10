# DBに格納されている情報を付随させて質問していく
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
import os
from dotenv import load_dotenv
import streamlit as st
import openai

load_dotenv()

# DB/llm等初期化
cassio.init(token=os.environ["ASTRA_DB_APPLICATION_TOKEN"], database_id=os.environ["ASTRA_DB_ID"])
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="switchbot_demo",
    session=None,
    keyspace=None,
)
print("--initialize DB/llm--")

def query_chatgpt_with_vector_db(user_query):
    """クエリに最も関係する文章と元のクエリをChatGPTに与える"""
    # ユーザーのクエリに基づいてベクトルDBから情報を検索
    vector_db_response = astra_vector_index.query(user_query).strip()

    # 検索結果とユーザーのクエリを組み合わせてChatGPTに問い合わせるクエリを生成
    combined_query = f"Answer the following question based on the provided information: {user_query} Relevant information: {vector_db_response}"

    chatgpt_response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "You are ChatBot answering questions about SwitchBot API v1.1."},
                  {"role": "user", "content": combined_query}]
    )

    return chatgpt_response.choices[0].message.content

def similarity_search_query_with_score(user_query):
    """クエリに関連した文章(関連度:0.8以上)と元のクエリをChatGPTに与える"""
    vector_db_response = ""
    for doc, score in astra_vector_store.similarity_search_with_score(user_query, k=4):
        if score >= 0.8:
            vector_db_response += doc.page_content + "¥n"

    combined_query = f"Answer the following question based on the provided information: {user_query} Relevant information: {vector_db_response}"
    chatgpt_response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "You are ChatBot answering questions about SwitchBot API."},
                  {"role": "user", "content": combined_query}]
    )
    return chatgpt_response.choices[0].message.content

embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
astra_vector_store = Cassandra(
    session=None,
    embedding=embedding,
    table_name="switchbot_table",
    keyspace=None,
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# StreamlitでGUI化
st.title("Modify Switchbot API specifications")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーの入力が送信された際に実行される処理
if prompt := st.chat_input("SwitchBot API v1.1について知りたいことはありますか?"):
    # ユーザの入力をチャット履歴に追加する
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ユーザの入力を表示する
    with st.chat_message("user"):
        st.markdown(prompt)

    response = similarity_search_query_with_score(prompt)

    # ChatBotの返答を表示する
    with st.chat_message("assistant"):
        st.markdown(response)

    # ChatBotの返答をチャット履歴に追加する
    st.session_state.messages.append({"role": "assistant", "content": response})
