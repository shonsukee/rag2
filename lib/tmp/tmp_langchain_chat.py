# DBに格納されている情報を付随させて質問していく
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import OpenAIEmbeddings
import cassio
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import numpy as np

load_dotenv()

cassio.init(token=os.environ["ASTRA_DB_APPLICATION_TOKEN"], database_id=os.environ["ASTRA_DB_ID"])

def query_chatgpt_with_vector_db(user_query):
    """クエリに最も関係する文章と元のクエリをChatGPTに与える"""
    # ユーザーのクエリに基づいてベクトルDBから情報を検索
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    vector_db_response = astra_vector_index.query(user_query).strip()

    # 検索結果とユーザーのクエリを組み合わせてChatGPTに問い合わせるクエリを生成
    combined_query = f"Answer the following question based on the provided information: {user_query} Relevant information: {vector_db_response}"

    chatgpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "You are ChatBot answering questions about SwitchBot API v1.1."},
                  {"role": "user", "content": combined_query}]
    )

    return chatgpt_response.choices[0].message.content

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
    query_embed = get_openai_embedding(user_query)

    similarities = []
    vector_db_response = ""
    for doc, score in astra_vector_store.similarity_search_with_score(user_query, k=4):
        ## TODO: 要削除！！！！！
        print("-----------score------------")
        print("score: ", score)
        print("page: ", doc.page_content)
        print("-----------------------")
        doc_embedding = get_openai_embedding(doc.page_content)
        similarity = cosine_similarity(query_embed, doc_embedding)

        print("--averageeee---------------------")
        print(similarity)
        print("-----------------------")
        if score >= 0.8:
            similarities.append(similarity)
            vector_db_response += doc.page_content + "¥n"

    average_similarity = np.mean(similarities)

    if vector_db_response == "":
        return "関連情報が見つかりませんでした", "", average_similarity

    combined_query = f"""
    Answer the following question based on the provided information: {user_query}
    Relevant information: {vector_db_response}
    """

    chatgpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "You are ChatBot answering questions about SwitchBot API."},
                  {"role": "user", "content": combined_query}]
    )
    return chatgpt_response.choices[0].message.content, vector_db_response, average_similarity

client = OpenAI()
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
astra_vector_store = Cassandra(
    session=None,
    embedding=embedding,
    table_name="switchbot_demo",
    keyspace=None,
)

# StreamlitでGUI化
st.title("関連するインデックスの抽出")

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
