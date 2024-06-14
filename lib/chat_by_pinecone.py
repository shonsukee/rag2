# pineconeから直接抽出
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

client = OpenAI()

# Pinecone APIキーの設定と初期化
pc = Pinecone(api_key=os.environ.get("PINECONE_IOT_API_KEY"))

# インデックスの名前
index_name = 'iot-api'

# Pineconeインデックスの作成（既に存在する場合はスキップ）
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Pineconeインデックスに接続
index = pc.Index(index_name)

def get_openai_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

def query_pinecone_index(embedding, top_k=5):
    # Pineconeインデックスにクエリを送信
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )

    return result['matches']

def query_chatgpt_with_vector_db(user_query):
    """クエリに最も関係する文章と元のクエリをChatGPTに与える"""
    embedding = get_openai_embedding(user_query)
    vector_db_response = query_pinecone_index(embedding)

    relevant_info = "\n".join([f"ID: {item['id']}, Score: {item['score']}, Content: {item['metadata']['text']}" for item in vector_db_response])

    combined_query = f"Answer the following question based on the provided information: {user_query} Relevant information: {relevant_info}"

    chatgpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are ChatBot answering questions about SwitchBot API v1.1."},
            {"role": "user", "content": combined_query}
        ]
    )

    return chatgpt_response.choices[0].message['content']

def similarity_search_query_with_score(user_query):
    """クエリに関連した文章(関連度:0.8以上)と元のクエリをChatGPTに与える"""
    embedding = get_openai_embedding(user_query)
    if not embedding:
        return "embedding error", ""

    vector_db_response = query_pinecone_index(embedding)

    for item in vector_db_response:
        print("---------------")
        print("score", item['score'])

    relevant_info = "\n".join([f"ID: {item['id']}, Score: {item['score']}, Content: {item['metadata']}" for item in vector_db_response if item['score'] >= 0.01])

    if not relevant_info:
        return "関連情報が見つかりませんでした", "no data"

    combined_query = f"""
    Answer the following question based on the provided information: {user_query}
    Relevant information: {relevant_info}
    """

    chatgpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are ChatBot answering questions about SwitchBot API."},
            {"role": "user", "content": combined_query}
        ]
    )

    return chatgpt_response.choices[0].message.content, relevant_info

# StreamlitでGUI化
st.title("Modify Switchbot API specifications")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("詳細"):
                st.write(message["expandar_content"])


if prompt := st.chat_input("SwitchBot API v1.1について知りたいことはありますか?"):
	# ユーザの入力をチャット履歴に追加
	st.session_state.messages.append({"role": "user", "content": prompt})

	# ユーザの入力を表示
	with st.chat_message("user"):
		st.markdown(prompt)

	with st.spinner('Wait for it...'):
		response, relevant_info = similarity_search_query_with_score(prompt)

	# ChatBotの返答を表示
	with st.chat_message("assistant"):
		st.markdown(response)
		with st.expander("詳細"):
			st.write(relevant_info)

	# ユーザの入力に関連する情報とChatBotの返答をチャット履歴に追加
	st.session_state.messages.append({"role": "assistant", "content": response, "expandar_content": relevant_info})
