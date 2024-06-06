import streamlit as st
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# 環境変数のロード
load_dotenv()

client = OpenAI()

# Pineconeの初期化
pc = Pinecone(api_key=os.environ.get("PINECONE_IOT_API_KEY"))

# Pineconeインデックスの設定
index_name = 'iot-api'
pinecone_index = pc.Index(index_name)

# Pinecone VectorStoreの初期化
storage_context = StorageContext()
pinecone_store = PineconeVectorStore(pinecone_index=pinecone_index)
vector_store_index = VectorStoreIndex(vector_store=pinecone_store, storage_context=storage_context)

def search_information(query):
    # クエリをベクトル化
    # embedding = get_embedding(query)
    embedding = "okkk"

    # Pineconeで検索
    results = vector_store_index.search(query_vector=embedding, top_k=10)
    return results

# Streamlit UI
st.title('情報検索アプリケーション')

user_query = st.text_input("質問を入力してください:", "")
if user_query:
    st.write("検索結果:")
    results = search_information(user_query)
    for result in results:
        st.write(result)  # 結果を表示
