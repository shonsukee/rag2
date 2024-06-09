from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from dotenv import load_dotenv
from pinecone import Pinecone
import os
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
import logging
import sys

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

pc = Pinecone(api_key=os.environ.get('PINECONE_LLAMA_API_KEY'))
pinecone_index = pc.Index("switchbot-llama")

# ファイルを読み込んでdocumentsオブジェクトに変換
documents = SimpleDirectoryReader(
    input_dir="../../data",
    excluded_llm_metadata_keys=["file_path"]
).load_data()

Settings.chunk_size = 350
Settings.chunk_overlap = 50

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
# documentsオブジェクトからベクトルインデックスを作成
# 内部でOpenAI Embedding APIを呼んでる
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# https://zenn.dev/kun432/scraps/81813cf6d4e359