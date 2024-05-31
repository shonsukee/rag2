from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Pineconeの初期化
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

# インデックスの作成
index_name = 'switchbot'
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

# インデックスの初期化
pinecone_index = pc.Index(index_name)

# ベクトルストアとストレージコンテキストの設定
vector_store = PineconeVectorStore(pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ドキュメントの読み込み
documents = SimpleDirectoryReader('../data').load_data()

# インデックスの作成
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
