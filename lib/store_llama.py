# Pineconeへインデックスを格納する
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from dotenv import load_dotenv
from pinecone import Pinecone
import os
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
import logging
import sys

pc = None

def initialize_pinecone():
	global pc

	load_dotenv()

	# ログレベルの設定
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
	logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
	pc = Pinecone(api_key=os.environ.get('PINECONE_LLAMA_API_KEY'))

# DBへ入力クエリとLLMからの回答を保存
def insert_query_response_to_db(index_name = "switchbot-llama", input_dir = "../data", chunk_size = 350, chunk_over_lap = 50):
	global pc

	if pc is None:
		initialize_pinecone()

	pinecone_index = pc.Index(index_name)

	# ファイルを読み込んでdocumentsオブジェクトに変換
	documents = SimpleDirectoryReader(
		input_dir=input_dir,
		exclude=["file_path"]
	).load_data()

	Settings.chunk_size = chunk_size
	Settings.chunk_overlap = chunk_over_lap

	vector_store = PineconeVectorStore(
		pinecone_index=pinecone_index,
		add_sparse_vector=True,
	)

	storage_context = StorageContext.from_defaults(vector_store=vector_store)
	# documentsオブジェクトからベクトルインデックスを作成
	# 内部でOpenAI Embedding APIを呼んでる
	VectorStoreIndex.from_documents(
		documents, storage_context=storage_context
	)

def main():
	insert_query_response_to_db()


if __name__ == "__main__":
	main()


# https://zenn.dev/kun432/scraps/81813cf6d4e359