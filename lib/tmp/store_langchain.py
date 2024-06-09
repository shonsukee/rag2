# 文章をDBに追加する処理
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings import OpenAIEmbeddings
import cassio
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

# ドキュメントロード
line = ""
with open("SwitchBot-API-v1.1.md", "r") as f:
    while True:
        content = f.readline()
        # ファイルの先頭から一行ずつ読み込む
        if content:
            line += content
        else:
            break

# ドキュメントを1文ずつに分割
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(line)

# DB/LLM等初期化
cassio.init(token=os.environ["ASTRA_DB_APPLICATION_TOKEN"], database_id=os.environ["ASTRA_DB_ID"])
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="switchbot_table", # 作成したいテーブル名に変更
    session=None,
    keyspace=None
)

# ベクトルDBにテキスト追加
astra_vector_store.add_texts(texts)
print("Inserted %i headlines." % len(texts))