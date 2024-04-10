# 初期化から質問までCUIで全て行う
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
import os
from dotenv import load_dotenv

load_dotenv()

# DB/LLM等初期化
cassio.init(token=os.environ["ASTRA_DB_APPLICATION_TOKEN"], database_id=os.environ["ASTRA_DB_ID"])
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])
embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="switchbot_demo",
    session=None,
    keyspace=None
)
print("initialize DB/LLM")
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

first_question = True
while True:
    if first_question:
        query = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query.lower() == "quit":
        break

    if query == "":
        continue

    first_question = False

    print("\nQUESTION: \"%s\"\n" % query)

    # クエリに直接関係する回答生成
    answer = astra_vector_index.query(query, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    # クエリ似関係するdocの上位4箇所(k=4)を抽出して，先頭84文字を出力
    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query, k=4):
        print(" 関連性: [%0.4f]  文章: \"%s ...\"\n\n" % (score, doc.page_content))