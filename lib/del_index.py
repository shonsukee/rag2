import os
from dotenv import load_dotenv
from pinecone import Pinecone

HISTORY_INDEX_NAME="revision-history"

load_dotenv()

def del_index():
	pc = Pinecone(api_key=os.environ.get("PINECONE_IOT_API_KEY"))
	pc.delete_index(HISTORY_INDEX_NAME)

def main():
	del_index()

if __name__ == "__main__":
	main()
