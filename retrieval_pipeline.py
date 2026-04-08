from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# database access
db = Chroma(
    persist_directory= persistent_directory,
    embedding_function = embedding_model,
    collection_metadata={"smsw:space" : "cosine"},
)

# sample query
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs = {"k":5}) #k = 5 means retriev the top highest reaults with higher cosine similarity

relevant_docs = retriever.invoke(query)

print(f"User Query : {query}")

print("--------context---------")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")

