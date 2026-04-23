# With just vector based search there is chance that user might not what he wants 
# beacause the vector based search uses embeddings which group all the similar words together
#With keyword Search, if a user asks about something though the document has listed it in different word, because the word is not similar to what the user has asked for it will not retireve any chunks
# so it is always better to use both the searches with which performance of the rag system will increase

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()
# connect to the hugging face
hf_token = os.getenv("HF-Token")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

#Creating small chunks of data for easier execution
chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.", 
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.", 
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]

# vector db only accepts langchain_docs so converting the chunks to langchain_docs
# Convert to Document objects for LangChain
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

print("Sample Data:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

print("\n" + "="*80)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

print("Vector Retriver setup done")

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Test semantic search
test_query = "space exploration company" #works in vector search but wouldn't work with keyword search

print(f"Testing: '{test_query}'")
test_docs = vector_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

# Keyword search retriever
print("Setting up BM25 Retriever...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3
test_query = "Cybertruck"

print(f"Testing: '{test_query}'")
test_docs = bm25_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

# Hybrud search retriever => both vector search(70%) and keyword seearch(30%)
#  3. Hybrid Retriever (Combination)
print("Setting up Hybrid Retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Equal weight to vector and keyword search
)

print("Setup complete!\n")

test_query = "purchase cost 7.5 billion"

retrieved_chunks = hybrid_retriever.invoke(test_query)
for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")
print()

print("Query 1 shows how hybrid finds exact financial info using both semantic understanding and keyword matching")

# Query 2: Semantic concept + specific product name  

# Vector search understands "electric vehicle manufacturing"
# BM25 search finds exact "Cybertruck"
# Hybrid gets the best of both worlds

test_query = "electric vehicle manufacturing Cybertruck"

retrieved_chunks = hybrid_retriever.invoke(test_query)

for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")
print()

print("Query 2 demonstrates combining product-specific terms with broader concepts")

# Query 3: Where neither alone would be perfect

# "Company performance" is semantic, "Tesla" is exact keyword
# Hybrid should find the most relevant Tesla performance info

test_query = "company performance Tesla"

retrieved_chunks = hybrid_retriever.invoke(test_query)
for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")


print("Query 3 shows how hybrid handles mixed semantic/keyword queries better than either approach alone")