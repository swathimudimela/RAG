import os
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

db = Chroma(
    persist_directory= persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"smsw:space":"cosine"},
)

query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs = {"k":5})

relevant_docs = retriever.invoke(query)

print(f"User Query : {query}")

print("--------context---------")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document{i}:\n{doc.page_content}\n")

# combine the query and the relevant document contents. The documents are formated as 
#   -Document1 PageContent
#   -Document2 PageContent
#   - 
combined_input = f""" Based on the following documents, Please answer this question : {query}
Documents : 
    {chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear , helpful anser using only the information from these documents, also cite the document number you found the answer from (e.g.,[Document 1]). If you can't find the answer in the documents, say "I dont have enough data to answer thar question.
"""

# Now the augmented data is passed to the LLM model and the answer is generated for the query
# for the LLM we are using phi3.5 model

# connect to the hugging face
hf_token = os.getenv("HF-Token")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

model_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id= model_id,
    task = "text-generation",
    max_new_tokens = 512,
    huggingfacehub_api_token=hf_token,
)

chat_model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = "You are a helpful assistant."),
    HumanMessage(content = combined_input),
]

response = chat_model.invoke(messages)

print(f"-------Generated response---------")
print(response.content)
