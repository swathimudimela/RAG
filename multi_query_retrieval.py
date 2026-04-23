# MutliQueryRetrieval improves the performance of the RAG System
# The logic is simple , for a given query it generates variations to it and the chunks are retrived
# Now using Reciprocal Rank Fusion(RRF), the retrived chunks are processed and top n chunks are selected to pass it to LLM

import os
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from collections import defaultdict

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks


load_dotenv()
# connect to the hugging face
hf_token = os.getenv("HF-Token")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

persistent_directory = "db/chroma_db"
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# get db access
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)


model_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id= model_id,
    task = "conversational",
    max_new_tokens = 512,
    huggingfacehub_api_token=hf_token,
)

chat_model = ChatHuggingFace(llm=llm)


# 2. Define Structured Output Schema
class QueryVariations(BaseModel):
    queries: List[str] 

parser = JsonOutputParser(pydantic_object=QueryVariations)

# 3. Multi-Query Prompt
template = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.

Original question: {question}

{format_instructions}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

chain = prompt | chat_model | parser
query_variations = chain.invoke({"question": original_query})["queries"]

print(f"Generated Query Variations:")
for i, variation in enumerate(query_variations,1):
    print(f"{i} : {variation}")

print("\n"+"-"*60)

# search db for each query variation and store results
retriever = db.as_retriever(search_kwargs={"k":5})
all_retrieved_results = []

for i, query in enumerate(query_variations,1):
    print(f"\n------ Results for Query {i} : {query}--------")
    docs = retriever.invoke(query)
    all_retrieved_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")
    for j, doc in enumerate(docs,1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:100]}....\n")
     
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")

fused_results = reciprocal_rank_fusion(all_retrieved_results, k=60, verbose=True)

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f" RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")