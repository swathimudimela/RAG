# conversation based RAG . The RAG has to be aware of the context of the conversation because as we 
# chat in the conversatins we start using pronouns instead of using the names, so the system has
# to be aware of what the client is talking about

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

load_dotenv()

# loading vector db
persistent_directory = "db/chroma_db"

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

db = Chroma(
    persist_directory= persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"},
)

# initialize LLM, We will use huggingFaceEndpoint to use the LLM that is already hosted by huggingFace instead of loading the model locally
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

chat_history = []

def ask_question(user_question):
    print(f"\n ---- You Asked :{user_question}-----")

    # Make the question clear using the conversation history
    if chat_history:
        # ask the ai to make the question standalone
        messages = [
            SystemMessage(content = "Given the chat history , rewrite the question to be standalone and searchable.Just return the rewritten question"),
        ] + chat_history+[
            HumanMessage(content = f"New Question: {user_question}")
        ]

        result = chat_model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for {search_question}")
    else:
        search_question = user_question

    # Now search RAG for the answer
    retriver = db.as_retriever(search_kwargs = {"k":3})
    docs = retriver.invoke(search_question)

    # now pass this info to the llm and get the answer
    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        # just show the 2 lines of the retrived docs
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f" Doc {i} : {preview}....")

    combined_input = f"""Based on the following documents, please answer this question:{user_question}

    Documents :
    {"\n".join([f"-{doc.page_content} for doc in docs"])}

    Please provide a clear, helpful answer using only the information from these documents. If you cant find the answer , say "I dont have enough information"
    """

    # get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers question based on the provided documents and conversation history"),
    ]+chat_history+[
        HumanMessage(content=combined_input)
    ]

    result = chat_model.invoke(messages)
    answer = result.content

    # update the chat history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content = answer))

    print(f"Answer : {answer}")

    return answer

def start_chat():
    print("Ask me a question! Type quit to exit")
    
    while(True):
        question = input("\n Your Question:")

        if question.lower()=='quit':
            print('GoodBye. Have a great rest of the day')
            break
        else:
           answer =  ask_question(question)

if __name__ == "__main__":
    start_chat()