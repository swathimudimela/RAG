RAG : Retrival Augmented Generation, used when we need LLM to answer the query's based on the heavy documentation that is provided to LLM. 
It is expensive to use an LLM to process heavy documents and answer the user queries. Instead with RAG we can can upload the documents, RAG then processes the document into meaningful smaller chunks and store them in vector
database. Now when the user query's something related to the doument, the RAG will search the document to find the chunks that has information similar to the query(we used cosine similarity to retrive the chunk with the info similar to the query)
and passes these chunks to the LLM . Now LLM can easily process the chunks and generate the answer to the user_query.

Here in this is repository we seperated individual modules of the RAG .
 1. Ingestion Pipeline : Takes documents as inputs, process them into chunks, creates a vector database and stores into it
 2. Retrieval pipeline : When the user asks a query, this retrieval pipeline is activated , it goes to the vector db and retrives all teh chunks that has info related to the query
 3. Answer_generation : The chunks retrievd are then passed to the LLM, which the LLM will process and ANSERS THE USER_QUERY.
 4. history_aware_generation: This supports chat based back and forth question answering, RAG keeps track of what the user is talking about from the pevious query's , this way the user dont have to always specify what is talking about.

MultiModalRag : Documents dont always have to be simple texts, documents can have images, tables as well. Multi Modal Rag is used for such document processing. 
  Steps in MultiModalRag : 
      1. The documents are processed usign the unstructed library , which divides document into atomic elements like, header, footer, table, text, formula, images, title. 
      2. These atomic elements are then grouped into chunks which are composite elemts => can contain images, tables, text etc.
      3. We then pass these chunks to the LLM to summarize the text, images, tables so we get the chunks of searchable and readable of data, but we make sure we store the original image and data as metadata 
      4. These summarized chunks are then stored in the vector DB
      5. Then the rest of the pipeline is similar to the regular RAG
