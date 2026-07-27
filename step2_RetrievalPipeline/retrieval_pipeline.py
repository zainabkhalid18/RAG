# import the chroma database
from langchain_chroma import Chroma
#import hugging face embedding model
from langchain_huggingface import HuggingFaceEmbeddings

#load the vector database
persistant_directory = "E:/RAG/step1_IngestionPipeline/db/chroma_db"

#load the embedding model . Use the same model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma( 
    persist_directory= persistant_directory,
    embedding_function= embedding_model,
    collection_metadata= {"hnsw:space": "cosine"}
)

query = "What island does SpaceX lease for it slaunches in the Pacific?"

#now use this db as a retriever and retrieve top 5 chunks with the highest similarity scores to the user query embedding
retriever = db.as_retriever(search_kwargs = {"k":5})

#invoke the retriever and pass the query to get top 5 chunks
relevant_documents = retriever.invoke(query)

print(f"User Query: {query}")

print("Context")

#print results
for i, docs in enumerate(relevant_documents,1):
    print(f"Document {i}:\n{docs.page_content}\n")




