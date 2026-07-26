import os
# from langchain community document loaders module import textloader and directoryloader classes
#both these classes are for reading text files 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# to chunk the loaded source files use charactertextsplitter class
from langchain_text_splitters import CharacterTextSplitter
# now to convert the chunks to embeddings using OpenAIEmbeddings model
from langchain_huggingface import HuggingFaceEmbeddings
# After embeddings process store in vector database which is chroma db here
from langchain_chroma import Chroma
# to load api from env use load_dotenv
from dotenv import load_dotenv

load_dotenv()
 # function to load all the source documents from the data directory
def load_documents(docs_path="data"):
    print(f"Loading the source documnets from {docs_path}...")

#check if the data directory exists else raise error
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists.")

    # now use DIrectoryLoader class from langchain to load all the documents
    loader = DirectoryLoader(
        path=docs_path,  # provide the path
        glob="*.txt",    # tell the class to only look for .txt files
        loader_cls= TextLoader,  # since we are using text files so use langchain textloader class
        loader_kwargs={"encoding": "utf-8"} 
    )

# after loading documnets now envoke load method which gives us a list of langchain documents
    documents = loader.load()

# raise error if no documents
    if len(documents) == 0:
        raise FileNotFoundError(f"No files found in {docs_path}")

    
    for i, doc in enumerate(documents[:2]): #print first two documents
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']} characters")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]} ...")
        print(f"metadata: {doc.metadata}")

    return documents


def split_documents(documents , chunk_size=800, chunk_overlap=0):
    print(f"Splitting documents into chunks")
    # use text_splitter class from langchain to split
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    #see the chunk
    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n-- Chunk {i+1} --")
            print(f"Source: {chunk.metadata["source"]}")
            print(f"Length: {len(chunk.page_content)}characters ")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) >5:
            print(f"\n.. and {len(chunks)-5} more chunks")

    return chunks       #basically langchain documents with smaller size

#function to create embeddings and storing in vector db
def create_vector_db(chunks , persist_directory="E:/RAG/step1_IngestionPipeline/db/chroma_db"):
    print(f"Creating embeddings and storing it in Chromadb...")
    #use small model for embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Now creating teh vector database")
    #create vector database
    vectordatabase = Chroma.from_documents(
        documents=chunks, #take langchain documents in the form of chunks
        embedding = embedding_model, #specify the embedding model
        persist_directory= persist_directory, #storing it locally
        collection_metadata = {"hnsw:space":"cosine"} #the algorithm use cosine similarity
    )

    print("Vector database created")
    print(f"Vector databse created and stored to {persist_directory}")
    return vectordatabase


def main():
    #load all the documents 
    documents = load_documents(docs_path="E:/RAG/data")

    #chunk the documents
    chunks = split_documents(documents)

    # make embeddings and store in the vector database
    vectordb = create_vector_db(chunks)

if __name__ == "__main__":
    main()


