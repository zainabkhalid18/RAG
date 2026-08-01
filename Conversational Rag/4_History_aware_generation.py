from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

#connect to the database
persistant_directory = "E:/RAG/step1_IngestionPipeline/db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embeddings
)

model = ChatGroq(model="llama-3.1-8b-instant")

#store all the conversation as history
chat_history = []

def ask_question(user_question):
    print(f"\n User Question: {user_question}---")
#make the question clear using conversation history
    if chat_history:  #if it is empty then no need to do the reformulation and go to the else block

        messages = [
            SystemMessage(content="Given the Chat history. rewrite the new asked question to be standalone and searchable.Just return the rewritten question")
        ] + chat_history + [
            HumanMessage(content=f"New Question: {search_question}")
        ]

    else:
        search_question = user_question

    #step 2 Find the relevant documents/chunks

    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents")
    for i , doc in enumerate(docs ,1):

        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f" Doc{i}: {preview}...")

    #step3: Create Final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\\n".join(f"- {doc.page_content}" for doc in docs)}

    Please provide a clear, helpful answer using only the information from these documents, If you can't find the answer then say it
    """

    #step 4 get the answer

    messages = [
        SystemMessage(content="You are a helpfull assistant that answers questions based on provided documents and convversations")
    ] + chat_history + [
        HumanMessage(content= combined_input)
    ]

    result= model.invoke(messages)
    answer = result.content

    #step 5: Remember the conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer



def start_chat():
    print("Ask me a question. If you dont have any question type 'quit' to exit ")

    while True:
        question = input("\n Your question")

        if question.lower()=='quit':
             print("Goddbye!")
             break
    
        ask_question(question)
        

if __name__=="__main__":
    start_chat()