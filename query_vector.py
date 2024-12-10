import os

from dotenv import load_dotenv
from constants import RED, YELLOW, GREEN, BLUE, MAGENTA, CYAN, RESET # Import constants for color-coding text

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import  ChatOllama

# Load environment vars (fron .env file)
load_dotenv()

# Define persistent path for db/vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db", "chroma_db_docs")

# Define the embedding model
# Make sure you use the same embeddings used to populate the vector store (vectorize.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

# Load the existing vector store with embedding function
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity", # Other options: ....
    search_kwargs={"k": 1}, # return only one result
)

# Create a ChatModel (Ollama or OpenAI)
#llm = ChatOllama(model="llama3.2")
llm = ChatOpenAI(model="gpt-4o-mini")

# Here's where it starts to get interesting
# Contextualize question prompt
contextualize_q_system_prompt = (
    "given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT anser the question, jsut "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Answer question promt
qa_system_prompt = (
    "You are an assistant for question-answer tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use five sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questios
qa_prompt = ChatPromptTemplate(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}") 
    ]
)

# Create a chain to combine documents for question answering
# 'create_stuff_documents_chain' feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to simulate a continual conversation
def continual_chat():
    print(f"{YELLOW}What would you konw to know about your documents? Type 'exit' to end the conversation.{RESET}")
    chat_history = [] # Collect chat history (sequence of messages)
    while True:
        query = input(f"{RED}You: {RESET}")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"{GREEN}AI:{RESET} {result['answer']} ")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result['answer']))


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
