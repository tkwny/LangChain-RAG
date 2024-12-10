import os
import glob

from dotenv import load_dotenv
from constants import RED, YELLOW, GREEN, BLUE, MAGENTA, CYAN, RESET # Import constants for color-coding text
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Define the directory that contains the text (.txt) files and the persistent (vector db) directory
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(current_dir, "docs")
db_dir = os.path.join(current_dir, "db", "chroma_db_docs")

# Check if the vector store already exists
if not os.path.exists(db_dir):
    print(f"{RED}The DB directory does not exist{RESET}. {GREEN}Initializing vector store...{RESET}")

    # Ensure the docs directory exists
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(
            f"The directory {docs_dir} does not exist. Please check the path."
        )
    
    # Find all text files in the docs directory
    text_files = glob.glob(os.path.join(docs_dir, "*.txt"))
    if not text_files:
        raise FileNotFoundError(
            f"No text (*.txt) files found in the directory {docs_dir}."
        )
    
    all_docs = []
    for file_path in text_files:
        # Read the text content from the file
        loader = TextLoader(file_path)
        documents = loader.load()
        all_docs.extend(documents)

    # Split the text content from the file
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print(f"{MAGENTA}\n--- Document Chunks Inforamiton ---{RESET}")
    print(f"Number of document chunks: {YELLOW}{len(docs)}{RESET}")
    print(f"Sample chunk:{BLUE}\n{docs[0].page_content}\n{RESET}")

    # Create embeddings
    print(f"{GREEN}\n--- Creating embeddings ---{RESET}")
    embeddings = OpenAIEmbeddings(
        # Update to your preferred embedding model if needed
        model="text-embedding-3-small" 
    ) 
    print(f"{GREEN}\n--- Finished creating embeddings ---{RESET}")

    # Create vector store (persists automatically)
    print(f"{CYAN}\n--- Creating vector store (chromadb) ---")
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)
    print(f"\n--- Finished creating vector store ---{RESET}")

else:
    print(f"{BLUE}Vector stora already exists. No need to initialize.{RESET}")



