import os
import glob
import time
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, JSONLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"

def load_documents() -> List:
    documents = []
    # Support .txt, .pdf, .docx, .json, .csv
    for file_path in glob.glob(os.path.join(DATA_DIR, "*")):
        try:
            if file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith(".json"):
                loader = JSONLoader(file_path, jq_schema='.', text_content=False)
                documents.extend(loader.load())
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return documents

import shutil

def ingest_data():
    if os.path.exists(DB_DIR):
        print(f"Removing existing vector store at {DB_DIR}...")
        shutil.rmtree(DB_DIR)
        
    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("No documents found in 'data/' directory. Please add your manuscript files.")
        return

    print(f"Loaded {len(documents)} documents. Splitting text...")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Create Vector Store
    print("Creating vector store with Local Embeddings (this may take a moment to download the model first)...")
    
    # Use a multilingual local model, force CPU to avoid MPS/Meta tensor errors
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'}
    )
    
    # Initialize Chroma
    # With local embeddings, we don't need to worry about rate limits, so we can process faster.
    # However, passing all at once might spike memory, so we'll still batch but with larger size and no sleep.
    
    batch_size = 100
    total_chunks = len(chunks)
    
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    print(f"Starting ingestion of {total_chunks} chunks...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
            print(f"Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")

    print(f"Vector store updated in '{DB_DIR}'.")

if __name__ == "__main__":
    ingest_data()
