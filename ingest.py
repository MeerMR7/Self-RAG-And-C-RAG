import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_documents():
    print("Loading PDFs from data/ folder...")
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    
    if not docs:
        print("No documents found in data/ folder!")
        return
    
    print(f"Loaded {len(docs)} pages from PDFs.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")
    
    print("Creating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    print("Done! Knowledge base created successfully.")

if __name__ == "__main__":
    ingest_documents()
