import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    print("Done! Knowledge base created successfully.")

if __name__ == "__main__":
    ingest_documents()
