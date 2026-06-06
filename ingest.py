import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCS_DIR   = "./documents"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_documents():
    os.makedirs(DOCS_DIR, exist_ok=True)
    files = os.listdir(DOCS_DIR)
    if not files:
        print("⚠️ No documents found in ./documents folder!")
        return

    documents = []
    for file in files:
        filepath = os.path.join(DOCS_DIR, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"✅ Loaded: {file}")
        elif file.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
            print(f"✅ Loaded: {file}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR, collection_name="rag_documents")
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")

if __name__ == "__main__":
    ingest_documents()