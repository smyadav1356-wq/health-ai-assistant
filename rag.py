import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_file = "data/nutrition.pdf"
DB_FOLDER = "vector_db"

def create_rag():
    if not os.path.exists(PDF_file):
        raise FileNotFoundError(f"PDF file nahi mila: {PDF_file} - Is folder me rakho: {os.getcwd()}")

    print("PDF load ho raha hai...")
    documents = PyPDFLoader(PDF_file).load()

    print(f"Total pages: {len(documents)}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks bane: {len(chunks)}")

    print("Embedding ban raha hai, thoda time lagega...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embedding)
    db.save_local(DB_FOLDER)
    print("RAG database ban gaya! vector_db folder me save ho gaya.")

def load_rag():
    if not os.path.exists(DB_FOLDER):
        raise FileNotFoundError("vector_db nahi mila, pehle create_rag() chalao")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FOLDER,
        embedding,
        allow_dangerous_deserialization=True
    )
    return db

