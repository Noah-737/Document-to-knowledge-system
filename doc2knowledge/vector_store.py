import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_and_store_embeddings(chunks: list[str], unique_id: str):
    """
    Creates embeddings for the given chunks and stores them in a FAISS vector store.

    Args:
        chunks: A list of text chunks.
        unique_id: A unique identifier for the document.
    """
    if not os.path.exists('./db'):
        os.makedirs('./db')

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vector_store.save_local(os.path.join('./db', f"{unique_id}.faiss"))
