import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import pickle
from src.config import settings

def load_chunks():
    with open("data/processed/chunks.jsonl") as f:
        return [json.loads(line) for line in f]

def to_documents(chunks):
    return [
        Document(page_content=c["text"], metadata={"pmid": c["pmid"], "id": c["id"]})
        for c in chunks
    ]

def build_chroma(documents):
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    print(f"Indexed {len(documents)} chunks into Chroma")
    return vectorstore

def build_bm25(documents):
    # BM25Retriever builds its index in memory from Documents; we pickle the
    # Document list so retrieval.py can rebuild the same retriever without re-ingesting
    with open("chroma_db/bm25_documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    print("Saved documents for BM25Retriever")

if __name__ == "__main__":
    chunks = load_chunks()
    documents = to_documents(chunks)
    build_chroma(documents)
    build_bm25(documents)