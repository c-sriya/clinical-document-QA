import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    GEN_MODEL = os.getenv("GEN_MODEL", "llama-3.3-70b-versatile")
    GRADE_MODEL = os.getenv("GRADE_MODEL", "llama-3.1-8b-instant")
    EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-1.5-flash")

    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = "pubmed_abstracts"

    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
    TOP_K = int(os.getenv("TOP_K", 5))

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, CPU-friendly

settings = Settings()