from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings

llm = ChatGroq(model=settings.GEN_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name=settings.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_PERSIST_DIR,
    collection_metadata={"hnsw:space": "cosine"}, # if we do not configure this, it calculates euclidean distance by default
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})

prompt = ChatPromptTemplate.from_template(
    "Answer the clinical question using the provided context.\n\n"
    "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
)

def run_naive_pipeline(question):
    docs = dense_retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return {
        "answer": response.content.strip(),
        "sources": [{"text": d.page_content, "pmid": d.metadata.get("pmid")} for d in docs],
        "trace": [{"attempt": 0, "query": question, "retrieved": len(docs)}],
    }