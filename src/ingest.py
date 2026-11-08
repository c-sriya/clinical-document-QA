import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def download_pubmedqa():
    """Downloads the full pqa_labeled split (~1,000 rows, expert-labeled)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ds.to_json(RAW_DIR / "pubmed_raw.jsonl")
    return ds

def chunk_abstract(text, chunk_size=500, overlap=50):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

def process():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ds = download_pubmedqa()
    records = []
    for i, row in enumerate(tqdm(ds, desc="Chunking")):
        context = " ".join(row["context"]["contexts"]) if row.get("context") else ""
        if not context:
            continue
        for j, chunk in enumerate(chunk_abstract(context)):
            records.append({
                "id": f"doc_{i}_chunk_{j}",
                "text": chunk,
                "source_question": row.get("question", ""),
                "pmid": row.get("pubid", f"unknown_{i}"),
            })
    out_path = PROCESSED_DIR / "chunks.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} chunks to {out_path}")

if __name__ == "__main__":
    process()