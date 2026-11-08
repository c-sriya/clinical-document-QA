import json
import random

random.seed(42)
rows = [json.loads(l) for l in open("data/raw/pubmed_raw.jsonl")]
sample = random.sample(rows, 50)

qa_pairs = [
    {"question": r["question"], "ground_truth": r.get("long_answer", r.get("final_decision", ""))}
    for r in sample
]

with open("eval/qa_pairs.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)