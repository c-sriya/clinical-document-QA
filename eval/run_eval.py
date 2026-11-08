import json
import argparse
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

PIPELINES = {
    "naive": "src.pipeline_naive:run_naive_pipeline",
    "hybrid": "src.pipeline_hybrid:run_hybrid_pipeline",
    "crag": "src.pipeline_graph:run_pipeline",
}

ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
)

def load_pipeline_fn(key):
    module_path, fn_name = PIPELINES[key].split(":")
    module = __import__(module_path, fromlist=[fn_name])
    return getattr(module, fn_name)

def build_eval_dataset(pipeline_fn):
    with open("eval/qa_pairs.json") as f:
        qa_pairs = json.load(f)
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    raw_answers = []
    for pair in qa_pairs:
        result = pipeline_fn(pair["question"])
        rows["question"].append(pair["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([s["text"] for s in result["sources"]] or ["(no context retrieved)"])
        rows["ground_truth"].append(pair["ground_truth"])
        raw_answers.append({"question": pair["question"], "answer": result["answer"],
                             "ground_truth": pair["ground_truth"]})
    return Dataset.from_dict(rows), raw_answers

def judge_irrelevance(raw_answers, judge_llm):
    irrelevant_count = 0
    tagged = []
    for item in raw_answers:
        prompt = f"""Question: {item['question']}
Expected answer covers: {item['ground_truth']}
Given answer: {item['answer']}

Does the given answer directly and correctly address the question, consistent
with the expected answer? Respond with exactly one word: "relevant" or "irrelevant"."""
        verdict = judge_llm.invoke(prompt).content.strip().lower()
        is_irrelevant = "irrelevant" in verdict
        irrelevant_count += int(is_irrelevant)
        tagged.append({**item, "verdict": verdict})
    return irrelevant_count / len(raw_answers) * 100, tagged

def run_eval(version_tag, pipeline_key):
    pipeline_fn = load_pipeline_fn(pipeline_key)
    dataset, raw_answers = build_eval_dataset(pipeline_fn)

    judge_llm = ChatGoogleGenerativeAI(model=settings.EVAL_JUDGE_MODEL, google_api_key=settings.GEMINI_API_KEY)
    ragas_judge = LangchainLLMWrapper(judge_llm)

    ragas_scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=ragas_judge,
        embeddings=ragas_embeddings,
    )
    irrelevant_pct, tagged = judge_irrelevance(raw_answers, judge_llm)

    out = {
        "version": version_tag,
        "pipeline": pipeline_key,
        "timestamp": datetime.utcnow().isoformat(),
        "ragas_scores": ragas_scores.to_pandas().mean(numeric_only=True).to_dict(),
        "irrelevant_pct": irrelevant_pct,
    }
    Path("eval/results").mkdir(exist_ok=True)
    with open(f"eval/results/{version_tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(f"eval/results/{version_tag}_tagged_answers.json", "w") as f:
        json.dump(tagged, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version_tag")
    parser.add_argument("--pipeline", required=True, choices=PIPELINES.keys())
    args = parser.parse_args()
    run_eval(args.version_tag, args.pipeline)




