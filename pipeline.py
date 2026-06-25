"""
pipeline.py
-----------
Orchestrates the full LLM evaluation pipeline.

For each question in the dataset:
  1. Calls all 3 models to get responses
  2. Runs all 4 evaluators on each response
  3. Stores everything in a results list
  4. Saves to results/results.json when done

Run with: python pipeline.py
To test with fewer questions first: python pipeline.py --samples 5
"""

import json
import time
import os
import argparse
from tqdm import tqdm

from config import (
    ALL_MODELS,
    MODEL_DISPLAY_NAMES,
    API_SLEEP_SECONDS,
    RESULTS_PATH,
    NUM_SAMPLES,
    validate_keys,
)
from dataset_loader import load_truthfulqa
from models.groq_model import generate
from evaluators.accuracy import score_accuracy
from evaluators.faithfulness import score_faithfulness
from evaluators.hallucination import score_hallucination
from evaluators.latency_cost import calculate_cost


def run_pipeline(n_samples: int = NUM_SAMPLES) -> list[dict]:
    """
    Run the full evaluation pipeline on n_samples questions.

    For each question, calls every model and scores every response.
    Handles individual failures gracefully — if one model fails on
    one question, logs it and continues rather than crashing.

    Args:
        n_samples: Number of TruthfulQA questions to evaluate.

    Returns:
        List of result dicts, one per question, each containing
        the question metadata and all model responses + scores.
    """
    validate_keys()

    dataset = load_truthfulqa(n_samples)
    results = []

    print(f"\n🚀 Starting evaluation pipeline")
    print(f"   Questions : {n_samples}")
    print(f"   Models    : {[MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS]}")
    print(f"   Metrics   : accuracy, faithfulness, hallucination, latency, cost\n")

    for item in tqdm(dataset, desc="Evaluating questions"):
        question     = item["question"]
        best_answer  = item["best_answer"]
        category     = item["category"]

        question_result = {
            "question":    question,
            "best_answer": best_answer,
            "category":    category,
            "models":      {}
        }

        for model_id in ALL_MODELS:
            display_name = MODEL_DISPLAY_NAMES[model_id]
            tqdm.write(f"  → {display_name}")

            try:
                # ── Step 1: Get model response ────────────────────────────
                model_output = generate(question, model=model_id)
                response     = model_output["response"]
                latency      = model_output["latency_seconds"]
                input_tok    = model_output["input_tokens"]
                output_tok   = model_output["output_tokens"]

                # ── Step 2: Run all evaluators ────────────────────────────
                accuracy     = score_accuracy(response, best_answer)
                faithfulness = score_faithfulness(response, best_answer)
                hallucination= score_hallucination(response, best_answer, question)
                cost         = calculate_cost(model_id, input_tok, output_tok)

                question_result["models"][display_name] = {
                    "response":          response,
                    "latency_seconds":   latency,
                    "input_tokens":      input_tok,
                    "output_tokens":     output_tok,
                    "accuracy":          accuracy,
                    "faithfulness":      faithfulness,
                    "hallucination":     hallucination,
                    "cost_usd":          cost,
                }

            except Exception as e:
                # Log failure but continue — one bad call shouldn't stop everything
                tqdm.write(f"  ❌ {display_name} failed on this question: {e}")
                question_result["models"][display_name] = {
                    "response":        "ERROR",
                    "latency_seconds": 0,
                    "input_tokens":    0,
                    "output_tokens":   0,
                    "accuracy":        0.0,
                    "faithfulness":    0.0,
                    "hallucination":   1.0,
                    "cost_usd":        0.0,
                }

            # ── Rate limit protection ─────────────────────────────────────
            time.sleep(API_SLEEP_SECONDS)

        results.append(question_result)

    return results


def save_results(results: list[dict]) -> None:
    """
    Save the results list to a JSON file.

    Creates the results/ directory if it doesn't exist.

    Args:
        results: The list of result dicts from run_pipeline().
    """
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {RESULTS_PATH}")
    print(f"   Total questions evaluated: {len(results)}")


def print_summary(results: list[dict]) -> None:
    """
    Print a quick summary table of average scores per model.

    Args:
        results: The list of result dicts from run_pipeline().
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for model_name in MODEL_DISPLAY_NAMES.values():
        scores = {
            "accuracy":      [],
            "faithfulness":  [],
            "hallucination": [],
            "latency":       [],
        }

        for item in results:
            if model_name in item["models"]:
                m = item["models"][model_name]
                scores["accuracy"].append(m["accuracy"])
                scores["faithfulness"].append(m["faithfulness"])
                scores["hallucination"].append(m["hallucination"])
                scores["latency"].append(m["latency_seconds"])

        if not scores["accuracy"]:
            continue

        avg = lambda lst: round(sum(lst) / len(lst), 4)

        print(f"\n{model_name}")
        print(f"  Accuracy      : {avg(scores['accuracy'])}")
        print(f"  Faithfulness  : {avg(scores['faithfulness'])}")
        print(f"  Hallucination : {avg(scores['hallucination'])}")
        print(f"  Avg Latency   : {avg(scores['latency'])}s")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of questions to evaluate (default: {NUM_SAMPLES})"
    )
    args = parser.parse_args()

    results = run_pipeline(n_samples=args.samples)
    save_results(results)
    print_summary(results)