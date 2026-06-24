"""
dataset_loader.py
-----------------
Loads the TruthfulQA benchmark dataset from HuggingFace and returns
a clean, consistent list of question dicts for the pipeline to use.

TruthfulQA is a dataset of 817 questions designed to test whether
language models generate truthful answers. Many questions are tricky
because the "obvious" answer is actually a common misconception.

Dataset card: https://huggingface.co/datasets/truthful_qa
"""

from datasets import load_dataset
from config import NUM_SAMPLES


def load_truthfulqa(n_samples: int = NUM_SAMPLES) -> list[dict]:
    """
    Load the TruthfulQA dataset and return a clean sample.

    We use the 'generation' split (as opposed to 'multiple_choice')
    because it gives us free-text correct answers, which is what we
    need to compare against model responses for scoring.

    Each returned dict has exactly three keys so every downstream
    module gets a consistent, predictable structure:
      - question    : the prompt we send to each LLM
      - best_answer : the single best correct answer (used for scoring)
      - category    : topic area e.g. "Science", "History", "Myths"

    Args:
        n_samples: How many questions to return. Defaults to NUM_SAMPLES
                   from config.py (50). Keep this small while testing.

    Returns:
        List of dicts, each with keys: question, best_answer, category.

    Raises:
        ValueError: If n_samples is larger than the dataset size.
    """
    print(f"📥 Loading TruthfulQA dataset ({n_samples} samples)...")

    # Load only the validation split — TruthfulQA has no train split
    # for the generation task. trust_remote_code=False is safer.
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    total_available = len(dataset)
    print(f"✅ Dataset loaded. Total available questions: {total_available}")

    if n_samples > total_available:
        raise ValueError(
            f"Requested {n_samples} samples but dataset only has "
            f"{total_available} questions."
        )

    # Slice to the requested number of samples
    dataset = dataset.select(range(n_samples))

    # Flatten into a simple list of dicts.
    # The raw dataset has many fields we don't need (incorrect answers,
    # source URLs, etc.) — we keep only what the pipeline uses.
    samples = []
    for row in dataset:
        samples.append({
            "question":    row["question"],
            "best_answer": row["best_answer"],   # single best correct answer
            "category":    row["category"],
        })

    print(f"✅ Returning {len(samples)} clean samples.\n")
    return samples


if __name__ == "__main__":
    # ── Quick test ────────────────────────────────────────────────────
    # Run this file directly to verify the dataset loads correctly:
    #   python dataset_loader.py
    # You should see the first 3 questions printed cleanly.
    # -----------------------------------------------------------------
    samples = load_truthfulqa(n_samples=50)

    print("=" * 60)
    print("FIRST 3 SAMPLES")
    print("=" * 60)
    for i, sample in enumerate(samples[:3], start=1):
        print(f"\n[{i}] Category : {sample['category']}")
        print(f"    Question : {sample['question']}")
        print(f"    Answer   : {sample['best_answer']}")

    print("\n" + "=" * 60)
    print(f"Total samples loaded: {len(samples)}")
    print("=" * 60)