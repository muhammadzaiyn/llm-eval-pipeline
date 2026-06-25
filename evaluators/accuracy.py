"""
evaluators/accuracy.py
----------------------
Scores how accurate a model response is compared to the correct answer.

We combine two approaches and average them:
  1. Exact match  — did the response contain the correct answer verbatim?
  2. Semantic similarity — are the meanings similar even if words differ?

Why both? Exact match alone is too strict. "Paris" and "The capital is Paris"
mean the same thing but exact match would score the second one as wrong.
Semantic similarity alone can be fooled by confident-sounding wrong answers.
Together they give a more balanced score.
"""

from sentence_transformers import SentenceTransformer, util

# Load the model once at module level — it's ~80MB and takes a few seconds.
# Loading inside the function would reload it on every single call (very slow).
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _exact_match_score(response: str, correct_answer: str) -> float:
    """
    Check if the correct answer appears anywhere in the response.

    Returns 1.0 if found, 0.0 if not. Case-insensitive.

    Args:
        response:       The model's response text.
        correct_answer: The ground truth answer.

    Returns:
        1.0 if correct_answer is contained in response, else 0.0.
    """
    return 1.0 if correct_answer.lower() in response.lower() else 0.0


def _semantic_similarity_score(response: str, correct_answer: str) -> float:
    """
    Measure how semantically similar the response is to the correct answer.

    Uses sentence-transformers to encode both texts into embedding vectors,
    then computes cosine similarity. Score ranges from 0.0 to 1.0.

    Args:
        response:       The model's response text.
        correct_answer: The ground truth answer.

    Returns:
        Float between 0.0 (completely different meaning) and 1.0 (identical meaning).
    """
    embeddings = _embedder.encode(
        [response, correct_answer],
        convert_to_tensor=True
    )
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    # cos_sim returns a tensor — .item() converts it to a plain Python float
    return round(float(similarity.item()), 4)


def score_accuracy(model_response: str, correct_answer: str) -> float:
    """
    Score the accuracy of a model response against the correct answer.

    Combines exact match and semantic similarity with equal weighting.
    This is the main function the pipeline calls.

    Args:
        model_response: The text returned by the LLM.
        correct_answer: The ground truth answer from TruthfulQA.

    Returns:
        Float between 0.0 (completely wrong) and 1.0 (perfectly correct).
    """
    if not model_response or model_response.startswith("ERROR"):
        return 0.0

    exact    = _exact_match_score(model_response, correct_answer)
    semantic = _semantic_similarity_score(model_response, correct_answer)

    return round((exact + semantic) / 2, 4)


if __name__ == "__main__":
    # Quick test — run `python -m evaluators.accuracy` to verify
    tests = [
        ("Paris", "Paris"),
        ("The capital of France is Paris.", "Paris"),
        ("I think it might be Lyon.", "Paris"),
        ("ERROR: rate limit", "Paris"),
    ]
    print("Accuracy scores:")
    for response, answer in tests:
        score = score_accuracy(response, answer)
        print(f"  Response: {response!r:45} → {score}")