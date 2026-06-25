"""
evaluators/faithfulness.py
--------------------------
Scores how faithful a model response is to a source context.

"Faithfulness" asks: does the response stick to what the source says,
or does it introduce outside claims?

We use a simple word overlap approach: what fraction of meaningful words
in the response also appear in the source context? This is a lightweight
proxy for "did the model stay on topic" without needing another API call.

Limitation: this approach doesn't understand meaning, only word presence.
A more advanced version would use NLI (Natural Language Inference) models.
For TruthfulQA we use the correct answer as the source context.
"""

import re


def _tokenize(text: str) -> set:
    """
    Convert text to a set of lowercase words, removing stopwords and punctuation.

    We remove common stopwords because "the", "is", "a" etc. appear in
    almost every sentence and would inflate the overlap score artificially.

    Args:
        text: Raw input string.

    Returns:
        Set of meaningful lowercase word strings.
    """
    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "be", "are", "was", "were",
        "that", "this", "with", "as", "by", "from", "its", "has", "have"
    }
    # Extract only alphabetic words
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return {w for w in words if w not in stopwords}


def score_faithfulness(model_response: str, source_context: str) -> float:
    """
    Score how faithful the model response is to the source context.

    Computes what fraction of meaningful words in the response
    also appear in the source context (precision-style overlap).

    Args:
        model_response:  The text returned by the LLM.
        source_context:  The reference text to check faithfulness against.
                         In our pipeline this is the correct answer.

    Returns:
        Float between 0.0 (no overlap) and 1.0 (fully faithful).
        Returns 0.0 for empty or error responses.
    """
    if not model_response or model_response.startswith("ERROR"):
        return 0.0

    response_words = _tokenize(model_response)
    context_words  = _tokenize(source_context)

    if not response_words:
        return 0.0

    overlap = response_words & context_words  # set intersection
    score   = len(overlap) / len(response_words)

    return round(score, 4)


if __name__ == "__main__":
    # Quick test — run `python -m evaluators.faithfulness` to verify
    context = "Paris is the capital and largest city of France."
    tests = [
        "Paris is the capital of France.",
        "The capital is Paris, a large city.",
        "I believe it could be Lyon or Marseille.",
        "ERROR: rate limit",
    ]
    print("Faithfulness scores (context: 'Paris is the capital and largest city of France.'):")
    for response in tests:
        score = score_faithfulness(response, context)
        print(f"  Response: {response!r:45} → {score}")