"""
evaluators/latency_cost.py
--------------------------
Calculates the estimated cost of a model API call in USD.

Since we are using Groq's free tier, all costs are $0.00 for now.
The pricing structure is kept in place so this project can be easily
extended to paid APIs (OpenAI, Anthropic) in the future — just update
the PRICING dict in config.py and this module handles the rest.

Cost formula:
    cost = (input_tokens / 1000 * input_rate)
         + (output_tokens / 1000 * output_rate)
"""

from config import PRICING


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Calculate the estimated USD cost for a single API call.

    Looks up the per-1k-token rates from config.PRICING and applies
    the standard formula used by all major LLM providers.

    Args:
        model_name:    The model ID string (must match a key in PRICING).
        input_tokens:  Number of tokens in the prompt.
        output_tokens: Number of tokens in the response.

    Returns:
        Estimated cost in USD as a float.
        Returns 0.0 if the model is not found in PRICING.
    """
    if model_name not in PRICING:
        print(f"⚠️  Model '{model_name}' not found in PRICING config. Returning 0.0.")
        return 0.0

    rates = PRICING[model_name]
    input_cost  = (input_tokens  / 1000) * rates["input_per_1k"]
    output_cost = (output_tokens / 1000) * rates["output_per_1k"]

    return round(input_cost + output_cost, 8)


if __name__ == "__main__":
    # Quick test — run `python -m evaluators.latency_cost` to verify
    from config import GROQ_MODEL_GPT, GROQ_MODEL_LLAMA, GROQ_MODEL_QWEN, MODEL_DISPLAY_NAMES

    print("Cost estimates for a sample API call (100 input, 50 output tokens):")
    for model in [GROQ_MODEL_GPT, GROQ_MODEL_LLAMA, GROQ_MODEL_QWEN]:
        cost = calculate_cost(model, input_tokens=100, output_tokens=50)
        name = MODEL_DISPLAY_NAMES[model]
        print(f"  {name:20} → ${cost:.8f}")