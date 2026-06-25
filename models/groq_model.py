"""
models/groq_model.py
--------------------
Wrapper for the Groq API.

All three models in this pipeline now run through Groq (free tier):
  - openai/gpt-oss-20b       (GPT-family)
  - llama-3.3-70b-versatile  (Meta Llama family)
  - qwen/qwen3.6-27b         (Alibaba Qwen family)

A single generate() function handles all three — just pass the model
name as an argument. The pipeline calls this function three times per
question, once with each model name.
"""

import time
import re
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL_LLAMA

# Instantiate the Groq client once at module level
client = Groq(api_key=GROQ_API_KEY)


def _clean_response(text: str) -> str:
    """
    Remove <think>...</think> blocks from reasoning models like Qwen.
    Handles both properly closed tags and truncated output with no closing tag.
    """
    # Case 1: properly closed <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Case 2: opening tag with no closing tag — strip everything from <think> onward
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def generate(prompt: str, model: str = None) -> dict:
    """
    Send a prompt to a Groq-hosted model and return the response with metadata.

    Args:
        prompt: The question or instruction to send to the model.
        model:  The Groq model ID to use. Defaults to GROQ_MODEL_LLAMA
                from config.py if not specified.

    Returns:
        A dict with keys:
            - response        (str)  : the model's text response
            - latency_seconds (float): how long the API call took
            - input_tokens    (int)  : tokens used in the prompt
            - output_tokens   (int)  : tokens used in the response

    On failure, returns a dict with response="ERROR: <message>"
    and all numeric fields set to 0 so the pipeline can continue.
    """
    if model is None:
        model = GROQ_MODEL_LLAMA

    # Qwen is a reasoning model — it needs more tokens and an explicit
    # instruction to skip showing its thinking process.
    is_qwen = "qwen" in model.lower()

    system_prompt = (
        "Answer the question truthfully and concisely. "
        "Do not show your reasoning or thinking process. "
        "Give only the final answer."
        if is_qwen
        else "Answer the question truthfully and concisely."
    )

    try:
        start_time = time.time()

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=512 if is_qwen else 256,  # Qwen needs more room
            temperature=0.0,
        )

        latency = round(time.time() - start_time, 3)

        return {
            "response":        _clean_response(completion.choices[0].message.content),
            "latency_seconds": latency,
            "input_tokens":    completion.usage.prompt_tokens,
            "output_tokens":   completion.usage.completion_tokens,
        }

    except Exception as e:
        error_type = type(e).__name__
        print(f"❌ Groq [{model}] ({error_type}): {e}")
        return {
            "response":        f"ERROR: {str(e)}",
            "latency_seconds": 0,
            "input_tokens":    0,
            "output_tokens":   0,
        }


# ── Manual test ───────────────────────────────────────────────────────────────
# Uncomment and run `python -m models.groq_model` to test all 3 models:
#
# if __name__ == "__main__":
#     from config import GROQ_MODEL_GPT, GROQ_MODEL_LLAMA, GROQ_MODEL_QWEN
#     for m in [GROQ_MODEL_GPT, GROQ_MODEL_LLAMA, GROQ_MODEL_QWEN]:
#         print(f"\nTesting {m}...")
#         result = generate("What is the capital of France?", model=m)
#         print(result)