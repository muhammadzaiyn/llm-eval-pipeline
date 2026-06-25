"""
config.py
---------
Central configuration for the LLM Evaluation Pipeline.

Loads API keys from the .env file and exposes them as clean
Python constants. Also defines shared pipeline settings like
which model names to use and how many dataset samples to run.

Any module that needs a key or setting imports from here —
never from os.environ directly — so there's one place to change things.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into the process environment.
# This must run before we call os.getenv() below.
load_dotenv()


# ── API Keys ──────────────────────────────────────────────────────────────────
# We now use Groq exclusively (free). OpenAI and Anthropic keys are
# no longer needed but kept as empty strings so .env.template still
# makes sense if someone wants to extend this project later.

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


# ── Model identifiers ─────────────────────────────────────────────────────────
# All three models run on Groq's free tier.
# These are the exact model ID strings the Groq SDK expects.

GROQ_MODEL_GPT:   str = "openai/gpt-oss-20b"        # GPT-family model
GROQ_MODEL_LLAMA: str = "llama-3.3-70b-versatile"   # Meta Llama family
GROQ_MODEL_QWEN:  str = "qwen/qwen3.6-27b"          # Alibaba Qwen family

# Human-readable display names for the dashboard
MODEL_DISPLAY_NAMES = {
    GROQ_MODEL_GPT:   "GPT-OSS 20B",
    GROQ_MODEL_LLAMA: "Llama 3.3 70B",
    GROQ_MODEL_QWEN:  "Qwen 3.6 27B",
}

# Ordered list — used by the pipeline to loop over all models
ALL_MODELS = [GROQ_MODEL_GPT, GROQ_MODEL_LLAMA, GROQ_MODEL_QWEN]


# ── Pipeline settings ─────────────────────────────────────────────────────────

# How many TruthfulQA questions to evaluate (keep small while testing).
NUM_SAMPLES: int = 50

# Seconds to sleep between API calls to avoid rate limits.
API_SLEEP_SECONDS: float = 0.5

# Where to write the final results file.
RESULTS_PATH: str = "results/results.json"


# ── Pricing (all free via Groq) ───────────────────────────────────────────────
PRICING = {
    GROQ_MODEL_GPT:   {"input_per_1k": 0.0, "output_per_1k": 0.0},
    GROQ_MODEL_LLAMA: {"input_per_1k": 0.0, "output_per_1k": 0.0},
    GROQ_MODEL_QWEN:  {"input_per_1k": 0.0, "output_per_1k": 0.0},
}


# ── Startup validation ────────────────────────────────────────────────────────

def validate_keys() -> None:
    """
    Check that the Groq API key is present and not a placeholder.

    Call this at the top of main.py so the pipeline fails fast
    with a clear message instead of crashing mid-run.

    Raises:
        EnvironmentError: If the Groq key is missing or still a placeholder.
    """
    if not GROQ_API_KEY or "your_" in GROQ_API_KEY:
        raise EnvironmentError(
            "Missing or placeholder GROQ_API_KEY detected.\n"
            "Copy .env.template → .env and fill in your real Groq key.\n"
            "Get one free at: https://console.groq.com/keys"
        )


if __name__ == "__main__":
    # Quick sanity-check: run `python config.py` to verify your .env is loaded.
    try:
        validate_keys()
        print("✅ Groq API key loaded successfully.")
        print(f"\n📋 Models configured:")
        for model in ALL_MODELS:
            print(f"   - {MODEL_DISPLAY_NAMES[model]} ({model})")
    except EnvironmentError as e:
        print(f"❌ Config error: {e}")