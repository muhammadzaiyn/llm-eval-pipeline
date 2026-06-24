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

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


# ── Model identifiers ─────────────────────────────────────────────────────────

# These are the exact strings the respective SDKs expect.
OPENAI_MODEL: str = "gpt-4o-mini"
ANTHROPIC_MODEL: str = "claude-haiku-4-5"
GROQ_MODEL: str = "llama3-8b-8192"


# ── Pipeline settings ─────────────────────────────────────────────────────────

# How many TruthfulQA questions to evaluate (keep small while testing).
NUM_SAMPLES: int = 50

# Seconds to sleep between API calls to stay under rate limits.
API_SLEEP_SECONDS: float = 0.5

# Where to write the final results file.
RESULTS_PATH: str = "results/results.json"


# ── Pricing (USD per 1 000 tokens) ────────────────────────────────────────────
# Source: published pricing pages as of mid-2025.
# Update here if prices change; latency_cost.py reads from this dict.

PRICING = {
    OPENAI_MODEL: {
        "input_per_1k": 0.000150,   # $0.150 / 1M tokens
        "output_per_1k": 0.000600,  # $0.600 / 1M tokens
    },
    ANTHROPIC_MODEL: {
        "input_per_1k": 0.000080,   # $0.080 / 1M tokens
        "output_per_1k": 0.000400,  # Haiku output rate
    },
    GROQ_MODEL: {
        "input_per_1k": 0.000005,   # Groq is extremely cheap
        "output_per_1k": 0.000008,
    },
}


# ── Startup validation ────────────────────────────────────────────────────────

def validate_keys() -> None:
    """
    Check that all required API keys are present.

    Call this at the top of main.py so the pipeline fails fast
    with a clear message instead of crashing mid-run.

    Raises:
        EnvironmentError: If any required key is missing or still
                          set to the placeholder value.
    """
    required = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
    }
    missing = [
        name for name, value in required.items()
        if not value or "your_" in value
    ]
    if missing:
        raise EnvironmentError(
            f"Missing or placeholder API keys detected: {missing}\n"
            "Copy .env.template → .env and fill in your real keys."
        )


if __name__ == "__main__":
    # Quick sanity-check: run `python config.py` to verify your .env is loaded.
    try:
        validate_keys()
        print("✅ All API keys loaded successfully.")
    except EnvironmentError as e:
        print(f"❌ Config error: {e}")