# models/__init__.py
# All three models now run through Groq.
# Import generate once — pass different model names to compare models.

from .groq_model import generate as groq_generate
from config import GROQ_MODEL_GPT, GROQ_MODEL_LLAMA, GROQ_MODEL_QWEN