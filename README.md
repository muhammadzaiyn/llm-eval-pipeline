# 🤖 LLM Evaluation Pipeline

> A production-quality benchmarking pipeline that evaluates and compares large language models on truthfulness, accuracy, faithfulness, and hallucination rate — with an interactive visual dashboard.

---

## 📌 What This Project Does

Most LLM comparisons are subjective. This pipeline makes them **measurable**.

It loads the [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) benchmark dataset, runs each question through three different LLMs simultaneously, scores every response across five metrics, and displays the results in an interactive Streamlit dashboard — so you can see exactly which model performs best and why.

---

## 🧰 Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| **LLM APIs** | Groq SDK | Free inference for all 3 models |
| **Models** | GPT-OSS 20B, Llama 3.3 70B, Qwen 3.6 27B | Models being evaluated |
| **Dataset** | HuggingFace `datasets` | TruthfulQA benchmark (817 questions) |
| **Semantic Scoring** | `sentence-transformers` | Semantic similarity for accuracy scoring |
| **LLM-as-Judge** | Llama 3.3 70B via Groq | Hallucination detection |
| **Data Handling** | Pandas + JSON | Result storage and aggregation |
| **Dashboard** | Streamlit + Plotly | Interactive visual comparison |
| **Config** | `python-dotenv` | Secure API key management |

---

## 📁 Project Structure

```
llm-eval-pipeline/
├── config.py                # API keys, model names, pricing config
├── dataset_loader.py        # Loads TruthfulQA from HuggingFace
├── evaluators/
│   ├── accuracy.py          # Exact match + semantic similarity scoring
│   ├── faithfulness.py      # Word overlap faithfulness scoring
│   ├── hallucination.py     # LLM-as-judge hallucination detection
│   └── latency_cost.py      # Latency and cost calculation
├── models/
│   └── groq_model.py        # Unified Groq wrapper for all 3 models
├── pipeline.py              # Orchestrates full evaluation run
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── results/                 # Auto-generated JSON results (gitignored)
├── requirements.txt
├── .env.template            # Safe template — copy to .env and add keys
└── .gitignore
```

---

## ⚙️ How It Works

```
TruthfulQA Dataset
       │
       ▼
  50 Questions
       │
  ┌────┴───────────────────────┐
  │      For each question      │
  │                             │
  │  GPT-OSS 20B  ──┐           │
  │  Llama 3.3 70B ──┼──► Evaluate ──► Store
  │  Qwen 3.6 27B  ──┘           │
  │                             │
  └─────────────────────────────┘
       │
       ▼
  results/results.json
       │
       ▼
  Streamlit Dashboard
```

**5 metrics scored per response:**
- **Accuracy** — exact match + semantic similarity (sentence-transformers)
- **Faithfulness** — word overlap against the correct answer
- **Hallucination** — LLM-as-judge score (0 = none, 1 = heavy)
- **Latency** — API response time in seconds
- **Cost** — estimated USD per call (currently $0 on Groq free tier)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/muhammadzaiyn/llm-eval-pipeline.git
cd llm-eval-pipeline
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp .env.template .env
```
Open `.env` and add your [Groq API key](https://console.groq.com/keys) (free):
```
GROQ_API_KEY=your_key_here
```

### 5. Run the pipeline
```bash
# Test with 5 questions first
python pipeline.py --samples 5

# Full run (50 questions, ~15-20 mins)
python pipeline.py --samples 50
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Key Findings

Evaluated on **50 questions** from the TruthfulQA benchmark across 3 models.

| Metric | GPT-OSS 20B | Llama 3.3 70B | Qwen 3.6 27B |
|--------|-------------|----------------|--------------|
| 🎯 Accuracy | 27.84% | **41.59%** | 6.28% |
| 📖 Faithfulness | 14.76% | **24.99%** | 10.13% |
| 🔍 Hallucination | 47.60% | **38.40%** ✅ | 83.20% |
| ⚡ Avg Latency | **0.67s** ✅ | 1.45s | 1.31s |

**Winner per category:**

| Category | Winner |
|----------|--------|
| Most Accurate | 🥇 Llama 3.3 70B |
| Most Faithful | 🥇 Llama 3.3 70B |
| Least Hallucination | 🥇 Llama 3.3 70B |
| Fastest Response | 🥇 GPT-OSS 20B |

**Observations:**
- **Llama 3.3 70B** is the clear overall winner — highest accuracy (41.59%), highest faithfulness (24.99%), and lowest hallucination rate (38.40%) across all 50 questions
- **GPT-OSS 20B** is the fastest model at 0.67s average latency, making it the best choice when speed matters more than accuracy
- **Qwen 3.6 27B** underperformed significantly — only 6.28% accuracy and 83.20% hallucination rate. This is likely because Qwen is a reasoning model that outputs internal `<think>` blocks which interfere with scoring even after stripping
- All models struggled with TruthfulQA — this dataset is specifically designed to be tricky, targeting common misconceptions that LLMs tend to get wrong
- Hallucination rates across all models are high (38-83%), highlighting that even modern LLMs fabricate information on factual benchmarks

---

## 📸 Dashboard Preview

![Dashboard](screenshot.png)

---

## 🔧 Extending This Project

Ideas for taking this further:
- Add OpenAI or Anthropic models by restoring the API wrappers in `models/`
- Swap in a different benchmark dataset (e.g. MMLU, HellaSwag)
- Upgrade hallucination scoring to use NLI models instead of LLM-as-judge
- Add a CI/CD pipeline that re-runs evaluation on a schedule
- Deploy the dashboard to Streamlit Cloud for public access

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <sub>Built by Muhammad Zaiyn · Powered by Groq · Dataset: TruthfulQA</sub>
</div>