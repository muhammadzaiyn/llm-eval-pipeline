"""
dashboard/app.py
----------------
Streamlit dashboard for visualizing LLM evaluation results.

Run with: streamlit run dashboard/app.py

Sections:
  1. Model Overview Cards  — avg scores per model at a glance
  2. Bar Charts            — visual comparison across all metrics
  3. Radar Chart           — multi-metric spider chart
  4. Results Table         — full question-level breakdown
  5. Best Model Summary    — which model wins on each metric
"""

import json
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #141722;
        border-right: 1px solid #2e3450;
    }

    h2 { color: #e2e8f0 !important; }
    h3 { color: #cbd5e0 !important; }
    hr { border-color: #2e3450 !important; }

    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    [data-testid="stMetric"] {
        background: #1e2130;
        border: 1px solid #2e3450;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

RESULTS_PATH = "results/results.json"

MODEL_COLORS = {
    "GPT-OSS 20B":   "#667eea",
    "Llama 3.3 70B": "#48bb78",
    "Qwen 3.6 27B":  "#ed8936",
}
DEFAULT_COLOR = "#a0aec0"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_results() -> list[dict]:
    """Load and cache results JSON."""
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_results(results: list[dict]) -> pd.DataFrame:
    """
    Flatten nested results JSON into a flat pandas DataFrame.
    Each row = one model's response to one question.
    """
    rows = []
    for item in results:
        for model_name, scores in item["models"].items():
            rows.append({
                "question":      item["question"],
                "best_answer":   item["best_answer"],
                "category":      item["category"],
                "model":         model_name,
                # Truncate long responses so the table doesn't overflow
                "response":      scores["response"][:120] + "…"
                                 if len(scores["response"]) > 120
                                 else scores["response"],
                "accuracy":      scores["accuracy"],
                "faithfulness":  scores["faithfulness"],
                "hallucination": scores["hallucination"],
                "latency":       scores["latency_seconds"],
                "cost_usd":      scores["cost_usd"],
                "input_tokens":  scores["input_tokens"],
                "output_tokens": scores["output_tokens"],
            })
    return pd.DataFrame(rows)


def make_bar_chart(
    agg: pd.DataFrame,
    y_col: str,
    title: str,
    text_format: str = ".2%",
    y_range: list = None,
) -> go.Figure:
    """
    Create a styled Plotly bar chart for a single metric.

    Args:
        agg:         Aggregated DataFrame with one row per model.
        y_col:       Column name to plot on the y-axis.
        title:       Chart title string.
        text_format: Format string for bar value labels.
        y_range:     Optional [min, max] for y-axis.

    Returns:
        Plotly Figure object.
    """
    colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in agg["model"]]

    fig = go.Figure(go.Bar(
        x=agg["model"],
        y=agg[y_col],
        marker_color=colors,
        marker_line_width=0,
        text=agg[y_col].apply(
            lambda v: f"{v:.2%}" if "%" in text_format else f"{v:.2f}"
        ),
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13, family="monospace"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=14), x=0.02),
        paper_bgcolor="#1e2130",
        plot_bgcolor="#1e2130",
        font=dict(color="#a0aec0"),
        xaxis=dict(showgrid=False, tickfont=dict(color="#cbd5e0", size=12),
                   linecolor="#2e3450"),
        yaxis=dict(showgrid=True, gridcolor="#2e3450",
                   tickfont=dict(color="#a0aec0"), range=y_range,
                   linecolor="#2e3450"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
    <h1 style="color:#f7fafc;font-size:2rem;font-weight:800;margin:0;">
        🤖 LLM Evaluation Dashboard
    </h1>
    <p style="color:#718096;font-size:1rem;margin-top:6px;">
        Benchmarking
        <b style="color:#667eea">GPT-OSS 20B</b>,
        <b style="color:#48bb78">Llama 3.3 70B</b>, and
        <b style="color:#ed8936">Qwen 3.6 27B</b>
        on the <b style="color:#a0aec0">TruthfulQA</b> dataset
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
results = load_results()

if not results:
    st.error("⚠️ No results found. Run: `python pipeline.py --samples 50`")
    st.stop()

df_full = flatten_results(results)
total_questions = len(results)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    st.markdown("---")

    categories = ["All"] + sorted(df_full["category"].unique().tolist())
    selected_category = st.selectbox("📂 Category", categories)

    n_to_show = st.slider(
        "📋 Questions in table",
        min_value=1,
        max_value=total_questions,
        value=total_questions,   # default = show all
    )

    st.markdown("---")
    st.markdown(f"**📊 Total questions:** `{total_questions}`")
    st.markdown(f"**🤖 Models compared:** `{df_full['model'].nunique()}`")
    st.markdown(f"**🏷️ Categories:** `{df_full['category'].nunique()}`")
    st.markdown("---")
    st.caption("Built with Streamlit + Plotly\nDataset: TruthfulQA · Models via Groq")

# Apply category filter to a copy — keep df_full intact for the table
df = df_full.copy()
if selected_category != "All":
    df = df[df["category"] == selected_category]

# ── Aggregate ─────────────────────────────────────────────────────────────────
agg = df.groupby("model").agg(
    avg_accuracy=("accuracy",      "mean"),
    avg_faithfulness=("faithfulness", "mean"),
    avg_hallucination=("hallucination", "mean"),
    avg_latency=("latency",        "mean"),
    avg_cost=("cost_usd",          "mean"),
).round(4).reset_index()


# ── Section 1: Model Overview Cards ──────────────────────────────────────────
st.markdown("## 📊 Model Overview")

cols = st.columns(len(agg))
for col, (_, row) in zip(cols, agg.iterrows()):
    color = MODEL_COLORS.get(row["model"], DEFAULT_COLOR)
    with col:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}11);
                    border:1px solid {color}55;border-radius:14px;
                    padding:18px;text-align:center;margin-bottom:8px;">
            <div style="font-size:0.75rem;color:{color};font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;">MODEL</div>
            <div style="font-size:1.1rem;color:#f7fafc;font-weight:800;
                        margin-top:4px;">{row["model"]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("🎯 Accuracy",      f"{row['avg_accuracy']:.2%}")
        st.metric("📖 Faithfulness",  f"{row['avg_faithfulness']:.2%}")
        st.metric("🔍 Hallucination", f"{row['avg_hallucination']:.2%}")
        st.metric("⚡ Avg Latency",   f"{row['avg_latency']:.2f}s")
        st.metric("💰 Avg Cost",      f"${row['avg_cost']:.6f}")

st.markdown("---")


# ── Section 2: Bar Charts ─────────────────────────────────────────────────────
st.markdown("## 📈 Metric Comparison")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        make_bar_chart(agg, "avg_accuracy",
                       "🎯 Accuracy  (higher = better)", y_range=[0, 1.15]),
        use_container_width=True)
with col2:
    st.plotly_chart(
        make_bar_chart(agg, "avg_hallucination",
                       "🔍 Hallucination Rate  (lower = better)", y_range=[0, 1.15]),
        use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(
        make_bar_chart(agg, "avg_latency",
                       "⚡ Avg Latency in seconds  (lower = better)",
                       text_format=".2f"),
        use_container_width=True)
with col4:
    st.plotly_chart(
        make_bar_chart(agg, "avg_faithfulness",
                       "📖 Faithfulness  (higher = better)", y_range=[0, 1.15]),
        use_container_width=True)

st.markdown("---")


# ── Section 3: Radar Chart ────────────────────────────────────────────────────
st.markdown("## 🕸️ Multi-Metric Radar")

radar_categories = ["Accuracy", "Faithfulness", "Non-Hallucination", "Speed Score"]
fig_radar = go.Figure()

for _, row in agg.iterrows():
    color       = MODEL_COLORS.get(row["model"], DEFAULT_COLOR)
    max_latency = agg["avg_latency"].max() or 1
    values = [
        row["avg_accuracy"],
        row["avg_faithfulness"],
        1 - row["avg_hallucination"],          # invert: higher = better
        1 - (row["avg_latency"] / max_latency), # invert: higher = better
    ]
    values += [values[0]]  # close the polygon

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=radar_categories + [radar_categories[0]],
        fill="toself",
        name=row["model"],
        line=dict(color=color, width=2),
        fillcolor=color,
        opacity=0.25,
    ))

fig_radar.update_layout(
    polar=dict(
        bgcolor="#1e2130",
        radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2e3450",
                        tickfont=dict(color="#718096"), linecolor="#2e3450"),
        angularaxis=dict(gridcolor="#2e3450",
                         tickfont=dict(color="#cbd5e0", size=12),
                         linecolor="#2e3450"),
    ),
    paper_bgcolor="#1e2130",
    font=dict(color="#a0aec0"),
    legend=dict(font=dict(color="#cbd5e0"), bgcolor="#262b3e",
                bordercolor="#2e3450", borderwidth=1),
    height=420,
    margin=dict(l=60, r=60, t=40, b=40),
)
st.plotly_chart(fig_radar, use_container_width=True)
st.markdown("---")


# ── Section 4: Results Table ──────────────────────────────────────────────────
st.markdown("## 📋 Detailed Results")
st.caption(f"Showing {n_to_show} of {total_questions} questions · "
           f"3 rows per question (one per model) · "
           f"Responses truncated to 120 characters")

# Slice by number of unique questions, then get all 3 model rows for each
questions_shown = df_full["question"].unique()[:n_to_show]
df_table = df_full[df_full["question"].isin(questions_shown)].copy()

display_cols = ["question", "category", "model", "response",
                "accuracy", "faithfulness", "hallucination", "latency"]

st.dataframe(
    df_table[display_cols]
        .sort_values(["question", "model"])
        .reset_index(drop=True)
        .style
        .background_gradient(subset=["accuracy", "faithfulness"], cmap="Greens")
        .background_gradient(subset=["hallucination"], cmap="Reds")
        .format({
            "accuracy":      "{:.2%}",
            "faithfulness":  "{:.2%}",
            "hallucination": "{:.2%}",
            "latency":       "{:.2f}s",
        }),
    use_container_width=True,
    height=500,
    column_config={
        "question":      st.column_config.TextColumn("Question",     width="medium"),
        "category":      st.column_config.TextColumn("Category",     width="small"),
        "model":         st.column_config.TextColumn("Model",        width="small"),
        "response":      st.column_config.TextColumn("Response",     width="large"),
        "accuracy":      st.column_config.TextColumn("Accuracy",     width="small"),
        "faithfulness":  st.column_config.TextColumn("Faithfulness", width="small"),
        "hallucination": st.column_config.TextColumn("Hallucination",width="small"),
        "latency":       st.column_config.TextColumn("Latency",      width="small"),
    }
)

st.markdown("---")


# ── Section 5: Best Model Summary ────────────────────────────────────────────
st.markdown("## 🏆 Best Model Summary")

best_accuracy      = agg.loc[agg["avg_accuracy"].idxmax(),      "model"]
best_faithfulness  = agg.loc[agg["avg_faithfulness"].idxmax(),  "model"]
best_hallucination = agg.loc[agg["avg_hallucination"].idxmin(), "model"]
best_latency       = agg.loc[agg["avg_latency"].idxmin(),       "model"]


col_a, col_b, col_c, col_d = st.columns(4)
winners = [
    (col_a, "🎯", "Most Accurate",       best_accuracy),
    (col_b, "📖", "Most Faithful",       best_faithfulness),
    (col_c, "✅", "Least Hallucination", best_hallucination),
    (col_d, "⚡", "Fastest Response",    best_latency),
]

for col, trophy, label, name in winners:
    color = MODEL_COLORS.get(name, DEFAULT_COLOR)
    with col:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}11);
                    border:1px solid {color}66;border-radius:14px;
                    padding:22px;text-align:center;">
            <div style="font-size:2rem;">{trophy}</div>
            <div style="color:{color};font-size:0.72rem;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;
                        margin-top:8px;">{label}</div>
            <div style="color:#f7fafc;font-size:1rem;font-weight:800;
                        margin-top:6px;">{name}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("🤖 LLM Evaluation Pipeline · TruthfulQA Benchmark · "
           "Models hosted on Groq · Built with Streamlit + Plotly")