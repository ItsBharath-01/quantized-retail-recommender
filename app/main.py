"""Streamlit dashboard for Quantized Retail Recommender.

Features:
- Project overview and system architecture
- Interactive shopping basket input & real-time product recommendations
- Choice of inference engine: Optimized FP32, Optimized Dynamic INT8, Optimized Static INT8, and Baselines
- Dataset statistics & frequent itemset mining
- Model training metrics & loss progression
- Quantization benchmarking: size reduction, accuracy retention, and latency comparison
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root and app directory are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import torch

try:
    from app import core
    from app.core import BasketTokenizer, predict_next_items
    from app.storage import Store
except ModuleNotFoundError:
    import core
    from core import BasketTokenizer, predict_next_items
    from storage import Store

_DATA_DIR = _PROJECT_ROOT / "data"
_REPORTS_DIR = _PROJECT_ROOT / "reports" / "final_results"

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Quantized Retail Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #2563EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data & Model Loaders (Cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_all_artifacts(data_dir: str = None) -> Dict[str, Any]:
    """Ensure milestones are executed and load saved models and catalog."""
    if data_dir is None:
        p_data = _DATA_DIR
    else:
        p_data = Path(data_dir)
        if not p_data.is_absolute():
            p_data = _PROJECT_ROOT / p_data
    str_data_dir = str(p_data)
    catalog_path = p_data / "catalog.json"
    vocab_path = p_data / "vocab.json"
    base_model_path = p_data / "baseline_model.pt"
    opt_model_path = p_data / "optimized_model.pt"
    opt_dyn_path = p_data / "optimized_quantized_dynamic.pt"
    opt_stat_path = p_data / "optimized_quantized_static.pt"

    # If dataset not yet prepared, run milestone 1
    if not (catalog_path.exists() and vocab_path.exists()):
        core.milestone_1(n_baskets=10000, seed=42, data_dir=str_data_dir)

    tokenizer = BasketTokenizer.load(vocab_path)
    catalog = json.loads(catalog_path.read_text())
    meta = json.loads((p_data / "meta.json").read_text()) if (p_data / "meta.json").exists() else {}
    truth = json.loads((p_data / "truth.json").read_text()) if (p_data / "truth.json").exists() else []

    # Ensure models exist
    if not base_model_path.exists():
        core.milestone_2(epochs=2, batch_size=64, data_dir=str_data_dir)
    if not opt_model_path.exists():
        core.train_optimized_recommender_model(epochs=6, data_dir=str_data_dir)

    base_model = torch.load(base_model_path, weights_only=False)
    opt_model = torch.load(opt_model_path, weights_only=False)

    if not opt_dyn_path.exists():
        core.apply_dynamic_quantization(opt_model, str(opt_dyn_path))
    if not opt_stat_path.exists():
        train_baskets = json.loads((p_data / "train_baskets.json").read_text())
        core.apply_static_quantization(opt_model, train_baskets[:50], tokenizer, str(opt_stat_path))

    opt_dyn_model = torch.load(opt_dyn_path, weights_only=False)
    opt_stat_model = torch.load(opt_stat_path, weights_only=False)

    return {
        "tokenizer": tokenizer,
        "catalog": catalog,
        "meta": meta,
        "truth": truth,
        "models": {
            "Optimized FP32 (Recommended)": opt_model,
            "Optimized Dynamic INT8": opt_dyn_model,
            "Optimized Static INT8": opt_stat_model,
            "Baseline FP32": base_model,
        },
        "paths": {
            "optimized": str(opt_model_path),
            "opt_dyn": str(opt_dyn_path),
            "opt_stat": str(opt_stat_path),
            "baseline": str(base_model_path),
        },
    }


def get_cached_benchmarks() -> pd.DataFrame:
    csv_path = _REPORTS_DIR / "benchmark_results.csv"
    if not csv_path.exists():
        core.run_comprehensive_benchmark(n_eval_cases=100, seed=42, data_dir=str(_DATA_DIR), output_dir=str(_REPORTS_DIR))
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Load System Data
# ---------------------------------------------------------------------------

try:
    artifacts = load_all_artifacts()
    tokenizer = artifacts["tokenizer"]
    catalog = artifacts["catalog"]
    meta = artifacts["meta"]
    truth = artifacts["truth"]
    models = artifacts["models"]
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------

st.sidebar.title("🛒 Control Panel")
st.sidebar.markdown("**Language Models from Scratch: Quantized Retail Recommender**")

model_choice = st.sidebar.selectbox(
    "Active Inference Engine",
    options=list(models.keys()),
    index=0,
    help="Select the model to evaluate next-item predictions.",
)

top_k_val = st.sidebar.slider("Number of Recommendations (Top-K)", min_value=1, max_value=10, value=5)

st.sidebar.divider()
st.sidebar.markdown("### Pipeline Actions")

if st.sidebar.button("🔄 Retrain & Re-benchmark Models"):
    with st.spinner("Retraining optimized model and regenerating benchmarks..."):
        core.train_optimized_recommender_model(epochs=6)
        core.run_comprehensive_benchmark(n_eval_cases=100)
        st.cache_resource.clear()
        st.sidebar.success("Models and benchmarks refreshed successfully!")
        st.rerun()

st.sidebar.info(
    "**MLOps Insights:**\n"
    "- **Optimized FP32:** 198k parameter Causal Transformer with causal masking & permutation training.\n"
    "- **Dynamic INT8:** ~47% storage compression with runtime weight quantization.\n"
    "- **Static INT8:** Quantizes weights & activations using offline calibration."
)


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">🛒 Quantized Retail Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Compact Causal Transformer Language Model for Retail Shopping Baskets with INT8 Post-Training Quantization</div>',
    unsafe_allow_html=True,
)

tab_demo, tab_data = st.tabs([
    "🚀 Live Recommendations",
    "📦 Dataset & Frequent Itemsets",
])


# ===========================================================================
# TAB 1: Live Recommendations
# ===========================================================================
with tab_demo:
    st.subheader("🛍️ Interactive Basket Next-Item Prediction")
    st.write(
        "Select or search grocery items to populate the basket. The active model processes "
        "the basket sequence and predicts ranked next-item recommendations."
    )

    all_item_names = [item["name"] for item in catalog]
    item_metadata = {item["name"]: item for item in catalog}

    # Preset selection
    col_preset, col_clear = st.columns([4, 1])
    preset_choice = col_preset.selectbox(
        "Choose a sample basket preset or build your own:",
        options=[
            "Custom Basket",
            "Morning Breakfast (Milk, Bread, Butter)",
            "South Indian Kitchen (Toor Dal, Mustard Seeds, Rice)",
            "Chai & Snacks (Tea, Sugar, Biscuits)",
        ],
    )

    if preset_choice == "Morning Breakfast (Milk, Bread, Butter)":
        default_items = [
            it for it in all_item_names
            if any(k in it.lower() for k in ["milk", "bread", "butter"])
        ][:3]
    elif preset_choice == "South Indian Kitchen (Toor Dal, Mustard Seeds, Rice)":
        default_items = [
            it for it in all_item_names
            if any(k in it.lower() for k in ["dal", "mustard", "rice"])
        ][:3]
    elif preset_choice == "Chai & Snacks (Tea, Sugar, Biscuits)":
        default_items = [
            it for it in all_item_names
            if any(k in it.lower() for k in ["tea", "sugar", "biscuit", "chips"])
        ][:3]
    else:
        default_items = all_item_names[:2]

    selected_items = st.multiselect(
        "Current Items in Basket:",
        options=all_item_names,
        default=default_items,
        placeholder="Search for grocery products...",
    )

    if not selected_items:
        st.warning("Please select at least one item in the basket to receive recommendations.")
    else:
        # Run prediction
        active_model = models[model_choice]
        t0 = time.perf_counter()
        recs = predict_next_items(
            model=active_model,
            visible_items=selected_items,
            tokenizer=tokenizer,
            top_k=top_k_val,
            exclude_input=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Active Model", model_choice)
        col_m2.metric("Basket Items", f"{len(selected_items)} items")
        col_m3.metric("Inference Latency", f"{latency_ms:.2f} ms")

        st.markdown(f"#### Ranked Product Recommendations (Top-{top_k_val}):")
        if recs:
            rec_rows = []
            for r in recs:
                meta_info = item_metadata.get(r["item"], {})
                rec_rows.append({
                    "Rank": f"#{r['rank']}",
                    "Product Name": r["item"],
                    "Category": meta_info.get("category", "General"),
                    "Brand": meta_info.get("brand", "Standard"),
                    "Pack Size": meta_info.get("size", "-"),
                    "Probability Score": f"{r['score']:.4f}",
                    "Score Bar": r["score"],
                })

            df_recs = pd.DataFrame(rec_rows)
            st.dataframe(
                df_recs[["Rank", "Product Name", "Category", "Brand", "Pack Size", "Probability Score"]],
                use_container_width=True,
                hide_index=True,
            )

            # Bar chart of recommendation scores
            chart_df = pd.DataFrame({
                "Product": [r["item"] for r in recs],
                "Score": [r["score"] for r in recs],
            }).set_index("Product")
            st.bar_chart(chart_df)
        else:
            st.info("No candidate recommendations above threshold.")


# ===========================================================================
# TAB 2: Dataset & Frequent Itemsets
# ===========================================================================
with tab_data:
    st.subheader("📦 Retail Dataset & Apriori Ground Truth")
    st.write(
        "Synthetically generated shopping baskets created using `thilimarketbasket`. "
        "Planted itemsets represent natural grocery affinity patterns (e.g. Chai, Sambar ingredients, Baking)."
    )

    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Total Baskets", f"{meta.get('n_baskets', 10000):,}")
    stat2.metric("Train Split", f"{meta.get('n_train', 8000):,}")
    stat3.metric("Test Split", f"{meta.get('n_test', 2000):,}")
    stat4.metric("Catalog Products", f"{len(catalog):,}")

    st.divider()
    col_cat, col_truth = st.columns(2)

    with col_cat:
        st.markdown("#### 🛒 Product Catalog Sample")
        df_cat = pd.DataFrame(catalog)
        st.dataframe(df_cat[["name", "category", "brand", "size", "share"]].head(15), use_container_width=True)

    with col_truth:
        st.markdown("#### 🔗 Planted & Mined Frequent Itemsets (Ground Truth)")
        if truth:
            df_truth = pd.DataFrame([
                {
                    "Itemset": " + ".join(row["items"]),
                    "Items Count": len(row["items"]),
                    "Support": f"{row['support']:.4f}",
                }
                for row in truth[:15]
            ])
            st.dataframe(df_truth, use_container_width=True)

st.markdown("---")
st.caption("Quantized Retail Recommender • Language Models from Scratch • Built with PyTorch, Streamlit & thilimarketbasket")
