"""
LoRA / DPO Fine-tuning for Function Calling — Streamlit Demo
=============================================================
Run:  streamlit run streamlit_app.py

No GPU required — this app explores the experiment results, shows how the
prompt formatter and DPO pair builder work, and visualises the evaluation
metrics — all without loading the actual fine-tuned model weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data.formatter import build_system_prompt, extract_function_call_from_output
from src.data.preference_builder import build_dpo_pairs
from src.evaluation.metrics import FunctionCallMetrics, evaluate_single

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LoRA / DPO Function Calling",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_PATH = Path("experiments/results.json")
SCHEMAS_PATH = Path("data/samples/function_schemas.json")

PURPLE = "#6A1B9A"
BLUE = "#1565C0"
GREEN = "#2E7D32"
ORANGE = "#E65100"
RED = "#C62828"

MODEL_COLORS = {"Base Model": RED, "SFT + LoRA": BLUE, "DPO": GREEN}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


@st.cache_data
def load_schemas() -> list[dict]:
    return json.loads(SCHEMAS_PATH.read_text())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(results: dict) -> None:
    st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=72)
    st.sidebar.title("LoRA / DPO Fine-tuning")
    st.sidebar.markdown("---")
    r = results["results"]
    st.sidebar.markdown("**Best model (DPO)**")
    st.sidebar.metric("Valid JSON Rate", f"{r['dpo']['valid_json_rate']:.1f}%",
                       delta=f"+{r['dpo']['valid_json_rate'] - r['base']['valid_json_rate']:.1f} pp vs base")
    st.sidebar.metric("Tool Match Rate", f"{r['dpo']['exact_tool_match_rate']:.1f}%",
                       delta=f"+{r['dpo']['exact_tool_match_rate'] - r['base']['exact_tool_match_rate']:.1f} pp vs base")
    st.sidebar.metric("Hallucination Rate", f"{r['dpo']['hallucination_rate']:.1f}%",
                       delta=f"{r['dpo']['hallucination_rate'] - r['base']['hallucination_rate']:.1f} pp vs base",
                       delta_color="inverse")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Model: `{results['model']}`  \n"
        f"Dataset: {results['train_samples']:,} train samples  \n"
        f"Eval on {results['test_samples']:,} test samples"
    )


# ---------------------------------------------------------------------------
# Tab 1 — Results Dashboard
# ---------------------------------------------------------------------------
def tab_results(results: dict) -> None:
    st.header("📊 Experiment Results Dashboard")
    st.markdown(
        f"Model **`{results['model']}`** fine-tuned on "
        f"[{results['dataset']}]({results['dataset_url']}) with "
        f"**{results['train_samples']:,}** training samples."
    )

    r = results["results"]
    models = ["Base Model", "SFT + LoRA", "DPO"]
    keys = ["base", "sft_lora", "dpo"]

    # KPI delta cards
    c1, c2, c3, c4 = st.columns(4)
    base_vjr = r["base"]["valid_json_rate"]
    dpo_vjr = r["dpo"]["valid_json_rate"]
    c1.metric("Valid JSON (base → DPO)", f"{dpo_vjr:.1f}%", f"+{dpo_vjr - base_vjr:.1f} pp")
    base_tm = r["base"]["exact_tool_match_rate"]
    dpo_tm = r["dpo"]["exact_tool_match_rate"]
    c2.metric("Tool Match (base → DPO)", f"{dpo_tm:.1f}%", f"+{dpo_tm - base_tm:.1f} pp")
    base_hall = r["base"]["hallucination_rate"]
    dpo_hall = r["dpo"]["hallucination_rate"]
    c3.metric("Hallucination (base → DPO)", f"{dpo_hall:.1f}%",
              f"{dpo_hall - base_hall:.1f} pp", delta_color="inverse")
    base_f1 = r["base"]["parameter_f1"]
    dpo_f1 = r["dpo"]["parameter_f1"]
    c4.metric("Param F1 (base → DPO)", f"{dpo_f1:.3f}", f"+{dpo_f1 - base_f1:.3f}")

    st.markdown("---")
    left, right = st.columns(2)

    # Valid JSON & Tool Match grouped bar
    with left:
        metrics_bar = {
            "Model": models * 2,
            "Value": (
                [r[k]["valid_json_rate"] for k in keys]
                + [r[k]["exact_tool_match_rate"] for k in keys]
            ),
            "Metric": (["Valid JSON Rate (%)"] * 3) + (["Tool Match Rate (%)"] * 3),
        }
        fig = px.bar(
            metrics_bar,
            x="Model",
            y="Value",
            color="Model",
            barmode="group",
            facet_col="Metric",
            color_discrete_map=MODEL_COLORS,
            title="Valid JSON & Tool Match Rates",
            text_auto=".1f",
        )
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Hallucination rate
    with right:
        hall_data = {
            "Model": models,
            "Hallucination Rate (%)": [r[k]["hallucination_rate"] for k in keys],
        }
        fig = px.bar(
            hall_data,
            x="Model",
            y="Hallucination Rate (%)",
            color="Model",
            color_discrete_map=MODEL_COLORS,
            title="Hallucination Rate (lower is better)",
            text_auto=".1f",
        )
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Parameter metrics radar
    st.subheader("Parameter-level Metrics (Precision / Recall / F1 / ROUGE-L)")
    categories = ["Precision", "Recall", "F1", "ROUGE-L"]
    fig = go.Figure()
    for model_label, key in zip(models, keys):
        vals = [
            r[key]["parameter_precision"],
            r[key]["parameter_recall"],
            r[key]["parameter_f1"],
            r[key]["rouge_l"],
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=model_label,
                line_color=MODEL_COLORS[model_label],
                opacity=0.6,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        title="Parameter Metrics Radar",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Observations
    with st.expander("Key Observations from Experiment"):
        for obs in results.get("observations", []):
            st.markdown(f"- {obs}")

    # Full metrics table
    with st.expander("Full Metrics Table"):
        rows = []
        for label, key in zip(models, keys):
            rv = r[key]
            rows.append({
                "Model": label,
                "Valid JSON %": rv["valid_json_rate"],
                "Tool Match %": rv["exact_tool_match_rate"],
                "Param Precision": rv["parameter_precision"],
                "Param Recall": rv["parameter_recall"],
                "Param F1": rv["parameter_f1"],
                "Hallucination %": rv["hallucination_rate"],
                "ROUGE-L": rv["rouge_l"],
            })
        import pandas as pd
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Prompt Formatter Explorer
# ---------------------------------------------------------------------------
def tab_formatter(schemas: list[dict]) -> None:
    st.header("🔧 Prompt Formatter Explorer")
    st.markdown(
        "See exactly how function schemas are embedded into the system prompt "
        "and how training examples are structured — without needing a real tokenizer."
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Available Functions")
        fn_names = [s["name"] for s in schemas]
        selected_fns = st.multiselect(
            "Select functions to include",
            fn_names,
            default=fn_names[:2],
            key="fn_select",
        )
        selected_schemas = [s for s in schemas if s["name"] in selected_fns]

    with col_b:
        st.subheader("User Request")
        user_request = st.text_area(
            "Enter a natural language request",
            value="What's the weather like in London right now?",
            height=100,
        )

    st.markdown("---")
    st.subheader("Generated System Prompt")
    if selected_schemas:
        system_prompt = build_system_prompt(selected_schemas)
        st.code(system_prompt, language="markdown")
    else:
        st.info("Select at least one function above.")

    st.subheader("ChatML-style Training Format (manual fallback)")
    if selected_schemas and user_request:
        messages = [
            {"role": "system", "content": build_system_prompt(selected_schemas)},
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": ''},
        ]
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        training_str = "\n".join(parts)
        st.code(training_str, language="markdown")

    # JSON extraction demo
    st.markdown("---")
    st.subheader("Function Call Extraction")
    st.markdown(
        "Paste a raw model output below — the extractor will parse and highlight "
        "any embedded JSON function call."
    )
    raw_output = st.text_area(
        "Raw model output",
        value='Sure! <functioncall> {"name": "get_weather", "arguments": {"location": "London", "units": "celsius"}}',
        height=90,
        key="raw_out",
    )
    extracted = extract_function_call_from_output(raw_output)
    if extracted:
        try:
            parsed = json.loads(extracted)
            st.success("Valid function call extracted:")
            st.json(parsed)
        except json.JSONDecodeError:
            st.warning(f"Extracted string (not valid JSON): `{extracted}`")
    else:
        st.warning("No function call found in the output.")


# ---------------------------------------------------------------------------
# Tab 3 — DPO Pair Builder
# ---------------------------------------------------------------------------
def tab_dpo(schemas: list[dict]) -> None:
    st.header("🧪 DPO Preference Pair Builder")
    st.markdown(
        "DPO needs **(prompt, chosen, rejected)** triplets. Because "
        "[glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) "
        "only provides correct completions, rejected completions are generated "
        "synthetically using four strategies."
    )

    # Strategy explanation
    strategy_info = {
        "wrong_function": "Swap the function name for a different one in the schema",
        "wrong_params": "Corrupt argument values while keeping the function name",
        "missing_params": "Drop half of the required parameters",
        "hallucinate": "Call a function name that does not exist in the schema",
    }
    c1, c2, c3, c4 = st.columns(4)
    for col, (strategy, desc) in zip([c1, c2, c3, c4], strategy_info.items()):
        col.info(f"**{strategy.replace('_', ' ').title()}**  \n{desc}")

    st.markdown("---")

    # DPO pair statistics from results
    results = load_results()
    dpo_pairs_info = results["results"]["dpo"].get("dpo_pairs", {})
    if dpo_pairs_info:
        st.subheader("DPO Pairs Generated During Training")
        pair_labels = [k.replace("_", " ").title() for k in list(dpo_pairs_info.keys())[1:]]
        pair_counts = list(dpo_pairs_info.values())[1:]
        fig = px.pie(
            names=pair_labels,
            values=pair_counts,
            title=f"Total DPO pairs: {dpo_pairs_info.get('total', 0):,}",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Live pair generation
    st.markdown("---")
    st.subheader("Live Pair Generation Demo")
    fn_names = [s["name"] for s in schemas]
    selected_fn = st.selectbox("Pick a function to call correctly", fn_names, index=0)
    fn_schema = next(s for s in schemas if s["name"] == selected_fn)

    # Build a minimal chosen example
    sample_args: dict = {}
    for param, spec in fn_schema["parameters"]["properties"].items():
        pt = spec.get("type", "string")
        if pt == "string":
            sample_args[param] = spec.get("enum", [f"example_{param}"])[0]
        elif pt == "number":
            sample_args[param] = 42.0
        elif pt == "integer":
            sample_args[param] = 60
        elif pt == "boolean":
            sample_args[param] = True

    chosen_call = json.dumps({"name": selected_fn, "arguments": sample_args})
    example = {
        "prompt": "User request that matches the function above",
        "completion": chosen_call,
        "available_functions": schemas,
    }

    seed = st.slider("Random seed", 0, 200, 42, key="dpo_seed")
    pairs = build_dpo_pairs([example], seed=seed)

    if pairs:
        pair = pairs[0]
        rc, rj = st.columns(2)
        with rc:
            st.markdown("**Chosen (correct)**")
            try:
                st.json(json.loads(pair["chosen"]))
            except Exception:
                st.code(pair["chosen"])
        with rj:
            st.markdown(f"**Rejected** — strategy: `{pair.get('rejection_strategy', 'n/a')}`")
            try:
                st.json(json.loads(pair["rejected"]))
            except Exception:
                st.code(pair["rejected"])
    else:
        st.info("Could not generate a rejection for this example — try a different seed.")

    with st.expander("Show all available functions (used as rejection pool)"):
        st.json(schemas)


# ---------------------------------------------------------------------------
# Tab 4 — Architecture & Pipeline
# ---------------------------------------------------------------------------
def tab_pipeline(results: dict) -> None:
    st.header("⚙️ Architecture & Training Pipeline")

    st.markdown(
        """
### Fine-tuning Pipeline Overview

```
Raw Dataset (glaive-function-calling-v2)
         │
         ▼
  DataLoader + Formatter          ← src/data/
  (ChatML prompt, system prompt with schemas)
         │
         ├──── SFT Training ─────► SFT LoRA Checkpoint
         │     (TRL SFTTrainer)        (r=16, α=32, QLoRA 4-bit NF4)
         │
         └──── DPO Pair Builder ──► DPO Training ──► DPO LoRA Checkpoint
               (4 rejection          (TRL DPOTrainer,   (β=0.1, sigmoid loss)
                strategies)           β=0.1)
                                         │
                                         ▼
                                  FunctionCallingPipeline ← src/inference/
                                  (adapter merge + generation)
                                         │
                                         ▼
                                  ModelEvaluator ← src/evaluation/
                                  (JSON validity, tool match, param F1, ROUGE-L)
```
"""
    )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("LoRA Configuration (SFT)")
        sft_lora = results["results"]["sft_lora"].get("lora_config", {})
        sft_train = results["results"]["sft_lora"].get("training_config", {})
        params = results["results"]["sft_lora"].get("params", {})
        st.markdown(
            f"""
| Parameter | Value |
|---|---|
| Rank (r) | {sft_lora.get('r', '–')} |
| Alpha | {sft_lora.get('lora_alpha', '–')} |
| Dropout | {sft_lora.get('lora_dropout', '–')} |
| Target modules | q/k/v/o + gate/up/down proj |
| RSLoRA | {sft_lora.get('use_rslora', False)} |
| Trainable params | {params.get('trainable', '–')} / {params.get('total', '–')} ({params.get('trainable_pct', '–')}%) |
| Quantisation | {sft_train.get('quantisation', '–')} |
| Max seq length | {sft_train.get('max_seq_length', '–')} |
| Epochs | {sft_train.get('num_train_epochs', '–')} |
| Learning rate | {sft_train.get('learning_rate', '–')} |
| Effective batch | {sft_train.get('effective_batch_size', '–')} |
| LR scheduler | {sft_train.get('lr_scheduler', '–')} |
| Packing | {sft_train.get('packing', '–')} |
| Training time | {results['results']['sft_lora'].get('training_time', '–')} |
| Hardware | {results['results']['sft_lora'].get('hardware', '–')} |
| Peak VRAM | {results['results']['sft_lora'].get('peak_vram_gb', '–')} GB |
"""
        )

    with c2:
        st.subheader("DPO Configuration")
        dpo_c = results["results"]["dpo"].get("dpo_config", {})
        st.markdown(
            f"""
| Parameter | Value |
|---|---|
| Beta (β) | {dpo_c.get('beta', '–')} |
| Loss type | {dpo_c.get('loss_type', '–')} |
| Label smoothing | {dpo_c.get('label_smoothing', '–')} |
| Epochs | {dpo_c.get('num_train_epochs', '–')} |
| Learning rate | {dpo_c.get('learning_rate', '–')} |
| Effective batch | {dpo_c.get('effective_batch_size', '–')} |
| LR scheduler | {dpo_c.get('lr_scheduler', '–')} |
| Training time | {results['results']['dpo'].get('training_time', '–')} |
| Hardware | {results['results']['dpo'].get('hardware', '–')} |
| Peak VRAM | {results['results']['dpo'].get('peak_vram_gb', '–')} GB |
"""
        )

    # Improvement summary
    st.markdown("---")
    st.subheader("Step-by-Step Improvement Summary")
    imps = results.get("improvements", {})

    imp_data = {
        "Stage": ["Base → SFT", "SFT → DPO", "Base → DPO"],
        "Valid JSON": [
            imps["base_to_sft"]["valid_json_rate"],
            imps["sft_to_dpo"]["valid_json_rate"],
            imps["base_to_dpo"]["valid_json_rate"],
        ],
        "Tool Match": [
            imps["base_to_sft"]["exact_tool_match_rate"],
            imps["sft_to_dpo"]["exact_tool_match_rate"],
            imps["base_to_dpo"]["exact_tool_match_rate"],
        ],
        "Param F1": [
            imps["base_to_sft"]["parameter_f1"],
            imps["sft_to_dpo"]["parameter_f1"],
            imps["base_to_dpo"]["parameter_f1"],
        ],
        "Hallucination": [
            imps["base_to_sft"]["hallucination_rate"],
            imps["sft_to_dpo"]["hallucination_rate"],
            imps["base_to_dpo"]["hallucination_rate"],
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(imp_data).set_index("Stage"), use_container_width=True)

    # Live metrics evaluation playground
    st.markdown("---")
    st.subheader("Metrics Playground")
    st.markdown(
        "Evaluate a predicted function call against a reference using the same "
        "metrics used in the paper."
    )
    mc1, mc2 = st.columns(2)
    with mc1:
        ref_text = st.text_area(
            "Reference (ground truth)",
            value='{"name": "get_weather", "arguments": {"location": "London", "units": "celsius"}}',
            height=100,
            key="ref_text",
        )
    with mc2:
        pred_text = st.text_area(
            "Prediction (model output)",
            value='{"name": "get_weather", "arguments": {"location": "London"}}',
            height=100,
            key="pred_text",
        )

    schemas = load_schemas()
    allowed_fns = [s["name"] for s in schemas]
    if st.button("Compute Metrics", type="primary"):
        try:
            metrics = FunctionCallMetrics()
            evaluate_single(pred_text, ref_text, allowed_fns, metrics)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Valid JSON", "Yes" if metrics.valid_json_rate == 100 else "No")
            mc2.metric("Tool Match", "Yes" if metrics.exact_tool_match_rate == 100 else "No")
            mc3.metric("Param F1", f"{metrics.parameter_f1:.4f}")
            mc4.metric("Hallucination", "Yes" if metrics.hallucination_rate == 100 else "No")
        except Exception as exc:
            st.error(f"Error computing metrics: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    results = load_results()
    schemas = load_schemas()

    render_sidebar(results)

    st.title("🤖 LoRA / DPO Fine-tuning for Function Calling")
    st.markdown(
        "Fine-tuning **Qwen2.5-1.5B-Instruct** for structured function-call generation "
        "using **QLoRA** (4-bit NF4) for SFT and **Direct Preference Optimisation (DPO)** "
        "with synthetically generated preference pairs — achieving **93.7% valid JSON** "
        "and **79.8% exact tool match** from a 23.4% / 18.7% base."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Results Dashboard", "🔧 Formatter Explorer", "🧪 DPO Pair Builder", "⚙️ Pipeline & Architecture"]
    )
    with tab1:
        tab_results(results)
    with tab2:
        tab_formatter(schemas)
    with tab3:
        tab_dpo(schemas)
    with tab4:
        tab_pipeline(results)


if __name__ == "__main__":
    main()
