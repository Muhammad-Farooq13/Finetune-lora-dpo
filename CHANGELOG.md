# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] - 2026-03-12

### Added
- **`streamlit_app.py`** — interactive demo with four tabs (no GPU required):
  - **📊 Results Dashboard** — grouped bar charts, radar chart, and full metrics
    table comparing Base / SFT-LoRA / DPO across valid JSON rate, tool match,
    parameter F1, hallucination rate, and ROUGE-L
  - **🔧 Formatter Explorer** — live system-prompt generator from function schemas;
    ChatML training-format preview; JSON function-call extractor
  - **🧪 DPO Pair Builder** — visualises chosen/rejected pair generation using
    all four rejection strategies (wrong_function, wrong_params, missing_params,
    hallucinate); pie chart of training-pair distribution
  - **⚙️ Pipeline & Architecture** — ASCII pipeline diagram, full LoRA/DPO config
    tables, improvement summary table, live metrics evaluation playground
- **`.streamlit/config.toml`** — purple AI theme (`#6A1B9A`)
- **`runtime.txt`** and **`packages.txt`** for Streamlit Cloud deployment
- `streamlit>=1.36.0` and `plotly>=5.18.0` added to `requirements.txt`

### Changed
- **CI** (`ci.yml`): `codecov/codecov-action@v4` upgraded to `@v5`

### Fixed
- Nothing broken in this release — all 46 unit tests passed from the start.

### Notes
- The Streamlit demo uses `src.data.formatter`, `src.data.preference_builder`,
  and `src.evaluation.metrics` directly — no model weights are loaded.
- Heavy training dependencies (torch, bitsandbytes, trl, peft) remain in
  `requirements.txt`; CI continues to use the lightweight `requirements-ci.txt`.