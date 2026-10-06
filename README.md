# HealthWise Insurance — Smart Premium Engine
**Sparsh Saraya · Machine Learning (MAIB) · 30 September 2026**

## Headline result
The current flat-pricing model misprices by **$6,167** per policy on held-out
customers. The two-stage engine misprices by **$1,535** — a **75.1% reduction**,
with R² rising from **0.715 to 0.919**. Measured end-to-end, with Stage 1
predicting its own tiers rather than being handed them.

## Files

| File | What it is |
|---|---|
| `HealthWise_Sparsh_Saraya.ipynb` | Submission notebook. All 61 code cells executed with output visible; Tasks A1–E3, all 20 Reflections, board memo, both bonus tasks. |
| `app.py` | Eight-tab underwriting console (Streamlit), with a live quote calculator. |
| `hw_theme.py` | Design system — palettes, chart chrome, CSS. Every colour choice is justified against a colour-vision validator. |
| `.streamlit/config.toml` | App theme so Streamlit's own widgets match the chart palette. |
| `HealthWise_Report_Sparsh_Saraya.docx` | 40-page report with compliance matrix and 19 embedded figures. |
| `healthwise_model.joblib` | Saved two-stage engine (1 classifier + 3 per-tier regressors, each in a Pipeline with its encoder). |
| `healthwise_enriched.csv` | All 1,200 policies with Health Score, predicted tier, cost, premium and an individual explanation. |
| `confidence_calibration.json` | Out-of-fold calibration behind the auto-quote threshold. |
| `data_dictionary.csv`, `healthwise_cleaned.csv`, `results.json` | Supporting data layers and every computed metric. |
| `figures/` | 19 analysis figures + dashboard screenshots. |

## Running the console

```bash
pip install streamlit plotly scikit-learn pandas joblib
streamlit run app.py
```

`healthwise_model.joblib`, `healthwise_enriched.csv`, `results.json`,
`confidence_calibration.json` and `hw_theme.py` must sit beside `app.py`.

## Running the notebook

```bash
jupyter notebook HealthWise_Sparsh_Saraya.ipynb
```

`healthwise.csv` must be in the same folder. A single fixed `RANDOM_STATE = 42`
is used throughout, so results reproduce exactly.

## Three decisions worth flagging to the marker

1. **Stage-1 tree depth is 4, not the template's 5.** CV accuracy peaks at depth 4
   (0.9242 vs 0.9150), test accuracy is higher, and the train–test gap is smaller.
   Full sweep in Task B3.

2. **Every headline figure uses the *predicted* tier, not the true one.** Scoring
   the tier models against the actual `risk_tier` label reports a flattering MAE
   of $816, but that label does not exist at quote time — the oracle figure
   understates real pricing error by 47%. Both are reported side by side.

3. **The console's colour palette is validated, not chosen by eye.** The earlier
   palette failed a colour-vision check: its mid-blue and teal sat only 11.3
   Delta-E apart in normal vision, below the 15 floor, so two series were
   genuinely hard to separate. See `hw_theme.py` for the replacement and its
   justification.
