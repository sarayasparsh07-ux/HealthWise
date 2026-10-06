# HealthWise Insurance — Smart Premium Engine
**Sparsh Saraya · Machine Learning (MAIB) · 29 September 2026**

## Headline result
The current flat-pricing model misprices by **$6,167** per policy on held-out customers.
The two-stage engine misprices by **$1,535** — a **75.1% reduction**, with R² rising from
**0.715 to 0.919**. These figures are measured end-to-end, with Stage 1 predicting its own
tiers rather than being given them.

## Files

| File | What it is |
|---|---|
| `HealthWise_Sparsh_Saraya.ipynb` | The submission notebook. All 61 code cells executed with output visible; Tasks A1–E3, all 20 Reflections, board memo and both bonus tasks. |
| `app.py` | Eight-tab Streamlit dashboard with a live premium calculator. |
| `HealthWise_Report_Sparsh_Saraya.docx` | 40-page consolidated report with compliance matrix and 19 embedded figures. |
| `healthwise_model.joblib` | The saved two-stage engine (1 classifier + 3 per-tier regressors, each in a Pipeline with its encoder). |
| `healthwise_enriched.csv` | All 1,200 customers with Health Score, predicted tier, predicted cost, final premium and an individual explanation. |
| `healthwise_cleaned.csv` | The cleaned layer, before feature engineering. |
| `data_dictionary.csv` | Every final column documented by layer and meaning. |
| `test_customers.csv` / `unseen_customers_scored.csv` | Task E3 unseen-customer template and its scored output. |
| `healthwise_core.py` / `results.json` | Shared analysis library and every computed metric. |
| `figures/` | 19 analysis figures + 2 dashboard screenshots. |

## Running the app

```bash
pip install streamlit plotly scikit-learn pandas joblib
streamlit run app.py
```

`healthwise_model.joblib`, `healthwise_enriched.csv` and `results.json` must sit in the same
folder as `app.py`.

## Running the notebook

```bash
jupyter notebook HealthWise_Sparsh_Saraya.ipynb
```

`healthwise.csv` must be in the same folder. A single fixed `RANDOM_STATE = 42` is used
throughout, so all results reproduce exactly.

## Two decisions worth flagging to the marker

1. **Stage-1 tree depth is 4, not the template's 5.** Cross-validated accuracy peaks at
   depth 4 (0.9242 vs 0.9150), test accuracy is higher (0.9458 vs 0.9375), and the
   train–test gap is smaller. The full sweep is in Task B3.

2. **Every headline figure uses the *predicted* tier, not the true one.** Evaluating the
   per-tier models against the actual `risk_tier` label gives a far more flattering MAE of
   $816, but that assumes perfect tier assignment, which is not available at quote time.
   The oracle figure understates the engine's true pricing error by 47%. Both numbers are
   reported side by side throughout.
