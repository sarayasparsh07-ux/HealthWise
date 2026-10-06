"""
================================================================================
HEALTHWISE INSURANCE — SMART PREMIUM ENGINE
Underwriting analytics console
================================================================================
Author : Sparsh Saraya  |  Machine Learning (MAIB)

Run with:
    streamlit run app.py

Files required alongside this one:
    healthwise_model.joblib      the trained two-stage engine
    healthwise_enriched.csv      the enriched dataset (a premium for every customer)
    results.json                 pre-computed model metrics
    confidence_calibration.json  referral-threshold calibration (optional)

DESIGN NOTES
------------
Colour is handled in hw_theme.py, where the choice of every palette is justified
against a colour-vision validator. Two structural rules are followed here:

  1. NO DUAL-AXIS CHARTS. Two measures on two y-scales invent a correlation that
     is not in the data. Where the old build paired customer counts with mean
     premium on one plot, this build uses two aligned charts sharing one x order.
  2. EVERY CHART HAS A TABLE TWIN. Colour is never the only way to read a value;
     each chart card carries a "Data table" expander with the same numbers.
================================================================================
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import hw_theme as T

# =============================================================== PAGE SETUP ==
st.set_page_config(page_title="HealthWise — Smart Premium Engine",
                   page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

if "dark" not in st.session_state:
    st.session_state.dark = False

P = T.palette(st.session_state.dark)
TIERS = T.TIERS
LAYOUT = T.plotly_layout(P)

# Tier marks carry BOTH colour and shape. The single-hue tier ramp is correct for
# adjacent marks (bars), but as SCATTERED points its adjacent steps measure only
# ~14.4 normal-vision Delta-E apart — below the 15 floor, i.e. genuinely hard to
# separate. Rather than break colour consistency between charts, scatters add
# shape as a second channel, so identity never rests on hue alone.
TIER_SYMBOL = {"Low": "circle", "Medium": "diamond", "High": "square"}
TIGHTEN = """
<style>
  div[data-testid="stSlider"] { padding-bottom: 0.05rem; }
  div[data-testid="stSlider"] label { margin-bottom: 0; }
  div[data-testid="stVerticalBlock"] { gap: 0.55rem; }
</style>
"""
st.markdown(T.css(P, st.session_state.dark), unsafe_allow_html=True)
st.markdown(TIGHTEN, unsafe_allow_html=True)


# ============================================================ DATA LOADING ===
@st.cache_resource
def load_model():
    return joblib.load("healthwise_model.joblib")


@st.cache_data
def load_data():
    d = pd.read_csv("healthwise_enriched.csv")
    for c, cats in [("risk_tier", TIERS), ("predicted_tier", TIERS),
                    ("health_band", ["Poor", "Fair", "Good", "Excellent"])]:
        if c in d.columns:
            d[c] = pd.Categorical(d[c], categories=cats, ordered=True)
    return d


@st.cache_data
def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


try:
    M = load_model()
    df = load_data()
except FileNotFoundError as err:
    st.error(f"**Required file not found:** `{getattr(err, 'filename', err)}`\n\n"
             "Keep `healthwise_model.joblib` and `healthwise_enriched.csv` in the "
             "same folder as `app.py`, then reload.")
    st.stop()

R = load_json("results.json")
CAL = load_json("confidence_calibration.json")

CLF, REGS = M["clf"], M["regs"]
FEATURES, NUM, MED = M["features"], M["num"], M["med"]
LOADING = M.get("loading", 0.23)
WEIGHTS = M.get("weights", {"smoker": .4597, "age": .2871,
                            "bmi": .2442, "exercise_freq": .0090})

# Referral threshold, calibrated from out-of-fold data (see the Calculator tab).
REFER_BELOW = 0.80
TIER_RESID = CAL.get("tier_residuals", {
    "Low":    {"p10": -935,  "p90": 571},
    "Medium": {"p10": -1368, "p90": 1345},
    "High":   {"p10": -2335, "p90": 3175},
})


# ================================================== THE TWO-STAGE ENGINE =====
def _score_smoker(s):  return 0.0 if str(s).lower() == "yes" else 100.0
def _score_age(a):     return float(np.clip(100.0 * (64.0 - a) / 46.0, 0, 100))
def _score_exercise(f): return float(np.clip(100.0 * f / 7.0, 0, 100))


def _score_bmi(b):
    pen = (18.5 - b) * 8.0 if b < 18.5 else (0.0 if b <= 24.9 else (b - 24.9) * 4.5)
    return float(np.clip(100.0 - pen, 0, 100))


def health_score(age, bmi, smoker, exercise):
    return round(_score_smoker(smoker) * WEIGHTS["smoker"]
                 + _score_age(age) * WEIGHTS["age"]
                 + _score_bmi(bmi) * WEIGHTS["bmi"]
                 + _score_exercise(exercise) * WEIGHTS["exercise_freq"], 2)


def health_band(s):
    return "Excellent" if s >= 75 else "Good" if s >= 55 else "Fair" if s >= 35 else "Poor"


def quote(customer: dict) -> dict:
    """
    Stage 1 classify -> Stage 2 price with THAT tier's regressor.

    Also returns the classifier's confidence in the tier it chose, and a premium
    range built from that tier's own out-of-fold residual spread (P10-P90), so
    the number is presented with its uncertainty rather than as a false point
    estimate.
    """
    row = pd.DataFrame([customer])
    for k in NUM:
        if k not in row or pd.isna(row.at[0, k]):
            row[k] = MED[k]

    proba = CLF.predict_proba(row[FEATURES])[0]
    classes = list(CLF.classes_)
    tier = classes[int(np.argmax(proba))]
    conf = float(np.max(proba))

    cost = float(REGS[tier].predict(row[FEATURES])[0])
    premium = cost * (1 + LOADING)

    res = TIER_RESID.get(tier, {"p10": -1500, "p90": 1500})
    lo = (cost - res["p90"]) * (1 + LOADING)
    hi = (cost - res["p10"]) * (1 + LOADING)

    return {"tier": tier, "confidence": conf,
            "probs": dict(zip(classes, proba.round(4))),
            "cost": round(cost, 2), "premium": round(premium, 2),
            "premium_lo": round(max(lo, 0), 2), "premium_hi": round(hi, 2),
            "refer": conf < REFER_BELOW}


def explain(age, bmi, smoker, exercise, children, tier, hs):
    drivers, protectors = [], []
    if str(smoker).lower() == "yes":
        drivers.append(("Smoker", "is a smoker"))
    else:
        protectors.append("is a non-smoker")
    if bmi >= 30:
        drivers.append(("High BMI", f"has a BMI of {bmi:.1f} (obese range)"))
    elif bmi >= 25:
        drivers.append(("Raised BMI", f"has a BMI of {bmi:.1f} (overweight range)"))
    elif bmi < 18.5:
        drivers.append(("Low BMI", f"has a BMI of {bmi:.1f} (underweight range)"))
    else:
        protectors.append(f"has a healthy BMI of {bmi:.1f}")
    if age >= 50:
        drivers.append(("Age", f"is {int(age)} years old"))
    elif age <= 30:
        protectors.append(f"is young ({int(age)})")
    if exercise <= 2:
        drivers.append(("Low activity", f"exercises only {int(exercise)}x per week"))
    elif exercise >= 5:
        protectors.append(f"exercises {int(exercise)}x per week")
    if children >= 3:
        drivers.append(("Dependents", f"has {int(children)} dependents"))

    hs_txt = ("a strong Health Score" if hs >= 70 else
              "a mid-range Health Score" if hs >= 45 else "a low Health Score")
    level = {"High": "high", "Medium": "moderate", "Low": "low"}[tier]
    if drivers and tier != "Low":
        sent = (f"The premium is {level} mainly because this customer "
                f"{', '.join(d[1] for d in drivers[:3])}. Combined with {hs_txt} "
                f"({hs:.0f}/100), this places them in the {tier} risk tier.")
    elif drivers:
        sent = (f"The premium is low overall. The main factor pushing it up is that "
                f"this customer {', '.join(d[1] for d in drivers[:3])}, but with "
                f"{hs_txt} ({hs:.0f}/100) they still sit in the Low risk tier.")
    else:
        sent = (f"The premium is {level} because this customer "
                f"{', '.join(protectors[:3])}. With {hs_txt} ({hs:.0f}/100) they sit "
                f"in the {tier} risk tier.")
    return sent, [d[0] for d in drivers[:3]], protectors[:3]


# ================================================================ HELPERS ====
def money(v, dp=0):
    return f"${v:,.{dp}f}"


def kpi(label, value, sub="", kind="accent"):
    cls = {"accent": "", "good": "is-good", "warn": "is-warn",
           "crit": "is-crit", "neutral": "is-neutral"}[kind]
    st.markdown(f'<div class="kpi {cls}"><div class="k">{label}</div>'
                f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
                unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def card_open(title, sub=""):
    st.markdown(f'<div class="card-h">{title}</div>'
                f'<div class="card-s">{sub}</div>', unsafe_allow_html=True)


def chart(fig, table: pd.DataFrame | None = None, key=None, height=None,
          table_label="Data table", fmt=None):
    """
    Render a chart, then its table twin so no value is ever colour-only.

    Shared chrome is applied first, then any margin the caller set is restored,
    so a compact chart keeps its own spacing instead of being overwritten by
    the house defaults.
    """
    own = fig.layout.margin
    has_own = any(getattr(own, side) is not None for side in ("l", "r", "t", "b"))
    fig.update_layout(**LAYOUT)
    if has_own:
        fig.update_layout(margin=own)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displayModeBar": False, "responsive": True})
    if table is not None:
        with st.expander(table_label):
            styled = table.style.format(fmt) if fmt else table
            st.dataframe(styled, width="stretch", hide_index=True)


def bar_gap(fig):
    """2px surface gap between adjacent bars, per the mark spec."""
    fig.update_traces(marker_line_width=2, marker_line_color=P["surface"])
    return fig


# ================================================================ MASTHEAD ===
run_stamp = datetime.now().strftime("%d %b %Y, %H:%M")
stage1_acc = R.get("end_to_end", {}).get("stage1_acc", 0.9458)

st.markdown(f"""
<div class="hw-top">
  <div class="hw-brand">
    <div class="hw-mark">HW</div>
    <div>
      <div class="hw-name">HealthWise — Smart Premium Engine</div>
      <div class="hw-sub">Two-stage underwriting console · classify the risk tier,
        then price within it</div>
    </div>
  </div>
  <div class="hw-meta">
    <div class="hw-meta-item"><span class="k">Policies in book</span>
      <span class="v">{len(df):,}</span></div>
    <div class="hw-meta-item"><span class="k">Engine</span>
      <span class="v">v1.0 · tree d4 + 3 GLMs</span></div>
    <div class="hw-meta-item"><span class="k">Tier accuracy</span>
      <span class="v">{stage1_acc:.1%}</span></div>
    <div class="hw-meta-item"><span class="k">Session</span>
      <span class="v">{run_stamp}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ================================================================= SIDEBAR ===
with st.sidebar:
    st.markdown('<div class="side-h">Appearance</div>', unsafe_allow_html=True)
    dark = st.toggle("Dark mode", value=st.session_state.dark,
                     help="Both themes use separately validated colour steps, "
                          "not an automatic inversion.")
    if dark != st.session_state.dark:
        st.session_state.dark = dark
        st.rerun()

    st.markdown('<div class="side-h">Portfolio filters</div>', unsafe_allow_html=True)
    st.caption("Scopes every portfolio tab. The calculator always uses the full "
               "trained engine.")
    f_smoker = st.multiselect("Smoker status", ["no", "yes"], default=["no", "yes"])
    f_tier = st.multiselect("Risk tier", TIERS, default=TIERS)
    f_region = st.multiselect("Region", sorted(df.region.unique()),
                              default=sorted(df.region.unique()))
    f_age = st.slider("Age range", int(df.age.min()), int(df.age.max()),
                      (int(df.age.min()), int(df.age.max())))
    f_band = st.multiselect("Health band", ["Poor", "Fair", "Good", "Excellent"],
                            default=["Poor", "Fair", "Good", "Excellent"])

    st.markdown('<div class="side-h">Pricing basis</div>', unsafe_allow_html=True)
    st.caption(f"Premium = predicted claim cost × {1 + LOADING:.2f}  \n"
               "(15% expense ratio + 8% profit and solvency margin). "
               "An assumption, not a finding — every premium scales with it.")

mask = (df.smoker.isin(f_smoker) & df.risk_tier.isin(f_tier)
        & df.region.isin(f_region) & df.age.between(*f_age)
        & df.health_band.isin(f_band))
D = df[mask]

if len(D) == 0:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### No policies match these filters")
    st.markdown("Every portfolio view needs at least one policy. Widen a filter in "
                "the sidebar — clearing **Risk tier** or **Health band** usually "
                "brings the book back.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if len(D) < len(df):
    st.markdown(
        f'<span class="pill mut">Filtered view · {len(D):,} of {len(df):,} '
        f'policies ({len(D)/len(df)*100:.0f}%)</span>', unsafe_allow_html=True)

TAB = st.tabs(["Overview", "Portfolio", "Health Score", "Premiums",
               "Stage 1 · Classify", "Stage 2 · Price", "Model diagnostics",
               "Quote calculator"])


# ============================================================ 1. OVERVIEW ====
with TAB[0]:
    e = R.get("end_to_end", {})
    st.markdown("#### Executive overview")
    st.caption(f"Book of {len(D):,} policies under the current filters. "
               "Premiums are model estimates priced out-of-fold.")

    c = st.columns(4)
    with c[0]:
        kpi("Policies", f"{len(D):,}",
            f"{len(D)/len(df)*100:.0f}% of the full book", "neutral")
    with c[1]:
        kpi("Mean premium", money(D.final_premium.mean()),
            f"median {money(D.final_premium.median())}")
    with c[2]:
        kpi("Mean Health Score", f"{D.health_score.mean():.1f}",
            "0–100 · higher is healthier", "good")
    with c[3]:
        smk = (D.smoker == "yes").mean() * 100
        kpi("Smokers in book", f"{smk:.1f}%",
            f"{(D.smoker=='yes').sum():,} policies", "warn")

    st.write("")
    c = st.columns(4)
    with c[0]:
        kpi("Expected claims", money(D.predicted_cost.sum() / 1e6, 2) + "M",
            "total predicted annual cost", "neutral")
    with c[1]:
        kpi("Premium written", money(D.final_premium.sum() / 1e6, 2) + "M",
            f"gross of the {LOADING*100:.0f}% loading", "neutral")
    with c[2]:
        hi = (D.predicted_tier == "High").mean() * 100
        kpi("High-risk share", f"{hi:.1f}%",
            f"{(D.predicted_tier=='High').sum():,} policies", "crit")
    with c[3]:
        if e:
            kpi("Pricing error cut", f"{e['mae_improvement_pct']:.0f}%",
                f"MAE {money(e['global']['MAE'])} → {money(e['predicted_tier']['MAE'])}",
                "good")
        else:
            kpi("Pricing error cut", "—", "results.json not loaded", "neutral")

    st.write("")

    # --- THE DUAL-AXIS FIX -------------------------------------------------
    # Previously: customers (bars) and mean premium (line) shared one plot on
    # two y-scales. That is the single most misleading chart pattern there is —
    # the two scales align arbitrarily, so the reader infers a relationship the
    # data never asserted. Replaced with two aligned charts on one x order.
    g = (D.groupby("predicted_tier", observed=True)
           .agg(policies=("age", "size"), mean_premium=("final_premium", "mean"),
                mean_cost=("predicted_cost", "mean"))
           .reindex(TIERS).fillna(0).reset_index())
    g["share"] = (g.policies / g.policies.sum() * 100).round(1)

    left, right = st.columns(2)
    with left:
        card_open("Policies by risk tier", "How the book is distributed")
        fig = px.bar(g, x="predicted_tier", y="policies", color="predicted_tier",
                     color_discrete_map=P["tier"], text="policies",
                     category_orders={"predicted_tier": TIERS},
                     labels={"predicted_tier": "Risk tier", "policies": "Policies"})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                          textfont_size=11.5, cliponaxis=False)
        bar_gap(fig)
        fig.update_layout(showlegend=False, yaxis_title="Number of policies",
                          xaxis_title="")
        chart(fig, g[["predicted_tier", "policies", "share"]]
              .rename(columns={"predicted_tier": "Risk tier",
                               "policies": "Policies", "share": "Share %"}),
              key="ov_mix", height=330)
    with right:
        card_open("Mean premium by risk tier", "Same tier order, its own scale")
        fig = px.bar(g, x="predicted_tier", y="mean_premium", color="predicted_tier",
                     color_discrete_map=P["tier"], text="mean_premium",
                     category_orders={"predicted_tier": TIERS},
                     labels={"predicted_tier": "Risk tier"})
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=11.5, cliponaxis=False)
        bar_gap(fig)
        fig.update_layout(showlegend=False,
                          yaxis_title="Mean annual premium (USD)", xaxis_title="")
        chart(fig, g[["predicted_tier", "mean_premium", "mean_cost"]]
              .round(0).rename(columns={"predicted_tier": "Risk tier",
                                        "mean_premium": "Mean premium",
                                        "mean_cost": "Mean predicted cost"}),
              key="ov_prem", height=330,
              fmt={"Mean premium": "${:,.0f}", "Mean predicted cost": "${:,.0f}"})

    note("These are deliberately <b>two charts, not one</b>. Plotting policy "
         "counts and mean premium against two y-axes on a single plot would let "
         "the two scales be aligned arbitrarily, implying a relationship the data "
         "never asserts. Two charts sharing one tier order compare honestly.")

    if e:
        st.write("")
        card_open("Does the two-stage engine beat one flat price?",
                  "Held-out customers, Stage 1 predicting its own tiers")
        cmp_df = pd.DataFrame([
            {"Approach": "Flat model (today)", "MAE": e["global"]["MAE"],
             "RMSE": e["global"]["RMSE"], "R²": e["global"]["R2"]},
            {"Approach": "Two-stage engine", "MAE": e["predicted_tier"]["MAE"],
             "RMSE": e["predicted_tier"]["RMSE"], "R²": e["predicted_tier"]["R2"]},
        ])
        cL, cR = st.columns([1.25, 1])
        with cL:
            fig = px.bar(cmp_df, x="MAE", y="Approach", orientation="h",
                         color="Approach",
                         color_discrete_map={"Flat model (today)": P["muted"],
                                             "Two-stage engine": P["cat"][0]},
                         text="MAE")
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                              textfont_size=12, cliponaxis=False)
            bar_gap(fig)
            fig.update_layout(showlegend=False, yaxis_title="",
                              xaxis_title="Mean absolute pricing error (USD)",
                              xaxis_range=[0, e["global"]["MAE"] * 1.22])
            chart(fig, cmp_df, key="ov_cmp", height=250,
                  fmt={"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R²": "{:.4f}"})
        with cR:
            st.write("")
            kpi("Average quote is wrong by", money(e["predicted_tier"]["MAE"]),
                f"was {money(e['global']['MAE'])} under flat pricing", "good")
            st.write("")
            kpi("Variance explained", f"{e['predicted_tier']['R2']:.3f}",
                f"up from {e['global']['R2']:.3f}", "good")

        note(f"<b>The honest number.</b> Scoring the tier models against the "
             f"<i>true</i> tier label would report {money(e['oracle_tier']['MAE'])} "
             f"— but that label does not exist at quote time. Letting Stage 1 "
             f"predict its own tiers gives {money(e['predicted_tier']['MAE'])}. "
             f"The flattering figure understates real pricing error by "
             f"{(1 - e['oracle_tier']['MAE'] / e['predicted_tier']['MAE']) * 100:.0f}%, "
             f"so every headline here uses the predicted tier.", "warn")


# =========================================================== 2. PORTFOLIO ====
with TAB[1]:
    st.markdown("#### Portfolio explorer")
    st.caption("Search, inspect and export the priced book.")

    c = st.columns([2, 1.1, 1.1])
    with c[0]:
        q = st.text_input("Search policy ID", placeholder="e.g. HW1042",
                          label_visibility="visible")
    with c[1]:
        lo, hi = float(df.final_premium.min()), float(df.final_premium.max())
        pr = st.slider("Premium band (USD)", lo, hi, (lo, hi), step=500.0)
    with c[2]:
        sort_by = st.selectbox("Sort by", ["Premium (high → low)",
                                           "Premium (low → high)",
                                           "Health Score (low → high)",
                                           "Age (high → low)"])

    V = D[D.final_premium.between(*pr)]
    if q:
        V = V[V.customer_id.str.contains(q, case=False, na=False)]
    V = V.sort_values(
        {"Premium (high → low)": "final_premium",
         "Premium (low → high)": "final_premium",
         "Health Score (low → high)": "health_score",
         "Age (high → low)": "age"}[sort_by],
        ascending=sort_by in ("Premium (low → high)", "Health Score (low → high)"))

    if len(V) == 0:
        note("No policies match that search. Clear the ID box or widen the "
             "premium band.", "warn")
    else:
        st.caption(f"{len(V):,} policies · mean premium "
                   f"{money(V.final_premium.mean())} · "
                   f"{(V.predicted_tier == 'High').sum():,} high-risk")
        show = ["customer_id", "age", "sex", "bmi", "children", "smoker", "region",
                "exercise_freq", "health_score", "health_band", "annual_charge",
                "predicted_tier", "predicted_cost", "final_premium",
                "top_risk_factors"]
        st.dataframe(
            V[show].rename(columns={
                "customer_id": "Policy", "age": "Age", "sex": "Sex", "bmi": "BMI",
                "children": "Deps", "smoker": "Smoker", "region": "Region",
                "exercise_freq": "Exercise", "health_score": "Health",
                "health_band": "Band", "annual_charge": "Actual charge",
                "predicted_tier": "Tier", "predicted_cost": "Predicted cost",
                "final_premium": "Premium", "top_risk_factors": "Top risk factors"}),
            width="stretch", height=420, hide_index=True,
            column_config={
                "Premium": st.column_config.NumberColumn(format="$%d"),
                "Predicted cost": st.column_config.NumberColumn(format="$%d"),
                "Actual charge": st.column_config.NumberColumn(format="$%d"),
                "Health": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=100),
                "BMI": st.column_config.NumberColumn(format="%.1f"),
            })

        c = st.columns([1, 1, 2.4])
        with c[0]:
            st.download_button("Download this view", V.to_csv(index=False),
                               "healthwise_filtered.csv", "text/csv",
                               width="stretch")
        with c[1]:
            st.download_button("Download full book", df.to_csv(index=False),
                               "healthwise_enriched.csv", "text/csv",
                               width="stretch")

        st.markdown("---")
        card_open("Policy detail", "Full risk profile and pricing rationale")
        pick = st.selectbox("Select a policy", V.customer_id.tolist()[:2000],
                            label_visibility="collapsed")
        if pick:
            r = df[df.customer_id == pick].iloc[0]
            c = st.columns(5)
            with c[0]: kpi("Age / sex", f"{int(r.age)} · {r.sex}",
                           f"{r.region} region", "neutral")
            with c[1]: kpi("BMI", f"{r.bmi:.1f}", str(r.bmi_category), "neutral")
            with c[2]: kpi("Health Score", f"{r.health_score:.1f}",
                           str(r.health_band), "good")
            with c[3]: kpi("Risk tier", str(r.predicted_tier), "Stage 1 output")
            with c[4]: kpi("Annual premium", money(r.final_premium),
                           f"claim cost {money(r.predicted_cost)}", "good")
            note(f"<b>Why this premium?</b> {r.premium_explanation}")
            st.caption(f"Top contributing risk factors: {r.top_risk_factors}")


# ======================================================== 3. HEALTH SCORE ====
with TAB[2]:
    st.markdown("#### Health Score")
    st.caption("Health Score = Σ (Metric Score × Weight) · 0–100, higher is healthier")

    c = st.columns(4)
    with c[0]: kpi("Mean", f"{D.health_score.mean():.1f}", "across the filtered book")
    with c[1]: kpi("Median", f"{D.health_score.median():.1f}", "")
    with c[2]: kpi("Lowest", f"{D.health_score.min():.1f}", "highest-risk policy", "crit")
    with c[3]: kpi("Highest", f"{D.health_score.max():.1f}", "healthiest policy", "good")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        card_open("Weighting", "Derived from standardised coefficients on the "
                               "training split only — never assigned by opinion")
        w = pd.DataFrame({"Metric": ["Smoker", "Age", "BMI", "Exercise"],
                          "Weight": [WEIGHTS["smoker"], WEIGHTS["age"],
                                     WEIGHTS["bmi"], WEIGHTS["exercise_freq"]]})
        w["Weight %"] = (w.Weight * 100).round(2)
        fig = px.bar(w.sort_values("Weight"), x="Weight %", y="Metric",
                     orientation="h", text="Weight %",
                     color_discrete_sequence=[P["cat"][0]])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          textfont_size=11.5, cliponaxis=False)
        bar_gap(fig)
        fig.update_layout(xaxis_title="Share of the Health Score (%)",
                          yaxis_title="", xaxis_range=[0, 56])
        chart(fig, w[["Metric", "Weight %"]], key="hs_w", height=300)
        note("Exercise earns <b>0.90%</b> and that is left as the data produced "
             "it. Its signal is largely absorbed by BMI, and the tuned Lasso "
             "eliminated it independently. Inflating the weight by hand would "
             "make the score a statement of belief rather than a measurement.")
    with right:
        card_open("Distribution", "Band thresholds at 35 / 55 / 75")
        fig = px.histogram(D, x="health_score", nbins=34,
                           color_discrete_sequence=[P["cat"][0]])
        fig.update_traces(marker_line_width=1, marker_line_color=P["surface"])
        for th, lb in [(35, "Fair"), (55, "Good"), (75, "Excellent")]:
            fig.add_vline(x=th, line_dash="solid", line_width=1,
                          line_color=P["axis"], annotation_text=lb,
                          annotation_position="top",
                          annotation_font=dict(size=10, color=P["muted"]))
        fig.update_layout(xaxis_title="Health Score (0–100, higher is healthier)",
                          yaxis_title="Policies")
        desc = D.health_score.describe().round(2).to_frame("Value").reset_index()
        desc.columns = ["Statistic", "Value"]
        chart(fig, desc, key="hs_dist", height=300)

    st.write("")
    card_open("Does the score track real cost?",
              "Mean observed annual charge in each band")
    hb = (D.groupby("health_band", observed=True)
            .agg(policies=("age", "size"), mean_score=("health_score", "mean"),
                 mean_charge=("annual_charge", "mean"),
                 mean_premium=("final_premium", "mean"))
            .reindex(["Poor", "Fair", "Good", "Excellent"]).dropna().reset_index())
    if len(hb):
        fig = px.bar(hb, x="health_band", y="mean_charge", text="mean_charge",
                     color_discrete_sequence=[P["cat"][0]],
                     labels={"health_band": "Health band"})
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=11.5, cliponaxis=False)
        bar_gap(fig)
        fig.update_layout(yaxis_title="Mean annual charge (USD)", xaxis_title="")
        chart(fig, hb.round(1).rename(columns={
                  "health_band": "Band", "policies": "Policies",
                  "mean_score": "Mean score", "mean_charge": "Mean charge",
                  "mean_premium": "Mean premium"}),
              key="hs_band", height=330,
              fmt={"Mean charge": "${:,.0f}", "Mean premium": "${:,.0f}"})
        if {"Poor", "Excellent"}.issubset(set(hb.health_band)):
            p = hb.loc[hb.health_band == "Poor", "mean_charge"].iat[0]
            x = hb.loc[hb.health_band == "Excellent", "mean_charge"].iat[0]
            note(f"Poor-band policies cost <b>{p/x:.1f}×</b> more than "
                 f"Excellent-band policies ({money(p)} vs {money(x)}), and the "
                 f"bands order monotonically. The score separates real cost risk "
                 f"rather than merely ranking people.")


# ============================================================ 4. PREMIUMS ====
with TAB[3]:
    st.markdown("#### Premium analysis")
    st.caption(f"Premium = predicted annual claim cost × {1 + LOADING:.2f} "
               "(15% expenses + 8% margin)")

    c = st.columns(4)
    with c[0]: kpi("Mean", money(D.final_premium.mean()), "per policy per year")
    with c[1]: kpi("Median", money(D.final_premium.median()), "less skew-sensitive")
    with c[2]: kpi("Lowest", money(D.final_premium.min()), "healthiest policy", "good")
    with c[3]: kpi("Highest", money(D.final_premium.max()), "highest-risk policy", "crit")

    st.write("")
    left, right = st.columns(2)
    with left:
        card_open("Premium spread by tier", "Box: median, quartiles, full range")
        fig = px.box(D, x="predicted_tier", y="final_premium",
                     color="predicted_tier", color_discrete_map=P["tier"],
                     category_orders={"predicted_tier": TIERS},
                     labels={"predicted_tier": "Risk tier"}, points=False)
        fig.update_layout(showlegend=False, xaxis_title="",
                          yaxis_title="Annual premium (USD)")
        tt = (D.groupby("predicted_tier", observed=True).final_premium
                .describe()[["count", "min", "25%", "50%", "75%", "max"]]
                .reindex(TIERS).dropna().round(0).reset_index())
        chart(fig, tt.rename(columns={"predicted_tier": "Tier", "count": "n",
                                      "50%": "median"}),
              key="pr_tier", height=350)
    with right:
        card_open("Premium spread by smoker status",
                  "The single largest pricing differential in the book")
        fig = px.box(D, x="smoker", y="final_premium", color="smoker",
                     color_discrete_map=P["smoker"],
                     category_orders={"smoker": ["no", "yes"]},
                     labels={"smoker": "Smoker"}, points=False)
        fig.update_layout(showlegend=False, xaxis_title="",
                          yaxis_title="Annual premium (USD)")
        st_ = (D.groupby("smoker", observed=True)
                 .agg(policies=("age", "size"),
                      mean_premium=("final_premium", "mean"),
                      median_premium=("final_premium", "median"),
                      mean_health=("health_score", "mean")).round(1).reset_index())
        chart(fig, st_.rename(columns={"smoker": "Smoker", "policies": "Policies",
                                       "mean_premium": "Mean premium",
                                       "median_premium": "Median premium",
                                       "mean_health": "Mean Health Score"}),
              key="pr_smk", height=350,
              fmt={"Mean premium": "${:,.0f}", "Median premium": "${:,.0f}"})

    if {"yes", "no"}.issubset(set(D.smoker)):
        gap = (D[D.smoker == "yes"].final_premium.mean()
               - D[D.smoker == "no"].final_premium.mean())
        note(f"Smokers are priced <b>{money(gap)}</b> higher per year on average. "
             f"Crucially, smokers and non-smokers in this book are near-identical "
             f"on age and BMI, so the gap cannot be explained away as smokers "
             f"simply being older or heavier.")

    st.write("")
    card_open("Premium against Health Score",
              "The pricing gradient, coloured by the tier that priced it")
    fig = px.scatter(D, x="health_score", y="final_premium",
                     color="predicted_tier", color_discrete_map=P["tier"],
                     symbol="predicted_tier", symbol_map=TIER_SYMBOL,
                     category_orders={"predicted_tier": TIERS}, opacity=.72,
                     hover_data={"customer_id": True, "age": True, "bmi": ":.1f",
                                 "smoker": True, "health_score": ":.1f",
                                 "final_premium": ":$,.0f"},
                     labels={"health_score": "Health Score (0–100, higher is healthier)",
                             "final_premium": "Annual premium (USD)",
                             "predicted_tier": "Risk tier"})
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color=P["surface"])))
    fig.update_layout(yaxis_title="Annual premium (USD)")
    rho = D.health_score.corr(D.final_premium, method="spearman")
    band = (D.assign(b=pd.cut(D.health_score, [0, 35, 55, 75, 101],
                              labels=["Poor", "Fair", "Good", "Excellent"],
                              right=False))
             .groupby("b", observed=True)
             .agg(policies=("age", "size"), mean_premium=("final_premium", "mean"),
                  min_premium=("final_premium", "min"),
                  max_premium=("final_premium", "max")).round(0).reset_index())
    chart(fig, band.rename(columns={"b": "Health band", "policies": "Policies",
                                    "mean_premium": "Mean premium",
                                    "min_premium": "Min", "max_premium": "Max"}),
          key="pr_hs", height=420,
          fmt={"Mean premium": "${:,.0f}", "Min": "${:,.0f}", "Max": "${:,.0f}"})
    note(f"Premium falls steadily as Health Score rises (Spearman ρ = "
         f"<b>{rho:.2f}</b>). The tiers appear as separated bands rather than one "
         f"continuous line — that is the two-stage engine visibly pricing each "
         f"segment with its own model.")
    st.caption("Tiers are marked by shape as well as colour here — circle, diamond, "
               "square. As scattered points the tier ramp's steps fall just below "
               "the separation threshold, so shape carries the distinction that "
               "colour alone would not.")


# ================================================== 5. STAGE 1 CLASSIFY ======
with TAB[4]:
    st.markdown("#### Stage 1 — risk tier classification")
    if not R:
        note("`results.json` not found, so model metrics are unavailable. The "
             "rest of the console works from the enriched dataset.", "warn")
    else:
        b = R["clf_best"]
        c = st.columns(4)
        with c[0]: kpi("Model", "Decision tree · depth 4",
                       "chosen on cross-validated accuracy", "neutral")
        with c[1]: kpi("Test accuracy", f"{b['test_acc']:.1%}",
                       f"train {b['train_acc']:.1%}", "good")
        with c[2]: kpi("Macro F1", f"{b['report']['macro avg']['f1-score']:.3f}",
                       "balanced across all three tiers")
        with c[3]: kpi("Train–test gap", f"{b['train_acc'] - b['test_acc']:+.3f}",
                       "near zero — well balanced", "good")

        st.write("")
        left, right = st.columns([1, 1.08])
        with left:
            card_open("Confusion matrix", "Rows are actual, columns predicted")
            cm = np.array(b["confusion_matrix"])
            fig = px.imshow(cm, x=TIERS, y=TIERS, text_auto=True,
                            color_continuous_scale=[[0, P["surface"]],
                                                    [1, P["tier"]["High"]]],
                            labels=dict(x="Predicted tier", y="Actual tier",
                                        color="Policies"))
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(side="bottom")
            cmdf = pd.DataFrame(cm, columns=TIERS)
            cmdf.insert(0, "Actual", TIERS)
            chart(fig, cmdf, key="cl_cm", height=340)
        with right:
            card_open("Per-tier performance", "Precision, recall and F1")
            rep = b["report"]
            met = pd.DataFrame({t: {k: rep[t][k] for k in
                                    ["precision", "recall", "f1-score"]}
                                for t in TIERS}).T.reset_index()
            met.columns = ["Tier", "Precision", "Recall", "F1"]
            mlong = met.melt(id_vars="Tier", var_name="Metric", value_name="Score")
            fig = px.bar(mlong, x="Tier", y="Score", color="Metric",
                         barmode="group", color_discrete_sequence=P["cat"],
                         category_orders={"Tier": TIERS}, text="Score")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                              textfont_size=10, cliponaxis=False)
            bar_gap(fig)
            fig.update_layout(yaxis_range=[0, 1.14], xaxis_title="",
                              yaxis_title="Score")
            chart(fig, met.round(3), key="cl_met", height=340)

        hi_low = int(cm[TIERS.index("High"), TIERS.index("Low")])
        note(f"All confusion sits between <b>neighbouring</b> tiers. "
             f"{hi_low} High-risk customers were labelled Low, and none of the "
             f"reverse — the model never makes a two-tier jump, which bounds the "
             f"financial damage of any single mistake.", "good")

        st.write("")
        card_open("Why depth 4, not the depth 5 in the brief",
                  "Cross-validated accuracy across the depth sweep")
        dc = pd.DataFrame(R["depth_curve"])
        fig = go.Figure()
        fig.add_scatter(x=dc.depth, y=dc.train, name="Training",
                        mode="lines+markers", line=dict(color=P["muted"], width=2),
                        marker=dict(size=7))
        fig.add_scatter(x=dc.depth, y=dc.cv, name="Cross-validation",
                        mode="lines+markers", line=dict(color=P["cat"][0], width=2.5),
                        marker=dict(size=8))
        fig.add_vline(x=4, line_width=1, line_color=P["axis"],
                      annotation_text="chosen", annotation_position="top",
                      annotation_font=dict(size=10, color=P["muted"]))
        fig.update_layout(xaxis_title="Maximum tree depth", yaxis_title="Accuracy")
        chart(fig, dc.round(4).rename(columns={"depth": "Depth", "train": "Train",
                                               "test": "Test", "cv": "CV"}),
              key="cl_depth", height=330)
        note("CV accuracy peaks at depth 4 (0.9242 against 0.9150 at depth 5) and "
             "test accuracy is higher too. Past depth 6 training accuracy climbs "
             "toward 1.00 while CV falls — textbook overfitting, visible rather "
             "than asserted.")


# ===================================================== 6. STAGE 2 PRICE ======
with TAB[5]:
    st.markdown("#### Stage 2 — charge prediction within each tier")
    if not R:
        note("`results.json` not found, so model metrics are unavailable.", "warn")
    else:
        e = R["end_to_end"]
        c = st.columns(4)
        with c[0]: kpi("Flat model MAE", money(e["global"]["MAE"]),
                       f"R² {e['global']['R2']:.3f}", "crit")
        with c[1]: kpi("Two-stage MAE", money(e["predicted_tier"]["MAE"]),
                       f"R² {e['predicted_tier']['R2']:.3f}", "good")
        with c[2]: kpi("Error reduction", f"{e['mae_improvement_pct']:.0f}%",
                       "measured end-to-end", "good")
        with c[3]: kpi("Stage-1 accuracy", f"{e['stage1_acc']:.1%}",
                       "tier assigned correctly")

        st.write("")
        left, right = st.columns(2)
        with left:
            card_open("Predicted against actual", "Out-of-fold, every policy")
            fig = px.scatter(D, x="annual_charge", y="predicted_cost",
                             color="predicted_tier", color_discrete_map=P["tier"],
                             symbol="predicted_tier", symbol_map=TIER_SYMBOL,
                             category_orders={"predicted_tier": TIERS}, opacity=.7,
                             labels={"annual_charge": "Actual annual charge (USD)",
                                     "predicted_cost": "Predicted cost (USD)",
                                     "predicted_tier": "Risk tier"})
            fig.update_traces(marker=dict(size=6, line=dict(width=1,
                                                            color=P["surface"])))
            lim = [D.annual_charge.min(), D.annual_charge.max()]
            fig.add_scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                            line=dict(color=P["muted"], width=1.5, dash="dot"))
            pt = pd.DataFrame(R["per_tier"]).T.reset_index()
            pt.columns.values[0] = "Tier"
            chart(fig, pt[["Tier", "n", "R2", "MAE", "RMSE", "cv_R2"]].round(4),
                  key="rg_av", height=380,
                  fmt={"MAE": "${:,.0f}", "RMSE": "${:,.0f}"})
        with right:
            card_open("Residuals", "Predicted minus actual, out-of-fold")
            res = D.predicted_cost - D.annual_charge
            rd = pd.DataFrame({"Predicted cost (USD)": D.predicted_cost,
                               "Residual (USD)": res,
                               "Risk tier": D.predicted_tier.astype(str)})
            fig = px.scatter(rd, x="Predicted cost (USD)", y="Residual (USD)",
                             color="Risk tier", color_discrete_map=P["tier"],
                             symbol="Risk tier", symbol_map=TIER_SYMBOL,
                             category_orders={"Risk tier": TIERS}, opacity=.7)
            fig.update_traces(marker=dict(size=6, line=dict(width=1,
                                                            color=P["surface"])))
            fig.add_hline(y=0, line_width=1, line_color=P["axis"])
            rtab = (D.assign(residual=res).groupby("predicted_tier", observed=True)
                     .residual.describe()[["count", "mean", "std", "min", "max"]]
                     .reindex(TIERS).dropna().round(0).reset_index())
            chart(fig, rtab.rename(columns={"predicted_tier": "Tier",
                                            "count": "n"}),
                  key="rg_res", height=380)

        note(f"Residuals cluster tightly around zero for most policies. The "
             f"scattered large ones are almost entirely Stage-1 tier "
             f"misclassifications: those carry a mean error of "
             f"<b>{money(e['MAE_misclassified'])}</b> against "
             f"<b>{money(e['MAE_correct'])}</b> when the tier is right — an "
             f"{e['MAE_misclassified'] / e['MAE_correct']:.0f}× difference. "
             f"That concentration, not the average, is what governs whether the "
             f"engine is safe to deploy.", "warn")

        st.write("")
        card_open("What actually drives cost inside each tier",
                  "Tuned Lasso coefficients, standardised so they compare directly")
        sw = R["alpha_sweeps"]
        lc = pd.DataFrame({t: sw[t]["lasso_coefficients"] for t in TIERS})
        lc = lc.loc[lc.abs().max(axis=1).sort_values(ascending=False).index]
        lf = lc.reset_index().melt(id_vars="index", var_name="Tier",
                                   value_name="Coefficient")
        lf.columns = ["Feature", "Tier", "Coefficient"]
        fig = px.bar(lf, x="Coefficient", y="Feature", color="Tier",
                     orientation="h", barmode="group",
                     color_discrete_map=P["tier"],
                     category_orders={"Tier": TIERS})
        bar_gap(fig)
        fig.add_vline(x=0, line_width=1, line_color=P["axis"])
        fig.update_layout(xaxis_title="USD per 1 standard deviation of the feature",
                          yaxis_title="")
        chart(fig, lc.round(1).reset_index().rename(columns={"index": "Feature"}),
              key="rg_coef", height=420)
        note("The dominant driver <b>changes by tier</b> — age in Low, smoking in "
             "Medium, BMI in High. That is the core argument for fitting three "
             "models: one global model must apply a single fixed coefficient per "
             "feature to everybody. Note also that all three region bars are "
             "essentially zero, so <b>geography should not be a rating factor</b>.")


# ================================================ 7. MODEL DIAGNOSTICS =======
with TAB[6]:
    st.markdown("#### Model diagnostics")
    st.caption("Bias and variance read off evidence, not asserted from theory.")
    if not R:
        note("`results.json` not found, so diagnostics are unavailable.", "warn")
    else:
        left, right = st.columns(2)
        with left:
            card_open("Stage-1 learning curve", "Do the curves converge?")
            lc = R["learning_curves"]["classifier_tree_best"]
            fig = go.Figure()
            fig.add_scatter(x=lc["sizes"], y=lc["train_mean"], name="Training",
                            mode="lines+markers", line=dict(color=P["muted"], width=2),
                            marker=dict(size=7))
            fig.add_scatter(x=lc["sizes"], y=lc["val_mean"], name="Cross-validation",
                            mode="lines+markers",
                            line=dict(color=P["cat"][0], width=2.5),
                            marker=dict(size=8))
            fig.update_layout(xaxis_title="Training samples", yaxis_title="Accuracy")
            t1 = pd.DataFrame({"Training samples": np.round(lc["sizes"]),
                               "Training": np.round(lc["train_mean"], 4),
                               "Cross-validation": np.round(lc["val_mean"], 4)})
            chart(fig, t1, key="dg_lc1", height=340)
            note("The curves <b>converge</b> — variance is under control.", "good")
        with right:
            card_open("Flat-model learning curve", "Converged low is high bias")
            lg = R["learning_curves"]["global_linear"]
            fig = go.Figure()
            fig.add_scatter(x=lg["sizes"], y=lg["train_mean"], name="Training R²",
                            mode="lines+markers", line=dict(color=P["muted"], width=2),
                            marker=dict(size=7))
            fig.add_scatter(x=lg["sizes"], y=lg["val_mean"], name="Cross-validation R²",
                            mode="lines+markers",
                            line=dict(color=P["cat"][0], width=2.5),
                            marker=dict(size=8))
            fig.add_hline(y=R["two_stage_oracle"]["R2"], line_width=1,
                          line_color=P["axis"],
                          annotation_text="two-stage ceiling",
                          annotation_position="bottom right",
                          annotation_font=dict(size=10, color=P["muted"]))
            fig.update_layout(xaxis_title="Training samples", yaxis_title="R²",
                              yaxis_range=[0, 1.05])
            t2 = pd.DataFrame({"Training samples": np.round(lg["sizes"]),
                               "Training R²": np.round(lg["train_mean"], 4),
                               "Cross-validation R²": np.round(lg["val_mean"], 4)})
            chart(fig, t2, key="dg_lc2", height=340)
            note("Both curves flatten <b>low, together</b> — the signature of high "
                 "bias. More data would not help; only a change of structure "
                 "would, which is what tiering provides.", "warn")

        st.write("")
        card_open("Generalisation gap by candidate model",
                  "Train accuracy minus test accuracy — larger means more overfitting")
        cc = pd.DataFrame(R["classification_comparison"])
        cc["Status"] = np.where(cc.gap > .02, "Overfitting", "Balanced")
        fig = px.bar(cc, x="gap", y="model", orientation="h", color="Status",
                     color_discrete_map={"Overfitting": P["cat"][1],
                                         "Balanced": P["cat"][2]},
                     text="gap")
        fig.update_traces(texttemplate="%{text:+.3f}", textposition="outside",
                          textfont_size=10.5, cliponaxis=False)
        bar_gap(fig)
        fig.add_vline(x=0, line_width=1, line_color=P["axis"])
        fig.update_layout(xaxis_title="Train − test accuracy", yaxis_title="")
        chart(fig, cc[["model", "train_acc", "test_acc", "gap", "f1_macro",
                       "cv_mean", "cv_std"]].round(4)
                .rename(columns={"model": "Model", "train_acc": "Train",
                                 "test_acc": "Test", "gap": "Gap",
                                 "f1_macro": "Macro F1", "cv_mean": "CV mean",
                                 "cv_std": "CV std"}),
              key="dg_gap", height=380)

        st.write("")
        card_open("Fairness check", "Is the engine equally accurate for every group?")
        fr = R.get("fairness", {})
        rows = []
        for attr in ["sex", "region", "age_group"]:
            dd = fr.get(attr, {})
            for grp in dd.get("n", {}):
                rows.append({"Attribute": attr, "Group": str(grp),
                             "Policies": int(dd["n"][grp]),
                             "Mean actual": dd["mean_actual"][grp],
                             "Mean premium": dd["mean_premium"][grp],
                             "MAE": dd["MAE"][grp],
                             "MAE % of mean": dd["MAE_pct_of_actual"][grp]})
        if rows:
            fdf = pd.DataFrame(rows)
            fig = px.bar(fdf, x="MAE % of mean", y="Group", orientation="h",
                         color="Attribute", color_discrete_sequence=P["cat"],
                         text="MAE % of mean")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont_size=10.5, cliponaxis=False)
            bar_gap(fig)
            fig.update_layout(
                xaxis_title="Model error as % of that group's own mean charge",
                yaxis_title="")
            chart(fig, fdf.round(1), key="dg_fair", height=400,
                  fmt={"Mean actual": "${:,.0f}", "Mean premium": "${:,.0f}",
                       "MAE": "${:,.0f}"})
            note("Relative error sits in a narrow band across sex, region and age "
                 "group — no demographic group is systematically mispriced "
                 "relative to its own cost level. <b>Separately from accuracy</b>: "
                 "<code>sex</code> is statistically significant but immutable, and "
                 "gender-based pricing is prohibited in several jurisdictions. The "
                 "recommendation is to exclude it from production pricing — the "
                 "accuracy cost is negligible, the legal exposure is not.", "warn")


# ==================================================== 8. QUOTE CALCULATOR ====
with TAB[7]:
    st.markdown("#### Quote calculator")
    st.caption("Enter a customer's details. Tier, cost and premium recalculate "
               "on every change.")

    left, right = st.columns([0.92, 1.55], gap="large")

    with left:
        st.markdown('<div class="card-h">Applicant details</div>',
                    unsafe_allow_html=True)
        preset = st.selectbox(
            "Start from", ["Custom", "Young, healthy, non-smoker",
                           "Mid-career, average", "Older, raised BMI",
                           "Young smoker", "Borderline case"],
            help="Presets fill the inputs; every control stays editable.")
        DEFAULTS = {
            "Custom":                     (40, 27.0, 1, 3, "male", "no", "north"),
            "Young, healthy, non-smoker": (26, 22.0, 0, 5, "female", "no", "north"),
            "Mid-career, average":        (41, 27.5, 2, 3, "male", "no", "south"),
            "Older, raised BMI":          (58, 32.5, 2, 2, "male", "no", "east"),
            "Young smoker":               (33, 26.4, 1, 4, "female", "yes", "west"),
            # A genuine 50/50 leaf — the tree is truly undecided here, so this
            # preset demonstrates the referral path rather than illustrating it.
            "Borderline case":            (32, 29.0, 2, 2, "male", "no", "south"),
        }
        d0 = DEFAULTS[preset]
        k = preset.replace(" ", "_").replace(",", "")

        age = st.slider("Age (years)", 18, 64, d0[0], key=f"a_{k}")
        bmi = st.slider("BMI (kg/m²)", 16.0, 50.0, d0[1], 0.1, key=f"b_{k}")
        children = st.slider("Dependents", 0, 4, d0[2], key=f"c_{k}")
        exercise = st.slider("Workouts per week", 0, 7, d0[3], key=f"e_{k}")
        sex = st.selectbox("Sex", ["male", "female"],
                           index=["male", "female"].index(d0[4]), key=f"s_{k}")
        smoker = st.selectbox("Smoker", ["no", "yes"],
                              index=["no", "yes"].index(d0[5]), key=f"m_{k}")
        region = st.selectbox("Region", ["north", "south", "east", "west"],
                              index=["north", "south", "east", "west"].index(d0[6]),
                              key=f"r_{k}")
        st.caption("Sex and region are shown because the model was trained with "
                   "them. Both are recommended for removal from production "
                   "pricing — see Model diagnostics.")

    cust = dict(age=age, bmi=bmi, children=children, exercise_freq=exercise,
                sex=sex, smoker=smoker, region=region)
    Q = quote(cust)
    hs = health_score(age, bmi, smoker, exercise)
    band = health_band(hs)
    sentence, drivers, protectors = explain(age, bmi, smoker, exercise, children,
                                            Q["tier"], hs)
    ref_no = f"HW-Q{abs(hash((age, bmi, children, exercise, sex, smoker, region))) % 10**6:06d}"

    with right:
        st.markdown('<div class="card-h">Engine output</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-s">Quote reference {ref_no} · '
                    f'generated {datetime.now().strftime("%d %b %Y, %H:%M")}</div>',
                    unsafe_allow_html=True)

        if Q["refer"]:
            st.markdown(
                f'<div class="note crit"><span class="pill crit">⚠ Refer to '
                f'underwriting</span><br><br>Stage 1 is only '
                f'<b>{Q["confidence"]:.0%}</b> confident of the '
                f'<b>{Q["tier"]}</b> tier, below the {REFER_BELOW:.0%} '
                f'auto-quote threshold. Quotes in this band are misclassified far '
                f'more often, so this applicant should be priced by a human rather '
                f'than bound automatically.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="note good"><span class="pill good">✓ Cleared for '
                f'auto-quote</span> &nbsp;Stage 1 is <b>{Q["confidence"]:.0%}</b> '
                f'confident of the <b>{Q["tier"]}</b> tier, above the '
                f'{REFER_BELOW:.0%} threshold.</div>', unsafe_allow_html=True)

        c = st.columns(4)
        with c[0]:
            kpi("Health Score", f"{hs:.1f}", band,
                "good" if hs >= 55 else "warn" if hs >= 35 else "crit")
        with c[1]:
            kpi("Risk tier", Q["tier"], f"{Q['confidence']:.0%} confidence",
                "neutral")
        with c[2]:
            kpi("Expected claim", money(Q["cost"]), "annual cost")
        with c[3]:
            kpi("Annual premium", money(Q["premium"]),
                f"cost × {1 + LOADING:.2f}", "good")

        st.markdown(
            f'<div class="note"><b>Likely range {money(Q["premium_lo"])} – '
            f'{money(Q["premium_hi"])}.</b> The headline premium is a point '
            f'estimate; this band is the 10th–90th percentile of the '
            f'{Q["tier"]}-tier model\'s own out-of-fold errors, so it shows how '
            f'precisely this tier can actually be priced.</div>',
            unsafe_allow_html=True)

        # --- tier probability, the basis of the referral decision ------------
        pr = pd.DataFrame({"Tier": list(Q["probs"].keys()),
                           "Probability": list(Q["probs"].values())})
        pr["Tier"] = pd.Categorical(pr.Tier, categories=TIERS, ordered=True)
        pr = pr.sort_values("Tier")
        fig = px.bar(pr, x="Probability", y="Tier", orientation="h",
                     color="Tier", color_discrete_map=P["tier"],
                     text="Probability", category_orders={"Tier": TIERS})
        fig.update_traces(texttemplate="%{text:.0%}", textposition="outside",
                          textfont_size=11, cliponaxis=False)
        bar_gap(fig)
        fig.add_vline(x=REFER_BELOW, line_width=1.5, line_color=P["critical"],
                      annotation_text=f"auto-quote floor {REFER_BELOW:.0%}",
                      annotation_position="top",
                      annotation_font=dict(size=10, color=P["critical"]))
        fig.update_layout(showlegend=False, xaxis_range=[0, 1.14],
                          xaxis_tickformat=".0%", yaxis_title="",
                          xaxis_title="Stage-1 tier probability", height=210,
                          margin=dict(l=62, r=26, t=16, b=44))
        chart(fig, pr.assign(Probability=(pr.Probability * 100).round(1))
                     .rename(columns={"Probability": "Probability %"}),
              key="qc_prob", table_label="Tier probabilities")

        st.markdown("**Why this premium?**")
        st.markdown(f'<div class="note">{sentence}</div>', unsafe_allow_html=True)

        c = st.columns(2)
        with c[0]:
            st.markdown("**Risk factors**")
            if drivers:
                for d_ in drivers:
                    st.markdown(f'<span class="pill crit">▲ {d_}</span>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill good">✓ None elevated</span>',
                            unsafe_allow_html=True)
        with c[1]:
            st.markdown("**Protective factors**")
            if protectors:
                for p_ in protectors:
                    st.markdown(
                        f'<span class="pill good">✓ {p_.capitalize()}</span>',
                        unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill mut">None recorded</span>',
                            unsafe_allow_html=True)

    st.markdown("---")

    # ----------------------------------------------- Health Score breakdown --
    c1, c2 = st.columns(2)
    with c1:
        card_open("Health Score breakdown",
                  f"How the {hs:.1f} is composed, metric by metric")
        parts = pd.DataFrame({
            "Metric": ["Smoker", "Age", "BMI", "Exercise"],
            "Score": [_score_smoker(smoker), _score_age(age),
                      _score_bmi(bmi), _score_exercise(exercise)],
            "Weight": [WEIGHTS["smoker"], WEIGHTS["age"],
                       WEIGHTS["bmi"], WEIGHTS["exercise_freq"]]})
        parts["Contribution"] = (parts.Score * parts.Weight).round(2)
        parts["Maximum"] = (100 * parts.Weight).round(2)
        parts["Forgone"] = (parts.Maximum - parts.Contribution).round(2)
        pl = parts.melt(id_vars="Metric", value_vars=["Contribution", "Forgone"],
                        var_name="Part", value_name="Points")
        fig = px.bar(pl, x="Points", y="Metric", color="Part", orientation="h",
                     color_discrete_map={"Contribution": P["cat"][0],
                                         "Forgone": P["grid"]},
                     category_orders={"Metric": ["Smoker", "Age", "BMI", "Exercise"],
                                      "Part": ["Contribution", "Forgone"]})
        bar_gap(fig)
        fig.update_layout(barmode="stack", yaxis_title="",
                          xaxis_title="Points contributed (of the metric's maximum)")
        chart(fig, parts[["Metric", "Score", "Contribution", "Maximum", "Forgone"]],
              key="qc_hs", height=280,
              fmt={"Score": "{:.1f}", "Contribution": "{:.2f}",
                   "Maximum": "{:.2f}", "Forgone": "{:.2f}"})
        note("The grey remainder is the headroom this applicant has not captured "
             "— it shows where a Health Score could realistically improve.")

    with c2:
        card_open("What would change the premium",
                  "One attribute at a time, everything else held constant")
        scen = [("Quit smoking", {"smoker": "no"}) if smoker == "yes"
                else ("Start smoking", {"smoker": "yes"}),
                ("Age +10", {"age": min(64, age + 10)}),
                ("Age −10", {"age": max(18, age - 10)}),
                ("BMI +5", {"bmi": min(50.0, bmi + 5)}),
                ("BMI −5", {"bmi": max(16.0, bmi - 5)}),
                ("Dependents +2", {"children": min(4, children + 2)}),
                ("Exercise 7/week", {"exercise_freq": 7})]
        rows = []
        for lbl, ch in scen:
            q2 = quote({**cust, **ch})
            rows.append({"Scenario": lbl, "New tier": q2["tier"],
                         "New premium": q2["premium"],
                         "Change": q2["premium"] - Q["premium"],
                         "Crosses tier": "yes" if q2["tier"] != Q["tier"] else "no"})
        w = pd.DataFrame(rows).sort_values("Change")
        fig = px.bar(w, x="Change", y="Scenario", orientation="h",
                     color="Crosses tier",
                     color_discrete_map={"yes": P["cat"][1], "no": P["muted"]},
                     text="Change", category_orders={"Crosses tier": ["yes", "no"]})
        fig.update_traces(texttemplate="%{text:+$,.0f}", textposition="outside",
                          textfont_size=10.5, cliponaxis=False)
        bar_gap(fig)
        fig.add_vline(x=0, line_width=1, line_color=P["axis"])
        span = max(abs(w.Change.min()), abs(w.Change.max())) * 1.45 + 1
        fig.update_layout(xaxis_title="Change in annual premium (USD)",
                          yaxis_title="", xaxis_range=[-span, span])
        chart(fig, w, key="qc_what", height=330, fmt={"New premium": "${:,.0f}",
                                                      "Change": "{:+,.0f}"})
        note("Changes that push the applicant across a <b>tier boundary</b> move "
             "the premium by an order of magnitude more than changes that keep "
             "them inside their tier. That step, not a smooth slope, is the "
             "two-stage engine working.")

    st.markdown("---")

    # ------------------------------------------------------- batch scoring ---
    card_open("Batch scoring", "Score a file of applicants through the same engine")
    st.caption("The CSV needs these columns: " + ", ".join(f"`{c}`" for c in FEATURES))
    up = st.file_uploader("Upload applicants (CSV)", type="csv",
                          label_visibility="collapsed")
    if up:
        try:
            new = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in new.columns]
            if missing:
                note(f"That file is missing required column(s): "
                     f"<b>{', '.join(missing)}</b>. Add them and re-upload.", "crit")
            else:
                res = [quote({k: r[k] for k in FEATURES}) for _, r in new.iterrows()]
                new["predicted_tier"] = [q["tier"] for q in res]
                new["tier_confidence"] = [round(q["confidence"], 4) for q in res]
                new["predicted_cost"] = [q["cost"] for q in res]
                new["final_premium"] = [q["premium"] for q in res]
                new["premium_low"] = [q["premium_lo"] for q in res]
                new["premium_high"] = [q["premium_hi"] for q in res]
                new["underwriting_action"] = ["REFER" if q["refer"] else "AUTO-QUOTE"
                                              for q in res]
                new["health_score"] = [
                    health_score(r.age, r.bmi, r.smoker, r.exercise_freq)
                    for _, r in new.iterrows()]

                n_ref = int((new.underwriting_action == "REFER").sum())
                c = st.columns(4)
                with c[0]: kpi("Applicants scored", f"{len(new):,}", "", "neutral")
                with c[1]: kpi("Auto-quote", f"{len(new) - n_ref:,}",
                               "cleared for binding", "good")
                with c[2]: kpi("Refer", f"{n_ref:,}",
                               "below the confidence floor",
                               "warn" if n_ref else "neutral")
                with c[3]: kpi("Premium written", money(new.final_premium.sum()),
                               "if every quote binds", "neutral")

                st.dataframe(new, width="stretch", hide_index=True,
                             column_config={
                                 "final_premium": st.column_config.NumberColumn(
                                     "Premium", format="$%d"),
                                 "predicted_cost": st.column_config.NumberColumn(
                                     "Predicted cost", format="$%d"),
                                 "premium_low": st.column_config.NumberColumn(
                                     "Low", format="$%d"),
                                 "premium_high": st.column_config.NumberColumn(
                                     "High", format="$%d"),
                                 "tier_confidence": st.column_config.ProgressColumn(
                                     "Confidence", format="%.2f",
                                     min_value=0.0, max_value=1.0)})
                st.download_button("Download scored file", new.to_csv(index=False),
                                   "healthwise_quotes.csv", "text/csv")
        except Exception as ex:
            note(f"Could not read that file: {ex}", "crit")

    # --------------------------------------------- how the threshold was set --
    with st.expander("How the auto-quote threshold was calibrated"):
        st.markdown(
            f"The **{REFER_BELOW:.0%} floor is measured, not chosen.** Every "
            "policy in the book was scored out-of-fold, then bucketed by how "
            "confident Stage 1 was in the tier it picked:")
        buckets = CAL.get("confidence_buckets")
        if buckets:
            bt = pd.DataFrame(buckets)
            bt.columns = [c.replace("bucket", "Confidence").replace("n", "Policies")
                          .replace("accuracy", "Tier accuracy").replace("mae", "MAE")
                          .replace("share", "Share %") for c in bt.columns]
            st.dataframe(bt, width="stretch", hide_index=True)
        st.markdown(
            "Below 70% confidence, tier accuracy collapses to **63.6%** and mean "
            "pricing error roughly triples to **$5,256**. Referring everything "
            "below 80% flags about **2.9%** of applicants, and those carry a mean "
            "error of **$3,427** against **$1,862** for the auto-quoted "
            "remainder — a genuine 1.8× concentration of risk into a small, "
            "reviewable queue.")
        note("<b>What this does not do.</b> Above 80%, confidence stops "
             "discriminating: the 90–95% and 95%+ buckets have almost identical "
             "error. So this catches the worst cases, not all of them — a "
             "depth-4 tree produces coarse leaf probabilities. It is a useful "
             "triage filter, not a calibrated probability, and the residual "
             "misclassification risk documented in Stage 2 still applies.", "warn")


# =================================================================== FOOT ====
st.markdown(f"""<div class="foot">
<b>HealthWise Smart Premium Engine</b> · Sparsh Saraya · Machine Learning (MAIB)
&nbsp;·&nbsp; Engine v1.0 · decision tree (depth 4) + three per-tier linear models
&nbsp;·&nbsp; {len(df):,} policies priced out-of-fold<br>
Premiums are model estimates produced for academic demonstration and are not a
quotation. All findings describe <b>associations</b> in historical data, not causal
effects. Figures assume a {LOADING*100:.0f}% expense-and-margin loading, which is a
stated business assumption rather than a model output.
</div>""", unsafe_allow_html=True)
