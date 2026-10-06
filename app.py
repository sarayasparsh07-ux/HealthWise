"""
================================================================================
HEALTHWISE INSURANCE - SMART PREMIUM ENGINE
Interactive analytics dashboard
================================================================================
Author : Sparsh Saraya  |  Course: Machine Learning - MAIB

Run with:
    streamlit run app.py

Required files in the same folder:
    healthwise_model.joblib     the saved two-stage engine
    healthwise_enriched.csv     the enriched dataset (premium for every customer)
    results.json                pre-computed model metrics
================================================================================
"""
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================== PAGE SETUP ===
st.set_page_config(page_title="HealthWise — Smart Premium Engine",
                   page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

# Palette used consistently across every chart in the app
NAVY, SKY, GOOD, MID, BAD, GREY = "#1B4965", "#5FA8D3", "#2A9D8F", "#E9C46A", "#C1444F", "#8D99AE"
TIER_C = {"Low": GOOD, "Medium": MID, "High": BAD}
SMK_C = {"no": GOOD, "yes": BAD}
BAND_C = {"Poor": BAD, "Fair": "#E8825F", "Good": MID, "Excellent": GOOD}
TIERS = ["Low", "Medium", "High"]

st.markdown(f"""
<style>
  .stApp {{ background: #F7F9FB; }}
  .main .block-container {{ padding-top: 2rem; max-width: 1400px; }}
  h1, h2, h3 {{ color: {NAVY}; font-weight: 700; }}
  .hw-hero {{
     background: linear-gradient(120deg, {NAVY} 0%, #2C6E8F 60%, {SKY} 100%);
     padding: 1.6rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.4rem;
  }}
  .hw-hero h1 {{ color: white; margin: 0; font-size: 2rem; }}
  .hw-hero p  {{ color: #DCEBF5; margin: .35rem 0 0 0; font-size: .97rem; }}
  .kpi {{
     background: white; border-radius: 12px; padding: 1rem 1.15rem;
     border-left: 5px solid {SKY}; box-shadow: 0 1px 4px rgba(27,73,101,.10);
     height: 100%;
  }}
  .kpi .lab {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .06em;
               color: {GREY}; font-weight: 600; }}
  .kpi .val {{ font-size: 1.7rem; font-weight: 700; color: {NAVY}; line-height: 1.15; }}
  .kpi .sub {{ font-size: .76rem; color: {GREY}; }}
  .insight {{
     background: #EAF4FA; border-left: 4px solid {SKY}; padding: .8rem 1rem;
     border-radius: 8px; margin: .7rem 0 1.3rem 0; font-size: .92rem; color: #22384A;
  }}
  .warnbox {{
     background: #FFF4E6; border-left: 4px solid {BAD}; padding: .8rem 1rem;
     border-radius: 8px; margin: .7rem 0 1.3rem 0; font-size: .92rem; color: #4A2A22;
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 3px; }}
  .stTabs [data-baseweb="tab"] {{
     background: white; border-radius: 8px 8px 0 0; padding: .55rem 1rem;
     font-size: .88rem; font-weight: 600; color: {NAVY};
  }}
  .stTabs [aria-selected="true"] {{ background: {NAVY}; color: white; }}
</style>
""", unsafe_allow_html=True)

PLOTLY = dict(template="plotly_white",
              font=dict(family="system-ui, sans-serif", size=12, color="#22384A"),
              title_font=dict(size=15, color=NAVY),
              margin=dict(l=60, r=30, t=60, b=55))


# ============================================================ DATA LOADING ===
@st.cache_resource
def load_model():
    return joblib.load("healthwise_model.joblib")


@st.cache_data
def load_data():
    d = pd.read_csv("healthwise_enriched.csv")
    d["risk_tier"] = pd.Categorical(d["risk_tier"], categories=TIERS, ordered=True)
    d["predicted_tier"] = pd.Categorical(d["predicted_tier"], categories=TIERS, ordered=True)
    d["health_band"] = pd.Categorical(d["health_band"],
                                      categories=["Poor", "Fair", "Good", "Excellent"],
                                      ordered=True)
    return d


@st.cache_data
def load_results():
    try:
        return json.load(open("results.json"))
    except FileNotFoundError:
        return {}


try:
    M = load_model()
    df = load_data()
    R = load_results()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}. "
             "Please keep healthwise_model.joblib and healthwise_enriched.csv "
             "in the same folder as app.py.")
    st.stop()

CLF, REGS = M["clf"], M["regs"]
FEATURES, NUM, MED = M["features"], M["num"], M["med"]
LOADING = M.get("loading", 0.23)
WEIGHTS = M.get("weights", {"smoker": .4597, "age": .2871,
                            "bmi": .2442, "exercise_freq": .0090})


# ====================================================== ENGINE (two-stage) ===
def score_smoker(s):
    return 0.0 if str(s).lower() == "yes" else 100.0


def score_age(a):
    return float(np.clip(100.0 * (64.0 - a) / 46.0, 0, 100))


def score_bmi(b):
    pen = (18.5 - b) * 8.0 if b < 18.5 else (0.0 if b <= 24.9 else (b - 24.9) * 4.5)
    return float(np.clip(100.0 - pen, 0, 100))


def score_exercise(f):
    return float(np.clip(100.0 * f / 7.0, 0, 100))


def health_score(age, bmi, smoker, exercise):
    return round(score_smoker(smoker) * WEIGHTS["smoker"]
                 + score_age(age) * WEIGHTS["age"]
                 + score_bmi(bmi) * WEIGHTS["bmi"]
                 + score_exercise(exercise) * WEIGHTS["exercise_freq"], 2)


def health_band(s):
    return "Excellent" if s >= 75 else "Good" if s >= 55 else "Fair" if s >= 35 else "Poor"


def predict(customer: dict):
    """Stage 1 classify -> Stage 2 price with THAT tier's regressor."""
    row = pd.DataFrame([customer])
    for k in NUM:
        if k not in row or pd.isna(row.at[0, k]):
            row[k] = MED[k]
    tier = CLF.predict(row[FEATURES])[0]
    cost = float(REGS[tier].predict(row[FEATURES])[0])
    return tier, round(cost, 2), round(cost * (1 + LOADING), 2)


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
        lead = ", ".join(d[1] for d in drivers[:3])
        sent = (f"The premium is {level} mainly because this customer {lead}. "
                f"Combined with {hs_txt} ({hs:.0f}/100), this places them in the "
                f"{tier} risk tier.")
    elif drivers:
        lead = ", ".join(d[1] for d in drivers[:3])
        sent = (f"The premium is low overall. The main factor pushing it up is that "
                f"this customer {lead}, but with {hs_txt} ({hs:.0f}/100) they still "
                f"sit in the Low risk tier.")
    else:
        sent = (f"The premium is {level} because this customer "
                f"{', '.join(protectors[:3])}. With {hs_txt} ({hs:.0f}/100) they sit "
                f"in the {tier} risk tier.")
    return sent, [d[0] for d in drivers[:3]], protectors[:3]


def kpi(label, value, sub="", color=SKY):
    st.markdown(f"""<div class="kpi" style="border-left-color:{color}">
        <div class="lab">{label}</div><div class="val">{value}</div>
        <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight"><b>Key insight:</b> {text}</div>',
                unsafe_allow_html=True)


def warnbox(text):
    st.markdown(f'<div class="warnbox"><b>Caution:</b> {text}</div>',
                unsafe_allow_html=True)


# =================================================================== HERO ====
st.markdown("""<div class="hw-hero">
  <h1>🏥 HealthWise Insurance — Smart Premium Engine</h1>
  <p>Two-stage risk pricing: classify the customer into a risk tier, then price them
     with a regression model built for that tier.</p>
</div>""", unsafe_allow_html=True)

# ================================================================ SIDEBAR ====
with st.sidebar:
    st.markdown(f"### 🏥 HealthWise")
    st.caption("Smart Premium Engine · v1.0")
    st.markdown("---")
    st.markdown("**Portfolio filters**")
    f_smoker = st.multiselect("Smoker status", ["no", "yes"], default=["no", "yes"])
    f_tier = st.multiselect("Risk tier", TIERS, default=TIERS)
    f_region = st.multiselect("Region", sorted(df.region.unique()),
                              default=sorted(df.region.unique()))
    f_age = st.slider("Age range", int(df.age.min()), int(df.age.max()),
                      (int(df.age.min()), int(df.age.max())))
    st.markdown("---")
    st.caption(f"**Premium loading**\n\nPremium = predicted cost × {1+LOADING:.2f}  \n"
               f"(15% expenses + 8% margin)")
    st.caption("Filters apply to the portfolio tabs. "
               "The Premium Calculator always uses the full trained engine.")

mask = (df.smoker.isin(f_smoker) & df.risk_tier.isin(f_tier)
        & df.region.isin(f_region) & df.age.between(*f_age))
D = df[mask]
if len(D) == 0:
    st.warning("No customers match the current filters. Please widen the selection.")
    st.stop()

tabs = st.tabs(["📊 Overview", "🔍 Dataset", "❤️ Health Score", "💰 Premiums",
                "🎯 Classification", "📈 Regression", "⚖️ Bias–Variance", "🧮 Calculator"])

# ========================================================== TAB 1 OVERVIEW ===
with tabs[0]:
    st.subheader("Executive Overview")
    st.caption(f"Showing {len(D):,} of {len(df):,} customers under the current filters.")

    c = st.columns(4)
    with c[0]: kpi("Total individuals", f"{len(D):,}",
                   f"{len(D)/len(df)*100:.0f}% of portfolio", NAVY)
    with c[1]: kpi("Avg Health Score", f"{D.health_score.mean():.1f}",
                   "0–100, higher = healthier", GOOD)
    with c[2]: kpi("Average premium", f"${D.final_premium.mean():,.0f}",
                   "per customer per year", SKY)
    with c[3]: kpi("Median premium", f"${D.final_premium.median():,.0f}",
                   "less skew-sensitive", SKY)

    c = st.columns(4)
    smk_pct = (D.smoker == "yes").mean() * 100
    with c[0]: kpi("Smokers", f"{smk_pct:.1f}%", f"{(D.smoker=='yes').sum():,} customers", BAD)
    with c[1]: kpi("Non-smokers", f"{100-smk_pct:.1f}%",
                   f"{(D.smoker=='no').sum():,} customers", GOOD)
    with c[2]: kpi("Avg predicted cost", f"${D.predicted_cost.mean():,.0f}",
                   "expected annual claim", MID)
    with c[3]: kpi("Avg actual charge", f"${D.annual_charge.mean():,.0f}",
                   "observed in the data", GREY)

    st.markdown("---")
    l, r = st.columns([1.15, 1])
    with l:
        g = D.groupby("risk_tier", observed=True).agg(
            customers=("age", "size"), premium=("final_premium", "mean")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=g.risk_tier.astype(str), y=g.customers, name="Customers",
                    marker_color=[TIER_C[t] for t in g.risk_tier.astype(str)],
                    text=g.customers, textposition="outside")
        fig.add_scatter(x=g.risk_tier.astype(str), y=g.premium, name="Mean premium",
                        mode="lines+markers+text", line=dict(color=NAVY, width=3),
                        marker=dict(size=11),
                        text=[f"${v:,.0f}" for v in g.premium], textposition="top center",
                        secondary_y=True)
        fig.update_layout(title="Portfolio Mix and Mean Premium by Risk Tier",
                          height=400, **PLOTLY)
        fig.update_xaxes(title_text="Risk tier")
        fig.update_yaxes(title_text="Number of customers", secondary_y=False)
        fig.update_yaxes(title_text="Mean annual premium (USD)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = px.histogram(D, x="final_premium", color="smoker", nbins=42,
                           color_discrete_map=SMK_C, barmode="overlay", opacity=.75,
                           labels={"final_premium": "Final annual premium (USD)",
                                   "smoker": "Smoker"})
        fig.update_layout(title="Premium Distribution by Smoker Status",
                          yaxis_title="Number of customers", height=400, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

    gap = (D[D.smoker == "yes"].final_premium.mean()
           - D[D.smoker == "no"].final_premium.mean())
    insight(f"Smokers in the current selection are priced <b>${gap:,.0f}</b> higher per year "
            f"on average than non-smokers — the single largest pricing differential in the "
            f"portfolio, driven by an observed claim cost ratio of about 2.4×.")

    if R:
        st.markdown("#### Engine performance vs the current flat-pricing model")
        e = R.get("end_to_end", {})
        if e:
            comp = pd.DataFrame([
                {"Pricing approach": "Global flat model (today)",
                 "MAE": e["global"]["MAE"], "RMSE": e["global"]["RMSE"], "R²": e["global"]["R2"]},
                {"Pricing approach": "Two-stage engine (deployed, predicted tier)",
                 "MAE": e["predicted_tier"]["MAE"], "RMSE": e["predicted_tier"]["RMSE"],
                 "R²": e["predicted_tier"]["R2"]},
            ])
            c = st.columns([1.3, 1])
            with c[0]:
                st.dataframe(comp.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}",
                                                "R²": "{:.4f}"}),
                             use_container_width=True, hide_index=True)
            with c[1]:
                kpi("Pricing error reduction", f"{e['mae_improvement_pct']:.0f}%",
                    f"MAE ${e['global']['MAE']:,.0f} → ${e['predicted_tier']['MAE']:,.0f}", GOOD)
            insight(f"Measured end-to-end on held-out customers — with Stage 1 predicting its "
                    f"own tiers rather than being given them — the engine cuts average pricing "
                    f"error by <b>{e['mae_improvement_pct']:.0f}%</b> and lifts R² from "
                    f"{e['global']['R2']:.3f} to {e['predicted_tier']['R2']:.3f}.")

# =========================================================== TAB 2 DATASET ===
with tabs[1]:
    st.subheader("Dataset Explorer")
    st.caption("Search, filter, inspect and download the cleaned, enriched dataset.")

    c = st.columns([2, 1, 1])
    with c[0]:
        q = st.text_input("Search by customer ID", placeholder="e.g. HW1042")
    with c[1]:
        band_f = st.multiselect("Health band", ["Poor", "Fair", "Good", "Excellent"],
                                default=["Poor", "Fair", "Good", "Excellent"])
    with c[2]:
        prem_f = st.slider("Premium range ($000)",
                           float(df.final_premium.min() / 1000),
                           float(df.final_premium.max() / 1000),
                           (float(df.final_premium.min() / 1000),
                            float(df.final_premium.max() / 1000)))

    V = D[D.health_band.isin(band_f)
          & D.final_premium.between(prem_f[0] * 1000, prem_f[1] * 1000)]
    if q:
        V = V[V.customer_id.str.contains(q, case=False, na=False)]

    st.caption(f"{len(V):,} customers match.")
    show = ["customer_id", "age", "sex", "bmi", "children", "smoker", "region",
            "exercise_freq", "health_score", "health_band", "annual_charge",
            "risk_tier", "predicted_tier", "predicted_cost", "final_premium",
            "top_risk_factors"]
    st.dataframe(
        V[show].style.format({"bmi": "{:.1f}", "health_score": "{:.1f}",
                              "annual_charge": "${:,.0f}", "predicted_cost": "${:,.0f}",
                              "final_premium": "${:,.0f}", "exercise_freq": "{:.0f}"}),
        use_container_width=True, height=430)

    c = st.columns([1, 1, 2])
    with c[0]:
        st.download_button("⬇️ Download this view (CSV)",
                           V.to_csv(index=False), "healthwise_filtered.csv",
                           "text/csv", use_container_width=True)
    with c[1]:
        st.download_button("⬇️ Download full dataset (CSV)",
                           df.to_csv(index=False), "healthwise_enriched.csv",
                           "text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Inspect an individual record")
    pick = st.selectbox("Select a customer", V.customer_id.tolist()[:2000]
                        if len(V) else df.customer_id.tolist()[:2000])
    if pick:
        r = df[df.customer_id == pick].iloc[0]
        c = st.columns(5)
        with c[0]: kpi("Age / Sex", f"{int(r.age)} · {r.sex}", f"{r.region} region", NAVY)
        with c[1]: kpi("BMI", f"{r.bmi:.1f}", str(r.bmi_category), SKY)
        with c[2]: kpi("Health Score", f"{r.health_score:.1f}", str(r.health_band),
                       BAND_C.get(str(r.health_band), SKY))
        with c[3]: kpi("Risk tier", str(r.predicted_tier), "predicted",
                       TIER_C.get(str(r.predicted_tier), SKY))
        with c[4]: kpi("Final premium", f"${r.final_premium:,.0f}",
                       f"cost ${r.predicted_cost:,.0f}", GOOD)
        st.markdown(f"**Why is this premium assigned?**")
        st.info(r.premium_explanation)
        st.caption(f"Top contributing risk factors: {r.top_risk_factors}")

# ====================================================== TAB 3 HEALTH SCORE ===
with tabs[2]:
    st.subheader("Health Score Analysis")
    st.markdown(f"""<div class="insight">
    <b>Health Score = Σ (Metric Score × Weight)</b> · scored 0–100, where
    <b>higher = healthier</b>.<br>
    Weights derived from standardised regression coefficients on the training split only:
    <b>smoker {WEIGHTS['smoker']*100:.1f}%</b>, <b>age {WEIGHTS['age']*100:.1f}%</b>,
    <b>BMI {WEIGHTS['bmi']*100:.1f}%</b>,
    <b>exercise {WEIGHTS['exercise_freq']*100:.1f}%</b>.
    </div>""", unsafe_allow_html=True)

    c = st.columns(4)
    with c[0]: kpi("Mean Health Score", f"{D.health_score.mean():.1f}", "0–100 scale", GOOD)
    with c[1]: kpi("Median", f"{D.health_score.median():.1f}", "", SKY)
    with c[2]: kpi("Minimum", f"{D.health_score.min():.1f}", "least healthy", BAD)
    with c[3]: kpi("Maximum", f"{D.health_score.max():.1f}", "healthiest", GOOD)

    l, r = st.columns(2)
    with l:
        fig = px.histogram(D, x="health_score", nbins=34, color_discrete_sequence=[NAVY],
                           labels={"health_score": "Health Score (0–100)"})
        for th, lb in [(35, "Fair"), (55, "Good"), (75, "Excellent")]:
            fig.add_vline(x=th, line_dash="dash", line_color=BAD,
                          annotation_text=lb, annotation_position="top")
        fig.update_layout(title="Health Score Distribution",
                          yaxis_title="Number of customers", height=380, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        g = D.groupby("health_band", observed=True).agg(
            n=("age", "size"), charge=("annual_charge", "mean")).reset_index()
        fig = px.bar(g, x="health_band", y="charge", text=g.charge.map("${:,.0f}".format),
                     color="health_band", color_discrete_map=BAND_C,
                     labels={"health_band": "Health band", "charge": "Mean annual charge (USD)"})
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Mean Annual Charge by Health Band",
                          height=380, showlegend=False, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

    poor = D[D.health_band == "Poor"].annual_charge.mean()
    exc = D[D.health_band == "Excellent"].annual_charge.mean()
    if np.isfinite(poor) and np.isfinite(exc) and exc > 0:
        insight(f"Customers in the <b>Poor</b> health band cost <b>{poor/exc:.1f}×</b> more "
                f"than those in the <b>Excellent</b> band (${poor:,.0f} vs ${exc:,.0f}), "
                f"confirming the score separates real cost risk rather than merely ranking people.")

    l, r = st.columns(2)
    with l:
        g = D.groupby(["age_group", "smoker"], observed=True).health_score.mean().reset_index()
        fig = px.bar(g, x="age_group", y="health_score", color="smoker", barmode="group",
                     color_discrete_map=SMK_C,
                     labels={"age_group": "Age group", "health_score": "Mean Health Score",
                             "smoker": "Smoker"})
        fig.update_layout(title="Mean Health Score by Age Group and Smoker Status",
                          height=370, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        g = D.groupby(["bmi_category", "smoker"], observed=True).health_score.mean().reset_index()
        fig = px.bar(g, x="bmi_category", y="health_score", color="smoker", barmode="group",
                     color_discrete_map=SMK_C,
                     labels={"bmi_category": "BMI category",
                             "health_score": "Mean Health Score", "smoker": "Smoker"})
        fig.update_layout(title="Mean Health Score by BMI Category and Smoker Status",
                          height=370, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

# ========================================================== TAB 4 PREMIUMS ===
with tabs[3]:
    st.subheader("Insurance Premium Analysis")
    st.caption("Premium = predicted annual claim cost × "
               f"{1+LOADING:.2f} (15% expense ratio + 8% profit and solvency margin).")

    c = st.columns(4)
    with c[0]: kpi("Mean premium", f"${D.final_premium.mean():,.0f}", "", SKY)
    with c[1]: kpi("Median premium", f"${D.final_premium.median():,.0f}", "", SKY)
    with c[2]: kpi("Lowest", f"${D.final_premium.min():,.0f}", "healthiest customer", GOOD)
    with c[3]: kpi("Highest", f"${D.final_premium.max():,.0f}", "highest-risk customer", BAD)

    l, r = st.columns(2)
    with l:
        fig = px.box(D, x="risk_tier", y="final_premium", color="risk_tier",
                     color_discrete_map=TIER_C, category_orders={"risk_tier": TIERS},
                     labels={"risk_tier": "Risk tier",
                             "final_premium": "Final annual premium (USD)"})
        fig.update_layout(title="Premium by Risk Tier", height=380,
                          showlegend=False, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = px.box(D, x="smoker", y="final_premium", color="smoker",
                     color_discrete_map=SMK_C, category_orders={"smoker": ["no", "yes"]},
                     labels={"smoker": "Smoker status",
                             "final_premium": "Final annual premium (USD)"})
        fig.update_layout(title="Premium by Smoker Status", height=380,
                          showlegend=False, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

    l, r = st.columns(2)
    with l:
        fig = px.scatter(D, x="age", y="final_premium", color="smoker",
                         color_discrete_map=SMK_C, opacity=.6,
                         labels={"age": "Age (years)",
                                 "final_premium": "Final annual premium (USD)",
                                 "smoker": "Smoker"})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(title="Premium vs Age", height=370, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = px.scatter(D, x="bmi", y="final_premium", color="smoker",
                         color_discrete_map=SMK_C, opacity=.6,
                         labels={"bmi": "BMI (kg/m²)",
                                 "final_premium": "Final annual premium (USD)",
                                 "smoker": "Smoker"})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(title="Premium vs BMI", height=370, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(D, x="health_score", y="final_premium", color="risk_tier",
                     color_discrete_map=TIER_C, opacity=.65,
                     category_orders={"risk_tier": TIERS},
                     trendline="ols", trendline_color_override=NAVY,
                     labels={"health_score": "Health Score (0–100, higher = healthier)",
                             "final_premium": "Final annual premium (USD)",
                             "risk_tier": "Risk tier"})
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(title="Premium vs Health Score — the pricing gradient", height=420, **PLOTLY)
    st.plotly_chart(fig, use_container_width=True)
    rho = D.health_score.corr(D.final_premium, method="spearman")
    insight(f"Premium falls steadily as Health Score rises (Spearman ρ = <b>{rho:.2f}</b>). "
            f"The three tiers appear as distinct horizontal bands rather than one continuous "
            f"line — visible evidence of the two-stage engine pricing each segment with its "
            f"own model.")

# ==================================================== TAB 5 CLASSIFICATION ===
with tabs[4]:
    st.subheader("Stage 1 — Risk Tier Classification")
    if not R:
        st.info("results.json not found — model metrics unavailable.")
    else:
        b = R["clf_best"]
        c = st.columns(4)
        with c[0]: kpi("Model", b["name"].replace("Decision Tree", "Tree"),
                       "chosen by cross-validation", NAVY)
        with c[1]: kpi("Test accuracy", f"{b['test_acc']:.1%}",
                       f"train {b['train_acc']:.1%}", GOOD)
        with c[2]: kpi("Macro F1", f"{b['report']['macro avg']['f1-score']:.3f}",
                       "balanced across tiers", SKY)
        with c[3]: kpi("Train–test gap", f"{b['train_acc']-b['test_acc']:+.3f}",
                       "near zero = well balanced", GOOD)

        l, r = st.columns([1, 1.15])
        with l:
            cm = np.array(b["confusion_matrix"])
            fig = px.imshow(cm, x=TIERS, y=TIERS, text_auto=True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted risk tier", y="Actual risk tier",
                                        color="Customers"))
            fig.update_layout(title="Confusion Matrix (test set)", height=390, **PLOTLY)
            st.plotly_chart(fig, use_container_width=True)
        with r:
            rep = b["report"]
            met = pd.DataFrame({t: {k: rep[t][k] for k in
                                    ["precision", "recall", "f1-score"]} for t in TIERS}).T
            fig = px.bar(met.reset_index().melt(id_vars="index"),
                         x="index", y="value", color="variable", barmode="group",
                         color_discrete_sequence=[NAVY, SKY, GOOD],
                         labels={"index": "Risk tier", "value": "Score",
                                 "variable": "Metric"}, text_auto=".2f")
            fig.update_layout(title="Per-Tier Precision, Recall and F1",
                              height=390, yaxis_range=[0, 1.15], **PLOTLY)
            st.plotly_chart(fig, use_container_width=True)

        hi_low = cm[TIERS.index("High"), TIERS.index("Low")]
        insight(f"All confusion is between <b>neighbouring</b> tiers — "
                f"<b>{hi_low}</b> High-risk customers were classified as Low, and none of the "
                f"reverse. Every error is a single-step mistake, which bounds the financial "
                f"damage of any individual misclassification.")

        st.markdown("#### Model comparison")
        cc = pd.DataFrame(R["classification_comparison"])
        st.dataframe(cc.style.format({
            "train_acc": "{:.4f}", "test_acc": "{:.4f}", "gap": "{:+.4f}",
            "balanced_acc": "{:.4f}", "precision_macro": "{:.4f}",
            "recall_macro": "{:.4f}", "f1_macro": "{:.4f}",
            "roc_auc_ovr": "{:.4f}", "cv_mean": "{:.4f}", "cv_std": "{:.4f}"})
            .background_gradient(subset=["cv_mean"], cmap="Greens"),
            use_container_width=True, hide_index=True)

        st.markdown("#### Feature importance")
        fi = pd.Series(b["feature_importance"]).sort_values(ascending=True)
        fi = fi[fi > 0.0005]
        fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                     color_discrete_sequence=[NAVY], text=[f"{v:.3f}" for v in fi.values],
                     labels={"x": "Importance (mean decrease in impurity)", "y": "Feature"})
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Which attributes decide the risk tier",
                          height=330, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

# ======================================================== TAB 6 REGRESSION ===
with tabs[5]:
    st.subheader("Stage 2 — Charge Prediction")
    if not R:
        st.info("results.json not found — model metrics unavailable.")
    else:
        e = R["end_to_end"]
        c = st.columns(4)
        with c[0]: kpi("Flat model MAE", f"${e['global']['MAE']:,.0f}",
                       f"R² {e['global']['R2']:.3f}", BAD)
        with c[1]: kpi("Two-stage MAE", f"${e['predicted_tier']['MAE']:,.0f}",
                       f"R² {e['predicted_tier']['R2']:.3f}", GOOD)
        with c[2]: kpi("Error reduction", f"{e['mae_improvement_pct']:.0f}%",
                       "measured end-to-end", GOOD)
        with c[3]: kpi("Stage-1 accuracy", f"{e['stage1_acc']:.1%}",
                       "tier assigned correctly", SKY)

        cmpdf = pd.DataFrame([
            {"Approach": "Global flat model", "MAE": e["global"]["MAE"],
             "RMSE": e["global"]["RMSE"], "R²": e["global"]["R2"]},
            {"Approach": "Two-stage (predicted tier)", "MAE": e["predicted_tier"]["MAE"],
             "RMSE": e["predicted_tier"]["RMSE"], "R²": e["predicted_tier"]["R2"]},
            {"Approach": "Two-stage (true tier, oracle)", "MAE": e["oracle_tier"]["MAE"],
             "RMSE": e["oracle_tier"]["RMSE"], "R²": e["oracle_tier"]["R2"]},
        ])
        l, r = st.columns([1.1, 1])
        with l:
            fig = px.bar(cmpdf, x="Approach", y="MAE", color="Approach",
                         color_discrete_sequence=[BAD, GOOD, SKY],
                         text=cmpdf.MAE.map("${:,.0f}".format),
                         labels={"MAE": "Mean absolute error (USD)"})
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Pricing Error — Lower is Better", height=390,
                              showlegend=False, **PLOTLY)
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.dataframe(cmpdf.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}",
                                             "R²": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            warnbox("The <b>oracle</b> row assumes the true tier is known, which it never is "
                    "at quote time. The deployable figure is the <b>predicted tier</b> row — "
                    "quoting the oracle number would overstate real performance by roughly "
                    f"{(e['predicted_tier']['MAE']/e['oracle_tier']['MAE']-1)*100:.0f}%.")

        l, r = st.columns(2)
        with l:
            fig = px.scatter(D, x="annual_charge", y="predicted_cost", color="predicted_tier",
                             color_discrete_map=TIER_C, opacity=.6,
                             category_orders={"predicted_tier": TIERS},
                             labels={"annual_charge": "Actual annual charge (USD)",
                                     "predicted_cost": "Predicted annual cost (USD)",
                                     "predicted_tier": "Risk tier"})
            lim = [D.annual_charge.min(), D.annual_charge.max()]
            fig.add_scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                            line=dict(color="black", dash="dash"))
            fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
            fig.update_layout(title="Actual vs Predicted (out-of-fold)", height=400, **PLOTLY)
            st.plotly_chart(fig, use_container_width=True)
        with r:
            res = D.predicted_cost - D.annual_charge
            fig = px.scatter(x=D.predicted_cost, y=res, color=D.predicted_tier.astype(str),
                             color_discrete_map=TIER_C, opacity=.6,
                             labels={"x": "Predicted annual cost (USD)",
                                     "y": "Residual (predicted − actual, USD)",
                                     "color": "Risk tier"})
            fig.add_hline(y=0, line_color="black")
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(title="Residuals vs Fitted (out-of-fold)", height=400, **PLOTLY)
            st.plotly_chart(fig, use_container_width=True)

        insight(f"Residuals cluster tightly around zero for most customers. The scattered "
                f"large residuals are almost entirely Stage-1 tier misclassifications: "
                f"misclassified customers carry a mean error of "
                f"<b>${e['MAE_misclassified']:,.0f}</b> against "
                f"<b>${e['MAE_correct']:,.0f}</b> for correctly classified ones — an "
                f"{e['MAE_misclassified']/e['MAE_correct']:.0f}× difference.")

        st.markdown("#### Per-tier model performance")
        pt = pd.DataFrame(R["per_tier"]).T.reset_index().rename(columns={"index": "Tier"})
        st.dataframe(pt[["Tier", "n", "R2", "MAE", "RMSE", "train_R2", "cv_R2",
                         "strongest_feature", "simple_R2"]].style.format({
            "R2": "{:.4f}", "MAE": "${:,.0f}", "RMSE": "${:,.0f}",
            "train_R2": "{:.4f}", "cv_R2": "{:.4f}", "simple_R2": "{:.4f}"}),
            use_container_width=True, hide_index=True)

        st.markdown("#### Regression model comparison (global feature set)")
        rc = pd.DataFrame(R["regression_comparison"])
        st.dataframe(rc.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.4f}",
                                      "train_R2": "{:.4f}", "gap": "{:+.4f}", "cv_R2": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

# ===================================================== TAB 7 BIAS-VARIANCE ===
with tabs[6]:
    st.subheader("Bias–Variance Analysis")
    if not R:
        st.info("results.json not found — diagnostics unavailable.")
    else:
        dc = pd.DataFrame(R["depth_curve"])
        l, r = st.columns(2)
        with l:
            fig = go.Figure()
            fig.add_scatter(x=dc.depth, y=dc.train, name="Training accuracy",
                            mode="lines+markers", line=dict(color=SKY, width=2.5))
            fig.add_scatter(x=dc.depth, y=dc.test, name="Test accuracy",
                            mode="lines+markers", line=dict(color=NAVY, width=2.5))
            fig.add_scatter(x=dc.depth, y=dc.cv, name="5-fold CV accuracy",
                            mode="lines+markers", line=dict(color=GOOD, width=2.5, dash="dash"))
            fig.add_vline(x=R.get("clf_depth", 4), line_dash="dash", line_color=BAD,
                          annotation_text="Chosen depth")
            fig.update_layout(title="Model Complexity Curve — Tree Depth",
                              xaxis_title="Maximum tree depth", yaxis_title="Accuracy",
                              height=400, **PLOTLY)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Left of the line: both curves low together → **underfitting (high "
                       "bias)**. Right: training rises to 1.0 while CV falls → "
                       "**overfitting (high variance)**.")
        with r:
            lc = R["learning_curves"]["classifier_tree_best"]
            fig = go.Figure()
            fig.add_scatter(x=lc["sizes"], y=lc["train_mean"], name="Training score",
                            mode="lines+markers", line=dict(color=SKY, width=2.5))
            fig.add_scatter(x=lc["sizes"], y=lc["val_mean"], name="Cross-validation score",
                            mode="lines+markers", line=dict(color=NAVY, width=2.5))
            fig.update_layout(title="Learning Curve — Stage-1 Classifier",
                              xaxis_title="Number of training samples",
                              yaxis_title="Accuracy", height=400, **PLOTLY)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The two curves **converge**, which indicates variance is under "
                       "control. Validation is still inching up, so a little more data "
                       "would help marginally.")

        lg = R["learning_curves"]["global_linear"]
        fig = go.Figure()
        fig.add_scatter(x=lg["sizes"], y=lg["train_mean"], name="Training R²",
                        mode="lines+markers", line=dict(color=BAD, width=2.5))
        fig.add_scatter(x=lg["sizes"], y=lg["val_mean"], name="Cross-validation R²",
                        mode="lines+markers", line=dict(color=NAVY, width=2.5))
        fig.add_hline(y=R["two_stage_oracle"]["R2"], line_dash="dash", line_color=GOOD,
                      annotation_text=f"Two-stage ceiling R² = {R['two_stage_oracle']['R2']:.2f}")
        fig.update_layout(title="Learning Curve — Global Flat Model (converged low = high bias)",
                          xaxis_title="Number of training samples", yaxis_title="R²",
                          yaxis_range=[0, 1.05], height=400, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

        insight("The global model's training and validation curves converge at a "
                f"<b>low</b> R² of about {lg['val_mean'][-1]:.2f} and then flatten. Both "
                "scores being poor <i>together</i> is the signature of <b>high bias</b>, not "
                "high variance — which means more data or heavier regularisation would not "
                "help. The fix had to be structural, and segmenting into tiers lifts the "
                f"ceiling to {R['two_stage_oracle']['R2']:.2f}.")

        st.markdown("#### Generalisation gap by model")
        cc = pd.DataFrame(R["classification_comparison"])[["model", "train_acc", "test_acc",
                                                           "gap", "cv_mean"]]
        fig = px.bar(cc, x="model", y="gap",
                     color=cc.gap.apply(lambda v: "Overfitting" if v > .02 else "Balanced"),
                     color_discrete_map={"Overfitting": BAD, "Balanced": GOOD},
                     text=cc.gap.map("{:+.3f}".format),
                     labels={"model": "Model", "gap": "Train accuracy − test accuracy"})
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Train–Test Gap by Model (larger = more overfitting)",
                          height=400, legend_title="", **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

# ========================================================= TAB 8 CALCULATOR ===
with tabs[7]:
    st.subheader("🧮 Individual Risk & Premium Calculator")
    st.caption("Enter a customer's details. The tier, cost and premium recalculate "
               "instantly on every change.")

    l, r = st.columns([1, 1.65])
    with l:
        st.markdown("##### Customer details")
        age = st.slider("Age (years)", 18, 64, 40)
        bmi = st.slider("BMI (kg/m²)", 16.0, 50.0, 27.0, 0.1)
        children = st.slider("Number of dependents", 0, 4, 1)
        exercise = st.slider("Workouts per week", 0, 7, 3)
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["north", "south", "east", "west"])

    cust = dict(age=age, bmi=bmi, children=children, exercise_freq=exercise,
                sex=sex, smoker=smoker, region=region)
    tier, cost, premium = predict(cust)
    hs = health_score(age, bmi, smoker, exercise)
    band = health_band(hs)
    sentence, drivers, protectors = explain(age, bmi, smoker, exercise, children, tier, hs)

    with r:
        st.markdown("##### Engine output")
        c = st.columns(4)
        with c[0]: kpi("Health Score", f"{hs:.1f}", band, BAND_C[band])
        with c[1]: kpi("Risk tier", tier, "Stage 1 output", TIER_C[tier])
        with c[2]: kpi("Predicted cost", f"${cost:,.0f}", "expected claim", SKY)
        with c[3]: kpi("Final premium", f"${premium:,.0f}",
                       f"cost × {1+LOADING:.2f}", GOOD)

        st.markdown("**Why is this premium assigned?**")
        st.info(sentence)
        c = st.columns(2)
        with c[0]:
            st.markdown("**Key risk factors**")
            if drivers:
                for d in drivers: st.markdown(f"- 🔺 {d}")
            else:
                st.markdown("- ✅ No elevated risk factors")
        with c[1]:
            st.markdown("**Protective factors**")
            if protectors:
                for p in protectors: st.markdown(f"- ✅ {p.capitalize()}")
            else:
                st.markdown("- —")

        # Health Score breakdown
        parts = pd.DataFrame({
            "Metric": ["Smoker", "Age", "BMI", "Exercise"],
            "Score": [score_smoker(smoker), score_age(age),
                      score_bmi(bmi), score_exercise(exercise)],
            "Weight": [WEIGHTS["smoker"], WEIGHTS["age"],
                       WEIGHTS["bmi"], WEIGHTS["exercise_freq"]],
        })
        parts["Contribution"] = (parts.Score * parts.Weight).round(2)
        fig = px.bar(parts, x="Contribution", y="Metric", orientation="h",
                     color="Metric", color_discrete_sequence=[BAD, NAVY, SKY, GOOD],
                     text=parts.Contribution.map("{:.1f}".format),
                     labels={"Contribution": "Points contributed to the Health Score"})
        fig.update_traces(textposition="outside")
        fig.update_layout(title=f"Health Score breakdown — total {hs:.1f} / 100",
                          height=260, showlegend=False, **PLOTLY)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("##### What-if: the effect of changing one attribute")
    scen = [("Smoker → yes", {"smoker": "yes"}), ("Smoker → no", {"smoker": "no"}),
            ("Age +10", {"age": min(64, age + 10)}), ("Age −10", {"age": max(18, age - 10)}),
            ("BMI +5", {"bmi": min(50.0, bmi + 5)}), ("BMI −5", {"bmi": max(16.0, bmi - 5)}),
            ("Dependents +2", {"children": min(4, children + 2)}),
            ("Exercise → 7/wk", {"exercise_freq": 7})]
    rows = []
    for lbl, ch in scen:
        t2, c2, p2 = predict({**cust, **ch})
        rows.append({"Scenario": lbl, "Tier": t2, "Premium": p2,
                     "Change": p2 - premium})
    w = pd.DataFrame(rows)
    fig = px.bar(w, x="Change", y="Scenario", orientation="h",
                 color=w.Change.apply(lambda v: "Increase" if v > 0 else "Decrease"),
                 color_discrete_map={"Increase": BAD, "Decrease": GOOD},
                 text=w.Change.map("${:+,.0f}".format),
                 labels={"Change": "Change in annual premium (USD)", "Scenario": ""},
                 hover_data={"Tier": True, "Premium": ":$,.0f"})
    fig.update_traces(textposition="outside")
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(title=f"Premium sensitivity from the current quote of ${premium:,.0f}",
                      height=380, legend_title="", **PLOTLY)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Batch scoring — upload a CSV of unseen customers")
    st.caption("The file needs these columns: " + ", ".join(f"`{c}`" for c in FEATURES))
    up = st.file_uploader("Upload customers to score", type="csv")
    if up:
        try:
            new = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in new.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                res = [predict({k: r[k] for k in FEATURES}) for _, r in new.iterrows()]
                new["predicted_tier"] = [t for t, _, _ in res]
                new["predicted_cost"] = [round(c, 2) for _, c, _ in res]
                new["final_premium"] = [round(p, 2) for _, _, p in res]
                new["health_score"] = [
                    health_score(r.age, r.bmi, r.smoker, r.exercise_freq)
                    for _, r in new.iterrows()]
                st.success(f"Scored {len(new)} customers.")
                st.dataframe(new.style.format({"predicted_cost": "${:,.0f}",
                                               "final_premium": "${:,.0f}",
                                               "health_score": "{:.1f}"}),
                             use_container_width=True)
                st.download_button("⬇️ Download predictions",
                                   new.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as ex:
            st.error(f"Could not score that file: {ex}")

st.markdown("---")
st.caption("HealthWise Smart Premium Engine · Sparsh Saraya · Machine Learning (MAIB) · "
           "Premiums are model estimates for academic demonstration and are not a "
           "quotation. Model findings describe associations in historical data, "
           "not causal effects.")
