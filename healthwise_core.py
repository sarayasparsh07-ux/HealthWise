"""
================================================================================
HEALTHWISE INSURANCE - SMART PREMIUM ENGINE
Core analysis library: cleaning, feature engineering, Health Score,
two-stage model, and premium calculation.
================================================================================
Author : Sparsh Saraya
Course : Machine Learning - MAIB
Purpose: Single source of truth for the notebook, the Streamlit dashboard and
         the written report, so that every deliverable reports IDENTICAL numbers.
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------ CONFIG ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
TIERS = ["Low", "Medium", "High"]

# Stage-1 tree depth. NOT taken from the lab template (which suggests 5):
# chosen at depth=4 because 5-fold CV accuracy peaks there (0.9242 vs 0.9150 at
# depth 5), test accuracy is higher (0.9458 vs 0.9375) and the train-test gap is
# smaller. See Task B3 / the complexity curve for the full sweep.
CLF_DEPTH = 4

# Columns dropped as IRRELEVANT (no plausible causal link to medical cost)
IRRELEVANT = ["favorite_color", "zodiac_sign", "lucky_number",
              "preferred_contact", "marketing_opt_in"]
# Columns dropped as REDUNDANT (duplicate information already present)
REDUNDANT = ["weight_kg", "date_of_birth"]
IDENTIFIER = ["customer_id"]

NUM_FEATURES = ["age", "bmi", "children", "exercise_freq"]
CAT_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Premium loading assumption (STATED ASSUMPTION - see report Section 15).
# The dataset contains an incurred CLAIM COST (annual_charge), not a price.
# A premium must cover cost + expenses + margin.
EXPENSE_RATIO = 0.15   # administration, claims handling, distribution
PROFIT_MARGIN = 0.08   # underwriting profit + solvency buffer
LOADING = EXPENSE_RATIO + PROFIT_MARGIN   # => premium = cost * 1.23


# ============================================================== 1. LOADING ===
def load_raw(path="healthwise.csv"):
    """RAW layer - untouched copy of the source file."""
    return pd.read_csv(path)


# ============================================================= 2. CLEANING ===
def clean(df_raw):
    """
    CLEANED layer.

    Returns (df_clean, cleaning_log).
    Decisions:
      * exact duplicate rows removed (they are true re-exports, not twins)
      * bmi / exercise_freq imputed with the MEDIAN (robust to the right skew
        of bmi; exercise_freq is a bounded 0-7 count so the median stays in range)
      * categorical labels stripped + case-normalised defensively
    """
    log = {}
    df = df_raw.copy()
    log["rows_raw"] = len(df)
    log["cols_raw"] = df.shape[1]

    # --- 2.1 duplicates ---------------------------------------------------
    log["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log["rows_after_dedup"] = len(df)

    # --- 2.2 categorical standardisation (defensive) ----------------------
    for c in ["sex", "smoker", "region", "favorite_color",
              "preferred_contact", "marketing_opt_in"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["zodiac_sign"] = df["zodiac_sign"].astype(str).str.strip().str.title()
    df["risk_tier"] = df["risk_tier"].astype(str).str.strip().str.title()

    # --- 2.3 missing values ----------------------------------------------
    miss = df.isna().sum()
    log["missing_before"] = {k: int(v) for k, v in miss.items() if v > 0}
    log["bmi_median"] = float(df["bmi"].median())
    log["exercise_median"] = float(df["exercise_freq"].median())
    df["bmi"] = df["bmi"].fillna(log["bmi_median"])
    df["exercise_freq"] = df["exercise_freq"].fillna(log["exercise_median"])
    log["missing_after"] = int(df.isna().sum().sum())

    # --- 2.4 invalid-value audit (report only, nothing silently altered) ---
    log["invalid"] = {
        "age_out_of_range": int(((df.age < 18) | (df.age > 120)).sum()),
        "bmi_impossible": int(((df.bmi < 10) | (df.bmi > 70)).sum()),
        "negative_charge": int((df.annual_charge <= 0).sum()),
        "exercise_out_of_range": int(((df.exercise_freq < 0) |
                                      (df.exercise_freq > 7)).sum()),
        "children_negative": int((df.children < 0).sum()),
    }

    # --- 2.5 outlier audit (IQR) - flagged, NOT deleted -------------------
    log["outliers_iqr"] = {}
    for c in ["age", "bmi", "children", "exercise_freq", "annual_charge"]:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        log["outliers_iqr"][c] = int(((df[c] < lo) | (df[c] > hi)).sum())

    log["rows_clean"] = len(df)
    return df, log


# =================================================== 3. HEALTH SCORE ==========
def _score_smoker(smoker):
    """100 = non-smoker (healthiest), 0 = smoker."""
    s = pd.Series(smoker).astype(str).str.lower()
    return np.where(s == "yes", 0.0, 100.0)


def _score_age(age):
    """
    100 at the youngest observed adult age (18), 0 at the oldest (64).
    Linear: ageing is a gradual, monotonic driver of medical cost.
    """
    a = np.asarray(age, dtype=float)
    return np.clip(100.0 * (64.0 - a) / (64.0 - 18.0), 0, 100)


def _score_bmi(bmi):
    """
    Clinical anchoring, not statistical anchoring.
    100 inside the WHO healthy band (18.5-24.9); penalty grows outside it.
    Overweight side penalised 4.5 pts per BMI unit, underweight 8.0 pts per unit
    (underweight deteriorates faster clinically over a much narrower range).
    """
    b = np.asarray(bmi, dtype=float)
    penalty = np.where(b < 18.5, (18.5 - b) * 8.0,
                       np.where(b <= 24.9, 0.0, (b - 24.9) * 4.5))
    return np.clip(100.0 - penalty, 0, 100)


def _score_exercise(freq):
    """100 at 7 workouts/week, 0 at none. Linear over the stated 0-7 range."""
    f = np.asarray(freq, dtype=float)
    return np.clip(100.0 * f / 7.0, 0, 100)


METRIC_SCORERS = {
    "smoker": _score_smoker,
    "age": _score_age,
    "bmi": _score_bmi,
    "exercise_freq": _score_exercise,
}


def derive_health_weights(df_clean, random_state=RANDOM_STATE):
    """
    Data-driven weights, learned on the TRAINING SPLIT ONLY (no leakage).

    Method: fit a standardised linear regression of annual_charge on the four
    health metrics, take |standardised coefficient| as each metric's influence,
    and normalise so the weights sum to 1. This gives each metric a weight
    proportional to how much it actually moves cost in this portfolio, rather
    than an arbitrary analyst opinion.

    Returns (weights dict, diagnostics dict).
    """
    tmp = df_clean.copy()
    tmp["smoker_bin"] = (tmp["smoker"] == "yes").astype(int)
    cols = ["smoker_bin", "age", "bmi", "exercise_freq"]

    X_tr, _, y_tr, _ = train_test_split(
        tmp[cols], tmp["annual_charge"],
        test_size=TEST_SIZE, random_state=random_state)

    sc = StandardScaler().fit(X_tr)
    lr = LinearRegression().fit(sc.transform(X_tr), y_tr)

    infl = pd.Series(np.abs(lr.coef_), index=cols)
    w = (infl / infl.sum())
    weights = {
        "smoker": float(w["smoker_bin"]),
        "age": float(w["age"]),
        "bmi": float(w["bmi"]),
        "exercise_freq": float(w["exercise_freq"]),
    }
    diag = {
        "std_coefficients": dict(zip(cols, [float(c) for c in lr.coef_])),
        "abs_std_coefficients": {k: float(v) for k, v in infl.items()},
    }
    return weights, diag


def health_score(df, weights):
    """
    Health Score = SUM( Metric Score x Weight )   [the required formula]

    Convention: HIGHER Health Score = HEALTHIER (0 = worst, 100 = best).
    Returns (score Series, per-metric score DataFrame).
    """
    parts = pd.DataFrame(index=df.index)
    parts["smoker_score"] = _score_smoker(df["smoker"])
    parts["age_score"] = _score_age(df["age"])
    parts["bmi_score"] = _score_bmi(df["bmi"])
    parts["exercise_score"] = _score_exercise(df["exercise_freq"])

    score = (parts["smoker_score"] * weights["smoker"]
             + parts["age_score"] * weights["age"]
             + parts["bmi_score"] * weights["bmi"]
             + parts["exercise_score"] * weights["exercise_freq"])
    return score.round(2), parts.round(2)


def health_band(score):
    """Interpretable bands over the Health Score."""
    s = np.asarray(score, dtype=float)
    return pd.Categorical(
        np.select(
            [s >= 75, s >= 55, s >= 35],
            ["Excellent", "Good", "Fair"],
            default="Poor"),
        categories=["Poor", "Fair", "Good", "Excellent"], ordered=True)


# ============================================ 4. FEATURE ENGINEERING =========
def engineer(df_clean, weights):
    """
    FEATURE-ENGINEERED layer. Every derived column is documented here.

    NOTE ON LEAKAGE: health_score is used for SEGMENTATION, REPORTING and the
    customer-facing explanation layer only. It is deliberately NOT fed into the
    Stage-2 regressors, because its weights were themselves learned from
    annual_charge - using it as a predictor would be circular.
    """
    df = df_clean.copy()

    # age_from_dob: built purely to PROVE the age/date_of_birth redundancy
    df["age_from_dob"] = 2026 - pd.to_datetime(df["date_of_birth"]).dt.year

    # Health Score + component scores
    hs, parts = health_score(df, weights)
    df["health_score"] = hs
    df = pd.concat([df, parts], axis=1)
    df["health_band"] = health_band(hs)

    # Interpretable bands used for grouped reporting in the dashboard
    df["age_group"] = pd.cut(df["age"], [17, 29, 39, 49, 64],
                             labels=["18-29", "30-39", "40-49", "50-64"])
    df["bmi_category"] = pd.cut(
        df["bmi"], [0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"], right=False)
    df["is_smoker"] = (df["smoker"] == "yes").astype(int)

    df["risk_tier"] = pd.Categorical(df["risk_tier"], categories=TIERS,
                                     ordered=True)
    return df


def modeling_frame(df_eng):
    """
    MODELING layer: junk + redundant columns removed, categoricals encoded.
    Only genuine, available-at-quote-time predictors survive.
    """
    keep = FEATURES + ["annual_charge", "risk_tier"]
    dm = df_eng[keep].copy()
    dm = pd.get_dummies(dm, columns=CAT_FEATURES, drop_first=True)
    dm["risk_tier"] = dm["risk_tier"].astype(str)
    return dm


def encoded_feature_names(dm):
    return [c for c in dm.columns if c not in ("annual_charge", "risk_tier")]


# =================================================== 5. PREMIUM ENGINE =======
def premium_from_cost(cost):
    """
    Risk Factors -> Health Score -> Two-stage model -> Expected Cost -> Premium

    Premium = Expected Annual Claim Cost x (1 + expense ratio + profit margin)
    """
    return np.asarray(cost, dtype=float) * (1.0 + LOADING)


def explain_premium(row, portfolio_medians):
    """
    Plain-English, individualised explanation built from THIS person's own
    characteristics, compared against the portfolio medians. Returns a sentence
    plus the ranked list of their top contributing risk factors.
    """
    drivers, protectors = [], []

    if str(row["smoker"]).lower() == "yes":
        drivers.append(("Smoker", "is a smoker"))
    else:
        protectors.append("is a non-smoker")

    bmi = float(row["bmi"])
    if bmi >= 30:
        drivers.append(("High BMI", f"has a BMI of {bmi:.1f} (obese range)"))
    elif bmi >= 25:
        drivers.append(("Raised BMI", f"has a BMI of {bmi:.1f} (overweight range)"))
    elif bmi < 18.5:
        drivers.append(("Low BMI", f"has a BMI of {bmi:.1f} (underweight range)"))
    else:
        protectors.append(f"has a healthy BMI of {bmi:.1f}")

    age = float(row["age"])
    if age >= 50:
        drivers.append(("Age", f"is {age:.0f} years old"))
    elif age <= 30:
        protectors.append(f"is young ({age:.0f})")

    ex = float(row["exercise_freq"])
    if ex <= 2:
        drivers.append(("Low activity", f"exercises only {ex:.0f}x per week"))
    elif ex >= 5:
        protectors.append(f"exercises {ex:.0f}x per week")

    ch = int(row["children"])
    if ch >= 3:
        drivers.append(("Dependents", f"has {ch} dependents"))

    hs = float(row["health_score"])
    hs_txt = ("a strong Health Score" if hs >= 70 else
              "a mid-range Health Score" if hs >= 45 else
              "a low Health Score")

    level = ("high" if row["risk_tier"] == "High" else
             "moderate" if row["risk_tier"] == "Medium" else "low")
    if drivers and row["risk_tier"] != "Low":
        lead = ", ".join(d[1] for d in drivers[:3])
        sent = (f"The premium is {level} mainly because this customer {lead}. "
                f"Combined with {hs_txt} ({hs:.0f}/100), this places them in the "
                f"{row['risk_tier']} risk tier.")
    elif drivers:
        lead = ", ".join(d[1] for d in drivers[:3])
        sent = (f"The premium is low overall. The main factor pushing it up is that "
                f"this customer {lead}, but with {hs_txt} ({hs:.0f}/100) they still "
                f"sit in the Low risk tier.")
    else:
        sent = (f"The premium is {level} because this customer "
                f"{', '.join(protectors[:3])}. With {hs_txt} ({hs:.0f}/100) they sit "
                f"in the {row['risk_tier']} risk tier.")

    top = [d[0] for d in drivers[:3]] or ["No elevated risk factors"]
    return sent, "; ".join(top)


# ============================================== 6. DATA DICTIONARY ===========
DATA_DICTIONARY = [
    ("customer_id", "Identifier", "Unique customer reference (not a predictor)"),
    ("age", "Original", "Age in years (18-64)"),
    ("sex", "Original", "male / female"),
    ("bmi", "Original (imputed)", "Body Mass Index; 18 missing filled with median"),
    ("weight_kg", "Original (unused)", "Weight in kg - dropped from modelling, r=0.987 with bmi"),
    ("children", "Original", "Number of dependents (0-4)"),
    ("smoker", "Original", "yes / no - the dominant cost driver"),
    ("region", "Original", "north / south / east / west"),
    ("exercise_freq", "Original (imputed)", "Workouts per week 0-7; 18 missing filled with median"),
    ("date_of_birth", "Original (unused)", "Dropped - perfectly redundant with age (r=1.000)"),
    ("favorite_color", "Original (dropped)", "Irrelevant to medical cost"),
    ("zodiac_sign", "Original (dropped)", "Irrelevant; apparent spread is small-sample noise (ANOVA p=0.82)"),
    ("lucky_number", "Original (dropped)", "Irrelevant; r=-0.031 with charge"),
    ("preferred_contact", "Original (dropped)", "Servicing channel, not a health risk factor"),
    ("marketing_opt_in", "Original (dropped)", "Marketing consent, not a health risk factor"),
    ("annual_charge", "Original TARGET", "Observed annual medical claim cost (Stage-2 target)"),
    ("risk_tier", "Original TARGET", "Low / Medium / High risk band (Stage-1 target)"),
    ("age_from_dob", "Engineered", "Age recomputed from date_of_birth to evidence redundancy"),
    ("smoker_score", "Engineered", "Health metric score 0-100 (100 = non-smoker)"),
    ("age_score", "Engineered", "Health metric score 0-100 (100 = age 18)"),
    ("bmi_score", "Engineered", "Health metric score 0-100 (100 = WHO healthy band)"),
    ("exercise_score", "Engineered", "Health metric score 0-100 (100 = 7 workouts/week)"),
    ("health_score", "Engineered", "SUM(metric score x weight); HIGHER = HEALTHIER (0-100)"),
    ("health_band", "Engineered", "Poor / Fair / Good / Excellent band over health_score"),
    ("age_group", "Engineered", "18-29 / 30-39 / 40-49 / 50-64 reporting band"),
    ("bmi_category", "Engineered", "Underweight / Normal / Overweight / Obese (WHO)"),
    ("is_smoker", "Engineered", "Binary smoker flag for modelling and grouping"),
    ("predicted_tier", "Model output", "Stage-1 predicted risk tier (out-of-fold)"),
    ("predicted_cost", "Model output", "Stage-2 predicted annual claim cost (out-of-fold)"),
    ("final_premium", "Model output", "predicted_cost x 1.23 (expense + margin loading)"),
    ("premium_explanation", "Model output", "Plain-English, individualised reason for the premium"),
    ("top_risk_factors", "Model output", "Ranked top contributing risk factors for this person"),
]
