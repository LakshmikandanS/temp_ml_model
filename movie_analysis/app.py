import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(__file__)
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline_outputs")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "movies.csv")
MODEL_COMPARISON_PATH = os.path.join(PIPELINE_DIR, "model_comparison.csv")
PREDICTIONS_PATH = os.path.join(PIPELINE_DIR, "predictions.csv")


st.set_page_config(
    page_title="Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: "Manrope", "Segoe UI", sans-serif;
            }

            /* Make sure main content text is readable on light backgrounds */
            [data-testid="stAppViewContainer"] {
                color: #1e293b !important;
            }
            [data-testid="stAppViewContainer"] h1, 
            [data-testid="stAppViewContainer"] h2, 
            [data-testid="stAppViewContainer"] h3, 
            [data-testid="stAppViewContainer"] p, 
            [data-testid="stAppViewContainer"] span,
            [data-testid="stAppViewContainer"] label {
                color: #1e293b !important;
            }
            
            /* Sidebar background and text */
            [data-testid="stSidebar"] {
                background-color: #0f172a;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] p, 
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label {
                color: #f8fafc !important;
            }

            .stApp {
                background:
                    radial-gradient(circle at 10% 10%, rgba(10, 132, 255, 0.12), transparent 40%),
                    radial-gradient(circle at 90% 20%, rgba(255, 99, 71, 0.12), transparent 35%),
                    linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            }

            .hero-card {
                background: linear-gradient(120deg, #0f172a 0%, #1e293b 100%);
                border-radius: 16px;
                padding: 22px;
                color: #f8fafc !important;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.22);
                margin-bottom: 12px;
            }
            
            .hero-card h1, .hero-card h2, .hero-card h3, .hero-card p, .hero-card span {
                color: #f8fafc !important;
            }

            .metric-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 14px;
                color: #1e293b !important;
                box-shadow: 0 4px 10px rgba(30, 41, 59, 0.05);
            }
            
            .metric-card h1, .metric-card h2, .metric-card h3, .metric-card p, .metric-card span {
                color: #1e293b !important;
            }

            .pred-result {
                font-size: 2.25rem;
                font-weight: 800;
                color: #0b8f4d !important;
                margin-top: 4px;
                margin-bottom: 2px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_release_month(released_str: Any) -> int:
    if pd.isna(released_str):
        return 6

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    lower = str(released_str).lower()
    for name, num in months.items():
        if name in lower:
            return num
    return 6


def detect_franchise(name: Any) -> int:
    if pd.isna(name):
        return 0

    patterns = [
        r"\b(II|III|IV|V|VI|VII|VIII|IX|X)\b",
        r"\b[2-9]\b",
        r"\bpart\b",
        r"\bchapter\b",
        r"\bepisode\b",
        r"\breturn\b",
        r"\brevenge\b",
        r"\brises\b",
        r"\breloaded\b",
        r"\bawakens\b",
    ]
    lower = str(name).lower()
    for pat in patterns:
        if re.search(pat, lower):
            return 1
    return 0


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def first_existing(paths: List[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return ""


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    if not os.path.exists(MODEL_COMPARISON_PATH):
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not os.path.exists(PREDICTIONS_PATH):
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def build_test_metadata() -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return pd.DataFrame()

    dfx = df.dropna(subset=["gross"]).copy()
    dfx = dfx.dropna(subset=["budget"])
    dfx = dfx[dfx["budget"] > 0].copy()
    if dfx.empty:
        return pd.DataFrame()

    dfx["_strat_bin"] = pd.qcut(dfx["gross"], q=5, labels=False, duplicates="drop")
    _, test_df = train_test_split(
        dfx,
        test_size=0.2,
        random_state=42,
        stratify=dfx["_strat_bin"],
    )
    test_df = test_df.drop(columns=["_strat_bin"]).reset_index(drop=True)
    return test_df


@st.cache_data(show_spinner=False)
def load_predictions_with_context() -> pd.DataFrame:
    preds = load_predictions()
    if preds.empty:
        return pd.DataFrame()

    preds = preds.copy()
    preds.insert(0, "movie_id", np.arange(1, len(preds) + 1, dtype=int))

    meta = build_test_metadata()
    if meta.empty or len(meta) != len(preds):
        return preds

    try:
        matches = np.allclose(
            meta["gross"].to_numpy(dtype=float),
            preds["actual"].to_numpy(dtype=float),
            rtol=0,
            atol=1,
        )
    except Exception:
        matches = False

    if not matches:
        return preds

    useful_cols = [
        c
        for c in ["name", "year", "genre", "director", "company", "budget", "gross"]
        if c in meta.columns
    ]
    return pd.concat(
        [preds.reset_index(drop=True), meta[useful_cols].reset_index(drop=True)],
        axis=1,
    )


@st.cache_resource(show_spinner=False)
def load_model() -> Dict[str, Any]:
    model_path = first_existing(
        [
            os.path.join(BASE_DIR, "trained_model.pkl"),
            os.path.join(PIPELINE_DIR, "trained_model.pkl"),
            os.path.join(PIPELINE_DIR, "rf_model.pkl"),
            os.path.join(PIPELINE_DIR, "lgb_model.pkl"),
        ]
    )
    ohe_path = first_existing(
        [
            os.path.join(PIPELINE_DIR, "ohe.pkl"),
            os.path.join(BASE_DIR, "encoder.pkl"),
        ]
    )
    target_enc_path = first_existing(
        [
            os.path.join(PIPELINE_DIR, "target_encodings.pkl"),
            os.path.join(BASE_DIR, "target_encodings.pkl"),
        ]
    )
    exp_path = first_existing(
        [
            os.path.join(PIPELINE_DIR, "experience_dicts.pkl"),
            os.path.join(BASE_DIR, "experience_dicts.pkl"),
        ]
    )
    feat_path = first_existing(
        [
            os.path.join(PIPELINE_DIR, "feature_cols.pkl"),
            os.path.join(BASE_DIR, "feature_cols.pkl"),
        ]
    )

    artifacts: Dict[str, Any] = {
        "model": None,
        "model_name": "Unavailable",
        "ohe": None,
        "target_encodings": {},
        "experience_dicts": {},
        "feature_cols": [],
    }

    if model_path:
        artifacts["model"] = joblib.load(model_path)
        artifacts["model_name"] = os.path.basename(model_path)
    if ohe_path:
        artifacts["ohe"] = joblib.load(ohe_path)
    if target_enc_path:
        artifacts["target_encodings"] = joblib.load(target_enc_path)
    if exp_path:
        artifacts["experience_dicts"] = joblib.load(exp_path)
    if feat_path:
        artifacts["feature_cols"] = joblib.load(feat_path)

    return artifacts


def preprocess_input(user_input: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    df_in = pd.DataFrame([user_input]).copy()

    for col in ["budget", "runtime", "score", "votes", "year"]:
        df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0)

    df_in["log_budget"] = np.log1p(df_in["budget"])
    df_in["log_votes"] = np.log1p(df_in["votes"])
    df_in["budget_x_votes"] = df_in["log_budget"] * df_in["log_votes"]
    df_in["budget_x_score"] = df_in["log_budget"] * df_in["score"]
    df_in["popularity"] = df_in["votes"] * df_in["score"]
    df_in["budget_per_min"] = np.where(df_in["runtime"] > 0, df_in["budget"] / df_in["runtime"], 0)
    df_in["years_since_1980"] = df_in["year"] - 1980

    if "released" not in df_in.columns:
        df_in["released"] = f"June 1, {int(df_in['year'].iloc[0])} ({df_in.get('country', pd.Series(['United States'])).iloc[0]})"
    df_in["release_month"] = df_in["released"].apply(parse_release_month)
    df_in["is_summer"] = df_in["release_month"].isin([5, 6, 7]).astype(int)
    df_in["is_holiday"] = df_in["release_month"].isin([11, 12]).astype(int)

    df_in["is_franchise"] = df_in["name"].apply(detect_franchise)
    age = (datetime.now().year - df_in["year"]).clip(lower=1)
    df_in["votes_per_year"] = df_in["votes"] / age

    exp_dicts = artifacts.get("experience_dicts", {}) or {}
    for role in ["director", "star", "writer"]:
        role_map = exp_dicts.get(role, {})
        df_in[f"{role}_exp"] = df_in[role].map(role_map).fillna(1).astype(float)

    ohe = artifacts.get("ohe")
    if ohe is not None:
        low_card = ["rating", "genre"]
        for col in low_card:
            df_in[col] = df_in[col].fillna("Unknown")
        encoded = ohe.transform(df_in[low_card])
        try:
            ohe_cols = list(ohe.get_feature_names_out(low_card))
        except Exception:
            ohe_cols = [f"ohe_{i}" for i in range(encoded.shape[1])]
        ohe_df = pd.DataFrame(encoded, columns=ohe_cols, index=df_in.index)
        df_in = pd.concat([df_in, ohe_df], axis=1)

    target_encodings = artifacts.get("target_encodings", {}) or {}
    for col in ["director", "writer", "star", "company", "country"]:
        df_in[col] = df_in[col].fillna("Unknown")
        config = target_encodings.get(col, {})
        mapping = config.get("mapping", {})
        global_mean = config.get("global_mean", 0.0)
        df_in[f"{col}_enc"] = float(mapping.get(df_in[col].iloc[0], global_mean))

    feature_cols = artifacts.get("feature_cols", []) or []
    if feature_cols:
        final_df = df_in.reindex(columns=feature_cols, fill_value=0)
    else:
        final_df = df_in.select_dtypes(include=[np.number]).copy()

    final_df = final_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return final_df


def predict(features: pd.DataFrame, artifacts: Dict[str, Any]) -> Tuple[float, float]:
    model = artifacts.get("model")
    if model is None:
        raise ValueError("Model artifact is not available.")

    raw_pred = float(model.predict(features)[0])
    if raw_pred < 1000:
        revenue = float(np.expm1(raw_pred))
    else:
        revenue = raw_pred

    return max(revenue, 0.0), raw_pred


def render_home(df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0;">🎬 Movie Revenue Predictor</h1>
            <p style="margin-top:10px; margin-bottom:0; font-size:1.05rem;">
                A beginner-friendly ML dashboard that explains movie data, visualizes insights,
                compares model performance, and predicts global box office revenue for new movies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_model_name = "RandomForest"
    best_r2 = 0.73
    if not model_df.empty and "R2" in model_df.columns:
        best_row = model_df.sort_values("R2", ascending=False).iloc[0]
        best_model_name = str(best_row["model"])
        best_r2 = float(best_row["R2"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dataset Size", f"{len(df):,} rows" if not df.empty else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Project Goal", "Predict Gross Revenue")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best Model", f"{best_model_name} (R² {best_r2:.3f})")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("How to Use This App")
    st.write("1. Start with Dataset Overview to understand the raw movie data.")
    st.write("2. Explore EDA & Insights to see patterns behind box office outcomes.")
    st.write("3. Review Model Performance and Predictions Analysis for reliability.")
    st.write("4. Use Predict Movie Revenue to estimate revenue for your custom movie setup.")


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.header("📊 Dataset Overview")
    if df.empty:
        st.error("Dataset not found. Please ensure datasets/movies.csv exists.")
        return

    st.subheader("Sample Data")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("What Each Feature Means")
    feature_help = pd.DataFrame(
        {
            "Feature": [
                "budget",
                "votes",
                "score",
                "genre",
                "director",
                "writer",
                "star",
                "company",
                "rating",
                "runtime",
                "year",
                "gross",
            ],
            "Beginner Explanation": [
                "How much money was spent to produce the movie.",
                "How many IMDb users voted for this movie.",
                "Average IMDb score out of 10.",
                "Movie category like Action, Drama, Comedy.",
                "Person who directed the movie.",
                "Person who wrote the screenplay/story.",
                "Main actor/actress for the movie.",
                "Production studio behind the movie.",
                "Audience suitability label (PG, R, etc.).",
                "Movie length in minutes.",
                "Release year.",
                "Global box office revenue (target variable).",
            ],
        }
    )
    st.dataframe(feature_help, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Missing Values Summary")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values detected.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            missing.head(15).sort_values().plot(kind="barh", color="#ef4444", ax=ax)
            ax.set_xlabel("Missing Count")
            ax.set_ylabel("Feature")
            ax.set_title("Top Features with Missing Values")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Some fields like budget are missing for part of the dataset, which impacts modeling quality.")

    with c2:
        st.subheader("Target Distribution: Gross Revenue")
        gross = pd.to_numeric(df["gross"], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(gross, bins=60, color="#2563eb", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Gross Revenue ($)")
        ax.set_ylabel("Movie Count")
        ax.set_title("Gross Revenue Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Most movies earn relatively lower revenue, while a small number of blockbusters earn extremely high revenue.")


def render_eda_insights(df: pd.DataFrame) -> None:
    st.header("🔍 EDA & Insights")
    if df.empty:
        st.error("Dataset not found. Please ensure datasets/movies.csv exists.")
        return

    plot_df = df.dropna(subset=["budget", "gross", "votes", "genre", "director", "company"]).copy()
    plot_df = plot_df[(plot_df["budget"] > 0) & (plot_df["gross"] > 0) & (plot_df["votes"] > 0)].copy()
    if plot_df.empty:
        st.warning("Not enough clean records for EDA visualizations.")
        return

    sample_n = min(2500, len(plot_df))
    sampled = plot_df.sample(sample_n, random_state=42)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Budget vs Gross")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.scatter(sampled["budget"], sampled["gross"], alpha=0.35, color="#0ea5e9", s=20)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Budget ($, log scale)")
        ax.set_ylabel("Gross ($, log scale)")
        ax.set_title("Higher Budget Often Raises Revenue Ceiling")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Bigger budgets usually increase upside, but many expensive movies still underperform.")

    with c2:
        st.subheader("Votes vs Gross")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.scatter(sampled["votes"], sampled["gross"], alpha=0.35, color="#22c55e", s=20)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Votes ($, log scale)")
        ax.set_ylabel("Gross ($, log scale)")
        ax.set_title("Audience Buzz Correlates with Revenue")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("More audience engagement tends to align with stronger box office performance.")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Genre vs Average Revenue")
        genre_avg = (
            plot_df.groupby("genre", observed=False)["gross"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
        )
        fig, ax = plt.subplots(figsize=(7.5, 5))
        genre_avg.sort_values().plot(kind="barh", color="#f59e0b", ax=ax)
        ax.set_xlabel("Average Gross Revenue ($)")
        ax.set_ylabel("Genre")
        ax.set_title("Top Genres by Average Revenue")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Action and event-driven genres often lead due to global audience appeal.")

    with c4:
        st.subheader("Top Directors & Companies")
        top_directors = (
            plot_df.groupby("director", observed=False)
            .agg(avg_gross=("gross", "mean"), movies=("gross", "size"))
            .query("movies >= 3")
            .sort_values("avg_gross", ascending=False)
            .head(7)
            .reset_index()
        )
        top_companies = (
            plot_df.groupby("company", observed=False)
            .agg(avg_gross=("gross", "mean"), movies=("gross", "size"))
            .query("movies >= 5")
            .sort_values("avg_gross", ascending=False)
            .head(7)
            .reset_index()
        )

        sub1, sub2 = st.columns(2)
        with sub1:
            fig, ax = plt.subplots(figsize=(3.7, 4.5))
            ax.barh(top_directors["director"], top_directors["avg_gross"], color="#8b5cf6")
            ax.set_title("Directors")
            ax.set_xlabel("Avg Gross")
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
        with sub2:
            fig, ax = plt.subplots(figsize=(3.7, 4.5))
            ax.barh(top_companies["company"], top_companies["avg_gross"], color="#ef4444")
            ax.set_title("Companies")
            ax.set_xlabel("Avg Gross")
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)

        st.caption("Historical track records of directors and studios provide useful predictive signals.")


def render_model_performance(model_df: pd.DataFrame) -> None:
    st.header("🤖 Model Performance")
    if model_df.empty:
        st.error("model_comparison.csv not found in pipeline_outputs.")
        return

    sorted_df = model_df.sort_values("R2", ascending=False).reset_index(drop=True)
    best_row = sorted_df.iloc[0]

    st.subheader("All Models")
    st.dataframe(sorted_df, use_container_width=True)
    st.success(
        f"Best model: {best_row['model']} with R² = {best_row['R2']:.3f}, "
        f"RMSE = {format_currency(float(best_row['RMSE']))}, "
        f"MAE = {format_currency(float(best_row['MAE']))}."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(sorted_df["model"], sorted_df["R2"], color="#2563eb")
        ax.set_ylabel("R²")
        ax.set_xlabel("Model")
        ax.set_title("Model Accuracy (Higher is Better)")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("RMSE vs MAE")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        idx = np.arange(len(sorted_df))
        width = 0.4
        ax.bar(idx - width / 2, sorted_df["RMSE"], width=width, label="RMSE", color="#f97316")
        ax.bar(idx + width / 2, sorted_df["MAE"], width=width, label="MAE", color="#22c55e")
        ax.set_xticks(idx)
        ax.set_xticklabels(sorted_df["model"], rotation=30)
        ax.set_ylabel("Error ($)")
        ax.set_title("Error Metrics Comparison")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Interpretation")
    st.write("Tree-based models perform better because movie revenue patterns are highly non-linear and interaction-heavy.")
    st.write("Linear Regression underperforms because it assumes straight-line relationships and cannot capture complex market behavior.")


def render_predictions_analysis(pred_df: pd.DataFrame) -> None:
    st.header("📈 Predictions Analysis")
    if pred_df.empty:
        st.error("predictions.csv not found in pipeline_outputs.")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.35, color="#0ea5e9", s=18)
        lim_max = max(pred_df["actual"].max(), pred_df["predicted"].max())
        ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=1.5)
        ax.set_xlabel("Actual Revenue ($)")
        ax.set_ylabel("Predicted Revenue ($)")
        ax.set_title("Actual vs Predicted Revenue")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("Error Distribution")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.hist(pred_df["error"], bins=50, color="#f97316", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Error (actual - predicted)")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Error Histogram")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Top Overpredicted & Underpredicted Movies")
    display_cols = [c for c in ["movie_id", "name", "year", "genre", "actual", "predicted", "error"] if c in pred_df.columns]
    underpred = pred_df.sort_values("error", ascending=False).head(10)
    overpred = pred_df.sort_values("error", ascending=True).head(10)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Underpredicted (Actual >> Predicted)**")
        st.dataframe(underpred[display_cols], use_container_width=True)
    with cc2:
        st.markdown("**Overpredicted (Predicted >> Actual)**")
        st.dataframe(overpred[display_cols], use_container_width=True)

    st.subheader("Sample Predictions")
    st.dataframe(pred_df[display_cols].head(20), use_container_width=True)

    st.write("The model tends to underestimate large blockbuster outcomes and overestimate smaller films.")
    st.write("This is common when extreme high-revenue events are rare and hard to model precisely.")


def render_predict_page(df: pd.DataFrame, artifacts: Dict[str, Any]) -> None:
    st.header("🎯 Predict Movie Revenue")

    if artifacts.get("model") is None:
        st.error("Model file not found. Please add trained_model.pkl or rf_model.pkl.")
        return

    genre_options = sorted(df["genre"].dropna().unique().tolist()) if "genre" in df.columns and not df.empty else ["Action", "Comedy", "Drama"]
    rating_options = sorted(df["rating"].dropna().unique().tolist()) if "rating" in df.columns and not df.empty else ["PG", "PG-13", "R"]
    country_options = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns and not df.empty else ["United States"]
    top_directors = df["director"].value_counts().head(30).index.tolist() if "director" in df.columns and not df.empty else []

    st.caption(f"Loaded model: {artifacts.get('model_name', 'unknown')} | Preprocessing artifacts: {'available' if artifacts.get('feature_cols') else 'partial'}")

    with st.form("predict_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            name = st.text_input("Movie Title", value="My New Movie")
            budget = st.number_input("Budget ($)", min_value=1_000, value=50_000_000, step=500_000)
            genre = st.selectbox("Genre", options=genre_options, index=0)
            rating = st.selectbox("Rating", options=rating_options, index=0)
            runtime = st.number_input("Runtime (minutes)", min_value=40, max_value=300, value=115, step=1)
            year = st.number_input("Year", min_value=1980, max_value=2035, value=2026, step=1)

        with c2:
            score = st.slider("IMDb Score", min_value=1.0, max_value=10.0, value=6.8, step=0.1)
            votes = st.number_input("Votes", min_value=0, value=120_000, step=1000)
            director_mode = st.selectbox("Director Input", options=["Choose popular", "Type custom"], index=0)
            if director_mode == "Choose popular" and top_directors:
                director = st.selectbox("Director", options=top_directors, index=0)
            else:
                director = st.text_input("Director", value="Unknown")
            star = st.text_input("Star", value="Unknown")
            company = st.text_input("Company", value="Unknown")
            country = st.selectbox("Country", options=country_options, index=country_options.index("United States") if "United States" in country_options else 0)

        submitted = st.form_submit_button("Predict Revenue")

    if not submitted:
        return

    name = (name or "Unknown Movie").strip()
    director = (director or "Unknown").strip()
    star = (star or "Unknown").strip()
    company = (company or "Unknown").strip()
    country = (country or "United States").strip()

    if budget <= 0:
        st.error("Budget must be greater than zero.")
        return
    if runtime <= 0:
        st.error("Runtime must be greater than zero.")
        return

    input_payload = {
        "name": name,
        "rating": rating,
        "genre": genre,
        "year": int(year),
        "released": f"June 1, {int(year)} ({country})",
        "score": float(score),
        "votes": float(votes),
        "director": director,
        "writer": "Unknown",
        "star": star,
        "country": country,
        "budget": float(budget),
        "gross": 0.0,
        "company": company,
        "runtime": float(runtime),
    }

    try:
        features = preprocess_input(input_payload, artifacts)
        predicted_revenue, _ = predict(features, artifacts)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.markdown("---")
    st.subheader("Prediction Result")
    st.markdown(f'<div class="pred-result">🎯 {format_currency(predicted_revenue)}</div>', unsafe_allow_html=True)
    st.info("This is an estimate based on historical patterns.")

    multiple = predicted_revenue / float(budget)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted Revenue", format_currency(predicted_revenue))
    with c2:
        st.metric("Input Budget", format_currency(float(budget)))
    with c3:
        st.metric("Revenue / Budget", f"{multiple:.2f}x")

    if multiple >= 3:
        st.success("Strong upside scenario: this setup resembles high-performing movies in the dataset.")
    elif multiple >= 1:
        st.warning("Moderate scenario: expected to recover budget with limited upside.")
    else:
        st.error("Risk scenario: estimated revenue is below the input budget.")


def main() -> None:
    apply_custom_theme()

    df = load_data()
    model_df = load_model_comparison()
    pred_df = load_predictions_with_context()
    artifacts = load_model()

    with st.sidebar:
        st.title("Navigation")
        section = st.radio(
            "Go to",
            options=[
                "🏠 Home",
                "📊 Dataset Overview",
                "🔍 EDA & Insights",
                "🤖 Model Performance",
                "📈 Predictions Analysis",
                "🎯 Predict Movie Revenue",
            ],
            index=0,
        )
        st.markdown("---")
        st.caption("Movie Gross Revenue Prediction Dashboard")

    if section == "🏠 Home":
        render_home(df, model_df)
    elif section == "📊 Dataset Overview":
        render_dataset_overview(df)
    elif section == "🔍 EDA & Insights":
        render_eda_insights(df)
    elif section == "🤖 Model Performance":
        render_model_performance(model_df)
    elif section == "📈 Predictions Analysis":
        render_predictions_analysis(pred_df)
    elif section == "🎯 Predict Movie Revenue":
        render_predict_page(df, artifacts)


if __name__ == "__main__":
    main()
