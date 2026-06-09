# fiserv_complete_dashboard.py
# ✅ ML‑integrated version using pseudo‑data trained LightGBM models
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import os
import ast
from datetime import date
from groq import Groq
import shap
import matplotlib.pyplot as plt
import io
from PIL import Image


# ============================================================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ============================================================================
st.set_page_config(
    page_title="Fiserv Impact Estimator/Delivery Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================



st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7FA;
}
            
@font-face {
    font-family: 'Univers';
    src: url('assets/UniversCnBold.ttf') format('truetype');
}

/* KPI Cards */
.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    border-left: 6px solid #0E1117;
}

/* Section Containers */
.section-card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0E1117;
}
            
/* Make sidebar info boxes dark to match the slider background */
section[data-testid="stSidebar"] div[data-testid="stInfo"] {
    background-color: #0E1117 !important;
    color: white !important;
}

/* Sidebar date input field */
section[data-testid="stSidebar"] div[data-testid="stDateInput"] input {
    background-color: #0E1117 !important;      /* same dark background as slider */
    color: white !important;                   /* text white for contrast */
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div {
    background-color: #0E1117 !important;
    color: white !important;
}
            
/* Sidebar text area (the input field itself) */
section[data-testid="stSidebar"] textarea {
    background-color: #FF0000 !important;   /* dark background */
    color: white !important;
    border: 1px solid white !important;     /* white border */
    border-radius: 8px;
}

/* Sidebar button (the container) */
section[data-testid="stSidebar"] button {
    background-color: #FF0000 !important;   /* orange background */
    color: white !important;
    border: none !important;
    border-radius: 10px;
    font-weight: bold;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

/* Risk badges */
.risk-high {
    color: #ff4b4b;
    font-weight: bold;
}

.risk-medium {
    color: #f39c12;
    font-weight: bold;
}

.risk-low {
    color: #00c853;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================
@st.cache_data
def load_release_cadence(csv_path="data/release_cadence.csv"):
    df = pd.read_csv(csv_path)
    # Ensure each row has a list of stages (split by comma if multiple)

    df["week_start"] = pd.to_datetime(df["week_start"], format="%d/%m/%y", errors="coerce")
    df["week_end"] = pd.to_datetime(df["week_end"], format="%d/%m/%y", errors="coerce")
    # Drop rows where conversion failed (optional)
    df = df.dropna(subset=["week_start", "week_end"])

    # Ensure stage_list is a string (if it's already a string, split later)
    if "stage_list" in df.columns:
        # If the column contains lists (e.g., from a bad CSV), convert to string
        df["stage_list"] = df["stage_list"].astype(str)
    else:
        st.error("Column 'stage_list' not found in release_cadence.csv")
        return df
    return df


@st.cache_data
def get_historical_stats():
    """Load summary statistics from change_requests.csv (cached)."""
    if data is None or "change_requests" not in data:
        return {}
    df = data["change_requests"]
    stats = {
        "total_changes": len(df),
        "avg_delay": df["delay_days_caused"].mean(),
        "spillover_rate": df["caused_spillover"].mean(),
        "avg_story_points": df["story_points"].mean(),
        "priority_dist": df["priority"].value_counts().to_dict()
    }
    # Optional: add recent trend (e.g., last 30 days)
    if "request_date" in df.columns:
        df["request_date"] = pd.to_datetime(df["request_date"])
        recent = df[df["request_date"] > (pd.Timestamp.now() - pd.Timedelta(days=30))]
        stats["recent_avg_delay"] = recent["delay_days_caused"].mean() if not recent.empty else None
    return stats

def get_active_releases(cadence_df, current_date):
    mask = (cadence_df["week_start"] <= current_date) & (current_date <= cadence_df["week_end"])
    active_series = cadence_df.loc[mask, "active_releases"]
    releases = set()
    for val in active_series:
        if isinstance(val, str):
            for r in val.split(","):
                releases.add(r.strip())
    return sorted(releases)

def get_stages_for_release_from_cadence(original_df, release_id, current_date):
    """
    original_df: the raw cadence DataFrame (before expansion) containing columns:
        week_start, week_end, stage_list, active_releases (comma‑separated)
    Returns a list of stage names for the given release on the given date.
    """
    # Find the row(s) for the week containing the current date
    mask = (original_df["week_start"] <= current_date) & (current_date <= original_df["week_end"])
    rows = original_df.loc[mask]
    if rows.empty:
        return []
    # For simplicity, take the first row (there should be only one week per date)
    row = rows.iloc[0]
    stage_list_str = row["stage_list"]
    # Split into individual stage‑release items
    if pd.isna(stage_list_str):
        return []
    items = [item.strip() for item in stage_list_str.split(",")]
    stages = []
    for item in items:
        # Format: "StageName(ReleaseID)" e.g., "CAT(X+4)"
        if "(" in item and item.endswith(")"):
            stage, rel = item.rsplit("(", 1)
            rel = rel.rstrip(")")
            if rel == release_id:
                stages.append(stage)
    return stages

def get_total_active_stages(original_df, current_date):
    mask = (original_df["week_start"] <= current_date) & (current_date <= original_df["week_end"])
    rows = original_df.loc[mask]
    if rows.empty:
        return 0
    # Take the first matching row (should be unique per week)
    stage_str = rows.iloc[0]["stage_list"]
    # If it's a Series (unlikely), extract the scalar
    if hasattr(stage_str, "iloc"):
        stage_str = stage_str.iloc[0]
    if pd.isna(stage_str):
        return 0
    # Split by comma to count stages (each item like "Stage(Release)")
    items = [item.strip() for item in stage_str.split(",")]
    return len(items)

@st.cache_resource
def load_fiserv_data():
    data_dir = "data"   # adjust if different
    if not os.path.exists(data_dir):
        st.warning(f"Data directory '{data_dir}' not found. Run the data generator first.")
        return None
    try:
        return {
            "releases": pd.read_csv(f"{data_dir}/releases.csv"),
            "sprints": pd.read_csv(f"{data_dir}/sprints.csv"),
            "work_items": pd.read_csv(f"{data_dir}/work_items.csv"),
            "release_scope": pd.read_csv(f"{data_dir}/release_scope.csv"),
            "change_requests": pd.read_csv(f"{data_dir}/change_requests.csv"),
            "resources": pd.read_csv(f"{data_dir}/resources.csv"),
            "teams": pd.read_csv(f"{data_dir}/teams.csv"),
            "resource_types": pd.read_csv(f"{data_dir}/resource_types.csv"),
            "release_cadence": pd.read_csv(f"{data_dir}/release_cadence.csv")
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load the LightGBM models trained on pseudo‑data."""
    model_dir = "models"   # adjust if models are elsewhere
    models = {}
    if not os.path.exists(model_dir):
        st.info("Model directory not found – will use heuristic only.")
        return None
    try:
        models["spillover"] = joblib.load(os.path.join(model_dir, "classifier_spillover_lgb.pkl"))
        models["regression"] = joblib.load(os.path.join(model_dir, "regressor_delay_days.pkl"))
        # No scaler needed for LightGBM (tree‑based)
        models["scaler"] = None
        # Get the exact feature names the model expects
        if hasattr(models["spillover"], "feature_names_in_"):
            models["feature_names"] = models["spillover"].feature_names_in_.tolist()
        else:
            models["feature_names"] = None
        return models
    except Exception as e:
        st.warning(f"Could not load ML models: {e}")
        return None

data = load_fiserv_data()
ml_models = load_models()

def get_current_stages(cadence_df, current_date):
    active = []
    for _, row in cadence_df.iterrows():
        if row["week_start"] <= current_date <= row["week_end"]:
            active.append({
                "release": row["release_name"],
                "stage": row["stage_list"],
                "week_start": row["week_start"],
                "week_end": row["week_end"]
            })
    return active



def parallel_impact(target_release, delay_days, active_stages):
    impact = []
    for rel in active_stages:
        if rel["release"] == target_release:
            continue
        impact.append({
            "Release": rel["release"],
            "Current Stage(s)": ", ".join(rel["stage"]),
            "Additional Delay (days)": delay_days
        })
    return pd.DataFrame(impact)

priority_mult = {"Low":0.3, "Medium":0.6, "High":0.9, "Critical":1.2}

def build_full_timeline(cadence_df, start_date, finish_date, current_date, delay_days=0):
    """
    Build timeline between selected start and finish dates.
    If delay_days > 0, future end dates are shifted.
    """
    rows = []
    # Filter cadence window

    filtered_df = cadence_df[

        (cadence_df["week_end"] >= start_date) &

        (cadence_df["week_start"] <= finish_date)

    ]
    for _, row in filtered_df.iterrows():
        # Parse active releases and stages
        releases = [r.strip() for r in str(row["active_releases"]).split(",")]
        stages_list = [s.strip() for s in str(row["stage_list"]).split(",")]
        
        # For each stage item (format "Stage(Release)")
        for item in stages_list:
            if '(' not in item or not item.endswith(')'):
                continue
            stage, rel = item.rsplit('(', 1)
            rel = rel.rstrip(')')
            if rel not in releases:
                continue
            start = row["week_start"]

            end = row["week_end"]
            # Apply delay to all weeks that start after the current date
            if delay_days > 0 and start > current_date:
                start += pd.Timedelta(days=delay_days)
                end += pd.Timedelta(days=delay_days)
            rows.append({
                "Release": rel,
                "Stage": stage.strip(),
                "Start": start,
                "Finish": end
            })
    return pd.DataFrame(rows)



# Load teams with skills
teams_df = data["teams"].copy()
teams_df["headcount"] = pd.to_numeric(teams_df["headcount"], errors="coerce")

# Ensure skills column exists and is string
if "skills" not in teams_df.columns:
    st.error("Teams CSV missing 'skills' column. Please add it.")
    st.stop()

teams_df["skills"] = teams_df["skills"].fillna("").astype(str)

def get_total_capacity_for_components(teams_df, selected_components):
    """
    Returns total hours per week and list of contributing teams.
    """
    if not selected_components:
        # If no component selected, default to using all teams? 
        # Better to use the first team as fallback.
        return teams_df.iloc[0]["hours_per_week"], [teams_df.iloc[0]["name"]]
    
    total_hours = 0
    contributing_teams = []
    for _, team in teams_df.iterrows():
        team_skills = [s.strip().lower() for s in team["skills"].split(",")]
        if any(comp.lower() in team_skills for comp in selected_components):
            total_hours += team["hours_per_week"]
            contributing_teams.append(team["name"])
    # If no team matches, fallback to first team
    if total_hours == 0:
        return teams_df.iloc[0]["hours_per_week"], [teams_df.iloc[0]["name"]]
    return total_hours, contributing_teams


def get_current_sprint_from_cadence(cadence_df, current_date):
    """
    Find the sprint that contains the current_date using the release_cadence.
    Returns (sprint_name, sprint_duration_days, days_into_sprint)
    """
    # Make a copy to avoid modifying cached data
    df = cadence_df.copy()
    
    # Convert date columns to datetime using the correct format
    date_columns = ["week_start", "week_end", "sprint_start", "sprint_end"]
    for col in date_columns:
        if col in df.columns:
            # Try to convert; assume format day/month/year (e.g., 22/09/25)
            df[col] = pd.to_datetime(df[col], format="%d/%m/%y", errors="coerce")
    
    # Drop rows with failed conversion
    df = df.dropna(subset=["week_start", "week_end", "sprint_start", "sprint_end"])
    
    # Filter rows where current_date falls within week_start and week_end
    mask = (df["week_start"] <= current_date) & (current_date <= df["week_end"])
    if not mask.any():
        return None, None, None
    
    # Take the first matching row
    row = df.loc[mask].iloc[0]
    sprint_name = row["sprint_number"]
    sprint_start = row["sprint_start"]
    sprint_end = row["sprint_end"]
    #sprint_duration = (sprint_end - sprint_start).days
    sprint_duration = 10
    #days_into_sprint = (current_date - sprint_start).days
    days_into_sprint = np.busday_count(sprint_start.date(), current_date.date())+1 # count business days to be more realistic
    return sprint_name, sprint_duration, days_into_sprint

@st.cache_data
def load_skill_capacity():
    df = pd.read_csv("data/resource_types.csv")
    # Aggregate by skill (e.g., QA sum of all QA roles)
    skill_cap = df.groupby("skill")["count"].sum().to_dict()
    # Convert to weekly hours (40h per person)
    for skill in skill_cap:
        skill_cap[skill] = skill_cap[skill] * 40
    return skill_cap

@st.cache_resource
def load_shap_explainer():
    """Create SHAP explainers for both classifier and regressor."""
    explainer_clf = None
    explainer_reg = None
    if ml_models and "spillover" in ml_models:
        explainer_clf = shap.TreeExplainer(ml_models["spillover"])
    if ml_models and "regression" in ml_models:
        explainer_reg = shap.TreeExplainer(ml_models["regression"])
    return explainer_clf, explainer_reg

shap_explainer_clf, shap_explainer_reg = load_shap_explainer()

# ============================================================================
# FEATURE ENGINEERING – EXACT MATCH TO TRAINING (13 features)
# ============================================================================
def engineer_ml_features(story_points, days_into_sprint, sprint_duration,
                         priority, affected_components, is_mid_sprint,
                         team_headcount, base_remaining_capacity_hours,
                         utilisation_factor, available_capacity_ratio,
                         item_type="User Story"):
    """
    Generate the exact 13 features used during LightGBM training.
    All features are numeric; categoricals are encoded using the same mappings.
    """
    # Encode priority (Low=0, Medium=1, High=2, Critical=3)
    priority_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    priority_encoded = priority_map.get(priority, 1)

    # Encode item_type – mapping derived from training data
    # In the pseudo‑data generation, item_type is same as work_item_level:
    # Epic:0, Feature:1, Business Story:2, User Story:3, Task:4
    item_type_map = {"Epic": 0, "Feature": 1, "Business Story": 2,
                     "User Story": 3, "Task": 4}
    item_type_encoded = item_type_map.get(item_type, 3)   # default to User Story

    # Sprint progress & remaining percentage
    if sprint_duration <= 0:
        sprint_duration = 10
    sprint_progress = days_into_sprint / sprint_duration
    remaining_sprint_pct = max(0, 1 - sprint_progress)

    # Engineered features (same formulas as in training)
    complexity_score = float(story_points * (0.5*affected_components))
    predicted_risk_proxy = story_points * 0.5 + affected_components * 0.1 + sprint_progress * 10
    # sprint_task_load: we don't have actual sprint plan, use a heuristic based on story points
    # (the model was trained with values from generated data, but a reasonable estimate works)
    sprint_task_load = min(21, int(story_points * 1.5) + affected_components)

    # Derived flags
    has_estimate = 1
    has_story_points = 1 if story_points > 0 else 0
    story_points_log = np.log1p(story_points)

    # Build feature dictionary in the exact order expected by the model
    # (order does not matter for LightGBM as long as column names match)
    features = {
        "story_points": float(story_points),
        "days_into_sprint": float(days_into_sprint),
        "sprint_duration": float(sprint_duration),
        "sprint_progress_at_creation": float(sprint_progress),
        "remaining_sprint_pct": float(remaining_sprint_pct),
        "complexity_score": float(complexity_score),
        "predicted_risk_proxy": float(predicted_risk_proxy),
        "sprint_task_load": float(sprint_task_load),
        "has_estimate": float(has_estimate),
        "has_story_points": float(has_story_points),
        "story_points_log": float(story_points_log),
        "priority_encoded": float(priority_encoded),
        "item_type_encoded": float(item_type_encoded),
        "team_headcount" : float(team_headcount), 
        "base_remaining_capacity_hours": float(base_remaining_capacity_hours),
        "utilisation_factor": float(utilisation_factor),
        "available_capacity_ratio": float(available_capacity_ratio)
    }

    df = pd.DataFrame([features])

    # Ensure columns are in the same order as the model expects
    if ml_models and ml_models.get("feature_names"):
        expected_cols = ml_models["feature_names"]
        # Add missing columns with 0 (should not happen)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    return df



# ============================================================================
# PREDICTION FUNCTION – ML first, heuristic fallback
# ============================================================================
def predict_impact_ml(story_points, days_into_sprint, sprint_duration, priority,
                      affected_components, team_capacity, is_mid_sprint,
                      team_headcount=None,
                      base_remaining_capacity_hours=None, utilisation_factor=None,
                      available_capacity_ratio=None,
                      item_type="User Story"):
    """
    Hybrid prediction: use LightGBM models if available, otherwise heuristic.
    """
    # ---------- Heuristic fallback (same as original) ----------
    def heuristic():
        sprint_progress = days_into_sprint / sprint_duration if sprint_duration > 0 else 0
        priority_mult = {"Low":0.3, "Medium":0.6, "High":0.9, "Critical":1.2}
        multiplier = priority_mult.get(priority, 1.0)
        timing_penalty = 1 + (sprint_progress * 0.8)
        component_mult = 0.2 + (affected_components - 1) * 0.15
        component_mult = min(component_mult, 1.0)
        mid_sprint_penalty = 1.2 if is_mid_sprint else 1.0
        base_effort = story_points * 6
        adjusted_effort = base_effort * multiplier * timing_penalty * component_mult * mid_sprint_penalty
        remaining_capacity = team_capacity * (1 - sprint_progress)
        if adjusted_effort > remaining_capacity:
            spillover_prob = 1.0
            delay_days = max(1, round((adjusted_effort - remaining_capacity) / 40, 1))
        else:
            spillover_prob = adjusted_effort / remaining_capacity if remaining_capacity > 0 else 0
            delay_days = 0
        return spillover_prob, delay_days
    

    # ---------- ML prediction ----------
    if ml_models and ml_models.get("spillover") and ml_models.get("regression"):
        try:
            X = engineer_ml_features(story_points, days_into_sprint, sprint_duration,
                                     priority, affected_components, is_mid_sprint,
                                     team_headcount, base_remaining_capacity_hours,
                                    utilisation_factor, available_capacity_ratio,
                                     item_type=item_type)
            spillover_prob = ml_models["spillover"].predict_proba(X)[0][1]
            spillover_prob = 0.1 + 0.8*(spillover_prob ** 2)
            delay_days = ml_models["regression"].predict(X)[0]
            used_ml = True
            st.session_state.last_X = X  # Store for SHAP explanations
        except Exception as e:
            st.warning(f"ML prediction failed ({e}). Using heuristic.")
            spillover_prob, delay_days = heuristic()
            used_ml = False
    else:
        spillover_prob, delay_days = heuristic()
        used_ml = False

    # Ensure delay_days is not negative
    delay_days = max(0, delay_days)

    
    # Business logic for recommendation
    if spillover_prob < 0.50:
        recommendation = "Accept in current sprint"
        risk = "Low"
    elif spillover_prob < 0.75:
        recommendation = "Accept with monitoring"
        risk = "Medium"
    else:
        recommendation = "Consider deferring"
        risk = "High"

    return {
        "spillover_prob": spillover_prob,
        "delay_days": delay_days,
        "recommendation": recommendation,
        "risk": risk,
        "sprint_fit": spillover_prob < 0.75,
        "used_ml": used_ml
    }

# ============================================================================
# HELPER FUNCTIONS (unchanged – hierarchy, stage parsing, etc.)
# ============================================================================
def get_hierarchy_options(data):
    """Extract hierarchical options from work items"""
    if data is None or "work_items" not in data:
        return {}
    df = data["work_items"]
    hierarchy = {
        "epics": df[df["level"] == "Epic"]["name"].unique().tolist(),
        "features": {},
        "business_stories": {},
        "user_stories": {},
        "tasks": {}
    }
    for epic_name in hierarchy["epics"]:
        epic_id = df[(df["level"] == "Epic") & (df["name"] == epic_name)]["work_item_id"].values
        if len(epic_id) > 0:
            features = df[(df["level"] == "Feature") & (df["parent_id"] == epic_id[0])]["name"].tolist()
            hierarchy["features"][epic_name] = features
    for epic in hierarchy["epics"]:
        for feature in hierarchy["features"].get(epic, []):
            feature_id = df[(df["level"] == "Feature") & (df["name"] == feature)]["work_item_id"].values
            if len(feature_id) > 0:
                bs = df[(df["level"] == "Business Story") & (df["parent_id"] == feature_id[0])]["name"].tolist()
                hierarchy["business_stories"][f"{epic}|{feature}"] = bs
    for epic in hierarchy["epics"]:
        for feature in hierarchy["features"].get(epic, []):
            for bs in hierarchy["business_stories"].get(f"{epic}|{feature}", []):
                bs_id = df[(df["level"] == "Business Story") & (df["name"] == bs)]["work_item_id"].values
                if len(bs_id) > 0:
                    us = df[(df["level"] == "User Story") & (df["parent_id"] == bs_id[0])]["name"].tolist()
                    hierarchy["user_stories"][f"{epic}|{feature}|{bs}"] = us
    for epic in hierarchy["epics"]:
        for feature in hierarchy["features"].get(epic, []):
            for bs in hierarchy["business_stories"].get(f"{epic}|{feature}", []):
                for us in hierarchy["user_stories"].get(f"{epic}|{feature}|{bs}", []):
                    us_id = df[(df["level"] == "User Story") & (df["name"] == us)]["work_item_id"].values
                    if len(us_id) > 0:
                        tasks = df[(df["level"] == "Task") & (df["parent_id"] == us_id[0])]["name"].tolist()
                        hierarchy["tasks"][f"{epic}|{feature}|{bs}|{us}"] = tasks
    return hierarchy

def safe_parse_stages(stages_str):
    if pd.isna(stages_str):
        return []
    try:
        return ast.literal_eval(stages_str)
    except (SyntaxError, ValueError, TypeError):
        return []
    


    
# ----------------------------------------------------------------------
# Groq AI Assistant
# ----------------------------------------------------------------------
def init_groq():
    """Create Groq client using API key from secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception as e:
        st.error("Groq API key not configured. Please add GROQ_API_KEY to secrets.")
        return None

def get_dashboard_context():
    """Build a concise context for the AI assistant."""
    context = ""
    
    # Latest prediction (if available)
    if "result" in st.session_state and st.session_state.result:
        r = st.session_state.result
        context += f"Latest prediction: Spillover risk = {r['spillover_prob']:.0%}, Delay = {r['delay_days']:.1f} days, Recommendation = {r['recommendation']}.\n"
    
    # Active releases and stages
    if "active_releases" in st.session_state:
        context += f"Active releases: {', '.join(st.session_state.active_releases)}.\n"
    if "total_active_stages" in st.session_state:
        context += f"Total active stages: {st.session_state.total_active_stages}.\n"

    # SHAP classifier features (if available)
    if "top_shap_features" in st.session_state and st.session_state.top_shap_features:
        context += "Top factors influencing this prediction:\n"
        for name, val in st.session_state.top_shap_features:
            direction = "increased" if val > 0 else "decreased"
            context += f"- {name}: {direction} risk by {abs(val):.3f}\n"

    # SHAP regressor features - only if last_X exists
    if "last_X" in st.session_state and st.session_state.last_X is not None:
        X_input = st.session_state.last_X
        if "shap_values_reg" in st.session_state and st.session_state.shap_values_reg is not None:
            contributions_reg = st.session_state.shap_values_reg
            abs_contrib = np.abs(contributions_reg)
            top_idx = np.argsort(abs_contrib)[-5:]
            top_features_reg = [(X_input.columns[i], contributions_reg[i]) for i in top_idx]
            context += "Top factors influencing expected delay:\n"
            for name, val in top_features_reg:
                direction = "increased" if val > 0 else "decreased"
                context += f"- {name}: {direction} delay by {abs(val):.2f} days\n"

    # Current inputs (if any)
    if "current_inputs" in st.session_state:
        inp = st.session_state.current_inputs
        context += f"Current change: {inp.get('story_points', '?')} story points, priority={inp.get('priority', '?')}, affected components={inp.get('affected_components', '?')}, mid‑sprint={inp.get('is_mid_sprint', '?')}, days into sprint={inp.get('days_into_sprint', '?')}, sprint duration={inp.get('sprint_duration', '?')}.\n"
    
    # Historical statistics
    stats = get_historical_stats()
    if stats:
        context += f"Historical data (based on {stats['total_changes']} changes): average delay = {stats['avg_delay']:.1f} days, spillover rate = {stats['spillover_rate']:.1%}, average story points = {stats['avg_story_points']:.1f}.\n"
        if stats.get("recent_avg_delay"):
            context += f"Last 30 days average delay: {stats['recent_avg_delay']:.1f} days.\n"
    
    # Target release
    if "target_release" in st.session_state:
        context += f"Target release for this change: {st.session_state.target_release}.\n"
    
    context += "You are an expert assistant for a software delivery impact estimator. Answer questions concisely and helpfully based on the provided context."
    return context

def ask_groq(question, context):
    client = init_groq()
    """Response from Groq (Llama 3)."""
    prompt = f"Context: {context}\n\nUser question: {question}\n\nPlease answer concisely and helpfully."
    if client is None:
        return "Groq client not available. Please check your API key."
    prompt = f"""
Context:
{context}

User question: {question}

Please answer concisely and helpfully, using the context above if relevant.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # very capable, fast
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again later."
    
# ============================================================================
# DASHBOARD UI (same structure, only prediction call adapted to pass item_type)
# ============================================================================

# ============================================================================
# SIDEBAR - INPUT CONTROL CENTER
# ============================================================================

st.sidebar.title("⚙️ Change Request Control Center")

#st.sidebar.markdown("---")

# ----------------------------------------------------------------
# WORK ITEM HIERARCHY SUGGESTION WITH AI ASSISTANT
# ----------------------------------------------------------------

st.sidebar.subheader("📋 Work Item")

if data is not None:
    work_items_df = data["work_items"]

    # Build hierarchy mapping
    item_dict = {}
    for _, row in work_items_df.iterrows():
        item_dict[row["work_item_id"]] = {
            "name": row["name"],
            "level": row["level"],
            "parent_id": row["parent_id"]
        }

    def get_path(item_id):
        path = []
        cur = item_id
        while cur in item_dict:
            node = item_dict[cur]
            path.append(f"{node['level']}: {node['name']}")
            cur = node["parent_id"]
        return " -> ".join(reversed(path))

    
    #st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Work Item Suggestion")

    change_desc = st.sidebar.text_area("Describe the change", placeholder="e.g., Add biometric authentication for iOS users", height=80, key="ai_desc_2")

    if st.sidebar.button("Suggest Work Item"):
        if change_desc.strip():
            keywords = set(change_desc.lower().split()) - {"a","an","the","and","or","for","of","to","in","on","at","by","with","without","from","via","add","update","remove","fix","implement","create"}
            # Filter work items by name containing any keyword
            matching = work_items_df[work_items_df["name"].str.lower().str.contains('|'.join(keywords), na=False)]
            # Limit to 20 to keep prompt manageable
            top_items = matching.head(20)
            items_with_path = []
            for _, row in top_items.iterrows():
                items_with_path.append(get_path(row["work_item_id"]))

            with st.spinner("Analyzing..."):
                # Build prompt for Groq

                prompt = f"""
                You are an assistant that matches change descriptions to the most appropriate work item hierarchy.
                
        
                Available work items (with full hierarchy path):
                {chr(10).join(items_with_path)}

                Change description: "{change_desc}"
    
                Choose the most suitable Epic, Feature, Business Story, and User Story from the list above.
                If none fits, reply "None".
                Reply exactly in this format:
                EPIC: <epic name>
                FEATURE: <feature name>
                BUSINESS STORY: <business story name>
                USER STORY: <user story name>
                """
                client = init_groq()
                if client:
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=200
                        )
                        answer = response.choices[0].message.content
                        
                        # Parse the response
                        import re
                        epic_match = re.search(r"EPIC:\s*(.+)", answer, re.IGNORECASE)
                        feature_match = re.search(r"FEATURE:\s*(.+)", answer, re.IGNORECASE)
                        business_match = re.search(r"BUSINESS STORY:\s*(.+)", answer, re.IGNORECASE)
                        story_match = re.search(r"USER STORY:\s*(.+)", answer, re.IGNORECASE)
                        
                        if epic_match:
                            st.session_state.suggested_epic = epic_match.group(1).strip()
                        if feature_match:
                            st.session_state.suggested_feature = feature_match.group(1).strip()  # fix variable name
                        if business_match:
                            st.session_state.suggested_bs = business_match.group(1).strip()
                        if story_match:
                            st.session_state.suggested_us = story_match.group(1).strip()
                        
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Suggestion failed: {e}")
                else:
                    st.sidebar.error("Groq client not available.")
        else:
            st.sidebar.warning("Please enter a description.")
else:
    work_items_df = None

st.sidebar.info("💡 **Tip:** AI suggested story points may be off. Use the slider to match your actual change request size.")
# After loading hierarchy options, set default index
if data is not None:
    hierarchy = get_hierarchy_options(data)
    epics = hierarchy.get("epics", [])
    
    if epics:
    # Use suggested epic if available
        default_epic = st.session_state.get("suggested_epic", None)
        if default_epic in epics:
            epic_index = epics.index(default_epic)
        else:
            epic_index = 0
        selected_epic = st.sidebar.selectbox("**Epic**", epics, index=epic_index)

        features = hierarchy.get("features", {}).get(selected_epic, [])
        if features:
            default_feature = st.session_state.get("suggested_feature", None)
            if default_feature in features:
                feature_index = features.index(default_feature)
            else:        feature_index = 0
            selected_feature = st.sidebar.selectbox("**Feature**", features, index=feature_index)

            bs_key = f"{selected_epic}|{selected_feature}"
            business_stories = hierarchy.get("business_stories", {}).get(bs_key, [])
            if business_stories:
                default_bs = st.session_state.get("suggested_business", None)
                if default_bs in business_stories:
                    bs_index = business_stories.index(default_bs)
                else:        bs_index = 0
                selected_bs = st.sidebar.selectbox("**Business Story**", business_stories, index=bs_index)  

                us_key = f"{selected_epic}|{selected_feature}|{selected_bs}"
                user_stories = hierarchy.get("user_stories", {}).get(us_key, [])
                if user_stories:
                    default_us = st.session_state.get("suggested_story", None)
                    if default_us in user_stories:
                        us_index = user_stories.index(default_us)
                    else:        us_index = 0
                    selected_us = st.sidebar.selectbox("**User Story**", user_stories, index=us_index)
                    item_type = "User Story"
                            # Retrieve story points from the selected user story
                    df = data["work_items"]
                    us_row = df[(df["level"] == "User Story") & (df["name"] == selected_us)]
                    if not us_row.empty:
                        #story_points = us_row.iloc[0]["story_points"]
                        #st.sidebar.info(f"📊 **Story Points:** {story_points}")
                        default_story_points = int(us_row.iloc[0]["story_points"])
                        story_points = st.sidebar.slider("📊 Story Points", min_value=1, max_value=21, value=default_story_points, step=1, help= "Story points measure relative effort. 1–3 = small, 5–8 = medium, 13–21 = large.")
                        st.sidebar.write(f"Selected Story Points: {story_points}")
                    else:
                        story_points = st.number_input("Story Points", min_value=1, max_value=21, value=3, step=1)
                else:
                    story_points = st.number_input("Story Points", min_value=1, max_value=21, value=3, step=1)
            else:
                story_points = st.number_input("Story Points", min_value=1, max_value=21, value=3, step=1)
        else:
            story_points = st.number_input("Story Points", min_value=1, max_value=21, value=3, step=1)
    else:
        story_points = st.number_input("Story Points", min_value=1, max_value=21, value=3, step=1)
else:
    story_points = st.number_input("Story Points", min_value=1, max_value=21, value=3, step=1)


    


# ----------------------------------------------------------------
# IMPACT DETAILS
# ----------------------------------------------------------------

st.sidebar.subheader("🚨 Change Impact")

priority = st.sidebar.select_slider("**Priority**", options=["Low", "Medium", "High", "Critical"], value="Medium", help="Critical changes have the highest multiplier (1.2). Low priority changes (0.3) rarely cause spillover.")

st.sidebar.subheader("🔧 Affected Components")
component_options = ["iOSDev", "AndroidDev", "PlatformDev", "ManualQA", "AutomationQA", "PerformanceQA", "Delivery", "BA", "SM", "Architect"]
selected_components = st.sidebar.multiselect(
    "Select affected components",
    options=component_options,
    default=["PlatformDev"],
    help="Each additional component increases coordination effort and risk."
)
affected_components = len(selected_components) if selected_components else 1
st.sidebar.caption(f"Total components affected: {affected_components}")



# ----------------------------------------------------------------
# SPRINT CONTEXT
# ----------------------------------------------------------------
st.sidebar.subheader("📅 Sprint Context")



cadence_df = load_release_cadence()

current_date = st.sidebar.date_input("Current Date", date.today())


sprint_name, sprint_duration, days_into_sprint = get_current_sprint_from_cadence(cadence_df, pd.to_datetime(current_date))


if days_into_sprint is None:
    days_into_sprint = 0   
is_mid_sprint = days_into_sprint > 1 and days_into_sprint < sprint_duration if sprint_duration else False

if sprint_name:
    st.sidebar.write(f"Current Sprint: {sprint_name}")
    st.sidebar.write(f"⏱️ Sprint Duration: {sprint_duration} days")
    st.sidebar.write(f"📊 Days into Sprint: {days_into_sprint}")
    st.sidebar.write(f"**Mid‑Sprint Change:** {'Yes' if is_mid_sprint else 'No'}")
else:
    st.sidebar.warning("No active sprint found for this date.")
    # Fallback to manual inputs
    sprint_duration = st.sidebar.slider("Sprint Duration (days)", 5, 20, 10)
    days_into_sprint = st.sidebar.slider("Days into Sprint", 0, sprint_duration-1, 5)


test_date = pd.to_datetime("2026-05-15")
mask = (cadence_df["week_start"] <= test_date) & (test_date <= cadence_df["week_end"])
rows_in_week = cadence_df.loc[mask]


# Initialize target_release to None
target_release = None
active_releases = get_active_releases(cadence_df, pd.to_datetime(current_date))
if active_releases:
    
    target_release = st.sidebar.selectbox("Select the release for this change request", active_releases)
else:
    st.warning("No active releases found for the selected date. Please adjust the date.")

 
original_cadence = load_release_cadence()
current_date_ts = pd.to_datetime(current_date)
        
total_active_stages = get_total_active_stages(original_cadence, current_date_ts)
default_util_pct = min(90, (total_active_stages-1) * 10)

# Load skill capacity dictionary
skill_capacity = load_skill_capacity()


# ----------------------------------------------------------------
# TEAM & CAPACITY CALCULATION (runs every time, not only on button click)
# ----------------------------------------------------------------
if data is not None and "teams" in data:
    teams_df = data["teams"]
    selected_team = st.sidebar.selectbox("**Team**", teams_df["name"].tolist())
    team_row = teams_df[teams_df["name"] == selected_team]

    if not team_row.empty:
        team_row = team_row.iloc[0]  # now a Series
        weekly_team_capacity = float(team_row["hours_per_week"])
        team_headcount = int(team_row["headcount"])

        # Skills (if present)
        if "skills" in team_row.index:
            skills_val = team_row["skills"]
            if pd.notna(skills_val):
                team_skills = set(skill.strip() for skill in str(skills_val).split(","))
            else:
                team_skills = set()
        else:
            team_skills = set()

        # Extra capacity from selected components (if any)
        extra_weekly = 0
        if selected_components:
            for comp in selected_components:
                total_headcount = skill_capacity.get(comp, 0) // 40   # hours -> headcount
                if comp in team_skills:
                    extra_headcount = max(0, total_headcount - 1)
                else:
                    extra_headcount = total_headcount
                extra_weekly += extra_headcount * 40

        total_weekly_capacity = weekly_team_capacity + extra_weekly
        team_capacity = total_weekly_capacity * (sprint_duration / 5)

        # Compute capacity features for ML
        sprint_progress = days_into_sprint / sprint_duration if sprint_duration > 0 else 0
        #base_remaining_capacity_hours = team_capacity * (1 - sprint_progress)
        utilisation_factor = default_util_pct / 100.0
        

        # Used/remaining for display
        used_capacity = team_capacity * sprint_progress
        remaining_capacity = team_capacity - used_capacity
        availablecapacity = remaining_capacity * (1 - default_util_pct/100)
        base_remaining_capacity_hours = availablecapacity
        available_capacity_ratio = base_remaining_capacity_hours / team_capacity if team_capacity > 0 else 0.0


    else:
        # Fallback values
        st.warning(f"Team '{selected_team}' not found. Using defaults.")
        team_headcount = 5
        weekly_team_capacity = 200
        team_capacity = 200 * (sprint_duration / 5)
        sprint_progress = days_into_sprint / sprint_duration if sprint_duration > 0 else 0
        base_remaining_capacity_hours = team_capacity * (1 - sprint_progress)
        utilisation_factor = default_util_pct / 100.0
        available_capacity_ratio = base_remaining_capacity_hours / team_capacity if team_capacity > 0 else 0.0
        used_capacity = team_capacity * sprint_progress
        remaining_capacity = team_capacity - used_capacity
        availablecapacity = remaining_capacity * (1 - default_util_pct/100)
else:
    # No data loaded – fallback
    team_headcount = 5
    team_capacity = 400
    used_capacity = 0
    remaining_capacity = team_capacity
    availablecapacity = team_capacity
    base_remaining_capacity_hours = team_capacity
    utilisation_factor = 0.5
    available_capacity_ratio = 0.5
        # ================================================================

        # CURRENT ACTIVE WORK

        # ================================================================


st.sidebar.subheader("🔄 Current Active Work")

current_work_story_points = st.sidebar.slider("**Current Work Story Points**", min_value=0, max_value=13, value=3)

current_work_priority = st.sidebar.select_slider("**Current Work Priority**", options=["Low", "Medium", "High", "Critical"], value="Medium")
        

# ----------------------------------------------------------------
# ESTIMATE BUTTON
# ----------------------------------------------------------------

st.sidebar.markdown("---")

estimate_btn = st.sidebar.button("🚀 Run Impact Analysis", type="primary", use_container_width=True)


# ============================================================================
# HEADER
# ============================================================================

st.title("🏦 Fiserv Delivery Intelligence Suite")

st.subheader("""Impact, Planning & Analytics for Agile Project Delivery.""")
st.info(""" AI-powered delivery impact estimation for client scope changes. Use inputs on the left to fill in the change request details (story points, priority, affected components) and the sprint context.The model will predict **spillover risk** and **expected delay**.""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Impact Estimator", "🔍 Result Explanations", "🏗️ Planning View", "📈 Data Overview" , "🔮 What‑If Simulation"
])

# ===== TAB 1: IMPACT ESTIMATOR =====
with tab1:
# ============================================================================
# GLOBAL KPI STRIP
# ============================================================================

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric("🔄 Active Stages", f"{total_active_stages}")


    with k2:
        st.metric("💪 Team Capacity", f"{team_capacity} hrs/sprint", help= "Total sprint capacity (hours) × (1 − sprint % complete)")
    with k3:
        #st.metric("👥 Remaining Capacity", f"{remaining_capacity:.0f} hrs or " f"{remaining_capacity/team_capacity:.0%} ")
        st.metric("👥 Available Capacity", f"{availablecapacity:.0f} hrs or " f"{availablecapacity/team_capacity:.0%} ")
        



    st.markdown("<br>", unsafe_allow_html=True)


    # ============================================================================
# MOCK PREDICTION ENGINE
# ============================================================================
    if estimate_btn:
        # Original predictions
        result = predict_impact_ml(story_points, days_into_sprint, sprint_duration,
                                    priority, affected_components, team_capacity,
                                    is_mid_sprint, team_headcount=team_headcount,
                                    base_remaining_capacity_hours=base_remaining_capacity_hours,
                                    utilisation_factor=utilisation_factor,
                                    available_capacity_ratio=available_capacity_ratio, item_type=item_type)
        
        current_work_result = predict_impact_ml(current_work_story_points, days_into_sprint, sprint_duration, current_work_priority,
                                    affected_components, team_capacity, is_mid_sprint, team_headcount=team_headcount,
                                    base_remaining_capacity_hours=base_remaining_capacity_hours,
                                    utilisation_factor=utilisation_factor,
                                    available_capacity_ratio=available_capacity_ratio, item_type="User Story")

        #completion_pct = (days_into_sprint / sprint_duration if sprint_duration > 0 else 0)

        # Rollover correction for end‑of‑sprint scenarios

        
                
                #st.rerun()

        # If we reach here, no rollover needed → store original results
        #completion_pct = min(completion_pct, 1.0)
        completion_pct = min(days_into_sprint / sprint_duration, 1.0)
        

        remaining_current_work = (current_work_story_points* (1 - completion_pct))

        incoming_priority_value = (priority_mult.get(priority, 1.0))
        current_priority_value = (priority_mult.get(current_work_priority, 1.0))

        if completion_pct < 0.8:
            reprioritisation_triggered = (incoming_priority_value > current_priority_value)
        else:
            reprioritisation_triggered = False

        additional_delay = (result["delay_days"]if reprioritisation_triggered else 0)
        
        total_current_work_delay = (current_work_result["delay_days"]+ additional_delay)

            # Store in session state
        st.session_state.result = result
        st.session_state.current_work_result = current_work_result

        st.session_state.reprioritisation_triggered = reprioritisation_triggered
        st.session_state.additional_delay = additional_delay
        st.session_state.total_current_work_delay = total_current_work_delay
        st.session_state.remaining_current_work = remaining_current_work
        st.session_state.total_active_stages = total_active_stages   
        st.session_state.active_releases = active_releases
        st.session_state.target_release = target_release
        st.session_state.current_inputs = {"story_points": story_points, "priority": priority, "affected_components": affected_components,
                                            "is_mid_sprint": is_mid_sprint, "days_into_sprint": days_into_sprint, "sprint_duration": sprint_duration,
                                            "current_work_story_points": current_work_story_points, "current_work_priority": current_work_priority}

    # ------------------------------------------------------------
    # IMPACT ASSESSMENT KPI ROW
    # ------------------------------------------------------------
    if "result" in st.session_state:
        result = st.session_state.result
        current_work_result = st.session_state.current_work_result
        reprioritisation_triggered = st.session_state.reprioritisation_triggered
        additional_delay = st.session_state.additional_delay
        total_current_work_delay = st.session_state.total_current_work_delay
        remaining_current_work = st.session_state.remaining_current_work

        days_left = (sprint_duration - days_into_sprint) + sprint_duration  # time left in current sprint + next sprint (assuming spillover goes to next sprint)
        if result["delay_days"] > days_left and days_left >= 0:
            st.info(f"The model predicts a delay of {result['delay_days']:.1f} days, which exceeds the remaining time in this sprint ({sprint_duration - days_into_sprint} days) hence re-evaluating for next sprint.")
                # Reset parameters for next sprint
            next_sprint_days_into = 0
            next_sprint_duration = 10
            next_base_remaining = team_capacity   # available capacity for next sprint
            next_available_ratio = next_base_remaining / team_capacity if team_capacity > 0 else 0.0
            # Call ML again with reset parameters (no recursion guard needed because we are in a button)
            rollover_result = predict_impact_ml(story_points, next_sprint_days_into, next_sprint_duration,
                                    priority, affected_components, team_capacity,
                                    is_mid_sprint, team_headcount=team_headcount,
                                    base_remaining_capacity_hours=next_base_remaining,  # reset to full capacity for next sprint
                                    utilisation_factor=utilisation_factor,
                                    available_capacity_ratio=next_available_ratio, item_type=item_type)
            
                # Store all session state variables with the rollover result and derived metrics
            
            st.session_state.rollover_result = rollover_result

            st.session_state.rollover_inputs = {
                    "story_points": story_points,
                    "priority": priority,
                    "affected_components": affected_components,
                    "is_mid_sprint": is_mid_sprint,
                    "days_into_sprint": next_sprint_days_into,  # now 0
                    "sprint_duration": next_sprint_duration,
                    "current_work_story_points": current_work_story_points,
                    "current_work_priority": current_work_priority
                }
                

            if result.get("used_ml", False):
                st.info("🤖 Prediction based on trained LightGBM models.")
            else:
                st.warning("⚠️ Using fallback heuristic rules (ML models not available).")

            st.subheader("📊 Next Sprint Impact Assessment")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("📊 Spillover Risk Next Sprint", f"{rollover_result['spillover_prob']:.0%}")
            with c2:
                st.metric("⏱️ Expected Delay Next Sprint", f"{rollover_result['delay_days']:.1f} days")
            with c3:
                st.metric("📅 Next Sprint Fit", "✅ Yes" if rollover_result['sprint_fit'] else "❌ No")
            with c4:
                st.metric("⚠️ Next Sprint Risk Level", rollover_result['risk'])

            st.markdown("<br>", unsafe_allow_html=True)

        else:
            if result.get("used_ml", False):
                st.info("🤖 Prediction based on trained LightGBM models.")
            else:
                st.warning("⚠️ Using fallback heuristic rules (ML models not available).")

            st.subheader("📊 Impact Assessment")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("📊 Spillover Risk", f"{result['spillover_prob']:.0%}")
            with c2:
                st.metric("⏱️ Expected Delay", f"{result['delay_days']:.1f} days")
            with c3:
                st.metric("📅 Sprint Fit", "✅ Yes" if result['sprint_fit'] else "❌ No")
            with c4:
                st.metric("⚠️ Risk Level", result['risk'])

            st.markdown("<br>", unsafe_allow_html=True)


        # ------------------------------------------------------------
    # TWO‑COLUMN CHARTS
    # ------------------------------------------------------------
        left_col, right_col = st.columns([1, 1.2])

        with left_col:
        # Risk gauge
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("🎯 Spillover Risk Gauge", help= "The gauge shows the probability that the change will **spill over** into the next sprint. Above 75% = high risk.")
                

                fig1 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['spillover_prob'] * 100,
                    title={'text': "Current Sprint Risk %"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'thickness': 0.3},
                            'steps': [
                            {'range': [0, 35], 'color': "#00c853"},
                            {'range': [35, 75], 'color': "#fbc02d"},
                            {'range': [75, 100], 'color': "#ff5252"},]
                    }
                ))
                fig1.update_layout(height=220, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig1, use_container_width=True)

                if result["delay_days"] > days_left and days_left >= 0:
                    fig2 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=rollover_result['spillover_prob'] * 100,
                        title={'text': "Next Sprint Risk %"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'thickness': 0.3},
                            'steps': [
                            {'range': [0, 35], 'color': "#00c853"},
                            {'range': [35, 75], 'color': "#fbc02d"},
                            {'range': [75, 100], 'color': "#ff5252"},]
                        }
                    ))
                    fig2.update_layout(height=220, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig2, use_container_width=True)

                else:
                    st.info("The predicted delay can be adjusted in next sprint, so no next sprint recalculation is needed.")
                st.markdown('</div>', unsafe_allow_html=True)

        with right_col:
        # Capacity gauge
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("🌐 Release Cadence Timeline")
                current_date_ts = pd.to_datetime(current_date)

                finish_date = current_date_ts + pd.Timedelta(weeks=21)

                full_timeline_df = build_full_timeline(cadence_df=cadence_df, start_date=current_date_ts, finish_date=finish_date, current_date=current_date_ts, delay_days=result['delay_days'])
                
                if not full_timeline_df.empty:
                    fig = px.timeline(
                        full_timeline_df,
                        x_start= "Start",
                        x_end= "Finish",
                        y="Release",
                        color="Stage",
                        title="With Delay Impact",
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No timeline data available.")
                st.markdown('</div>', unsafe_allow_html=True)
  

       # ================================================================

        # REPRIORITISATION METRICS

        # ================================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🔄 Current Work Impact")
        if reprioritisation_triggered:
            reprior_col1, reprior_col2 = st.columns(2)
            with reprior_col1:
                st.metric("Remaining Current Work",f"{remaining_current_work:.1f} Story Points")
            with reprior_col2:
                st.metric("Reprioritisation", "✅ Triggered" )   
            reprior_col3, reprior_col4 = st.columns(2)
            with reprior_col3:
                st.metric("Interruption Delay", f"{additional_delay:.1f} days")
            with reprior_col4:
                st.metric("Current Work Total Delay", f"{total_current_work_delay:.1f} days")
        else:
            st.info("No reprioritisation is needed. Current work will not be delayed beyond the original estimate.")
                    
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("💡 Recommendation")
        if result['recommendation'] == "Accept in current sprint":
            st.success(f"✅ **{result['recommendation']}**")
            st.write(f"The {story_points}-point work item can be accommodated with minimal impact.")
        elif result['recommendation'] == "Accept with monitoring":
            st.info(f"📋 **{result['recommendation']}**")
            st.write(f"Estimated {result['delay_days']:.1f}-day delay. Monitor progress closely.")
        else:
            st.warning(f"📅 **{result['recommendation']}**")
            st.write(f"Estimated {result['delay_days']:.1f}-day delay. Better to plan in upcoming sprint.")

# ============================================================================
# AI COPILOT
# ============================================================================

    with st.expander("🤖 AI AI Assistant"):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

# Scrollable message container
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.markdown('</div>', unsafe_allow_html=True)


    # Single chat input
        if prompt := st.chat_input("Ask about release impact, sprint risk, or recommendations..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            context = get_dashboard_context()
            with st.spinner("Thinking..."):
                answer = ask_groq(prompt, context)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
            #st.rerun()
    


# ============================================================================
# TAB 2: Model Result Explanations (SHAP)
# ============================================================================

with tab2:
    st.subheader("🔍 Model Result Explanations (SHAP)")

    st.info("""After running a prediction, SHAP explains *why* the model made that decision.  
    - **Global feature importance** shows the most influential factors across all predictions.  
    - **Waterfall plots** break down the current change request – each bar shows how a feature pushed the prediction up or down.
    """)
    
    # ---- Classifier explainer check ----
    if shap_explainer_clf is None:
        st.info("SHAP explainer for spillover not available.")
    else:
        # Global Feature Importance (precomputed image)
        st.markdown("#### Global Feature Importance: Spillover Risk")
        try:
            st.image("models/shap_global_bar_classifier_spillover_lgb.png",
                     caption="Top Features Driving Spillover Predictions",
                     use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load precomputed SHAP chart: {e}")

        st.divider()
        
        # ---- Local explanations for current prediction ----
        if "last_X" in st.session_state and st.session_state.last_X is not None:
            X_input = st.session_state.last_X
            st.markdown("#### Explanation for the Current Change Request")
            
            with st.spinner("Computing SHAP values..."):
                # ---------- Classifier (Spillover) ----------
                shap_values_clf = shap_explainer_clf.shap_values(X_input)
                if isinstance(shap_values_clf, list):
                    shap_values_class = shap_values_clf[1]       # class 1 (spillover)
                else:
                    shap_values_class = shap_values_clf
                expected_value_clf = shap_explainer_clf.expected_value
                if isinstance(expected_value_clf, list):
                    expected_value_clf = expected_value_clf[1]
                
                # ---------- Regressor (Delay Days) ----------
                if shap_explainer_reg is not None:
                    shap_values_reg = shap_explainer_reg.shap_values(X_input)   # shape (1, n_features)
                    st.session_state.shap_values_reg = shap_values_reg[0]       # 1D array
                    #st.session_state.expected_value_reg = shap_explainer_reg.expected_value
                    st.session_state.expected_value_reg = float(np.squeeze(shap_explainer_reg.expected_value))
            
            # ---------- Waterfall for Classifier ----------
            st.markdown("##### Waterfall Plot (Spillover Risk)")
            fig_wf_clf = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(values=shap_values_class[0],
                                 base_values=expected_value_clf,
                                 data=X_input.iloc[0].values,
                                 feature_names=X_input.columns.tolist()),
                show=False
            )
            st.pyplot(fig_wf_clf)
            plt.close(fig_wf_clf)
            
            # Store top features for chatbot (classifier)
            contributions_clf = shap_values_class[0]
            abs_contrib = np.abs(contributions_clf)
            top_idx = np.argsort(abs_contrib)[-5:]
            top_features = [(X_input.columns[i], contributions_clf[i]) for i in top_idx]
            st.session_state.top_shap_features = top_features

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("#### Global Feature Importance: Expected Delay(days)")
            try:
                st.image("models/shap_global_bar_regressor_delay_days.png",
                     caption="Top Features Driving Delay Predictions",
                     use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load precomputed SHAP chart: {e}")

            st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" />""",unsafe_allow_html=True)

            # ---------- Waterfall for Regressor (if available) ----------
            if shap_explainer_reg is not None and "shap_values_reg" in st.session_state:
                st.markdown("#### Explanation for Expected Delay (days)")
                st.markdown("##### Waterfall Plot (Delay Days)")
                fig_wf_reg = plt.figure(figsize=(10, 6))
                #base_value = float(np.squeeze(st.session_state.expected_value_reg))
                shap.waterfall_plot(
                    shap.Explanation(values=st.session_state.shap_values_reg,
                                     base_values=st.session_state.expected_value_reg,
                                     data=X_input.iloc[0].values,
                                     feature_names=X_input.columns.tolist()),
                    show=False
                )
                st.pyplot(fig_wf_reg)
                plt.close(fig_wf_reg)
            else:
                st.info("SHAP explainer for delay days not available.")
        else:
            st.info("Run an impact estimate first to see explanations for that change.")

# ============================================================================
# TAB 3: Unified Planning View
# ============================================================================
with tab3:
    st.subheader("🏗️ Unified Planning View – Release Timeline & Work Breakdown")
    st.info("""The Gantt chart shows all parallel releases (Rel.X+2 … X+10) with their stages over time.  
    Orange dashed lines mark sprint boundaries (with sprint numbers).  
    Below, expand the **Work Item Hierarchy** to see the breakdown of Epics → Features → User Stories → Tasks.
    """)

    # ------------------------------------------------------------
    # Load and prepare release cadence data for Gantt
    # ------------------------------------------------------------
    @st.cache_data
    def load_cadence_for_gantt(csv_path="data/release_cadence.csv"):
        df = pd.read_csv(csv_path)
        # Convert date columns
        df["week_start"] = pd.to_datetime(df["week_start"], format="%d/%m/%y", errors="coerce")
        df["week_end"]   = pd.to_datetime(df["week_end"],   format="%d/%m/%y", errors="coerce")
        df["sprint_start"] = pd.to_datetime(df["sprint_start"], format="%d/%m/%y", errors="coerce")
        df["sprint_end"]   = pd.to_datetime(df["sprint_end"],   format="%d/%m/%y", errors="coerce")
        df = df.dropna(subset=["week_start", "week_end"])
        
        # Extract unique sprint boundaries (start and end)
        sprint_start_map = {}   # date -> sprint_number
        sprint_boundaries = set()
        for _, row in df.iterrows():
            if pd.notna(row["sprint_start"]) and pd.notna(row["sprint_number"]):
                sprint_start_map[row["sprint_start"]] = row["sprint_number"]
                sprint_boundaries.add(row["sprint_start"])
            #if pd.notna(row["sprint_end"]):
                #sprint_boundaries.add(row["sprint_end"])
        sprint_boundaries = sorted(sprint_boundaries)
        
        # Identify release columns
        release_cols = [col for col in df.columns if col.startswith("Rel.")]
        # Melt to long format
        long_rows = []
        for _, row in df.iterrows():
            for rel in release_cols:
                stage_str = row[rel]
                if pd.isna(stage_str) or stage_str == "":
                    continue
                long_rows.append({
                    "Release": rel,
                    "Stage": stage_str,
                    "Start": row["week_start"],
                    "End": row["week_end"]
                })
        gantt_df = pd.DataFrame(long_rows)
        gantt_df = gantt_df.sort_values("Start")
        return gantt_df, sprint_boundaries, sprint_start_map

    cadence_gantt, sprint_boundaries, sprint_start_map  = load_cadence_for_gantt()

    if not cadence_gantt.empty:
        # Date range filter
        min_date = cadence_gantt["Start"].min().date()
        max_date = cadence_gantt["End"].max().date()
        date_range = st.slider(
            "Zoom timeline (select date range)",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        start_filter, end_filter = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = cadence_gantt[(cadence_gantt["Start"] <= end_filter) & (cadence_gantt["End"] >= start_filter)]

        # Build Gantt chart
        fig = px.timeline(
            filtered_df,
            x_start="Start",
            x_end="End",
            y="Release",
            color="Stage",
            title="Release Cadence with Parallel Stages & Sprint Boundaries",
            labels={"Release": "Release", "Stage": "Stage"},
            height=600
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Release",
            legend_title="Stage",
            hovermode="closest",
            plot_bgcolor="white"
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="gray")

        # Add vertical lines for sprint boundaries (only those within filtered date range)
        for sprint_date in sprint_boundaries:
            if start_filter <= sprint_date <= end_filter:
                fig.add_vline(
                    x=sprint_date.to_pydatetime(),
                    line_dash="dash",
                    line_color="orange",
                    line_width=1
                    #annotation_text=f"Sprint {sprint_no}",
                    #annotation_position="top"
                )
                # Add sprint number label (only for sprint start dates)
                if sprint_date in sprint_start_map:
                    sprint_num = sprint_start_map[sprint_date]
                    fig.add_annotation(
                        x=sprint_date.to_pydatetime(),
                        y=1.06,          # near top of the plotting area
                        xref="x",
                        yref="paper",
                        text=sprint_num,
                        showarrow=False,
                        font=dict(size=9, color="black"),
                        textangle=-90,   # rotate to save horizontal space
                        yshift=5
                    )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each coloured bar represents release stages. Orange dashed lines mark sprint boundaries. Hover over bars for exact dates.")
    else:
        st.warning("No release cadence data available. Please check the CSV file.")

    # ------------------------------------------------------------
    # Work Item Hierarchy (collapsible)
    # ------------------------------------------------------------
    st.divider()
    #st.markdown("---")
    with st.expander("📂 Work Item Hierarchy (click to expand)", expanded=False):
        if data is not None and "work_items" in data:
            df = data["work_items"]
            st.markdown(f"📌 {len(df[df['level'] == 'Epic'])} Epics · "
                       f"🔹 {len(df[df['level'] == 'Feature'])} Features · "
                       f"📘 {len(df[df['level'] == 'Business Story'])} Business Stories · "
                       f"📄 {len(df[df['level'] == 'User Story'])} User Stories · "
                       f"⚙️ {len(df[df['level'] == 'Task'])} Tasks")

            epics = df[df["level"] == "Epic"]
            for _, epic in epics.iterrows():
                with st.expander(f"📌 **Epic:** {epic['name']} (SP {epic['story_points']})"):
                    features = df[(df["level"] == "Feature") & (df["parent_id"] == epic["work_item_id"])]
                    for _, feat in features.iterrows():
                        st.markdown(f"    🔹 **Feature:** {feat['name']} (SP {feat['story_points']})")
                        bss = df[(df["level"] == "Business Story") & (df["parent_id"] == feat["work_item_id"])]
                        for _, bs in bss.iterrows():
                            st.markdown(f"        📘 **Business Story:** {bs['name']} (SP {bs['story_points']})")
                            uss = df[(df["level"] == "User Story") & (df["parent_id"] == bs["work_item_id"])]
                            for _, us in uss.iterrows():
                                st.markdown(f"            📄 **User Story:** {us['name']} (SP {us['story_points']})")
                                tasks = df[(df["level"] == "Task") & (df["parent_id"] == us["work_item_id"])]
                                for i, task in enumerate(tasks.iterrows()):
                                    if i >= 3:
                                        st.markdown("                ⚙️ ... and more tasks")
                                        break
                                    task = task[1]
                                    st.markdown(f"                ⚙️ Task: {task['name']} (SP {task['story_points']})")
        else:
            st.info("Work item data not loaded.")

# ============================================================================
# TAB 4: DATA OVERVIEW
# ============================================================================
with tab4:
    st.subheader("📈 Team & Resource Intelligence")

    if data is not None:
        teams_df = data["teams"].copy()
        # Ensure numeric columns
        numeric_cols = ["headcount", "hours_per_week"]
        for col in numeric_cols:
            teams_df[col] = pd.to_numeric(teams_df[col], errors="coerce")
        
        # The skill matrix columns (all except the core team info)
        # Identify columns that are skill booleans (0/1)
        skill_cols = [col for col in teams_df.columns if col not in 
                      ["team_id", "name", "location", "headcount", "hours_per_week", "type", "Total"]]
        # Keep only those that contain 0/1 values (assume all are skills)
        
        # Overall KPIs
        
        #st.markdown("---")
        total_hc = teams_df["headcount"].sum()
        total_cap = teams_df["hours_per_week"].sum()
        nz_hc = teams_df[teams_df["location"] == "NZ"]["headcount"].sum()
        os_hc = teams_df[teams_df["location"] == "Overseas"]["headcount"].sum()
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("👥 Total Headcount", total_hc)
        col_b.metric("⏱️ Total Weekly Capacity", f"{total_cap} hrs")
        col_c.metric("🇳🇿 NZ Headcount", nz_hc)
        col_d.metric("🌏 Overseas Headcount", os_hc)

        st.divider()
        # ------------------------------------------------------------
        # Skill Coverage Heatmap
        # ------------------------------------------------------------
        st.markdown("#### 🔥 Team Skill Coverage Heatmap")
        # Prepare pivot: teams as rows, skills as columns
        heatmap_data = teams_df.set_index("name")[skill_cols]
        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Skill", y="Team", color="Has Skill"),
            title="Which teams have which skills? (1 = yes, 0 = no)",
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Dark blue = team has that skill. White = no coverage. Gaps in critical skills (e.g., PerformanceQA) may cause bottlenecks.")

        st.divider()
        
        # ------------------------------------------------------------
        # Headcount & Capacity Visuals
        # ------------------------------------------------------------
        col1, col2 = st.columns([0.5, 0.5])
        resources_df = data["resource_types"].copy()
        resources_df["count"] = pd.to_numeric(resources_df["count"], errors="coerce")
        st.markdown("#### 👥 Teamwise and Rolewise Resource count")

        with col1:
            
            fig_hc = px.bar(
                teams_df,
                x="name",
                y="headcount",
                color="location",
                title="Headcount per Team",
                text="headcount"
            )
            fig_hc.update_traces(textposition="outside")
            fig_hc.update_layout(height=400)
            st.plotly_chart(fig_hc, use_container_width=True)

        with col2:
            
            # Horizontal bar chart for better readability
            fig_res = px.bar(
                resources_df,
                y="role",
                x="count",
                color="skill",
                title="Resource Count by Role",
                labels={"count": "Number of People", "role": ""},
                orientation="h",
                text="count"
            )
            fig_res.update_traces(textposition="outside")
            fig_res.update_layout(height=400)
            st.plotly_chart(fig_res, use_container_width=True)


        st.divider()
        
        # ------------------------------------------------------------
        # Utilisation Heatmap (Robust version)
        # ------------------------------------------------------------
        if "change_requests" in data and data["change_requests"] is not None:
            cr_df = data["change_requests"].copy()
            required = ["team_id", "delay_days_caused", "sprint_name"]
            if all(col in cr_df.columns for col in required):
                # Clean data
                cr_df["delay_days_caused"] = pd.to_numeric(cr_df["delay_days_caused"], errors="coerce")
                cr_df["sprint_name"] = cr_df["sprint_name"].astype(str).str.strip()
                #cr_df["team_id"] = pd.to_numeric(cr_df["team_id"], errors="coerce")
                cr_df["team_id"] = (cr_df["team_id"].astype(str).str.strip())
                # Drop rows with missing numeric values
                cr_df = cr_df.dropna(subset=["delay_days_caused"])
                
                if not cr_df.empty:
                    # Aggregate
                    util_agg = (cr_df.groupby(["team_id", "sprint_name"])["delay_days_caused"].mean().reset_index())
                    util_agg = util_agg.rename(columns={"team_id": "team_name"})
                    # Map team_id to team name
                    #team_name_map = teams_df.set_index("team_id")["name"].to_dict()
                    #util_agg["team_name"] = util_agg["team_id"].map(team_name_map)
                    # Drop any rows where team_name is missing (shouldn't happen)
                    #util_agg = util_agg.dropna(subset=["team_name"])
                    
                    if not util_agg.empty:
                        # Pivot
                        pivot_util = (cr_df.groupby(["team_id", "sprint_name"])["delay_days_caused"].mean().reset_index().pivot_table(index="team_id",columns="sprint_name",values="delay_days_caused"))
                        # Optional: sort sprint columns chronologically
                        try:
                            # Extract sprint number (e.g., "Sprint 20" -> 20)
                            def sprint_num(x):
                                parts = x.split()
                                return int(parts[-1]) if parts and parts[-1].isdigit() else 0
                            pivot_util = pivot_util.reindex(
                                sorted(pivot_util.columns, key=sprint_num),
                                axis=1
                            )
                        except:
                            pass
                        
                        st.markdown("#### 📊 Team Delay Heatmap (by Sprint)")
                        fig_util = px.imshow(
                            pivot_util,
                            labels=dict(x="Sprint", y="Team", color="Avg days delay caused"),
                            title="Average Team Delay Days per Sprint (darker = more delay)",
                            color_continuous_scale="plasma_r",
                            aspect="auto",
                            zmin=pivot_util.min().min(),
                            zmax=pivot_util.max().max()
                        )
                        fig_util.update_layout(height=400)
                        st.plotly_chart(fig_util, use_container_width=True)
                    else:
                        st.info("No valid team mapping found for delay_days data.")
                else:
                    st.info("No rows with valid numeric delay_days or team_id.")
            else:
                st.info(f"Missing required columns: {[c for c in required if c not in cr_df.columns]}")
        else:
            st.info("Change requests data not loaded.")
 
        

        
    else:
        st.info("Data not loaded. Please run the data generator first.")

with tab5:

    st.subheader("🔮 What‑If Simulation")
    st.markdown("Adjust the parameters below to see how they affect **spillover risk** and **expected delay**. The simulation runs instantly.")

    # Use columns to organise controls
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Change Request Attributes")
        sim_story_points = st.slider("Story Points", 1, 21, 5, key="sim_sp")
        sim_priority = st.select_slider("Priority", options=["Low", "Medium", "High", "Critical"], value="Medium", key="sim_prio")
        sim_affected_components = st.multiselect(
            "Affected Components",
            options=["iOSDev", "AndroidDev", "PlatformDev", "ManualQA", "AutomationQA", "PerformanceQA", "Delivery", "BA", "SM", "Architect"],
            default=["PlatformDev"],
            key="sim_comp"
        )
        sim_affected_count = len(sim_affected_components) if sim_affected_components else 1
        st.caption(f"Total components affected: {sim_affected_count}")

        sim_is_mid_sprint = st.checkbox("Is this a mid‑sprint change?", value=True, key="sim_mid")
        sim_item_type = st.selectbox("Work Item Level", ["User Story", "Task", "Business Story", "Feature", "Epic"], index=0, key="sim_type")

    with col_right:
        st.markdown("#### Sprint & Team Context")
        sim_sprint_duration = st.slider("Sprint Duration (days)", 5, 20, 10, key="sim_dur")
        sim_days_into_sprint = st.slider("Days into Sprint", 0, sim_sprint_duration - 1, 5, key="sim_days")
        sim_team_capacity = st.number_input("Team Capacity (hours per sprint)", min_value=100, max_value=800, value=400, step=50, key="sim_cap")
        sim_team_headcount = st.number_input("Team Headcount", min_value=1, max_value=20, value=5, step=1, key="sim_hc")
        sim_utilisation_factor = st.slider("Team Utilisation Factor", 0.3, 0.95, 0.7, 0.05, key="sim_util")
        sim_available_capacity_ratio = st.slider("Available Capacity Ratio", 0.0, 1.0, 0.4, 0.05, key="sim_avail")

    # Derive base_remaining_capacity_hours from capacity and utilisation
    sim_base_remaining = sim_team_capacity * (1 - sim_days_into_sprint / sim_sprint_duration) * (1 - sim_utilisation_factor)

    # Run prediction
    sim_result = predict_impact_ml(
        story_points=sim_story_points,
        days_into_sprint=sim_days_into_sprint,
        sprint_duration=sim_sprint_duration,
        priority=sim_priority,
        affected_components=sim_affected_count,
        team_capacity=sim_team_capacity,
        is_mid_sprint=sim_is_mid_sprint,
        team_headcount=sim_team_headcount,
        base_remaining_capacity_hours=sim_base_remaining,
        utilisation_factor=sim_utilisation_factor,
        available_capacity_ratio=sim_available_capacity_ratio,
        item_type=sim_item_type
    )

    # Display results in nice cards
    st.markdown("---")
    st.subheader("📊 Simulation Results")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("📊 Spillover Risk", f"{sim_result['spillover_prob']:.0%}")
    with metric_col2:
        st.metric("⏱️ Expected Delay", f"{sim_result['delay_days']:.1f} days")
    with metric_col3:
        st.metric("📅 Sprint Fit", "✅ Yes" if sim_result['sprint_fit'] else "❌ No")

    st.info(f"**Recommendation:** {sim_result['recommendation']}  |  **Risk Level:** {sim_result['risk']}")

    # Optional: Compare with current baseline from Impact Estimator tab
    st.markdown("---")
    st.subheader("📈 Compare with Baseline")

    if "result" in st.session_state and st.session_state.result:
        baseline = st.session_state.result
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Current Baseline** (from Impact Estimator)")
            st.metric("Spillover Risk", f"{baseline['spillover_prob']:.0%}")
            st.metric("Delay Days", f"{baseline['delay_days']:.1f}")
        with col_b:
            st.markdown("**What‑If Scenario**")
            st.metric("Spillover Risk", f"{sim_result['spillover_prob']:.0%}",
                      delta=f"{sim_result['spillover_prob'] - baseline['spillover_prob']:.1%}")
            st.metric("Delay Days", f"{sim_result['delay_days']:.1f}",
                      delta=f"{sim_result['delay_days'] - baseline['delay_days']:.1f}")
    else:
        st.info("Run an estimate in the **Impact Estimator** tab first to enable comparison.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("🏦 **Fiserv Internal Tool** | Powered by LightGBM & Random Forest models | Spillover F1: 0.79 | Delay MAE: 1.36 days")
