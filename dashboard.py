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
        models["regression"] = joblib.load(os.path.join(model_dir, "regressor_delay_days_lgb.pkl"))
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

def build_full_timeline(cadence_df, start_date, finish_date,delay_days=0):
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
            if delay_days > 0 and start > pd.Timestamp.today():
                end = end + pd.Timedelta(days=delay_days)
                start = start + pd.Timedelta(days=delay_days)
            rows.append({
                "Release": rel,
                "Stage": stage.strip(),
                "Start": start,
                "Finish": end
            })
    return pd.DataFrame(rows)





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
    days_into_sprint = (current_date - sprint_start).days
    return sprint_name, sprint_duration, days_into_sprint

# ============================================================================
# FEATURE ENGINEERING – EXACT MATCH TO TRAINING (13 features)
# ============================================================================
def engineer_ml_features(story_points, days_into_sprint, sprint_duration,
                         priority, affected_components, is_mid_sprint,
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
    complexity_score = story_points * affected_components
    predicted_risk_proxy = story_points * 0.5 + affected_components * 0.3 + sprint_progress * 10
    # sprint_task_load: we don't have actual sprint plan, use a heuristic based on story points
    # (the model was trained with values from generated data, but a reasonable estimate works)
    sprint_task_load = min(20, int(story_points * 1.5) + affected_components)

    # Derived flags
    has_estimate = 1
    has_story_points = 1 if story_points > 0 else 0
    story_points_log = np.log1p(story_points)

    # Build feature dictionary in the exact order expected by the model
    # (order does not matter for LightGBM as long as column names match)
    features = {
        "story_points": story_points,
        "days_into_sprint": days_into_sprint,
        "sprint_duration": sprint_duration,
        "sprint_progress_at_creation": sprint_progress,
        "remaining_sprint_pct": remaining_sprint_pct,
        "complexity_score": complexity_score,
        "predicted_risk_proxy": predicted_risk_proxy,
        "sprint_task_load": sprint_task_load,
        "has_estimate": has_estimate,
        "has_story_points": has_story_points,
        "story_points_log": story_points_log,
        "priority_encoded": priority_encoded,
        "item_type_encoded": item_type_encoded,
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
        component_mult = 1 + (affected_components - 1) * 0.15
        component_mult = min(component_mult, 1.8)
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
                                     item_type=item_type)
            spillover_prob = ml_models["spillover"].predict_proba(X)[0][1]
            delay_days = ml_models["regression"].predict(X)[0]
            used_ml = True
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
    if spillover_prob < 0.33:
        recommendation = "Accept in current sprint"
        risk = "Low"
    elif spillover_prob < 0.66:
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
        "sprint_fit": spillover_prob < 0.5,
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
        "epics": df[df["level"] == "Epic"]["name"].tolist(),
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
    if "result" in st.session_state:
        r = st.session_state.result
        context += f"Latest prediction: Spillover risk = {r['spillover_prob']:.0%}, Delay = {r['delay_days']:.1f} days, Recommendation = {r['recommendation']}.\n"
    # Active releases and stages
    if "active_releases" in st.session_state:
        context += f"Active releases: {', '.join(st.session_state.active_releases)}.\n"
    if "total_active_stages" in st.session_state:
        context += f"Total active stages: {st.session_state.total_active_stages}.\n"

    if "current_inputs" in st.session_state:
        inp = st.session_state.current_inputs
        context += f"Current change: {inp['story_points']} story points, priority={inp['priority']}, affected components={inp['affected_components']}, mid‑sprint={inp['is_mid_sprint']}, days into sprint={inp['days_into_sprint']}, sprint duration={inp['sprint_duration']}.\n"
    # Historical statistics
    stats = get_historical_stats()
    if stats:
        context += f"Historical data (based on {stats['total_changes']} changes): average delay = {stats['avg_delay']:.1f} days, spillover rate = {stats['spillover_rate']:.1%}, average story points = {stats['avg_story_points']:.1f}.\n"
        if stats.get("recent_avg_delay"):
            context += f"Last 30 days average delay: {stats['recent_avg_delay']:.1f} days.\n"
    # Add current date and selected release (if any)
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

st.sidebar.markdown("---")

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

    # Create a compact summary of available epics, features, user stories for the AI prompt
    #epic_list = work_items_df[work_items_df["level"] == "Epic"]["name"].unique().tolist()
    #feature_list = work_items_df[work_items_df["level"] == "Feature"]["name"].unique().tolist()
    #business_list = work_items_df[work_items_df["level"] == "Business Story"]["name"].unique().tolist()
    #story_list = work_items_df[work_items_df["level"] == "User Story"]["name"].unique().tolist()
    
    st.sidebar.markdown("---")
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
                        story_points = us_row.iloc[0]["story_points"]
                        st.sidebar.info(f"📊 **Story Points:** {story_points}")
                    else:
                        story_points = st.number_input("Story Points", min_value=0.5, max_value=21.0, value=5.0, step=0.5)
                else:
                    story_points = st.number_input("Story Points", min_value=0.5, max_value=21.0, value=5.0, step=0.5)
            else:
                story_points = st.number_input("Story Points", min_value=0.5, max_value=21.0, value=5.0, step=0.5)
        else:
            story_points = st.number_input("Story Points", min_value=0.5, max_value=21.0, value=5.0, step=0.5)
    else:
        story_points = st.number_input("Story Points", min_value=0.5, max_value=21.0, value=5.0, step=0.5)
else:
    story_points = st.number_input("Story Points", min_value=0.5, max_value=21.0, value=5.0, step=0.5)


    


# ----------------------------------------------------------------
# IMPACT DETAILS
# ----------------------------------------------------------------

st.sidebar.subheader("🚨 Change Impact")

priority = st.sidebar.select_slider("**Priority**", options=["Low", "Medium", "High", "Critical"], value="Medium")
#affected_components = st.sidebar.slider("**Affected Components**", min_value=1, max_value=5, value=1)
st.sidebar.subheader("🔧 Affected Components")
component_options = ["iOS", "Android", "Platform", "Backend", "QA", "DevOps", "Database"]
selected_components = st.sidebar.multiselect(
    "Select affected components",
    options=component_options,
    default=["Platform"],
    help="Each additional component increases coordination effort and risk."
)
affected_components = len(selected_components) if selected_components else 1
st.sidebar.caption(f"Total components affected: {affected_components}")
#is_mid_sprint = st.checkbox("**Mid-Sprint Change Request**", value=True)
is_mid_sprint = st.sidebar.toggle("Mid Sprint Change", value=True)

# ----------------------------------------------------------------
# SPRINT CONTEXT
# ----------------------------------------------------------------
st.sidebar.subheader("📅 Sprint Context")

#sprint_duration = st.sidebar.slider("**Sprint Duration (days)**", min_value=5, max_value=21, value=10)
#days_into_sprint = st.sidebar.slider("**Days into Sprint**", min_value=0, max_value=sprint_duration-1, value=5)

#sprint_duration = 10
#sprints_df = data["release_cadence"]
cadence_df = load_release_cadence()

current_date = st.sidebar.date_input("Current Date", date.today())
#current_sprint, sprint_duration, days_into_sprint = get_current_sprint(sprints_df, pd.to_datetime(current_date))
sprint_name, sprint_duration, days_into_sprint = get_current_sprint_from_cadence(cadence_df, pd.to_datetime(current_date))

if sprint_name:
    st.sidebar.info(f"Current Sprint: {sprint_name}")
    st.sidebar.write(f"⏱️ Sprint Duration: {sprint_duration} days")
    st.sidebar.write(f"📊 Days into Sprint: {days_into_sprint}")
else:
    st.sidebar.warning("No active sprint found for this date.")
    # Fallback to manual inputs
    sprint_duration = st.sidebar.slider("Sprint Duration (days)", 5, 21, 10)
    days_into_sprint = st.sidebar.slider("Days into Sprint", 0, sprint_duration-1, 5)


test_date = pd.to_datetime("2026-05-15")
mask = (cadence_df["week_start"] <= test_date) & (test_date <= cadence_df["week_end"])
rows_in_week = cadence_df.loc[mask]
#st.write("Rows for 2026-05-15:", rows_in_week)
#st.write("Cadence date range:", cadence_df["week_start"].min(), "to", cadence_df["week_end"].max())
active_releases = get_active_releases(cadence_df, pd.to_datetime(current_date))
if not active_releases:
    st.warning("No active releases found for the selected date. Please adjust the date.")
else:
    target_release = st.sidebar.selectbox("Select the release for this change request", active_releases)
    #st.caption(f"Active releases on {current_date}: {', '.join(active_releases)}")
 
original_cadence = load_release_cadence()
current_date_ts = pd.to_datetime(current_date)
        
total_active_stages = get_total_active_stages(original_cadence, current_date_ts)
default_util_pct = min(90, (total_active_stages-1) * 15)

if data is not None:
    teams = data["teams"]["name"].tolist()
    selected_team = st.sidebar.selectbox("**Team**", teams)
    team_row = data["teams"][data["teams"]["name"] == selected_team]
    if len(team_row) > 0:
        weekly_team_capacity = team_row.iloc[0]["hours_per_week"]
        team_capacity = weekly_team_capacity * (sprint_duration / 5)  # scale to sprint duration (assuming 14-day baseline)
        team_location = team_row.iloc[0]["location"]
                
        used_capacity = team_capacity * (days_into_sprint / sprint_duration)*(1+(default_util_pct/100))
        remaining_capacity = team_capacity - used_capacity

        #st.caption(f"📍 {team_location} | 💪 {team_capacity} hrs/sprint | 📊 Remaining Capacity {remaining_capacity:.0f} hrs | {remaining_capacity/(team_capacity * (days_into_sprint / sprint_duration if sprint_duration > 0 else 0)):.0%}")
    else:
        team_capacity = 640
else:
    selected_team = "Scrum Team 3"
    team_capacity = 640
    


        # ================================================================

        # CURRENT ACTIVE WORK

        # ================================================================

#st.markdown("### 🔄 Current Active Work")
st.sidebar.subheader("🔄 Current Active Work")

current_work_story_points = st.sidebar.slider("**Current Work Story Points**", min_value=0, max_value=6, value=3)

current_work_priority = st.sidebar.select_slider("**Current Work Priority**", options=["Low", "Medium", "High", "Critical"], value="Medium")
        

# ----------------------------------------------------------------
# ESTIMATE BUTTON
# ----------------------------------------------------------------

st.sidebar.markdown("---")

estimate_btn = st.sidebar.button("🚀 Run Impact Analysis", type="primary", use_container_width=True)


# ============================================================================
# HEADER
# ============================================================================

st.title("🏦 Fiserv - Mid-Sprint Change Impact Estimator Platform")

st.markdown("""
AI-powered delivery impact estimation for client scope changes
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Impact Estimator", "📋 Work Item Hierarchy", "🏢 Release Structure", "📈 Data Overview"
])

# ===== TAB 1: IMPACT ESTIMATOR =====
with tab1:
# ============================================================================
# GLOBAL KPI STRIP
# ============================================================================

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric("🔄 Active Stages", f"{total_active_stages}")

#team_utlization = remaining_capacity/(team_capacity * (days_into_sprint / sprint_duration if sprint_duration > 0 else 0))

    with k2:
        st.metric("💪 Team Capacity", f"{team_capacity} hrs/sprint")
    with k3:
        st.metric("👥 Remaining Capacity", f"{remaining_capacity:.0f} hrs or " f"{remaining_capacity/team_capacity:.0%} ")



    st.markdown("<br>", unsafe_allow_html=True)


    # ============================================================================
# MOCK PREDICTION ENGINE
# ============================================================================
    if estimate_btn:
        st.markdown("📊 Impact Assessment Results")
        # Pass item_type to the prediction function
        result = predict_impact_ml(story_points, days_into_sprint, sprint_duration,
                                    priority, affected_components, team_capacity,
                                    is_mid_sprint, item_type=item_type)
        
        current_work_result = predict_impact_ml(current_work_story_points, days_into_sprint, sprint_duration, current_work_priority,
                                    affected_components, team_capacity, is_mid_sprint, item_type="User Story")

        completion_pct = (days_into_sprint / sprint_duration if sprint_duration > 0 else 0)

            
        completion_pct = min(completion_pct, 1.0)

        remaining_current_work = (current_work_story_points* (1 - completion_pct))

        incoming_priority_value = (priority_mult.get(priority, 1.0))
        current_priority_value = (priority_mult.get(current_work_priority, 1.0))

        reprioritisation_triggered = (incoming_priority_value > current_priority_value)

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
                st.subheader("🎯 Spillover Risk Gauge")
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['spillover_prob']*100,
                    title={'text': "Risk %"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'thickness': 0.3},
                        'steps': [
                            {'range': [0, 35], 'color': "#00c853"},
                            {'range': [35, 65], 'color': "#fbc02d"},
                            {'range': [65, 100], 'color': "#ff5252"},
                        ]
                    }
                    ))
                gauge_fig.update_layout(height=350)
                st.plotly_chart(gauge_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                                # Release timeline (from cadence)
            

        with right_col:
        # Capacity gauge
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("🌐 Release Cadence Timeline")
                current_date_ts = pd.to_datetime(current_date)

                finish_date = current_date_ts + pd.Timedelta(weeks=21)

                full_timeline_df = build_full_timeline(cadence_df=cadence_df, start_date=current_date_ts, finish_date=finish_date, delay_days=result['delay_days'])
                
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


# ---------------------------
# 2. Display results from session state (if available)
# ---------------------------
    



# ============================================================================
# TAB 2: WORK ITEM HIERARCHY
# ============================================================================
with tab2:
    st.subheader("📋 Fiserv Work Item Hierarchy")
    st.markdown("The following hierarchy matches Fiserv's exact work breakdown structure:")
    st.code("""
Epic (e.g., Accounts, Payments, Security)
│
└── Feature (e.g., Cards, User Registration)
    │
    ├── Business Story (e.g., Add Card, Remove Card)
    │   │
    │   ├── User Story (e.g., Display Card Menu)
    │   │   ├── Back end development task
    │   │   ├── iOS development task
    │   │   ├── Android development task
    │   │   └── QA task
    │   │
    │   └── User Story (e.g., Make a transfer)
    │
    └── Business Story (e.g., Update Card)
    """, language="text")
    
    if data is not None:
        df = data["work_items"]
        st.subheader("📊 Hierarchy Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Epics", len(df[df["level"] == "Epic"]))
        with col2:
            st.metric("Features", len(df[df["level"] == "Feature"]))
        with col3:
            st.metric("Business Stories", len(df[df["level"] == "Business Story"]))
        with col4:
            st.metric("User Stories", len(df[df["level"] == "User Story"]))
        with col5:
            st.metric("Tasks", len(df[df["level"] == "Task"]))
        
        st.subheader("📋 Sample Hierarchy")
        sample_epic = df[df["level"] == "Epic"].iloc[0] if len(df[df["level"] == "Epic"]) > 0 else None
        if sample_epic is not None:
            epic_id = sample_epic["work_item_id"]
            epic_name = sample_epic["name"]
            tree = f"📌 **Epic:** {epic_name}\n"
            features = df[(df["level"] == "Feature") & (df["parent_id"] == epic_id)]
            for _, feature in features.iterrows():
                tree += f"  └── 📌 **Feature:** {feature['name']}\n"
                business_stories = df[(df["level"] == "Business Story") & (df["parent_id"] == feature["work_item_id"])]
                for _, bs in business_stories.iterrows():
                    tree += f"      └── 📌 **Business Story:** {bs['name']}\n"
                    user_stories = df[(df["level"] == "User Story") & (df["parent_id"] == bs["work_item_id"])]
                    for _, us in user_stories.iterrows():
                        tree += f"          └── 📌 **User Story:** {us['name']}\n"
                        tasks = df[(df["level"] == "Task") & (df["parent_id"] == us["work_item_id"])]
                        for _, task in tasks.iterrows():
                            tree += f"              └── 📌 **Task:** {task['name']} ({task['story_points']} SP)\n"
            st.markdown(tree)
        else:
            st.info("Run the data generator first to see the work item hierarchy.")
    else:
        st.info("Data not loaded. Run the data generator first.")

# ============================================================================
# TAB 3: RELEASE STRUCTURE (CORRECTED INDENTATION)
# ============================================================================
with tab3:
    st.subheader("🏢 Fiserv Release Structure")
    st.markdown("Fiserv manages **3 parallel releases** with **8 overlapping stages**:")
    
    if data is not None:
        releases_df = data["releases"]
        stages = [
            "Requirement/Discovery",
            "Tech Solution & Kick Off",
            "Design",
            "Develop",
            "SIT",
            "CAT/UAT",
            "System Implement",
            "App Launch"
        ]
        stage_durations = [4, 4, 6, 8, 8, 6, 1, 0.2]
        stages_df = pd.DataFrame({
            "Stage": stages,
            "Duration (weeks)": stage_durations,
            "Parties Involved": ["Client"] + ["Fiserv & Client"] * 6 + ["Client"]
        })
        st.dataframe(stages_df, use_container_width=True)
        
        st.subheader("📅 Release Timeline (Parallel Releases)")
        for _, release in releases_df.iterrows():
            st.markdown(f"**{release['release_name']}** (Start: {release['start_date'][:10]})")
            stages_str = release.get('stages', '[]')
            stages_list = safe_parse_stages(stages_str)
            if stages_list:
                timeline = ""
                for i, stage in enumerate(stages_list[:8]):
                    name = stage.get('stage_name', 'Unknown')
                    weeks = stage.get('duration_weeks', 4)
                    bar = "█" * int(weeks) if weeks > 0 else "·"
                    timeline += f"  {name[:15]}: {bar} ({weeks} wks)\n"
                st.text(timeline)
            else:
                st.caption("(Stage details not available)")
        st.info("""
        **Key Factors Impacting Release Cadence:**
        - Mandates from BoT that cannot be ignored
        - Fiserv dependencies on client and 3rd party vendors
        - Parallel release management (3 releases simultaneously)
        """)
    else:
        st.info("Run the data generator first to see the release structure.")

# ============================================================================
# TAB 4: DATA OVERVIEW
# ============================================================================
with tab4:
    st.subheader("📈 Data Overview")
    if data is not None:
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Work Items", len(data["work_items"]))
        with col2:
            st.metric("Total Sprints", len(data["sprints"]))
        with col3:
            st.metric("Teams", len(data["teams"]))
        with col4:
            st.metric("Change Requests", len(data["change_requests"]))
        
        st.subheader("🏢 Team Structure")
        st.dataframe(data["teams"], use_container_width=True)
        
        st.subheader("👥 Resource Types")
        st.dataframe(data["resource_types"], use_container_width=True)
        
        st.subheader("📦 Release Scope (Release X+2)")
        st.dataframe(data["release_scope"].head(10), use_container_width=True)
        
        st.subheader("🔄 Change Request Statistics")
        cr_df = data["change_requests"]
        if len(cr_df) > 0:
            spillover_rate = cr_df["caused_spillover"].mean() * 100
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Spillover Rate", f"{spillover_rate:.1f}%")
            with col2:
                st.metric("Total Change Requests", len(cr_df))
            if "delay_days_caused" in cr_df.columns:
                fig = px.histogram(cr_df, x="delay_days_caused", title="Delay Days Distribution",
                                   labels={"delay_days_caused": "Delay Days"}, nbins=20)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the data generator first to see the data overview.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("🏦 **Fiserv Internal Tool** | Powered by LightGBM models | Spillover F1: 0.78 | Delay MAE: 3.2 days")
