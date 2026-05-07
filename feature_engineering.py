import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "fiserv_data"
INPUT_FILE = os.path.join(DATA_DIR, "change_requests.csv")
OUTPUT_DIR = "data/processed_pseudo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target columns
TARGET_REGRESSION = "delay_days_caused"
TARGET_CLASSIFICATION = "caused_spillover"
TARGET_MULTICLASS = "delay_severity"

# Columns to drop (leakage, IDs, or future information)
DROP_COLS = [
    "change_request_id", "sprint_id", "sprint_name", "release_id",
    "work_item_id", "work_item_name", "request_date",
    "timing_penalty", "priority_multiplier", "recommendation",
    "base_effort_hours", "impacted_effort_hours", "remaining_capacity_hours",
    "original_estimate_hours", "rework_hours", "actual_hours",
    "sprint_progress_pct",   # we'll re‑compute from days_into_sprint and sprint_duration if available
]

# ============================================================
# LOAD DATA
# ============================================================
print("Loading pseudo‑data...")
df = pd.read_csv(INPUT_FILE)
print(f"Shape: {df.shape}")

# Convert request_date to datetime
df["request_date"] = pd.to_datetime(df["request_date"], errors="coerce")

# Sort by date for time‑based split (optional)
df = df.sort_values("request_date").reset_index(drop=True)

# ============================================================
# BASIC CLEANING
# ============================================================
# Keep only rows with valid targets
df = df.dropna(subset=[TARGET_REGRESSION, TARGET_CLASSIFICATION])
print(f"After dropping null targets: {df.shape}")

# Ensure binary classification target is 0/1
df[TARGET_CLASSIFICATION] = df[TARGET_CLASSIFICATION].astype(int)

# ============================================================
# FEATURE ENGINEERING (non‑leaking)
# ============================================================
print("\nEngineering features...")

# 1. Sprint progress (from days_into_sprint and fixed sprint_duration = 14 days)
# In the generator, sprint duration is 14 days (2 weeks). We can infer sprint_duration = 14.
if "sprint_duration" not in df.columns:
    df["sprint_duration"] = 14
df["sprint_progress_at_creation"] = df["days_into_sprint"] / df["sprint_duration"]
df["remaining_sprint_pct"] = 1 - df["sprint_progress_at_creation"]

# 2. Complexity score (already present, but ensure it's numeric)
if "complexity_score" in df.columns:
    df["complexity_score"] = df["complexity_score"].astype(float)

# 3. Predicted risk proxy (already present)
if "predicted_risk_proxy" in df.columns:
    df["predicted_risk_proxy"] = df["predicted_risk_proxy"].astype(float)

# 4. Sprint task load (already present)
if "sprint_task_load" in df.columns:
    df["sprint_task_load"] = df["sprint_task_load"].astype(float)

# 5. Additional derived features
df["has_estimate"] = 1  # all rows have estimates
df["has_story_points"] = (df["story_points"] > 0).astype(int)
df["story_points_log"] = np.log1p(df["story_points"])

# 6. Encode categoricals
priority_map = {"Low":0, "Medium":1, "High":2, "Critical":3}
df["priority_encoded"] = df["priority"].map(priority_map)

# work_item_level and item_type are identical here; keep one
if "work_item_level" in df.columns:
    level_map = {"Epic":0, "Feature":1, "Business Story":2, "User Story":3, "Task":4}
    df["item_type_encoded"] = df["work_item_level"].map(level_map).fillna(0).astype(int)
elif "item_type" in df.columns:
    level_map = {"Epic":0, "Feature":1, "Business Story":2, "User Story":3, "Task":4}
    df["item_type_encoded"] = df["item_type"].map(level_map).fillna(0).astype(int)

# 7. One‑hot encoding for delay_severity (if needed for multi‑class)
# We'll keep it as label for now (target)

# ============================================================
# DROP LEAKAGE / NON‑FEATURE COLUMNS
# ============================================================
# Identify all columns that are not features or targets
feature_candidates = [
    "story_points", "days_into_sprint", "sprint_duration",
    "sprint_progress_at_creation", "remaining_sprint_pct",
    "complexity_score", "predicted_risk_proxy", "sprint_task_load",
    "has_estimate", "has_story_points", "story_points_log",
    "priority_encoded", "item_type_encoded"
]

# Keep only columns that exist
feature_cols = [c for c in feature_candidates if c in df.columns]

# Ensure target columns are kept
target_cols = [TARGET_REGRESSION, TARGET_CLASSIFICATION, TARGET_MULTICLASS]
existing_targets = [c for c in target_cols if c in df.columns]

# Final ML dataset
ml_df = df[feature_cols + existing_targets].copy()
print(f"Final feature columns: {feature_cols}")
print(f"Target columns: {existing_targets}")

# ============================================================
# TIME‑BASED SPLIT (by request_date)
# ============================================================
# Use date thresholds (e.g., 80% train, 10% val, 10% test based on time order)
# Here we sort by date and split chronologically
dates = df["request_date"].sort_values()
train_cutoff = dates.quantile(0.8)
val_cutoff = dates.quantile(0.9)

train_idx = df["request_date"] <= train_cutoff
val_idx = (df["request_date"] > train_cutoff) & (df["request_date"] <= val_cutoff)
test_idx = df["request_date"] > val_cutoff

# Create splits (using ml_df but aligning with same index)
train_df = ml_df[train_idx].copy()
val_df = ml_df[val_idx].copy()
test_df = ml_df[test_idx].copy()

print(f"\nSplit sizes:")
print(f"Train: {len(train_df)} ({len(train_df)/len(ml_df):.1%})")
print(f"Val:   {len(val_df)} ({len(val_df)/len(ml_df):.1%})")
print(f"Test:  {len(test_df)} ({len(test_df)/len(ml_df):.1%})")

# ============================================================
# SAVE SPLITS
# ============================================================
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "valid.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
print(f"\n✅ Saved splits to {OUTPUT_DIR}/")

# Optional: save feature list for dashboard alignment
feature_list_path = os.path.join(OUTPUT_DIR, "feature_names.txt")
with open(feature_list_path, "w") as f:
    for col in feature_cols:
        f.write(col + "\n")
print(f"✅ Saved feature list to {feature_list_path}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*60)
print("TARGET DISTRIBUTION")
print("="*60)
for name, split in [("Train", train_df), ("Valid", val_df), ("Test", test_df)]:
    print(f"\n{name}:")
    if TARGET_CLASSIFICATION in split:
        print(f"  Spillover rate: {split[TARGET_CLASSIFICATION].mean():.2%}")
    if TARGET_REGRESSION in split:
        print(f"  Avg delay days: {split[TARGET_REGRESSION].mean():.2f}")
    if TARGET_MULTICLASS in split:
        print(f"  Severity distribution:\n{split[TARGET_MULTICLASS].value_counts()}")