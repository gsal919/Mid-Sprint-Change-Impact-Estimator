import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import os
from config import (
    TRAIN_PATH, VALID_PATH, TEST_PATH,
    TARGET_CLASSIFICATION_SPILLOVER,
    TARGET_REGRESSION,
    TARGET_MULTICLASS
)

TARGET_REGRESSION = "delay_days_caused"
# Create models directory
os.makedirs('models', exist_ok=True)

print("="*60)
print("TRAINING FISERV DELAY PREDICTION MODELS (Pseudo‑data)")
print("="*60)

# ============================================
# LOAD DATA
# ============================================
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nTrain shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"Test shape: {test_df.shape}")

# Define feature columns (exclude targets and leakage)
# Targets: caused_spillover, delay_days_caused, delay_severity
leakage_cols = ["delay_days_caused"]   # remove from features
target_cols = [TARGET_CLASSIFICATION_SPILLOVER, TARGET_REGRESSION, TARGET_MULTICLASS]
feature_cols = [c for c in train_df.columns if c not in target_cols + leakage_cols]

print(f"\nUsing {len(feature_cols)} feature columns: {feature_cols}")

# Prepare data (full copies for consistent encoding)
X_train_full = train_df[feature_cols].copy()
X_valid_full = valid_df[feature_cols].copy()
X_test_full = test_df[feature_cols].copy()

# ============================================
# HANDLE CATEGORICAL VARIABLES (LightGBM compatible)
# ============================================
categorical_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    # Convert to category dtype
    X_train_full[col] = X_train_full[col].astype('category')
    X_valid_full[col] = X_valid_full[col].astype('category')
    X_test_full[col] = X_test_full[col].astype('category')

# Fill missing values (use train median for numerics)
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = X_train_full[col].median()
    X_train_full[col] = X_train_full[col].fillna(median_val)
    X_valid_full[col] = X_valid_full[col].fillna(median_val)
    X_test_full[col] = X_test_full[col].fillna(median_val)

# ============================================
# MODEL 1: Spillover Classification (Random Forest)
# ============================================
print("\n" + "="*60)
print("MODEL 1: Spillover Classification (Random Forest)")
print("="*60)

train_clf = train_df.dropna(subset=[TARGET_CLASSIFICATION_SPILLOVER])
valid_clf = valid_df.dropna(subset=[TARGET_CLASSIFICATION_SPILLOVER])
test_clf = test_df.dropna(subset=[TARGET_CLASSIFICATION_SPILLOVER])

X_train = X_train_full.loc[train_clf.index]
X_valid = X_valid_full.loc[valid_clf.index]
X_test = X_test_full.loc[test_clf.index]

y_train = train_clf[TARGET_CLASSIFICATION_SPILLOVER]
y_valid = valid_clf[TARGET_CLASSIFICATION_SPILLOVER]
y_test = test_clf[TARGET_CLASSIFICATION_SPILLOVER]

print(f"Train size: {len(train_clf)} | Valid size: {len(valid_clf)} | Test size: {len(test_clf)}")
print(f"Train spillover rate: {y_train.mean():.2%}")
print(f"Valid spillover rate: {y_valid.mean():.2%}")
print(f"Test spillover rate: {y_test.mean():.2%}")

# Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced')
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
print(f"RF Test Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"RF Test F1: {f1_score(y_test, y_pred_rf):.3f}")
joblib.dump(clf_rf, 'models/classifier_spillover.pkl')
print("✅ Saved models/classifier_spillover.pkl")

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=5, 
    class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
)
lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)
y_pred_lgb = lgb_clf.predict(X_test)
print(f"\nLGB Test Accuracy: {accuracy_score(y_test, y_pred_lgb):.3f}")
print(f"LGB Test F1: {f1_score(y_test, y_pred_lgb):.3f}")
joblib.dump(lgb_clf, 'models/classifier_spillover_lgb.pkl')
print("✅ Saved models/classifier_spillover_lgb.pkl")

# ============================================
# MODEL 2: Delay Days Regression (using delay_days_caused)
# ============================================
print("\n" + "="*60)
print("MODEL 2: Delay Days Regression")
print("="*60)

# Use TARGET_REGRESSION which should be 'delay_days_caused'
if TARGET_REGRESSION not in train_df.columns:
    raise KeyError(f"Regression target '{TARGET_REGRESSION}' not found. Available: {train_df.columns.tolist()}")

train_reg = train_df.dropna(subset=[TARGET_REGRESSION])
valid_reg = valid_df.dropna(subset=[TARGET_REGRESSION])
test_reg = test_df.dropna(subset=[TARGET_REGRESSION])

X_train_reg = X_train_full.loc[train_reg.index]
X_valid_reg = X_valid_full.loc[valid_reg.index]
X_test_reg = X_test_full.loc[test_reg.index]

y_train_reg = train_reg[TARGET_REGRESSION]
y_valid_reg = valid_reg[TARGET_REGRESSION]
y_test_reg = test_reg[TARGET_REGRESSION]

print(f"Train size: {len(train_reg)} | Valid size: {len(valid_reg)} | Test size: {len(test_reg)}")
print(f"Train delay mean: {y_train_reg.mean():.2f} days")
print(f"Valid delay mean: {y_valid_reg.mean():.2f} days")
print(f"Test delay mean: {y_test_reg.mean():.2f} days")

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg)
print(f"\nRF Test MAE: {mean_absolute_error(y_test_reg, y_pred_rf_reg):.2f} days")
print(f"RF Test R²: {r2_score(y_test_reg, y_pred_rf_reg):.3f}")
joblib.dump(rf_reg, 'models/regressor_delay_days.pkl')
print("✅ Saved models/regressor_delay_days.pkl")

# LightGBM Regressor
lgb_reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
lgb_reg.fit(
    X_train_reg, y_train_reg,
    eval_set=[(X_valid_reg, y_valid_reg)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)
y_pred_lgb_reg = lgb_reg.predict(X_test_reg)
print(f"\nLGB Test MAE: {mean_absolute_error(y_test_reg, y_pred_lgb_reg):.2f} days")
print(f"LGB Test R²: {r2_score(y_test_reg, y_pred_lgb_reg):.3f}")
joblib.dump(lgb_reg, 'models/regressor_delay_days_lgb.pkl')
print("✅ Saved models/regressor_delay_days_lgb.pkl")

# ============================================
# MODEL 3: Delay Severity Classification (Multi‑class)
# ============================================
print("\n" + "="*60)
print("MODEL 3: Delay Severity Classification")
print("="*60)

if TARGET_MULTICLASS not in train_df.columns:
    print(f"Warning: {TARGET_MULTICLASS} not found. Skipping multi‑class training.")
else:
    train_multi = train_df.dropna(subset=[TARGET_MULTICLASS])
    valid_multi = valid_df.dropna(subset=[TARGET_MULTICLASS])
    test_multi = test_df.dropna(subset=[TARGET_MULTICLASS])

    X_train_multi = X_train_full.loc[train_multi.index]
    X_valid_multi = X_valid_full.loc[valid_multi.index]
    X_test_multi = X_test_full.loc[test_multi.index]

    y_train_multi = train_multi[TARGET_MULTICLASS]
    y_valid_multi = valid_multi[TARGET_MULTICLASS]
    y_test_multi = test_multi[TARGET_MULTICLASS]

    # Encode string labels to integers (LightGBM requires numeric)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_multi_enc = le.fit_transform(y_train_multi)
    y_valid_multi_enc = le.transform(y_valid_multi)
    y_test_multi_enc = le.transform(y_test_multi)

    lgb_multi = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        objective='multiclass', num_class=len(le.classes_),
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_multi.fit(
        X_train_multi, y_train_multi_enc,
        eval_set=[(X_valid_multi, y_valid_multi_enc)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    y_pred_multi = lgb_multi.predict(X_test_multi)
    print(f"Test Accuracy: {accuracy_score(y_test_multi_enc, y_pred_multi):.3f}")
    joblib.dump(lgb_multi, 'models/classifier_severity_lgb.pkl')
    joblib.dump(le, 'models/severity_label_encoder.pkl')
    print("✅ Saved models/classifier_severity_lgb.pkl and label encoder")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)