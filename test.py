from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def test_LGBM(X, Y, test_size, **kwargs):
    print("[TEST] Testing LGBM")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42, **kwargs)

    # LightGBM model for standard scaled data
    lgb_model_scaled = lgb.LGBMClassifier(random_state=42)
    lgb_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = lgb_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)

    print(f"---- AUC score LGBM: {auc_scaled} ----")

def test_DTree(X, Y, test_size, **kwargs):
    print("[TEST] Testing Decision Tree")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42, **kwargs)

    # LightGBM model for standard scaled data
    lgb_model_scaled = DecisionTreeClassifier(random_state=42)
    lgb_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = lgb_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)

    print(f"---- AUC score DTree: {auc_scaled} ----")

def test_RForest(X, Y, test_size, n_estimators=100, **kwargs):
    print("[TEST] Testing Random Forest")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42)

    # LightGBM model for standard scaled data
    lgb_model_scaled = RandomForestClassifier(n_estimators, random_state=42, **kwargs)
    lgb_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = lgb_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)

    print(f"---- AUC score RForest: {auc_scaled} ----")