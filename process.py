from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def LGBM(X, Y, test_size, **kwargs):
    print("[TEST] Testing LGBM")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42, **kwargs)

    # LightGBM model for standard scaled data
    lgb_model_scaled = lgb.LGBMClassifier(n_jobs=-1, random_state=42)
    lgb_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = lgb_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)

    return auc_scaled

def DTree(X, Y, test_size, **kwargs):
    print("[TEST] Testing Decision Tree")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42, **kwargs)

    # LightGBM model for standard scaled data
    lgb_model_scaled = DecisionTreeClassifier(random_state=42)
    lgb_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = lgb_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)
    return auc_scaled

def RForest(X, Y, test_size, n_estimators=100, **kwargs):
    print("[TEST] Testing Random Forest")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42)

    # LightGBM model for standard scaled data
    lgb_model_scaled = RandomForestClassifier(n_estimators, n_jobs=-1, random_state=42, **kwargs)
    lgb_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = lgb_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)
    return auc_scaled