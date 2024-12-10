from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

    # Decision Tree model for standard scaled data
    dtree_model_scaled = DecisionTreeClassifier(random_state=42)
    dtree_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = dtree_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)
    return auc_scaled

def RForest(X, Y, test_size, n_estimators=100, **kwargs):
    print("[TEST] Testing Random Forest")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Random Forest model for standard scaled data
    rf_model_scaled = RandomForestClassifier(n_estimators, n_jobs=-1, random_state=42, **kwargs)
    rf_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = rf_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)
    return auc_scaled


def MLP(X, Y, test_size, hidden_layer_sizes=(100,), **kwargs):
    print("[TEST] Testing MLP")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=42, **kwargs)

    # Train MLP for standard scaled data
    mlp_model_scaled = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=42)
    mlp_model_scaled.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred_scaled = mlp_model_scaled.predict_proba(X_val)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)
    return auc_scaled
