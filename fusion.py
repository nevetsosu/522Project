from sklearn.metrics import roc_auc_score
import numpy as np


def fusion(prob_predictions, y, weights=None, **kwargs):
    prob_predictions = np.array(prob_predictions)

    if weights is None:
        weights = [1] * len(prob_predictions)  
    
    if (sum(weights) == 0):
        weight_sum = 1
    else:
        weight_sum = sum(weights)
    weights = np.array(weights) / weight_sum  # normalize weights
    fused_probs = np.average(prob_predictions, axis=0, weights=weights)

    auc_fused = roc_auc_score(y, fused_probs)
    print(f"[FUSION] AUC after fusion: {auc_fused:.4f}")
    print("bug")
    return auc_fused, fused_probs