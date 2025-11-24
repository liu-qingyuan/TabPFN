import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred_proba):
    """Calculate classification metrics (binary-friendly).

    Returns a dict containing AUC, Accuracy, F1, Precision, Recall with
    keys compatible with downstream visualization utilities.
    """
    metrics = calculate_detailed_metrics(y_true, y_pred_proba)
    # Preserve legacy keys
    return {
        'AUC': metrics['AUC'],
        'Accuracy': metrics['Accuracy'],
        'F1': metrics['F1'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
    }


def calculate_detailed_metrics(y_true, y_pred_proba):
    """Calculate AUC, Accuracy, F1, Precision, Recall for binary/one-vs-rest.

    Args:
        y_true: 1D array-like of true labels.
        y_pred_proba: array-like of predicted probabilities or decision scores.

    Returns:
        Dict with AUC/Accuracy/F1/Precision/Recall.
    """
    # Handle binary and multi-class probabilities
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
        y_prob = y_pred_proba[:, 1]
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_prob = y_pred_proba
        y_pred = (y_prob > 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        'AUC': auc,
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
    }

def sample_data(X, y, n_samples, random_state=42):
    """Subsample data."""
    if n_samples >= len(X):
        return X, y
    
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), n_samples, replace=False)
    
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices], y.iloc[indices]
    else:
        return X[indices], y[indices]
