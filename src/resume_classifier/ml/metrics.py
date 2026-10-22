import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def quality_metrics(expected: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "f1_macro": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(expected, predicted, average="weighted", zero_division=0)),
    }
