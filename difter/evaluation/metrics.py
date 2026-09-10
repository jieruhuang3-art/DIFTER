import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(truth, prediction, probabilities=None):
    result = {"accuracy": float(accuracy_score(truth, prediction)),
              "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
              "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0))}
    if probabilities is not None:
        probabilities = np.asarray(probabilities); truth = np.asarray(truth)
        confidence, predicted = probabilities.max(1), probabilities.argmax(1)
        boundaries = np.linspace(0, 1, 16); ece = 0.0
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            selected = (confidence > lower) & (confidence <= upper)
            if selected.any():
                ece += selected.mean() * abs((predicted[selected] == truth[selected]).mean() - confidence[selected].mean())
        result["ece"] = float(ece)
    return result
