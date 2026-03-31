def compute_metrics(df, decision_column):
    if len(df) == 0:
        raise ValueError("cannot compute metrics for an empty dataframe")

    tp = ((df["label"] == 1) & (df[decision_column] == 1)).sum()
    tn = ((df["label"] == 0) & (df[decision_column] == 0)).sum()
    fp = ((df["label"] == 0) & (df[decision_column] == 1)).sum()
    fn = ((df["label"] == 1) & (df[decision_column] == 0)).sum()

    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "support": int(len(df)),
    }
