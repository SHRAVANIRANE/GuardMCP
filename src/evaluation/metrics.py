def compute_metrics(df):
    tp = ((df["label"] == 1) & (df["directional_decision"] == 1)).sum()
    tn = ((df["label"] == 0) & (df["directional_decision"] == 0)).sum()
    fp = ((df["label"] == 0) & (df["directional_decision"] == 1)).sum()
    fn = ((df["label"] == 1) & (df["directional_decision"] == 0)).sum()

    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return accuracy, precision, recall