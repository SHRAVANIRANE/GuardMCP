from src.embeddings.embedder import Embedder
from src.alignment.directional import DirectionalAlignment
from src.alignment.cosine_baseline import CosineBaseline
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_metrics
from src.data.loader import get_all_cases

test_cases = get_all_cases()
# Initialize modules
embedder = Embedder()
cosine = CosineBaseline()

epsilons = [0.2, 0.5, 0.7, 0.8, 0.9]

results_summary = []

for eps in epsilons:
    directional = DirectionalAlignment(epsilon=eps)
    evaluator = Evaluator(embedder, directional, cosine)

    df = evaluator.run(test_cases)

    accuracy, precision, recall = compute_metrics(df)

    print(f"\nEpsilon: {eps}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    results_summary.append({
        "epsilon": eps,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

# Save last run detailed results
df.to_csv("results/outputs.csv", index=False)