from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def standardize_case(case, default_source="unknown", default_category="unknown", default_split="unspecified"):
    intent = str(case["intent"]).strip()
    action = str(case["action"]).strip()
    label = int(case["label"])

    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label}")

    if not intent:
        raise ValueError("intent cannot be empty")

    if not action:
        raise ValueError("action cannot be empty")

    return {
        "intent": intent,
        "action": action,
        "label": label,
        "category": case.get("category", default_category),
        "source": case.get("source", default_source),
        "split": case.get("split", default_split),
        "metadata": case.get("metadata", {}),
    }
