from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
EVAL_SPLITS = ("train", "dev", "test")


def standardize_case(case, default_source="unknown", default_category="unknown", default_split="unspecified"):
    intent = str(case["intent"]).strip()
    action = str(case["action"]).strip()
    label = int(case["label"])
    split = case.get("split", default_split)
    metadata = dict(case.get("metadata", {}))

    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label}")

    if not intent:
        raise ValueError("intent cannot be empty")

    if not action:
        raise ValueError("action cannot be empty")

    if split not in EVAL_SPLITS:
        metadata.setdefault("source_split", split)

    return {
        "intent": intent,
        "action": action,
        "label": label,
        "category": case.get("category", default_category),
        "source": case.get("source", default_source),
        "split": split,
        "metadata": metadata,
    }
