import hashlib
import json
from collections import Counter, defaultdict

from src.data.schema import DATA_DIR, EVAL_SPLITS, standardize_case


TOOLTALK_CASES_PATH = DATA_DIR / "tooltalk_cases.json"
AGENTDOJO_CASES_PATH = DATA_DIR / "agentdojo_cases.json"


def _case_identity(case):
    metadata = case.get("metadata", {})
    parts = [
        case["source"],
        case["category"],
        str(case["label"]),
        case["intent"],
        case["action"],
        str(metadata.get("suite", "")),
        str(metadata.get("source_split", case.get("split", ""))),
    ]
    return "||".join(parts)


def _stable_digest(case):
    return hashlib.sha256(_case_identity(case).encode("utf-8")).hexdigest()


def _compute_split_counts(group_size):
    if group_size <= 0:
        return {"train": 0, "dev": 0, "test": 0}
    if group_size == 1:
        return {"train": 1, "dev": 0, "test": 0}
    if group_size == 2:
        return {"train": 1, "dev": 0, "test": 1}

    dev_count = max(1, int(round(group_size * 0.15)))
    test_count = max(1, int(round(group_size * 0.15)))

    while dev_count + test_count > group_size - 1:
        if test_count >= dev_count and test_count > 1:
            test_count -= 1
        elif dev_count > 1:
            dev_count -= 1
        else:
            break

    train_count = group_size - dev_count - test_count
    return {"train": train_count, "dev": dev_count, "test": test_count}


def assign_eval_splits(cases):
    grouped_cases = defaultdict(list)
    for index, case in enumerate(cases):
        grouped_cases[(case["source"], case["label"])].append((index, case))

    split_lookup = {}
    for group_items in grouped_cases.values():
        ordered_items = sorted(group_items, key=lambda item: _stable_digest(item[1]))
        counts = _compute_split_counts(len(ordered_items))

        for offset, (index, _) in enumerate(ordered_items):
            if offset < counts["train"]:
                split_lookup[index] = "train"
            elif offset < counts["train"] + counts["dev"]:
                split_lookup[index] = "dev"
            else:
                split_lookup[index] = "test"

    assigned_cases = []
    for index, case in enumerate(cases):
        metadata = dict(case.get("metadata", {}))
        metadata.setdefault("source_split", case.get("split", "unspecified"))
        assigned_cases.append(
            {
                **case,
                "split": split_lookup[index],
                "metadata": metadata,
            }
        )

    return assigned_cases


def _print_dataset_summary(cases, include_tooltalk, include_agentdojo):
    source_counts = Counter(case["source"] for case in cases)
    split_counts = Counter(case["split"] for case in cases)

    print(f"Manual: {source_counts.get('manual', 0)}")
    print(f"Generated: {source_counts.get('generated', 0)}")
    if include_tooltalk:
        print(f"ToolTalk: {source_counts.get('tooltalk', 0)}")
    if include_agentdojo:
        print(f"AgentDojo: {source_counts.get('agentdojo', 0)}")
    print(f"Total: {len(cases)}")
    print(
        "Assigned splits -> "
        f"train: {split_counts.get('train', 0)}, "
        f"dev: {split_counts.get('dev', 0)}, "
        f"test: {split_counts.get('test', 0)}"
    )

def load_manual():
    from src.data.test_cases import test_cases
    return [
        standardize_case(
            {
                "intent": intent,
                "action": action,
                "label": label,
                "source": "manual",
                "category": "manual",
                "split": "local",
            },
            default_source="manual",
            default_category="manual",
            default_split="local",
        )
        for intent, action, label in test_cases
    ]


def load_generated():
    with (DATA_DIR / "generated_cases.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return [
        standardize_case(
            {
                **item,
                "source": "generated",
                "split": item.get("split", "local"),
            },
            default_source="generated",
            default_category=item.get("category", "generated"),
            default_split=item.get("split", "local"),
        )
        for item in data
    ]


def load_tooltalk():
    if not TOOLTALK_CASES_PATH.exists():
        return []

    with TOOLTALK_CASES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    return [
        standardize_case(
            item,
            default_source="tooltalk",
            default_category=item.get("category", "tooltalk_aligned"),
            default_split=item.get("split", "benchmark"),
        )
        for item in data
    ]


def load_agentdojo():
    if not AGENTDOJO_CASES_PATH.exists():
        return []

    with AGENTDOJO_CASES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    return [
        standardize_case(
            item,
            default_source="agentdojo",
            default_category=item.get("category", "agentdojo_injection"),
            default_split=item.get("split", "benchmark"),
        )
        for item in data
    ]


def get_all_cases(include_tooltalk=False, include_agentdojo=False, split=None, return_records=False):
    manual = load_manual()
    generated = load_generated()
    tooltalk = load_tooltalk() if include_tooltalk else []
    agentdojo = load_agentdojo() if include_agentdojo else []

    all_data = assign_eval_splits(manual + generated + tooltalk + agentdojo)
    if split is not None:
        if split not in EVAL_SPLITS:
            raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split}")
        all_data = [case for case in all_data if case["split"] == split]

    _print_dataset_summary(all_data, include_tooltalk, include_agentdojo)

    if return_records:
        return all_data

    return [
        (case["intent"], case["action"], case["label"], case.get("category", "unknown"))
        for case in all_data
    ]
