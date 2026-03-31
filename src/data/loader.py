import json

from src.data.schema import DATA_DIR, standardize_case


TOOLTALK_CASES_PATH = DATA_DIR / "tooltalk_cases.json"
AGENTDOJO_CASES_PATH = DATA_DIR / "agentdojo_cases.json"

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


def get_all_cases(include_tooltalk=False, include_agentdojo=False):
    manual = load_manual()
    generated = load_generated()
    tooltalk = load_tooltalk() if include_tooltalk else []
    agentdojo = load_agentdojo() if include_agentdojo else []

    all_data = manual + generated + tooltalk + agentdojo

    print(f"Manual: {len(manual)}")
    print(f"Generated: {len(generated)}")
    if include_tooltalk:
        print(f"ToolTalk: {len(tooltalk)}")
    if include_agentdojo:
        print(f"AgentDojo: {len(agentdojo)}")
    print(f"Total: {len(all_data)}")

    return [(d["intent"], d["action"], d["label"], d.get("category", "manual")) for d in all_data]
