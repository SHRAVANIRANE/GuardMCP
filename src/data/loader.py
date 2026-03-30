import json

def load_manual():
    from src.data.test_cases import test_cases
    return [{"intent": i, "action": a, "label": l, "source": "manual"} for i, a, l in test_cases]


def load_generated():
    with open("src/data/generated_cases.json") as f:
        data = json.load(f)
    for d in data:
        d["source"] = "generated"
    return data


def get_all_cases():
    manual = load_manual()
    generated = load_generated()

    all_data = manual + generated

    print(f"Manual: {len(manual)}")
    print(f"Generated: {len(generated)}")
    print(f"Total: {len(all_data)}")

    return [(d["intent"], d["action"], d["label"], d.get("category", "manual")) for d in all_data]