import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from src.data.schema import DATA_DIR, standardize_case


TOOLTALK_API_ROOT = "https://api.github.com/repos/microsoft/ToolTalk/contents/data"
SPLIT_CONFIG = {
    "easy": {
        "remote_dir": "easy",
        "label": 1,
        "category": "tooltalk_aligned",
    },
    "hard": {
        "remote_dir": "tooltalk",
        "label": 1,
        "category": "tooltalk_aligned",
    },
}
HIDDEN_PARAMETER_FRAGMENTS = ("password", "token")


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "GuardMCP-ToolTalk-Adapter"})
    with urlopen(request) as response:
        return json.load(response)


def list_split_files(split_name):
    remote_dir = SPLIT_CONFIG[split_name]["remote_dir"]
    contents_url = f"{TOOLTALK_API_ROOT}/{remote_dir}"
    items = fetch_json(contents_url)
    return [item for item in items if item.get("type") == "file" and item.get("download_url")]


def sanitize_text(text, sensitive_values):
    sanitized = str(text)
    for sensitive_value in sensitive_values:
        if sensitive_value:
            sanitized = sanitized.replace(sensitive_value, "[REDACTED]")
    return sanitized


def collect_sensitive_values(record):
    sensitive_values = set()

    for container in (record.get("user", {}), record.get("metadata", {})):
        for key, value in container.items():
            if any(fragment in key.lower() for fragment in HIDDEN_PARAMETER_FRAGMENTS):
                sensitive_values.add(str(value))

    for turn in record.get("conversation", []):
        for api_call in turn.get("apis", []):
            parameters = api_call.get("request", {}).get("parameters", {})
            for key, value in parameters.items():
                if any(fragment in key.lower() for fragment in HIDDEN_PARAMETER_FRAGMENTS):
                    sensitive_values.add(str(value))

    return sensitive_values


def format_parameter_value(value, sensitive_values):
    if isinstance(value, list):
        return ", ".join(sanitize_text(item, sensitive_values) for item in value)
    if isinstance(value, dict):
        return sanitize_text(json.dumps(value, sort_keys=True), sensitive_values)
    if value == "":
        return "empty"
    return sanitize_text(value, sensitive_values)


def verbalize_api_call(api_call, sensitive_values):
    request = api_call["request"]
    api_name = request["api_name"]
    parameters = request.get("parameters", {})

    visible_parameters = []
    for key, value in parameters.items():
        if any(fragment in key.lower() for fragment in HIDDEN_PARAMETER_FRAGMENTS):
            continue
        visible_parameters.append(f"{key}={format_parameter_value(value, sensitive_values)}")

    if visible_parameters:
        return f"Call {api_name} with " + "; ".join(visible_parameters)

    return f"Call {api_name}"


def conversation_to_case(record, split_name):
    conversation = record["conversation"]
    sensitive_values = collect_sensitive_values(record)
    api_turn_positions = [
        index
        for index, turn in enumerate(conversation)
        if turn.get("role") == "assistant" and turn.get("apis")
    ]

    if not api_turn_positions:
        return None

    last_api_turn = api_turn_positions[-1]
    user_messages = [
        sanitize_text(turn["text"].strip(), sensitive_values)
        for turn in conversation[:last_api_turn]
        if turn.get("role") == "user" and turn.get("text")
    ]
    api_actions = []

    for position in api_turn_positions:
        turn = conversation[position]
        for api_call in turn.get("apis", []):
            api_actions.append(verbalize_api_call(api_call, sensitive_values))

    if not user_messages or not api_actions:
        return None

    split_config = SPLIT_CONFIG[split_name]
    return standardize_case(
        {
            "intent": " ".join(user_messages),
            "action": " Then ".join(api_actions),
            "label": split_config["label"],
            "category": split_config["category"],
            "source": "tooltalk",
            "split": split_name,
            "metadata": {
                "benchmark": "ToolTalk",
                "conversation_id": record.get("conversation_id"),
                "name": record.get("name"),
                "scenario": record.get("scenario"),
                "suites_used": record.get("suites_used", []),
                "apis_used": record.get("apis_used", []),
            },
        },
        default_source="tooltalk",
        default_category=split_config["category"],
        default_split=split_name,
    )


def build_tooltalk_cases(splits):
    cases = []

    for split_name in splits:
        for item in list_split_files(split_name):
            record = fetch_json(item["download_url"])
            case = conversation_to_case(record, split_name)
            if case:
                cases.append(case)

    return cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ToolTalk benchmark conversations and convert them into GuardMCP cases."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_CONFIG),
        default=["easy", "hard"],
        help="ToolTalk splits to download and convert.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "tooltalk_cases.json"),
        help="Output JSON path for standardized GuardMCP cases.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = build_tooltalk_cases(args.splits)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2)

    print(f"Saved {len(cases)} ToolTalk cases to {output_path}")


if __name__ == "__main__":
    main()
