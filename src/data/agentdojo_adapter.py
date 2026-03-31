import argparse
import ast
import json
from pathlib import Path
from urllib.request import Request, urlopen

from src.data.schema import DATA_DIR, standardize_case


AGENTDOJO_RAW_ROOT = "https://raw.githubusercontent.com/ethz-spylab/agentdojo/main/src/agentdojo/default_suites"
DEFAULT_SUITES = ["workspace", "travel", "banking", "slack"]


def fetch_text(url):
    request = Request(url, headers={"User-Agent": "GuardMCP-AgentDojo-Adapter"})
    with urlopen(request) as response:
        return response.read().decode("utf-8")


def normalize_text(text):
    return " ".join(str(text).split())


def evaluate_expr(node, env):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return env.get(node.id, f"{{{node.id}}}")

    if isinstance(node, ast.Attribute):
        return node.attr

    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(evaluate_expr(value.value, env)))
        return "".join(parts)

    if isinstance(node, ast.List):
        return [evaluate_expr(element, env) for element in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(evaluate_expr(element, env) for element in node.elts)

    if isinstance(node, ast.Set):
        return {evaluate_expr(element, env) for element in node.elts}

    if isinstance(node, ast.Dict):
        return {
            evaluate_expr(key, env): evaluate_expr(value, env)
            for key, value in zip(node.keys, node.values)
        }

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = evaluate_expr(node.left, env)
        right = evaluate_expr(node.right, env)
        if isinstance(left, str) or isinstance(right, str):
            return f"{left}{right}"
        return left + right

    return None


def parse_task_definitions(source_code, field_name):
    module = ast.parse(source_code)
    tasks = []

    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue

        env = {}
        captured_text = None
        difficulty = "UNKNOWN"

        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue

            value = evaluate_expr(statement.value, env)
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    continue

                env[target.id] = value
                if target.id == field_name and value:
                    captured_text = normalize_text(value)
                if target.id == "DIFFICULTY" and value:
                    difficulty = str(value)

        if captured_text:
            tasks.append(
                {
                    "name": node.name,
                    "difficulty": difficulty,
                    field_name.lower(): captured_text,
                }
            )

    return tasks


def build_raw_url(version, suite, filename):
    return f"{AGENTDOJO_RAW_ROOT}/{version}/{suite}/{filename}"


def build_suite_cases(version, suite):
    user_source = fetch_text(build_raw_url(version, suite, "user_tasks.py"))
    injection_source = fetch_text(build_raw_url(version, suite, "injection_tasks.py"))

    user_tasks = parse_task_definitions(user_source, "PROMPT")
    injection_tasks = parse_task_definitions(injection_source, "GOAL")
    cases = []

    # AgentDojo security cases pair a benign task with a malicious injected goal.
    for user_task in user_tasks:
        for injection_task in injection_tasks:
            cases.append(
                standardize_case(
                    {
                        "intent": user_task["prompt"],
                        "action": injection_task["goal"],
                        "label": 0,
                        "category": "agentdojo_injection",
                        "source": "agentdojo",
                        "split": f"{version}_{suite}",
                        "metadata": {
                            "benchmark": "AgentDojo",
                            "version": version,
                            "suite": suite,
                            "user_task": user_task["name"],
                            "user_task_difficulty": user_task["difficulty"],
                            "injection_task": injection_task["name"],
                            "injection_task_difficulty": injection_task["difficulty"],
                        },
                    },
                    default_source="agentdojo",
                    default_category="agentdojo_injection",
                    default_split=f"{version}_{suite}",
                )
            )

    return cases


def build_agentdojo_cases(version, suites):
    cases = []

    for suite in suites:
        cases.extend(build_suite_cases(version, suite))

    return cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download AgentDojo task definitions and convert them into GuardMCP negative cases."
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="AgentDojo default suite version to use.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=DEFAULT_SUITES,
        choices=DEFAULT_SUITES,
        help="AgentDojo suites to adapt.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "agentdojo_cases.json"),
        help="Output JSON path for standardized GuardMCP cases.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = build_agentdojo_cases(args.version, args.suites)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2)

    print(f"Saved {len(cases)} AgentDojo cases to {output_path}")


if __name__ == "__main__":
    main()
