import argparse
import csv

from config import (
    BEST_THRESHOLDS_PATH,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_DIRECTIONAL_THRESHOLD,
    DEFAULT_MODEL_NAME,
)
from src.alignment.cosine_baseline import CosineBaseline
from src.alignment.directional import DirectionalAlignment
from src.embeddings.embedder import Embedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="GuardMCP CLI for checking whether a proposed action stays aligned with user intent."
    )
    parser.add_argument("--intent", help="User intent text to evaluate.")
    parser.add_argument("--action", help="Proposed agent action text to evaluate.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model to use for embeddings.",
    )
    parser.add_argument(
        "--directional-threshold",
        type=float,
        help="Override the directional rejection threshold.",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        help="Override the cosine similarity threshold.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run a prompt loop instead of a single one-shot evaluation.",
    )
    return parser.parse_args()


def load_thresholds():
    thresholds = {
        "directional": DEFAULT_DIRECTIONAL_THRESHOLD,
        "cosine": DEFAULT_COSINE_THRESHOLD,
    }
    source = "config defaults"

    if BEST_THRESHOLDS_PATH.exists():
        with BEST_THRESHOLDS_PATH.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                method = row.get("method")
                threshold = row.get("threshold")
                if method in thresholds and threshold:
                    thresholds[method] = float(threshold)
        source = str(BEST_THRESHOLDS_PATH)

    return thresholds, source


def evaluate_pair(embedder, intent, action, directional_threshold, cosine_threshold):
    intent_vector = embedder.encode(intent)
    action_vector = embedder.encode(action)

    directional = DirectionalAlignment(epsilon=directional_threshold)
    cosine = CosineBaseline(threshold=cosine_threshold)

    directional_result = directional.check(intent_vector, action_vector)
    cosine_result = cosine.check(intent_vector, action_vector)
    final_verdict = "ALLOW" if directional_result["allow"] else "BLOCK"

    return {
        "final_verdict": final_verdict,
        "directional": directional_result,
        "cosine": cosine_result,
        "directional_threshold": directional_threshold,
        "cosine_threshold": cosine_threshold,
    }


def print_result(intent, action, result, threshold_source):
    print("\nGuardMCP Demo")
    print(f"Threshold source: {threshold_source}")
    print(f"Intent: {intent}")
    print(f"Action: {action}")
    print(f"GuardMCP verdict: {result['final_verdict']}")
    print(
        "Directional -> "
        f"rejection={result['directional']['rejection_magnitude']:.6f}, "
        f"threshold={result['directional_threshold']:.6f}, "
        f"decision={'ALLOW' if result['directional']['allow'] else 'BLOCK'}"
    )
    print(
        "Cosine baseline -> "
        f"similarity={result['cosine']['similarity']:.6f}, "
        f"threshold={result['cosine_threshold']:.6f}, "
        f"decision={'ALLOW' if result['cosine']['allow'] else 'BLOCK'}"
    )

    if result["final_verdict"] == "BLOCK":
        print("Explanation: the action contains extra semantic content beyond the allowed directional threshold.")
    else:
        print("Explanation: the action stays within the allowed directional threshold for the current configuration.")


def run_single_demo(args, embedder, thresholds, threshold_source):
    if bool(args.intent) != bool(args.action):
        raise ValueError("Provide both --intent and --action together, or use --interactive.")

    intent = args.intent.strip()
    action = args.action.strip()
    result = evaluate_pair(
        embedder,
        intent,
        action,
        directional_threshold=args.directional_threshold or thresholds["directional"],
        cosine_threshold=args.cosine_threshold or thresholds["cosine"],
    )
    print_result(intent, action, result, threshold_source)


def run_interactive_demo(embedder, thresholds, threshold_source, args):
    directional_threshold = args.directional_threshold or thresholds["directional"]
    cosine_threshold = args.cosine_threshold or thresholds["cosine"]

    print("GuardMCP interactive mode. Press Enter on an empty intent to exit.")
    print(f"Threshold source: {threshold_source}")
    print(f"Directional threshold: {directional_threshold:.6f}")
    print(f"Cosine threshold: {cosine_threshold:.6f}")

    while True:
        intent = input("\nIntent: ").strip()
        if not intent:
            print("Exiting GuardMCP demo.")
            break

        action = input("Action: ").strip()
        if not action:
            print("Action cannot be empty. Try again.")
            continue

        result = evaluate_pair(
            embedder,
            intent,
            action,
            directional_threshold=directional_threshold,
            cosine_threshold=cosine_threshold,
        )
        print_result(intent, action, result, threshold_source)


def main():
    args = parse_args()
    thresholds, threshold_source = load_thresholds()
    embedder = Embedder(model_name=args.model)

    if args.interactive:
        run_interactive_demo(embedder, thresholds, threshold_source, args)
        return

    if not args.intent or not args.action:
        print("Use --intent and --action for one-shot mode, or pass --interactive.")
        return

    run_single_demo(args, embedder, thresholds, threshold_source)


if __name__ == "__main__":
    main()
