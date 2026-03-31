import argparse

from config import DEFAULT_MODEL_NAME
from src.demo_service import create_embedder, evaluate_pair, resolve_thresholds


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

    print(f"Explanation: {result['explanation']}")


def run_single_demo(args, embedder, thresholds, threshold_source):
    if bool(args.intent) != bool(args.action):
        raise ValueError("Provide both --intent and --action together, or use --interactive.")

    intent = args.intent.strip()
    action = args.action.strip()
    result = evaluate_pair(
        embedder,
        intent,
        action,
        directional_threshold=(
            thresholds["directional"] if args.directional_threshold is None else args.directional_threshold
        ),
        cosine_threshold=(
            thresholds["cosine"] if args.cosine_threshold is None else args.cosine_threshold
        ),
    )
    print_result(intent, action, result, threshold_source)


def run_interactive_demo(embedder, thresholds, threshold_source, args):
    directional_threshold = (
        thresholds["directional"] if args.directional_threshold is None else args.directional_threshold
    )
    cosine_threshold = (
        thresholds["cosine"] if args.cosine_threshold is None else args.cosine_threshold
    )

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
    threshold_context = resolve_thresholds(
        directional_override=args.directional_threshold,
        cosine_override=args.cosine_threshold,
    )
    embedder = create_embedder(model_name=args.model)

    if args.interactive:
        run_interactive_demo(embedder, threshold_context, threshold_context["source"], args)
        return

    if not args.intent or not args.action:
        print("Use --intent and --action for one-shot mode, or pass --interactive.")
        return

    run_single_demo(embedder=embedder, args=args, thresholds=threshold_context, threshold_source=threshold_context["source"])


if __name__ == "__main__":
    main()
