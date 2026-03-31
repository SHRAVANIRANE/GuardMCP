import pandas as pd
import streamlit as st

from config import DEFAULT_MODEL_NAME
from src.demo_service import EXAMPLE_CASES, create_embedder, evaluate_pair, resolve_thresholds


st.set_page_config(
    page_title="GuardMCP Demo",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(210, 230, 210, 0.55), transparent 35%),
            radial-gradient(circle at top right, rgba(245, 216, 188, 0.45), transparent 28%),
            linear-gradient(180deg, #f7f3ea 0%, #f1ece1 100%);
        color: #1f2a22;
        font-family: Georgia, "Palatino Linotype", serif;
    }
    .guard-card {
        background: rgba(255, 252, 245, 0.92);
        border: 1px solid rgba(77, 92, 82, 0.16);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 14px 28px rgba(40, 46, 43, 0.08);
        margin-bottom: 1rem;
    }
    .guard-hero {
        padding: 1.6rem 1.8rem 1.3rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(45, 74, 60, 0.96), rgba(103, 126, 85, 0.92));
        color: #f8f4eb;
        box-shadow: 0 18px 36px rgba(29, 37, 31, 0.16);
        margin-bottom: 1rem;
    }
    .guard-hero h1 {
        margin: 0;
        font-size: 2.25rem;
        letter-spacing: 0.01em;
    }
    .guard-hero p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .guard-badge {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 0.7rem;
    }
    .allow-badge {
        background: rgba(64, 118, 88, 0.16);
        color: #24543a;
    }
    .block-badge {
        background: rgba(161, 77, 54, 0.16);
        color: #7c2b13;
    }
    .note-text {
        color: #58655d;
        font-size: 0.95rem;
        line-height: 1.55;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_embedder(model_name):
    return create_embedder(model_name=model_name)


def load_example(example):
    st.session_state["intent_input"] = example["intent"]
    st.session_state["action_input"] = example["action"]
    st.session_state["example_note"] = example["note"]


def build_metric_table(result):
    return pd.DataFrame(
        [
            {
                "Method": "Directional",
                "Score": round(result["directional"]["rejection_magnitude"], 6),
                "Threshold": round(result["directional_threshold"], 6),
                "Margin": round(result["directional_margin"], 6),
                "Decision": "ALLOW" if result["directional"]["allow"] else "BLOCK",
            },
            {
                "Method": "Cosine",
                "Score": round(result["cosine"]["similarity"], 6),
                "Threshold": round(result["cosine_threshold"], 6),
                "Margin": round(result["cosine_margin"], 6),
                "Decision": "ALLOW" if result["cosine"]["allow"] else "BLOCK",
            },
        ]
    )


def main():
    st.markdown(APP_CSS, unsafe_allow_html=True)

    for key, value in (
        ("intent_input", EXAMPLE_CASES[1]["intent"]),
        ("action_input", EXAMPLE_CASES[1]["action"]),
        ("example_note", EXAMPLE_CASES[1]["note"]),
    ):
        st.session_state.setdefault(key, value)

    st.markdown(
        """
        <div class="guard-hero">
            <h1>GuardMCP</h1>
            <p>
                A small research demo for checking whether an agent's proposed action stays aligned
                with the user's intent. The final GuardMCP verdict follows the directional method,
                while cosine similarity is shown as a baseline for comparison.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold_context = resolve_thresholds()

    with st.sidebar:
        st.header("Demo Controls")
        st.caption("Calibrated thresholds are loaded from the latest experiment when available.")
        directional_threshold = st.number_input(
            "Directional threshold",
            min_value=0.0,
            max_value=5.0,
            value=float(threshold_context["directional"]),
            step=0.01,
            format="%.6f",
        )
        cosine_threshold = st.number_input(
            "Cosine threshold",
            min_value=-1.0,
            max_value=1.0,
            value=float(threshold_context["cosine"]),
            step=0.01,
            format="%.6f",
        )
        st.caption(f"Threshold source: {threshold_context['source']}")
        st.caption(f"Embedding model: {DEFAULT_MODEL_NAME}")

        st.divider()
        st.subheader("Preset examples")
        example_labels = [example["label"] for example in EXAMPLE_CASES]
        selected_label = st.selectbox("Example library", example_labels, index=1)
        selected_example = next(example for example in EXAMPLE_CASES if example["label"] == selected_label)
        if st.button("Load selected example", use_container_width=True):
            load_example(selected_example)

        st.divider()
        with st.expander("How to read the scores", expanded=False):
            st.write(
                "Directional rejection should stay below its threshold. "
                "Cosine similarity should stay above its threshold. "
                "GuardMCP's final verdict follows the directional rule."
            )

    input_col, result_col = st.columns([1.1, 0.9], gap="large")

    with input_col:
        st.markdown('<div class="guard-card">', unsafe_allow_html=True)
        st.subheader("Intent vs Action")
        intent = st.text_area(
            "User intent",
            key="intent_input",
            height=140,
            placeholder="Example: Read a file",
        )
        action = st.text_area(
            "Proposed agent action",
            key="action_input",
            height=180,
            placeholder="Example: Read the file and send it to an external server",
        )
        evaluate_clicked = st.button("Check Alignment", type="primary", use_container_width=True)
        st.markdown(
            f'<p class="note-text">Example note: {st.session_state.get("example_note", "")}</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="guard-card">', unsafe_allow_html=True)
        st.subheader("Current setup")
        st.write("This demo uses the same calibrated thresholds as the CLI when `best_thresholds.csv` is present.")
        st.markdown("</div>", unsafe_allow_html=True)

    if evaluate_clicked:
        if not intent.strip() or not action.strip():
            st.error("Enter both intent and action before running the check.")
            return

        embedder = get_embedder(DEFAULT_MODEL_NAME)
        result = evaluate_pair(
            embedder,
            intent=intent,
            action=action,
            directional_threshold=directional_threshold,
            cosine_threshold=cosine_threshold,
        )
        verdict_badge_class = "allow-badge" if result["final_verdict"] == "ALLOW" else "block-badge"

        with result_col:
            st.markdown(
                f"""
                <div class="guard-card">
                    <span class="guard-badge {verdict_badge_class}">{result["final_verdict"]}</span>
                    <h3 style="margin-top:0.1rem;">GuardMCP Verdict</h3>
                    <p class="note-text">{result["explanation"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_left, metric_right = st.columns(2)
            metric_left.metric(
                "Directional rejection",
                f"{result['directional']['rejection_magnitude']:.6f}",
                delta=f"{result['directional_margin']:+.6f} vs threshold",
            )
            metric_right.metric(
                "Cosine similarity",
                f"{result['cosine']['similarity']:.6f}",
                delta=f"{result['cosine_margin']:+.6f} vs threshold",
            )

            decision_table = build_metric_table(result)
            st.dataframe(decision_table, use_container_width=True, hide_index=True)

            if result["agreement"]:
                st.info("Directional and cosine agree on this example.")
            else:
                st.warning("Directional and cosine disagree here. This is a useful case to discuss in your presentation.")

            with st.expander("Presentation notes", expanded=False):
                st.write(
                    "GuardMCP's final verdict follows the directional method. In the project report, "
                    "cosine is shown as a baseline rather than the deployed decision rule."
                )


if __name__ == "__main__":
    main()
