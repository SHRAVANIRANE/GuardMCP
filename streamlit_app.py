import pandas as pd
import streamlit as st

from config import DEFAULT_MODEL_NAME, RESULTS_DIR
from src.demo_service import EXAMPLE_CASES, create_embedder, evaluate_pair, resolve_thresholds

SUMMARY_PATH = RESULTS_DIR / "results_summary.csv"
OUTPUTS_PATH = RESULTS_DIR / "outputs.csv"
SOURCE_REPORT_PATH = RESULTS_DIR / "reports" / "by_source_metrics.csv"
SUITE_REPORT_PATH = RESULTS_DIR / "reports" / "by_suite_metrics.csv"
ATTACK_REPORT_PATH = RESULTS_DIR / "reports" / "by_attack_type_metrics.csv"
PLOT_PATH = RESULTS_DIR / "plots" / "precision_recall_graph.png"


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
            radial-gradient(circle at top left, rgba(136, 167, 120, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(196, 144, 103, 0.16), transparent 22%),
            linear-gradient(180deg, #151a18 0%, #0f1312 100%);
        color: #ecede7;
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
        color: #ecede7 !important;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(27, 36, 31, 0.98) 0%, rgba(18, 24, 21, 0.98) 100%);
        border-right: 1px solid rgba(214, 219, 207, 0.08);
    }
    .block-shell {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(236, 237, 231, 0.08);
        border-radius: 24px;
        box-shadow: 0 26px 50px rgba(0, 0, 0, 0.24);
        padding: 1.25rem 1.25rem 1.1rem 1.25rem;
        backdrop-filter: blur(14px);
    }
    .hero-panel {
        background:
            linear-gradient(135deg, rgba(42, 63, 53, 0.95), rgba(74, 91, 63, 0.92) 60%, rgba(128, 94, 61, 0.88));
        border: 1px solid rgba(255, 245, 225, 0.12);
        border-radius: 30px;
        padding: 1.8rem 1.9rem 1.4rem 1.9rem;
        box-shadow: 0 28px 54px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
        overflow: hidden;
        position: relative;
    }
    .hero-panel::after {
        content: "";
        position: absolute;
        inset: auto -80px -70px auto;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255, 237, 201, 0.16), transparent 62%);
    }
    .hero-kicker {
        display: inline-block;
        font-size: 0.76rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.34rem 0.68rem;
        border-radius: 999px;
        background: rgba(255, 248, 229, 0.12);
        color: #f8f2de !important;
        margin-bottom: 0.85rem;
        font-weight: 700;
    }
    .hero-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.7rem;
        line-height: 1.02;
        letter-spacing: -0.03em;
        margin: 0;
        color: #fbf7ed !important;
    }
    .hero-copy {
        margin-top: 0.8rem;
        max-width: 44rem;
        line-height: 1.65;
        color: rgba(250, 245, 232, 0.92) !important;
        font-size: 1rem;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }
    .hero-chip {
        padding: 0.48rem 0.78rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.88rem;
        color: #f8f5ea !important;
    }
    .card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(236, 237, 231, 0.07);
        border-radius: 22px;
        padding: 1.15rem 1.1rem;
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.25rem;
        margin-bottom: 0.18rem;
        color: #fbf7ed !important;
    }
    .card-copy {
        color: rgba(230, 232, 224, 0.78) !important;
        line-height: 1.55;
        font-size: 0.95rem;
    }
    .small-note {
        color: rgba(214, 219, 207, 0.72) !important;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .stat-tile {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(236, 237, 231, 0.07);
        border-radius: 20px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.8rem;
    }
    .stat-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        color: rgba(214, 219, 207, 0.68) !important;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .stat-value {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.8rem;
        color: #fbf7ed !important;
        margin-bottom: 0.18rem;
    }
    .stat-note {
        color: rgba(228, 231, 223, 0.76) !important;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .architecture-band {
        margin: 0.2rem 0 1.2rem 0;
        padding: 1.05rem 1.05rem 0.2rem 1.05rem;
        border-radius: 26px;
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(236, 237, 231, 0.06);
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.16);
    }
    .architecture-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.35rem;
        margin-bottom: 0.15rem;
        color: #fbf7ed !important;
    }
    .architecture-copy {
        color: rgba(228, 231, 223, 0.76) !important;
        line-height: 1.6;
        margin-bottom: 0.9rem;
    }
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin-bottom: 0.85rem;
    }
    .flow-node {
        position: relative;
        padding: 1rem 0.9rem 0.95rem 0.9rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
        border: 1px solid rgba(236, 237, 231, 0.07);
        min-height: 154px;
    }
    .flow-kicker {
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(215, 221, 210, 0.72) !important;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .flow-name {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.15rem;
        color: #fbf7ed !important;
        margin-bottom: 0.38rem;
    }
    .flow-body {
        color: rgba(228, 231, 223, 0.8) !important;
        line-height: 1.55;
        font-size: 0.92rem;
    }
    .flow-arrow {
        text-align: center;
        color: rgba(213, 159, 102, 0.88) !important;
        font-size: 1.35rem;
        margin-top: -0.1rem;
        margin-bottom: 0.85rem;
        letter-spacing: 0.6rem;
    }
    .verdict-panel {
        border-radius: 26px;
        padding: 1.3rem 1.25rem 1.15rem 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.09);
        box-shadow: 0 20px 42px rgba(0, 0, 0, 0.22);
        margin-bottom: 1rem;
    }
    .verdict-allow {
        background: linear-gradient(135deg, rgba(42, 77, 55, 0.92), rgba(67, 111, 84, 0.92));
    }
    .verdict-block {
        background: linear-gradient(135deg, rgba(103, 39, 29, 0.94), rgba(146, 69, 44, 0.92));
    }
    .verdict-tag {
        display: inline-block;
        padding: 0.34rem 0.76rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.13);
        color: #fff7ea !important;
        margin-bottom: 0.8rem;
    }
    .verdict-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2rem;
        margin: 0;
        color: #fff9f0 !important;
    }
    .verdict-copy {
        margin-top: 0.55rem;
        line-height: 1.6;
        color: rgba(255, 248, 236, 0.92) !important;
    }
    .agree-chip, .split-chip {
        display: inline-block;
        margin-top: 0.8rem;
        margin-right: 0.45rem;
        padding: 0.32rem 0.68rem;
        border-radius: 999px;
        font-size: 0.8rem;
        background: rgba(255, 255, 255, 0.14);
        color: #fff8ec !important;
    }
    .meter-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(236, 237, 231, 0.08);
        border-radius: 22px;
        padding: 1rem 1rem 0.95rem 1rem;
        margin-bottom: 1rem;
    }
    .meter-title {
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(225, 230, 219, 0.7) !important;
        margin-bottom: 0.55rem;
        font-weight: 700;
    }
    .meter-score {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2rem;
        margin-bottom: 0.18rem;
        color: #fbf7ed !important;
    }
    .meter-line {
        height: 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
        margin-top: 0.6rem;
        margin-bottom: 0.55rem;
    }
    .meter-fill {
        height: 100%;
        border-radius: 999px;
    }
    .meter-meta {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        font-size: 0.86rem;
        color: rgba(223, 226, 216, 0.78) !important;
    }
    .sidebar-note {
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(236, 237, 231, 0.06);
        margin-top: 0.75rem;
        color: rgba(225, 229, 220, 0.82) !important;
        line-height: 1.55;
        font-size: 0.92rem;
    }
    .caption-quiet {
        color: rgba(214, 219, 207, 0.76) !important;
    }
    div[data-baseweb="input"],
    div[data-baseweb="textarea"],
    div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] > div,
    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input {
        color: #f0f2eb !important;
        background: transparent !important;
    }
    .stTextArea textarea {
        min-height: 150px !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #d39a5b, #b86a41) !important;
        color: #1f130d !important;
        border-radius: 999px !important;
        font-weight: 800 !important;
        border: none !important;
        box-shadow: 0 12px 24px rgba(181, 103, 59, 0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 28px rgba(181, 103, 59, 0.28);
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        color: #f3f2ec !important;
    }
    .stDataFrame, .stTable, .stInfo, .stWarning, .stSuccess, .stError {
        color: #ecede7 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 999px;
        color: #e5e7e1 !important;
        padding: 0.55rem 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(214, 154, 93, 0.18) !important;
        border: 1px solid rgba(214, 154, 93, 0.36) !important;
    }
    @media (max-width: 1100px) {
        .flow-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 700px) {
        .flow-grid {
            grid-template-columns: repeat(1, minmax(0, 1fr));
        }
        .hero-title {
            font-size: 2.2rem;
        }
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
    st.session_state["selected_example"] = example["label"]


@st.cache_data(show_spinner=False)
def load_csv_if_present(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def load_benchmark_bundle():
    return {
        "summary": load_csv_if_present(SUMMARY_PATH),
        "outputs": load_csv_if_present(OUTPUTS_PATH),
        "source": load_csv_if_present(SOURCE_REPORT_PATH),
        "suite": load_csv_if_present(SUITE_REPORT_PATH),
        "attack_type": load_csv_if_present(ATTACK_REPORT_PATH),
    }


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


def build_meter_html(title, score, threshold, decision, better_direction):
    if better_direction == "lower":
        max_reference = max(threshold * 1.5, score, 1.0)
        fill_ratio = min(score / max_reference, 1.0)
        threshold_ratio = min(threshold / max_reference, 1.0)
        accent = "#8dd3a5" if decision == "ALLOW" else "#f3a37c"
        subtitle = "Lower is safer"
    else:
        fill_ratio = min(max((score + 1.0) / 2.0, 0.0), 1.0)
        threshold_ratio = min(max((threshold + 1.0) / 2.0, 0.0), 1.0)
        accent = "#89c7ff" if decision == "ALLOW" else "#f4c26b"
        subtitle = "Higher is safer"

    marker_percent = max(min(threshold_ratio * 100.0, 100.0), 0.0)
    fill_percent = max(min(fill_ratio * 100.0, 100.0), 0.0)

    return f"""
    <div class="meter-card">
        <div class="meter-title">{title}</div>
        <div class="meter-score">{score:.6f}</div>
        <div class="small-note">{subtitle}</div>
        <div class="meter-line" style="position: relative;">
            <div class="meter-fill" style="width: {fill_percent:.2f}%; background: {accent};"></div>
            <div style="position:absolute; top:-3px; left: calc({marker_percent:.2f}% - 1px); width: 2px; height: 18px; background: rgba(255,255,255,0.92);"></div>
        </div>
        <div class="meter-meta">
            <span>Threshold: {threshold:.6f}</span>
            <span>Decision: {decision}</span>
        </div>
    </div>
    """


def render_intro_cards(threshold_context):
    info_col, note_col, rule_col = st.columns(3, gap="medium")

    with info_col:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Runtime Goal</div>
                <div class="card-copy">
                    Compare the user’s intent against the proposed action and decide whether
                    the action stays inside the allowed semantic scope.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with note_col:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Calibrated Setup</div>
                <div class="card-copy">
                    Directional threshold: <strong>{threshold_context['directional']:.6f}</strong><br>
                    Cosine threshold: <strong>{threshold_context['cosine']:.6f}</strong><br>
                    Source: <span class="caption-quiet">{threshold_context['source']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rule_col:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Final Rule</div>
                <div class="card-copy">
                    GuardMCP’s final verdict follows the directional method.
                    Cosine is shown as a baseline for comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_architecture_section():
    st.markdown(
        """
        <div class="architecture-band">
            <div class="architecture-title">How GuardMCP Works</div>
            <div class="architecture-copy">
                This section turns the codebase
                into a simple mental model: compare intent and action, measure alignment, then make a
                runtime decision.
            </div>
            <div class="flow-grid">
                <div class="flow-node">
                    <div class="flow-kicker">Step 1</div>
                    <div class="flow-name">Intent + Action</div>
                    <div class="flow-body">
                        The user intent and the agent’s proposed action are entered as natural-language
                        text. GuardMCP treats these as the two semantic objects to compare.
                    </div>
                </div>
                <div class="flow-node">
                    <div class="flow-kicker">Step 2</div>
                    <div class="flow-name">Embedding Layer</div>
                    <div class="flow-body">
                        Both texts are embedded with a sentence-transformer model so they can be compared
                        as vectors instead of only raw strings.
                    </div>
                </div>
                <div class="flow-node">
                    <div class="flow-kicker">Step 3</div>
                    <div class="flow-name">Two Safety Checks</div>
                    <div class="flow-body">
                        GuardMCP computes directional rejection as the main decision rule and cosine
                        similarity as the baseline for side-by-side comparison.
                    </div>
                </div>
                <div class="flow-node">
                    <div class="flow-kicker">Step 4</div>
                    <div class="flow-name">Verdict + Reports</div>
                    <div class="flow-body">
                        The app returns an allow or block verdict, while the experiment pipeline saves
                        threshold tuning, grouped metrics, and benchmark-backed evaluation tables.
                    </div>
                </div>
            </div>
            <div class="flow-arrow">→ → →</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_panel(result):
    verdict_class = "verdict-allow" if result["final_verdict"] == "ALLOW" else "verdict-block"
    agreement_label = "Methods agree" if result["agreement"] else "Methods disagree"
    disagreement_label = (
        "Interesting comparison case"
        if not result["agreement"]
        else "Consistent decision"
    )

    st.markdown(
        f"""
        <div class="verdict-panel {verdict_class}">
            <div class="verdict-tag">GuardMCP Verdict</div>
            <h2 class="verdict-title">{result["final_verdict"]}</h2>
            <div class="verdict-copy">{result["explanation"]}</div>
            <span class="agree-chip">{agreement_label}</span>
            <span class="split-chip">{disagreement_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    meter_left, meter_right = st.columns(2, gap="medium")
    with meter_left:
        st.markdown(
            build_meter_html(
                title="Directional Rejection",
                score=float(result["directional"]["rejection_magnitude"]),
                threshold=float(result["directional_threshold"]),
                decision="ALLOW" if result["directional"]["allow"] else "BLOCK",
                better_direction="lower",
            ),
            unsafe_allow_html=True,
        )
    with meter_right:
        st.markdown(
            build_meter_html(
                title="Cosine Similarity",
                score=float(result["cosine"]["similarity"]),
                threshold=float(result["cosine_threshold"]),
                decision="ALLOW" if result["cosine"]["allow"] else "BLOCK",
                better_direction="higher",
            ),
            unsafe_allow_html=True,
        )

    detail_left, detail_right = st.columns([1.1, 0.9], gap="medium")
    with detail_left:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Method Comparison</div>
                <div class="card-copy">
                    The table below is helpful in a presentation because it shows the raw score,
                    the threshold, and the margin to the decision boundary for both methods.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(build_metric_table(result), use_container_width=True, hide_index=True)

    with detail_right:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">How To Explain It</div>
                <div class="card-copy">
                    If the directional rejection goes above its threshold, GuardMCP blocks the
                    action because it detects extra semantic behavior beyond the intended scope.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["agreement"]:
            st.info("Directional and cosine agree on this example.")
        else:
            st.warning("Directional and cosine disagree here. This is a strong example to discuss in interviews.")

        with st.expander("Presentation notes", expanded=False):
            st.write(
                "Start with the verdict, then explain the directional score, then mention cosine "
                "as the baseline. If this is a limitation case, be honest about it and connect it "
                "to future work such as stronger calibration or richer data."
            )


def render_placeholder():
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Ready For A Live Check</div>
            <div class="card-copy">
                Enter an intent and a proposed action, then press <strong>Check Alignment</strong>.
                The right side will show a GuardMCP verdict, method comparison, and a short explanation
                you can reuse in a presentation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_tile(label, value, note):
    st.markdown(
        f"""
        <div class="stat-tile">
            <div class="stat-kicker">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_benchmark_section():
    bundle = load_benchmark_bundle()
    summary_df = bundle["summary"]
    outputs_df = bundle["outputs"]

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Benchmark Snapshot</div>
            <div class="card-copy">
                These panels turn the saved experiment artifacts into a presentation-friendly dashboard.
                They summarize the calibrated test metrics, the grouped report slices, and the latest
                evaluation plot generated by the experiment pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary_df is None or outputs_df is None:
        st.info(
            "Benchmark artifacts are not available yet. Run "
            "`venv\\Scripts\\python.exe experiments/run_experiments.py --include-tooltalk --include-agentdojo` "
            "to generate them."
        )
        return

    test_rows = summary_df[summary_df["stage"] == "test_best"].copy()
    split_counts = outputs_df["split"].value_counts().to_dict()

    total_col, train_col, dev_col, test_col = st.columns(4, gap="medium")
    with total_col:
        render_stat_tile("Dataset size", str(len(outputs_df)), "Rows across local data and adapted public benchmarks.")
    with train_col:
        render_stat_tile("Train split", str(split_counts.get("train", 0)), "Reserved for future learning or ablation work.")
    with dev_col:
        render_stat_tile("Dev split", str(split_counts.get("dev", 0)), "Used to choose the best directional and cosine thresholds.")
    with test_col:
        render_stat_tile("Test split", str(split_counts.get("test", 0)), "Used for the final unseen evaluation reported below.")

    metrics_tab, grouped_tab, plot_tab = st.tabs(["Final Metrics", "Grouped Reports", "Saved Plot"])

    with metrics_tab:
        if test_rows.empty:
            st.warning("No final test rows were found in results_summary.csv.")
        else:
            metric_cols = st.columns(len(test_rows), gap="medium")
            for column, row in zip(metric_cols, test_rows.itertuples()):
                with column:
                    render_stat_tile(
                        f"{row.method.title()} F1",
                        f"{row.f1:.2f}",
                        (
                            f"Accuracy {row.accuracy:.2f}, precision {row.precision:.2f}, "
                            f"recall {row.recall:.2f}, threshold {row.threshold:.6f}"
                        ),
                    )
            display_df = test_rows[
                ["method", "threshold", "accuracy", "precision", "recall", "f1", "support"]
            ].copy()
            for column_name in ["threshold", "accuracy", "precision", "recall", "f1"]:
                display_df[column_name] = display_df[column_name].map(lambda value: round(float(value), 4))
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with grouped_tab:
        report_lookup = {
            "By Source": bundle["source"],
            "By Suite": bundle["suite"],
            "By Attack Type": bundle["attack_type"],
        }
        selected_report_name = st.selectbox("Grouped report", list(report_lookup), index=0)
        selected_report_df = report_lookup[selected_report_name]

        if selected_report_df is None or selected_report_df.empty:
            st.warning(f"{selected_report_name} report is not available yet.")
        else:
            available_methods = list(selected_report_df["method"].dropna().unique())
            method_filter = st.selectbox("Method view", ["All", *available_methods], index=0)
            visible_df = selected_report_df.copy()
            if method_filter != "All":
                visible_df = visible_df[visible_df["method"] == method_filter].copy()

            sortable_df = visible_df.sort_values(["support", "accuracy"], ascending=[False, True]).reset_index(drop=True)
            hardest_slice = sortable_df.iloc[0]
            render_stat_tile(
                "Hardest slice",
                str(hardest_slice["group_value"]),
                (
                    f"{hardest_slice['method'].title()} accuracy {float(hardest_slice['accuracy']):.2f} "
                    f"on {int(hardest_slice['support'])} rows."
                ),
            )

            display_columns = [
                "group_value",
                "method",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "allow_rate",
                "block_rate",
                "support",
            ]
            formatted_df = sortable_df[display_columns].copy()
            for column_name in ["accuracy", "precision", "recall", "f1", "allow_rate", "block_rate"]:
                formatted_df[column_name] = formatted_df[column_name].map(lambda value: round(float(value), 4))
            st.dataframe(formatted_df, use_container_width=True, hide_index=True)
            st.caption(
                "These grouped tables come from the final test split after applying the dev-selected thresholds."
            )

    with plot_tab:
        if PLOT_PATH.exists():
            st.image(str(PLOT_PATH), use_container_width=True, caption="Directional vs cosine dev threshold sweep")
            st.caption(
                "The saved plot visualizes the dev-sweep metrics used to select the final thresholds."
            )
        else:
            st.warning("Saved plot not found. Run the plotting script after experiments to regenerate it.")


def main():
    st.markdown(APP_CSS, unsafe_allow_html=True)

    default_example = EXAMPLE_CASES[1]
    for key, value in (
        ("intent_input", default_example["intent"]),
        ("action_input", default_example["action"]),
        ("example_note", default_example["note"]),
        ("selected_example", default_example["label"]),
    ):
        st.session_state.setdefault(key, value)

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Semantic Runtime Guardrails</div>
            <h1 class="hero-title">GuardMCP</h1>
            <div class="hero-copy">
                A compact research demo for checking whether a tool-using agent stays faithful to the
                user’s intent or quietly introduces extra behavior. The interface is designed for live
                walkthroughs, resume demos, and college presentations.
            </div>
            <div class="chip-row">
                <span class="hero-chip">721 benchmark-backed examples</span>
                <span class="hero-chip">Directional verdict + cosine baseline</span>
                <span class="hero-chip">Dev-calibrated thresholds</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold_context = resolve_thresholds()
    render_intro_cards(threshold_context)
    render_architecture_section()

    with st.sidebar:
        st.header("Demo Controls")
        st.caption("Use the calibrated thresholds from the latest experiment, or override them manually.")

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
        st.markdown(
            f"""
            <div class="sidebar-note">
                Threshold source: <strong>{threshold_context["source"]}</strong><br>
                Embedding model: <strong>{DEFAULT_MODEL_NAME}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Preset examples")
        example_labels = [example["label"] for example in EXAMPLE_CASES]
        selected_label = st.selectbox(
            "Example library",
            example_labels,
            index=example_labels.index(st.session_state["selected_example"]),
        )
        selected_example = next(example for example in EXAMPLE_CASES if example["label"] == selected_label)
        if st.button("Load selected example", use_container_width=True):
            load_example(selected_example)

        st.markdown(
            f"""
            <div class="sidebar-note">
                <strong>Current note</strong><br>
                {st.session_state.get("example_note", "")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        with st.expander("How to read the scores", expanded=False):
            st.write(
                "Directional rejection should stay below its threshold. Cosine similarity should stay "
                "above its threshold. GuardMCP's final verdict always follows the directional rule."
            )

    input_col, result_col = st.columns([1.05, 0.95], gap="large")

    with input_col:
        with st.form("guardmcp_form", clear_on_submit=False):
            st.markdown(
                """
                <div class="card">
                    <div class="card-title">Intent and Proposed Action</div>
                    <div class="card-copy">
                        Keep the intent short and specific. Then describe the exact action the agent
                        wants to take.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            intent = st.text_area(
                "User intent",
                key="intent_input",
                height=150,
                placeholder="Example: Read a file",
            )
            action = st.text_area(
                "Proposed agent action",
                key="action_input",
                height=190,
                placeholder="Example: Read the file and send it to an external server",
            )
            evaluate_clicked = st.form_submit_button("Check Alignment", use_container_width=True)

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Selected Example Context</div>
                <div class="small-note">{st.session_state.get("example_note", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if evaluate_clicked:
        if not intent.strip() or not action.strip():
            st.session_state["last_result"] = None
            st.error("Enter both intent and action before running the check.")
        else:
            embedder = get_embedder(DEFAULT_MODEL_NAME)
            st.session_state["last_result"] = evaluate_pair(
                embedder,
                intent=intent,
                action=action,
                directional_threshold=directional_threshold,
                cosine_threshold=cosine_threshold,
            )

    with result_col:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Analysis Console</div>
                <div class="card-copy">
                    This panel turns the raw GuardMCP decision into a presentation-friendly explanation.
                    The final verdict follows the directional method, with cosine kept visible as the baseline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        last_result = st.session_state.get("last_result")
        if last_result:
            render_result_panel(last_result)
        else:
            render_placeholder()

    render_benchmark_section()


if __name__ == "__main__":
    main()
