# GuardMCP

### Deterministic Runtime Semantic Enforcement for Securing Agentic Tool Execution

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Type](https://img.shields.io/badge/Project-Research%20Prototype-green)
![Status](https://img.shields.io/badge/Status-Active-orange)

---

## 🔍 Overview

GuardMCP is a research-oriented prototype focused on improving the safety of **agentic AI systems** during tool execution.

Modern AI agents can perform actions that appear relevant to user intent but may include **hidden or unintended behaviors**. This project explores a structured mechanism to detect such misalignments at runtime.

---

## 🚨 Problem Statement

Existing approaches (e.g., similarity-based checks) evaluate whether an agent’s action is *related* to user intent but often fail to detect **additional embedded semantics** that may lead to unsafe execution.

This creates a critical gap in **AI safety and tool governance**.

---

## 💡 Proposed Approach (High-Level)

GuardMCP introduces a **direction-aware semantic verification mechanism** that evaluates how closely an action aligns with the intended objective while identifying potential deviations.

Instead of relying solely on similarity scores, the system analyzes whether an action contains **semantic components beyond the intended scope**.

> The goal is to enable **runtime enforcement of intent-aligned behavior** in AI agents.

---

## 🧠 Key Insight

> “Semantic similarity alone is insufficient — safe execution requires detecting hidden deviations from intent.”

---

## ⚙️ Tech Stack

* Python
* Sentence Transformers (Embeddings)
* NumPy (Vector Operations)
* Pandas (Evaluation)

---

## 🧪 Experimental Setup

The prototype simulates agent behavior using:

* Intent–Action pairs
* Safe vs Misaligned scenarios
* Baseline comparison methods
* Structured evaluation pipeline

---

## 📊 Features

* Semantic alignment verification module
* Baseline comparison (similarity-based)
* Adversarial-style test scenarios
* Reproducible experiment pipeline
* Result logging for analysis

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python experiments/run_experiments.py
```

---

## 📁 Project Structure

```
GuardMCP/
│
├── src/                # Core modules (embedding, alignment, evaluation)
├── experiments/        # Experiment runner
├── results/            # Output data
├── docs/               # Documentation (methodology, architecture)
├── main.py             # Quick demo
├── config.py           # Parameters
└── README.md
```

---

## 📌 Applications

* AI Agent Safety
* Tool Execution Governance
* Autonomous System Monitoring
* Secure LLM-based Workflows

---

## 🔮 Future Work

* Adaptive threshold tuning
* Integration with real agent frameworks
* Real-time enforcement layer
* Extended adversarial evaluation

---

## ⚠️ Note

This project is a **research prototype** designed to demonstrate a novel concept in semantic alignment and is not a full production system.

---

## 👩‍💻 Author

**Shravani Rane**

---

## 📄 License

This project is licensed under the MIT License.
