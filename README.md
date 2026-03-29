# GuardMCP

### Deterministic Runtime Semantic Enforcement for Agentic Tool Execution

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Research](https://img.shields.io/badge/Type-Research%20Prototype-green)
![Status](https://img.shields.io/badge/Status-Active-orange)

## 🔍 Overview

GuardMCP is a research prototype that detects semantic misalignment between user intent and agent actions using **directional vector projection** instead of traditional cosine similarity.

## 🚨 Problem

Cosine similarity is symmetric and fails to detect **hidden malicious intent** in agent actions.

## 💡 Proposed Solution

We introduce a **Directional Intent–Action Alignment** method:

- Project action vector onto intent vector
- Compute orthogonal component (semantic leakage)
- Block actions if leakage exceeds threshold

## 🧠 Key Insight

> “Not all similar actions are safe — hidden semantics matter.”

## ⚙️ Tech Stack

- Python
- Sentence Transformers
- NumPy
- Pandas

## 📊 Features

- Directional semantic verification
- Cosine similarity baseline comparison
- Adversarial test cases
- Evaluation pipeline

## 🧪 Example

Intent: "Read a file"
Action: "Read file and send data to server"

✔ Cosine Similarity → HIGH (incorrectly allows)
✔ GuardMCP → BLOCK (detects extra intent)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python experiments/run_experiments.py
```

## 📁 Project Structure

(see folders)

## 📌 Future Work

- Adaptive epsilon tuning
- Real agent integration
- Runtime enforcement layer

---

## 👩‍💻 Author

Shravani Rane
