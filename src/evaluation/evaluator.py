import pandas as pd


class Evaluator:
    def __init__(self, embedder, directional, cosine):
        self.embedder = embedder
        self.directional = directional
        self.cosine = cosine

    def _coerce_case(self, case):
        if isinstance(case, dict):
            return case

        intent, action, label, category = case
        return {
            "intent": intent,
            "action": action,
            "label": label,
            "category": category,
            "source": "unknown",
            "split": "unspecified",
            "metadata": {},
        }

    def score(self, test_cases):
        results = []

        for raw_case in test_cases:
            case = self._coerce_case(raw_case)
            intent = case["intent"]
            action = case["action"]
            label = case["label"]
            category = case["category"]
            metadata = case.get("metadata", {})
            intent_vec = self.embedder.encode(intent)
            action_vec = self.embedder.encode(action)

            dir_result = self.directional.check(intent_vec, action_vec)
            cos_result = self.cosine.check(intent_vec, action_vec)

            results.append(
                {
                    "intent": intent,
                    "action": action,
                    "label": label,
                    "category": category,
                    "source": case.get("source", "unknown"),
                    "split": case.get("split", "unspecified"),
                    "benchmark": metadata.get("benchmark"),
                    "source_split": metadata.get("source_split"),
                    "suite": metadata.get("suite"),
                    "rejection_magnitude": dir_result["rejection_magnitude"],
                    "cosine_similarity": cos_result["similarity"],
                }
            )

        return pd.DataFrame(results)

    def run(self, test_cases, directional_threshold=None, cosine_threshold=None):
        scored_df = self.score(test_cases)
        return self.apply_thresholds(
            scored_df,
            directional_threshold=directional_threshold,
            cosine_threshold=cosine_threshold,
        )

    def apply_thresholds(self, scored_df, directional_threshold=None, cosine_threshold=None):
        directional_threshold = (
            self.directional.epsilon if directional_threshold is None else directional_threshold
        )
        cosine_threshold = self.cosine.threshold if cosine_threshold is None else cosine_threshold

        df = scored_df.copy()
        df["directional_epsilon"] = directional_threshold
        df["directional_decision"] = (df["rejection_magnitude"] <= directional_threshold).astype(int)
        df["cosine_threshold"] = cosine_threshold
        df["cosine_decision"] = (df["cosine_similarity"] >= cosine_threshold).astype(int)
        return df
