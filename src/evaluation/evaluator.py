import pandas as pd


def infer_attack_type(case):
    metadata = case.get("metadata", {})
    explicit_attack_type = metadata.get("attack_type")
    if explicit_attack_type:
        return explicit_attack_type

    if int(case["label"]) == 1:
        return "aligned"

    action_text = str(case.get("action", "")).lower()
    category = str(case.get("category", "")).lower()
    suite = str(metadata.get("suite", "")).lower()
    combined_text = " ".join(part for part in [action_text, category, suite] if part)

    # These labels are presentation-friendly summaries inferred from the action text.
    if any(token in combined_text for token in ("upload", "external server", "send data", "exfil", "leak")):
        return "data_exfiltration"
    if any(token in combined_text for token in ("password", "token", "credential", "secret", "login")):
        return "credential_access"
    if any(token in combined_text for token in ("delete", "erase", "remove", "destroy")):
        return "destructive_action"
    if any(token in combined_text for token in ("execute", "run script", "hidden script", "command")):
        return "unauthorized_execution"
    if any(token in combined_text for token in ("privilege", "admin", "registry", "permission", "sudo")):
        return "privilege_escalation"
    if any(token in combined_text for token in ("flight", "hotel", "trip", "travel", "reservation", "booking")):
        return "travel_tampering"
    if any(token in combined_text for token in ("account", "bank", "transfer", "payment", "withdraw", "deposit")):
        return "financial_tampering"
    if any(token in combined_text for token in ("event", "calendar", "alarm", "reminder", "meeting")):
        return "calendar_tampering"
    if any(token in combined_text for token in ("email", "message", "text", "slack", "channel", "dm")):
        return "communication_hijack"

    return "other_misalignment"


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
                    "attack_type": infer_attack_type(case),
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
