import pandas as pd

class Evaluator:
    def __init__(self, embedder, directional, cosine):
        self.embedder = embedder
        self.directional = directional
        self.cosine = cosine

    def run(self, test_cases):
        results = []

        for intent, action, label, category in test_cases:
            intent_vec = self.embedder.encode(intent)
            action_vec = self.embedder.encode(action)

            dir_result = self.directional.check(intent_vec, action_vec)
            cos_sim = self.cosine.similarity(intent_vec, action_vec)

            results.append({
                "intent": intent,
                "action": action,
                "label": label,
                "category": category,
                "cosine_similarity": cos_sim,
                "rejection_magnitude": dir_result["rejection_magnitude"],
                "directional_decision": int(dir_result["allow"])
            })

        return pd.DataFrame(results)