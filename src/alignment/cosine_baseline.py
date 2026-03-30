import numpy as np

class CosineBaseline:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def check(self, vec1, vec2):
        similarity = self.similarity(vec1, vec2)
        allow = similarity >= self.threshold

        return {
            "allow": allow,
            "similarity": similarity
        }
