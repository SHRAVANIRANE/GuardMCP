import numpy as np

class CosineBaseline:
    def similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )