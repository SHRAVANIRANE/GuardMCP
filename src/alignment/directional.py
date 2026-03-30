import numpy as np

class DirectionalAlignment:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def check(self, intent_vec, action_vec):
        intent_vec = np.array(intent_vec)
        action_vec = np.array(action_vec)

        projection = (
            np.dot(action_vec, intent_vec) /
            np.dot(intent_vec, intent_vec)
        ) * intent_vec

        rejection = action_vec - projection
        rejection_magnitude = np.linalg.norm(rejection)

        allow = rejection_magnitude <= self.epsilon

        return {
            "allow": allow,
            "rejection_magnitude": rejection_magnitude
        }