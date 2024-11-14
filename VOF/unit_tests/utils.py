
class Around:
    def __init__(self, v: float, epsilon: float = 0.1):
        self.value = v
        self.epsilon = epsilon

    def __eq__(self, f: float):
        return abs(self.value - f) <= self.epsilon
