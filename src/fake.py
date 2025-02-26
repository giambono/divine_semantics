import numpy as np

class FakeModel:
    def __init__(self, embedding_dim=1024):
        self.embedding_dim = embedding_dim  # Simulate the real model's output dimension

    def encode(self, text):
        """Simulate encoding by returning a random vector of the expected dimension."""
        return np.random.rand(self.embedding_dim)
