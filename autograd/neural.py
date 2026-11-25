import random
from value import Value
class Neural:
    def __init__(self, n_in):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # wx + b
        action = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        out = action.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]