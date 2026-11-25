from layer import Layer

class MultiLayerPerceptron:
    def __init__(self, n_in, n_outs):
        size = [n_in] + n_outs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # Gradients accumulate with +=, so reset them before every backward()
        for p in self.parameters():
            p.grad = 0.0