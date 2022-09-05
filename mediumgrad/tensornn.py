from micrograd import Tensor
import numpy as np

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

class Linear(Module):

    def __init__(self, nin, nout, nonlin=True, **kwargs):
        self.w = Tensor(np.random.uniform(-1,1, size=(nin, nout)))
        self.b = Tensor(np.random.uniform(-1,1, size=(1, nout)))
        self.nonlin = nonlin

    def __call__(self, x):
        z = x @ self.w + self.b
        return z.relu() if self.nonlin else z

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"Linear({self.w.data.shape[0]}, {self.w.data.shape[1]}, nonlin={self.nonlin})"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
