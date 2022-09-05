from __future__ import annotations
import numpy as np

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        assert len(self.shape) != 1

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

class Tensor:
    """ stores a single tensor value and its gradient """

    def __init__(self, data: np.ndarray, _children=(), _op=''):
        self.data = data
        self.grad = np.zeros_like(data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
    
    @property
    def T(self) -> Tensor:
        out = Tensor(self.data.T, (self,), 'T')
    
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
    
        return out
    
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad = np.ones_like(self.data)
        out._backward = _backward 

        return out
    
    def __matmul__(self, other) -> Tensor:
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __add__(self, other: Tensor) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        # basic broadcasting support http://coldattic.info/post/116/
        if other.shape[0] == 1 and self.shape[0] != 1:
            other = Tensor(np.ones((self.shape[0], 1))) @ other
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other) -> Tensor:
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self) -> Tensor:
        out = Tensor(np.clip(self.data, 0, np.inf), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self) -> None:
        assert self.data.shape == ()

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> Tensor: # -self
        return self * -1

    def __radd__(self, other) -> Tensor: # other + self
        return self + other

    def __sub__(self, other) -> Tensor: # self - other
        return self + (-other)

    def __rsub__(self, other) -> Tensor: # other - self
        return other + (-self)

    def __rmul__(self, other) -> Tensor: # other * self
        return self * other

    def __truediv__(self, other) -> Tensor: # self / other
        return self * other**-1

    def __rtruediv__(self, other) -> Tensor: # other / self
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data.__repr__()}, grad={self.grad.__repr__()}, op={self._op})"
