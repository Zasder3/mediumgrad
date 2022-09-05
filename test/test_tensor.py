import numpy as np
import jax.numpy as jnp
from jax import grad
from mediumgrad import Tensor

def _is_close(t1, t2, eps=1e-4):
    return (np.abs(t1-t2) < eps).all()

def test_identity():
    t = np.arange(4).reshape(2, 2) + 0.0 
    t1 = Tensor(t)
    t2 = jnp.array(t)

    # test our gradient
    t1.sum().backward()
    g1 = t1.grad

    # compute gold truth gradient
    g2 = np.array(grad(lambda x: x.sum())(t2))

    assert _is_close(g1, g2)

def test_matmul():
    a = np.arange(4).reshape(2, 2) + 0.0 
    b = np.arange(4).reshape(2, 2) + 4.0 
    a1 = Tensor(a)
    b1 = Tensor(b)
    a2 = jnp.array(a)
    b2 = jnp.array(b)

    # test our gradient
    (a1 @ b1).sum().backward()
    ga1, gb1 = a1.grad, b1.grad

    # compute gold truth gradient
    ga2, gb2 = grad(lambda x, y: (x @ y).sum(), (0, 1))(a2, b2)
    ga2, gb2 = np.array(ga2), np.array(gb2)

    # print the gradients
    assert _is_close(ga1, ga2)
    assert _is_close(gb1, gb2)

def test_broadcast():
    # test broadcasting
    a = np.arange(4).reshape(2, 2) + 0.0
    b = np.arange(2).reshape(1, 2) + 4.0
    a1 = Tensor(a)
    b1 = Tensor(b)
    a2 = jnp.array(a)
    b2 = jnp.array(b)

    # test our gradient
    (a1 + b1).sum().backward()
    ga1, gb1 = a1.grad, b1.grad

    # compute gold truth gradient
    ga2, gb2 = grad(lambda x, y: (x + y).sum(), (0, 1))(a2, b2)
    ga2, gb2 = np.array(ga2), np.array(gb2)

    assert _is_close(ga1, ga2)
    assert _is_close(gb1, gb2)

def test_nn():
    # test a simple neural network
    # 3 inputs, 2 hidden units, 1 output
    # 3x2 weights, 2x1 weights
    x = np.arange(3).reshape(1, 3) + 0.0
    w1 = np.arange(6).reshape(3, 2) + 4.0
    w2 = np.arange(2).reshape(2, 1) + 8.0
    b1 = np.arange(2).reshape(1, 2) + 10.0
    b2 = np.arange(1).reshape(1, 1) + 12.0

    x1 = Tensor(x)
    w11 = Tensor(w1)
    w21 = Tensor(w2)
    b11 = Tensor(b1)
    b21 = Tensor(b2)

    x2 = jnp.array(x)
    w12 = jnp.array(w1)
    w22 = jnp.array(w2)
    b12 = jnp.array(b1)
    b22 = jnp.array(b2)

    # test our gradient
    z1 = x1 @ w11 + b11
    a1 = z1.relu()
    z2 = a1 @ w21 + b21
    ((z2 - np.array([[1.0]]))**2).sum().backward()
    gx1, gw11, gw21, gb11, gb21 = x1.grad, w11.grad, w21.grad, b11.grad, b21.grad

    # compute gold truth gradient
    def jax_loss(x, w1, w2, b1, b2):
        z1 = x @ w1 + b1
        a1 = jnp.maximum(z1, 0)
        z2 = a1 @ w2 + b2
        return ((z2 - np.array([[1.0]]))**2).sum()
    
    gx2, gw12, gw22, gb12, gb22 = grad(jax_loss, (0, 1, 2, 3, 4))(x2, w12, w22, b12, b22)
    gx2, gw12, gw22, gb12, gb22 = np.array(gx2), np.array(gw12), np.array(gw22), np.array(gb12), np.array(gb22)

    print(gx1, gx2)
    assert _is_close(gx1, gx2)
    assert _is_close(gw11, gw12)
    assert _is_close(gw21, gw22)
    assert _is_close(gb11, gb12)
    assert _is_close(gb21, gb22)


if __name__ == "__main__":
    test_nn()