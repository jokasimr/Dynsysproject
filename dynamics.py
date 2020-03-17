import numpy as np
import matplotlib.pyplot as plt


def henon(x, a=1.4, b=0.3):
    x = np.array(x)
    x, y = x[:, 0], x[:, 1]
    return np.stack([1-a*x**2+y, b*x], -1)


def attractor_points(f, n, k=100, condition=None):
    if condition is None:
        condition = lambda x: np.sum(x**2, 1) < 1000
    x = 5*np.random.randn(n, 2)
    for i in range(k):
        x = f(x)
        x = x[condition(x)]
    return x[:n, :]


def iter(f, x, k):
    """
    f tar en matris av form NxD
    där N är antalet punkter och D är dimensionen av X
    """
    x = np.array(x)
    for i in range(k):
        x = f(x)
    return x


def itercat(f, x, k):
    x = np.array(x)
    xs = [x]
    for i in range(k):
        xs.append(f(xs[-1]))
    return np.stack(xs, axis=0)

