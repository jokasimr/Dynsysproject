import numpy as np
import matplotlib.pyplot as plt


def henon(x, a=1.4, b=0.3):
    x = np.array(x)
    x, y = x[:, 0], x[:, 1]
    return np.stack([1-a*x**2+y, b*x], -1)


def iter(f, x, k):
    """
    f tar en matris av form NxD
    där N är antalet punkter och D är dimensionen av X
    """
    x = np.array(x).reshape(-1, len(x))
    for i in range(k):
        x = f(x)
    return x


def itercat(f, x, k):
    x = np.array(x).reshape(-1, len(x))
    xs = [x]
    for i in range(k):
        xs.append(f(xs[-1]))
    return np.stack(xs, axis=0)

