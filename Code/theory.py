from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt


def softmax(x, beta, axis=-1):
    kw = dict(axis=axis, keepdims=True)
    # make every value 0 or below, as exp(0) won't overflow
    xrel = x - x.max(**kw)
    exp_xrel = np.exp(beta * xrel)
    return exp_xrel / exp_xrel.sum(**kw)


def asymptote(beta):
    f = lambda x: (np.exp(beta*x) / (np.exp(beta*x) + np.exp(beta*(1-x)))) - x
    return optimize.newton(f, .75, maxiter=1000)


def steadystate(noise, beta):
    Q = 1 - noise
    H = asymptote(beta)
    if H < Q:
        H = softmax(np.array([Q, 1-Q]), beta)[0]
    p = H
    r = p * (1 - noise) + (1 - p) * noise
    return Q, H, p, r
