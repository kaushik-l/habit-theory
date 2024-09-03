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
    if noise > 0.5:
        H = 1 - H
    if np.abs(H-0.5) < np.abs(Q-0.5):
        H = softmax(np.array([Q, 1-Q]), beta)[0]
    p = H
    r = p * (1 - noise) + (1 - p) * noise
    return Q, H, p, r


def criticalbeta(noise):
    p = 1-noise
    return (np.log(p) - np.log(1-p)) / (p - (1-p))


def criticalbeta_sequential(noise, p):
    p1 = 1-noise
    p2 = p1 * p
    b1 = (np.log(p1) - np.log(1 - p1)) / (p1 - (1 - p1))
    return


def criticalbeta_schedule(p, r):
    return np.log(p*r) * ((p*r + 1) / (p*r - 1))


def criticalbeta_distal(p1, p2, r):
    return np.log(p1*p2*r) * ((p1*p2*r + 1) / (p1*p2*r - 1))


def criticalbeta_delay(p, r, delay, tau):
    return (np.log(p*r) - (delay/tau)) * ((p*r*np.exp(-delay/tau) + 1) / (p*r*np.exp(-delay/tau) - 1))


def gonogo(p, r, beta):
    q1 = (p * r) / (p * r + 1)
    q2 = 1 / (p * r + 1)
    hval = softmax(np.array([q1, q2]), beta)[0]
    return q1, hval


def distal(p1, p2, r, beta):
    qval_distal = (p1 * p2 * r) / (p1 * p2 * r + 1)
    hval_distal = softmax(np.array([qval_distal, 1 - qval_distal]), beta)[0]
    return qval_distal, hval_distal


def delay(p, r, delay, tau, beta):
    qval = (p*r*np.exp(-delay/tau)) / (p*r*np.exp(-delay/tau) + 1)
    hval = softmax(np.array([qval, 1 - qval]), beta)[0]
    return qval, hval


def schedule(p, r, t, tau, dt, beta):
    ti = np.linspace(dt, dt * len(p), len(p))
    pr = np.sum(p[ti>=t] * np.exp(-(ti[ti>=t]-t)/tau))
    qval = (pr*r) / (pr*r + 1)
    hval = softmax(np.array([qval, 1 - qval]), beta)[0]
    return qval, hval
