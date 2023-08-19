import numpy as np
import math
from math import sqrt, pi
import numpy.random as npr
import torch


class Network:
    def __init__(self, name='cortex', N=128, S=3, R=1, Rc=1, Ra=2, g=1.1, seed=1):
        self.name = name
        npr.seed(seed)
        if self.name == 'ventral_striatum':
            # network parameters
            self.N = N  # hidden units
            self.S = S  # input
            self.Rc = Rc  # readout critic
            self.Ra = Ra  # readout actor
            self.sa, self.ha, self.ca, self.aa = [], [], [], []  # input, activity, critic output, actor output
            self.ws = 1 * (npr.random([N, S]))  # input weights
            self.wc = (2 * npr.random((Rc, N)) - 1) / sqrt(N)  # readout weights critic
            self.wa = (2 * npr.random((Ra, N)) - 1) / sqrt(N)  # readout weights actor
            self.var = 0.05
            self.lam = .25
            self.gam = 1
        elif self.name == 'tail_striatum':
            # network parameters
            self.N = N  # hidden units
            self.S = S  # input
            self.Rc = Rc  # readout critic
            self.Ra = Ra  # readout actor
            self.sa, self.ha, self.ca, self.aa = [], [], [], []  # input, activity, critic output, actor output
            self.ws = 1 * (npr.random([N, S]))  # input weights
            self.wc = (2 * npr.random((Rc, N)) - 1) / sqrt(N)  # readout weights critic
            self.wa = (2 * npr.random((Ra, N)) - 1) / sqrt(N)  # readout weights actor
            self.var = 0.05
            self.lam = .25
            self.gam = 1
        elif self.name == 'ventral_tail_striatum':
            # network parameters
            self.N = N  # hidden units
            self.S = S  # input
            self.Rc = Rc  # readout critic
            self.Ra = Ra  # readout actor
            self.sa, self.ha, self.ca, self.aa = [], [], [], []  # input, activity, critic output, actor output
            self.ws_v = npr.random([N, S])  # cortico-striatal weights (dorsal/ventral)
            self.wc_v = (2 * npr.random((Rc, N)) - 1) / sqrt(N)  # readout weights critic (dorsal/ventral)
            self.wa_v = (2 * npr.random((Ra, N)) - 1) / sqrt(N)  # readout weights actor (dorsal/ventral)
            self.ws_t = npr.random([N, S])  # cortico-striatal weights (tail)
            self.wc_t = (2 * npr.random((Rc, N)) - 1) / sqrt(N)  # readout weights critic (tail)
            self.wa_t = (2 * npr.random((Ra, N)) - 1) / sqrt(N)  # readout weights actor (tail)
            self.var = 0.05
            self.lam = .25
            self.gam = 1
        elif self.name == 'ventral_dorsal_striatum':
            # network parameters
            self.N = N  # hidden units
            self.S = S  # input
            self.Rc = Rc  # readout critic
            self.Ra = Ra  # readout actor
            self.sa, self.ha, self.ca, self.aa = [], [], [], []  # input, activity, critic output, actor output
            self.w_cs_v = npr.random([N, S])  # cortico-striatal weights (ventral)
            self.w_zs_v = npr.random([N, 1]) - 0.5  # iti cortico-striatal weights (ventral)
            self.w_out_v = 1 * np.ones((Rc, N)) / sqrt(N)  # readout weights critic (ventral)
            self.w_cs_d = npr.random([Ra, N, S])  # cortico-striatal weights (dorsal)
            self.w_zs_d = npr.random([Ra, N, 1])  # iti cortico-striatal weights (dorsal)
            self.w_out_d = np.ones((Ra, N)) / sqrt(N)  # readout weights actor (dorsal)
            self.var = 0.1
            self.lam = .25
            self.gam = 1
        elif self.name == 'ventral_dorsal_tail_striatum':
            # network parameters
            self.N = N  # hidden units
            self.S = S  # input
            self.Rc = Rc  # readout critic
            self.Ra = Ra  # readout actor
            self.sa, self.ha, self.ca, self.aa = [], [], [], []  # input, activity, critic output, actor output
            self.w_cs_v = npr.random([N, S])  # cortico-striatal weights (ventral)
            self.w_zs_v = npr.random([N, 1]) - 0.5  # iti cortico-striatal weights (ventral)
            self.w_out_v = 1 * np.ones((Rc, N)) / sqrt(N)  # readout weights critic (ventral)
            self.w_cs_d = npr.random([Ra, N, S])  # cortico-striatal weights (dorsal)
            self.w_zs_d = npr.random([Ra, N, 1])  # iti cortico-striatal weights (dorsal)
            self.w_out_d = np.ones((Ra, N)) / N  # readout weights actor (dorsal)
            self.w_cs_t = npr.random([N, S])  # cortico-striatal weights (tail)
            self.w_zs_t = npr.random([N, 1]) - 0.5   # cortico-striatal weights (tail)
            self.w_out_t = 1 * np.ones((Rc, N)) / N  # readout weights prior (tail)
            self.var = 0.1
            self.lam = 5
            self.gam = 1
        elif self.name == 'ventral_dorsal_DI_striatum':
            # network parameters
            self.N = N  # hidden units
            self.S = S  # input
            self.Rc = Rc  # readout critic
            self.Ra = Ra  # readout actor
            self.sa, self.ha, self.ca, self.aa = [], [], [], []  # input, activity, critic output, actor output
            self.ws_v = npr.random([N, S])  # cortico-striatal weights (ventral)
            self.wc_v = (2 * npr.random((Rc, N)) - 1) / sqrt(N)  # readout weights critic (ventral)
            self.ws_d_D = npr.random([round(N/2), S])  # cortico-striatal weights (dorsal direct)
            self.ws_d_I = npr.random([round(N/2), S])  # cortico-striatal weights (dorsal indirect)
            self.wa_d_D = npr.random((Ra, round(N/2))) / sqrt(N/2)  # readout weights actor (dorsal direct)
            self.wa_d_I = npr.random((Ra, round(N/2))) / sqrt(N/2)  # readout weights actor (dorsal indirect)
            # self.wa_d_D = np.ones((Ra, round(N/2))) / sqrt(N/2)  # readout weights actor (dorsal direct)
            # self.wa_d_I = np.ones((Ra, round(N/2))) / sqrt(N/2)  # readout weights actor (dorsal indirect)
            self.var = 0.05
            self.lam = .25
            self.gam = 1

    # nlin
    def f(self, x):
        return np.tanh(x) if not torch.is_tensor(x) else torch.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 / (np.cosh(10*np.tanh(x/10)) ** 2) if not torch.is_tensor(x) else 1 / (torch.cosh(10*torch.tanh(x/10)) ** 2)


class Task:
    def __init__(self, name='detect', duration=200, noise=0.2, signal=0.1, snr=(1, 2, 3), dur_a=10, dur_r=10,
                 prob_signal=(0.25, 0.75), rewards=(10, 10, -10, -10), tb=30, tf=70, ta=100, tr=140, dt=0.2):
        self.name = name
        self.tb, self.tf, self.ta, self.tr, self.dur_a, self.dur_r, self.noise, self.signal, self.snr, \
            self.prob_signal, self.rewards = tb, tf, ta, tr, dur_a, dur_r, noise, signal, snr, prob_signal, rewards
        NT = int(duration / dt)
        # task parameters
        if self.name == 'detect':
            self.T, self.dt, self.NT = duration, dt, NT
            self.s = 0.0 * np.ones((3, NT))
            self.s[0, round(tb / dt):round(tf / dt)] = self.signal
            self.s[1, round(tb / dt):round(tf / dt)] = self.prob_signal[0]
            self.s[2, :round((tr + dur_r) / dt)] = 1

    def loss(self, err):
        mse = (err ** 2).mean() / 2
        return mse


class Algorithm:
    def __init__(self, name='Adam', Nepochs=10000, lr=(0, 1e-1, 1e-1)):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.Nstart_anneal = 10000
        self.lr = lr  # learning rate
        self.annealed_lr = 1e-5
