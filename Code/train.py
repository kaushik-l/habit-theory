import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli, norm
import numpy.random as npr


def softmax(x, beta, axis=-1):
    kw = dict(axis=axis, keepdims=True)
    # make every value 0 or below, as exp(0) won't overflow
    xrel = x - x.max(**kw)
    exp_xrel = np.exp(beta * xrel)
    return exp_xrel / exp_xrel.sum(**kw)


def winner(Q, H):
    return np.array([Q[a] if np.abs(Q[a] - 0.5) > np.abs(H[a] - 0.5) else H[a] for a in range(len(Q))])


def train_single(S, A, R, params):
    learning = {'params': [], 'state': [], 'action': [], 'rew': [],
                'Gval': [], 'rpe': [], 'rpe_var': [], 'ape': [], 'ape_var': [], 'pe': []}
    G, beta, p_n, lr, Ntrials, eps = (params['G_init'], params['beta'],
                                      params['p_n'], params['lr'], params['Ntrials'], params['eps'])
    for idx in range(Ntrials):
        # run the trial
        s = npr.choice(S)                                           # true state
        s_hat = np.abs(s - bernoulli.rvs(p_n))                      # subjective state
        p_a = softmax(G[S == s_hat].flatten(), beta)
        a = np.nonzero(npr.multinomial(1, p_a))[0][0]           # action
        r = R[S == s, A == a][0]                                    # reward
        # calculate prediction errors
        rpe = r - G[S == s_hat, A == a][0]
        var_rpe = p_n * (1 - p_n) * np.diff(G[:, A == a].flatten())[0] ** 2 + eps
        ape = 1 - p_a[a]
        var_ape = p_a[a] * (1 - p_a[a]) + eps
        pe = (rpe/var_rpe + ape/var_ape)
        # update Q value
        G[S == s_hat, A == a] += lr * pe
        # save variables for plotting
        learning['state'].append(s)
        learning['action'].append(a)
        learning['rew'].append(r)
        learning['Gval'].append(G.copy())
        learning['rpe'].append(rpe)
        learning['ape'].append(ape)
        learning['rpe_var'].append(var_rpe)
        learning['ape_var'].append(var_ape)
        learning['pe'].append(pe)
    for key in learning.keys():
        learning[key] = np.array(learning[key])
    learning['params'] = params
    return learning


def train_dual(S, A, R, params, task='learning'):
    learning = {'params': [], 'state': [], 'action': [], 'rew': [],
                'Qval': [], 'Hval': [], 'rpe': [], 'rpe_var': [], 'ape': [], 'ape_var': [], 'pe': []}
    Q, H, beta, p_n, lr, Ntrials, eps = (params['Q_init'], params['H_init'], params['beta'],
                                         params['p_n'], params['lr'], params['Ntrials'], params['eps'])
    for idx in range(Ntrials):
        if task == 'reversal' and idx == int(Ntrials / 2):
            R = np.flip(R, axis=1)
        # run the trial
        s = npr.choice(S)                                           # true state
        s_hat = np.abs(s - bernoulli.rvs(p_n))                      # subjective state
        p_a = softmax(winner(Q[S == s_hat].flatten(), H[S == s_hat].flatten()), beta)
        a = np.nonzero(npr.multinomial(1, p_a))[0][0]               # action
        r = R[S == s, A == a][0]                                    # reward
        # calculate prediction errors
        rpe = r - Q[S == s_hat, A == a][0]
        var_rpe = p_n * (1 - p_n) * np.diff(Q[:, A == a].flatten())[0] ** 2 + eps
        ape = 1 - H[S == s_hat, A == a][0]
        var_ape = p_a[a] * (1 - p_a[a]) + eps
        # update Q value
        Q[S == s_hat, A == a] += lr * rpe / var_rpe
        Q[S == s_hat, A != a] -= lr * rpe / var_rpe
        # update H value
        H[S == s_hat, A == a] += lr * ape / var_ape
        H[S == s_hat, A != a] -= lr * ape / var_ape
        # save variables for plotting
        learning['state'].append(s)
        learning['action'].append(a)
        learning['rew'].append(r)
        learning['Qval'].append(Q.copy())
        learning['Hval'].append(H.copy())
        learning['rpe'].append(rpe)
        learning['ape'].append(ape)
        learning['rpe_var'].append(var_rpe)
        learning['ape_var'].append(var_ape)
    for key in learning.keys():
        learning[key] = np.array(learning[key])
    learning['params'] = params
    return learning
