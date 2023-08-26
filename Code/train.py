import numpy as np
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


def train(S, A, R, params, task='learning'):
    model = {'params': [], 'state': [], 'action': [], 'rew': [],
                'Qval': [], 'Hval': [], 'p_a': [], 'rpe': [], 'rpe_var': [], 'ape': [], 'ape_var': [], 'pe': []}
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
        var_rpe = Q[S == s_hat, A == a][0] * (1 - Q[S == s_hat, A == a][0]) + eps
        # var_rpe = p_n * (1 - p_n) * np.diff(Q[:, A == a].flatten())[0] ** 2 + eps
        ape = 1 - H[S == s_hat, A == a][0]
        var_ape = p_a[a] * (1 - p_a[a]) + eps
        # update Q value
        Q[S == s_hat, A == a] += lr * rpe / var_rpe
        Q[S == s_hat, A != a] -= lr * rpe / var_rpe       # comment unless Q_init = 0.5, 0.5
        # update H value
        H[S == s_hat, A == a] += lr * ape / var_ape
        H[S == s_hat, A != a] -= lr * ape / var_ape
        # save variables for plotting
        model['state'].append(s)
        model['action'].append(a)
        model['rew'].append(r)
        model['Qval'].append(Q.copy())
        model['Hval'].append(H.copy())
        model['p_a'].append(p_a)
        model['rpe'].append(rpe)
        model['ape'].append(ape)
        model['rpe_var'].append(var_rpe)
        model['ape_var'].append(var_ape)
    for key in model.keys():
        model[key] = np.array(model[key])
    model['params'] = params
    return model
