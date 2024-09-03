import numpy as np
from scipy.stats import bernoulli, norm
import numpy.random as npr


def softmax(x, beta, axis=-1):
    kw = dict(axis=axis, keepdims=True)
    # make every value 0 or below, as exp(0) won't overflow
    xrel = x - x.max(**kw)
    exp_xrel = np.exp(beta * xrel)
    return exp_xrel / exp_xrel.sum(**kw)


def winner(Q, H, mean=0.5):
    return np.array([Q[a] if np.abs(Q[a] - mean) > np.abs(H[a] - mean) else H[a] for a in range(len(Q))])


def likelihood(s, a, r, p_n):
    if s == 0 and a == 0:
        return 1-p_n if r else p_n
    elif s == 0 and a == 1:
        return p_n if r else 1-p_n
    elif s == 1 and a == 1:
        return 1-p_n if r else p_n
    elif s == 1 and a == 0:
        return p_n if r else 1-p_n


def discretize(b):
    binedges = [0, .125, .25, .375, .625, .75, .875, 1]
    b_discrete = [np.where(np.histogram(b[j], binedges)[0])[0][0] for j in range(len(b))]
    return b_discrete


def accumulate(stim, s, p_n, Q, H, beta, method='bayesian', adapt=False):
    if method == 'bayesian':
        bel = (np.cumprod((1 - (s == stim)) * p_n + (s == stim) * (1 - p_n)) /
                         (np.cumprod((1 - (s == stim)) * p_n + (s == stim) * (1 - p_n)) +
                          np.cumprod((1 - (s == stim)) * (1-p_n) + (s == stim) * p_n)))
    elif method == 'wta':
        bel = [0.5]
        for idx in range(len(stim)):
            num = bel[-1] * ((1 - (s == stim[idx])) * p_n + (s == stim[idx]) * (1 - p_n))
            den = (bel[-1] * ((1 - (s == stim[idx])) * p_n + (s == stim[idx]) * (1 - p_n)) +
                   (1 - bel[-1]) * ((1 - (s == stim[idx])) * (1 - p_n) + (s == stim[idx]) * p_n))
            bel.append(num / den)
        bel = bel[1:]
        # winner(Q[b[-1]].flatten(), H[b[-1]].flatten())
    elif method == 'confirmation':
        bel = [0.5]
        for idx in range(len(stim)):
            num = bel[-1] * ((1 - (s == stim[idx])) * p_n + (s == stim[idx]) * (1 - p_n))
            den = (bel[-1] * ((1 - (s == stim[idx])) * p_n + (s == stim[idx]) * (1 - p_n)) +
                   (1 - bel[-1]) * ((1 - (s == stim[idx])) * (1 - p_n) + (s == stim[idx]) * p_n))
            p = num / den
            if adapt == 0:
                p_a = softmax(np.array([p, 1 - p]), beta)
                bel.append(p_a[0])
            elif adapt == 1:
                p_a = softmax(np.array([p, 1 - p]), beta)
                bel.append(0.5 + 0.8 * (p_a[0] - 0.5))
            else:
                if idx < 4:
                    p_a = softmax(np.array([p, 1 - p]), beta)
                    bel.append(0.5 + 0.8 * (p_a[0] - 0.5))
                else:
                    beta = 3.5
                    p_a = softmax(np.array([p, 1 - p]), beta)
                    bel.append(0.5 + 0.8 * (p_a[0] - 0.5))
        bel = bel[1:]
    return np.array(bel)


def train(S, A, R, params, task='learning'):
    model = {'params': [], 'stim': [], 'belief': [], 'state': [], 'action': [], 'rew': [],
                'Qval': [], 'Hval': [], 'p_a': [], 'rpe': [], 'rpe_var': [], 'ape': [], 'ape_var': [], 'pe': [], 'p_c': []}
    Q, H, beta, p_n, lr, Ntrials, eps, adapt, p_c = (params['Q_init'], params['H_init'], params['beta'],
                                         params['p_n'], params['lr'], params['Ntrials'], params['eps'], params['adapt'], 0)
    for idx in range(Ntrials):
        if task == 'reversal' and idx == int(Ntrials / 2):
            R = np.flip(R, axis=1)
        if task == 'sequential_reversal' and idx == int(Ntrials / 2):
            R = np.flip(R, axis=1)
        if task == 'inference_reversal' and idx == int(Ntrials / 2):
            R = np.flip(R, axis=1)
        # run the trial
        if task == 'learning' or task == 'reversal':
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
        elif task == 'inference_reversal':
            s = npr.choice(S)   # true state
            s_hat = np.abs(s - bernoulli.rvs(p_n))  # subjective state
            p_a = softmax(winner(Q[S == s_hat].flatten(), H[S == s_hat].flatten()), beta)
            a = np.nonzero(npr.multinomial(1, p_a))[0][0]  # action
            r = R[S == s, A == a][0]  # reward
            # calculate prediction errors
            rpe = r - Q[S == s_hat, A == a][0]
            var_rpe = Q[S == s_hat, A == a][0] * (1 - Q[S == s_hat, A == a][0]) + eps
            # var_rpe = p_n * (1 - p_n) * np.diff(Q[:, A == a].flatten())[0] ** 2 + eps
            ape = 1 - H[S == s_hat, A == a][0]
            var_ape = p_a[a] * (1 - p_a[a]) + eps
            # update Q value
            Q[S == s_hat, A == a] += lr * rpe / var_rpe
            Q[S == s_hat, A != a] -= lr * rpe / var_rpe  # comment unless Q_init = 0.5, 0.5
            # update H value
            H[S == s_hat, A == a] += lr * ape / var_ape
            H[S == s_hat, A != a] -= lr * ape / var_ape
            # update context probability
            l_c = likelihood(s, a, r, p_n)
            p_c = np.max((-7, np.min((7, p_c + np.log(l_c / (1-l_c))))))
        elif task == 'sequential_learning' or task == 'sequential_reversal':
            s = [0]
            p_a = [softmax(winner(Q[S == s[-1]].flatten(), H[S == s[-1]].flatten(), mean=0.5), beta)]
            a = [np.nonzero(npr.multinomial(1, p_a[-1]))[0][0]]  # action
            if a[-1] == 1:
                s.append(1)
                p_a.append(softmax(winner(Q[S == s[-1]].flatten(), H[S == s[-1]].flatten(), mean=0.5), beta))
                a.append(np.nonzero(npr.multinomial(1, p_a[-1]))[0][0])
                r = np.abs(R[1, 1] - bernoulli.rvs(p_n)) if a[-1] == 1 else 0 #np.abs(R[1, 0] - bernoulli.rvs(p_n))
            else:
                r = 0
            # calculate prediction errors
            rpe = [r - Q[S == s[0], A == a[0]][0]]
            var_rpe = [Q[S == s[0], A == a[0]][0] * (1 - Q[S == s[0], A == a[0]][0]) + eps]
            ape = [1 - H[S == s[0], A == a[0]][0]]
            var_ape = [p_a[0][a[0]] * (1 - p_a[0][a[0]]) + eps]
            # update Q and value
            Q[S == s[0], A == a[0]] += lr * rpe[0] / var_rpe[0]
            Q[S == s[0], A != a[0]] -= lr * rpe[0] / var_rpe[0]       # comment unless Q_init = 0.5, 0.5
            H[S == s[0], A == a[0]] += lr * ape[0] / var_ape[0]
            H[S == s[0], A != a[0]] -= lr * ape[0] / var_ape[0]
            if len(a) == 2:
                rpe.append(r - Q[S == s[1], A == a[1]][0])
                var_rpe.append(Q[S == s[1], A == a[1]][0] * (1 - Q[S == s[1], A == a[1]][0]) + eps)
                ape.append(1 - H[S == s[1], A == a[1]][0])
                var_ape.append(p_a[1][a[1]] * (1 - p_a[1][a[1]]) + eps)
                # update Q and value
                Q[S == s[1], A == a[1]] += lr * rpe[1] / var_rpe[1]
                # Q[S == s[1], A != a[1]] -= lr * rpe[1] / var_rpe[1]  # comment unless Q_init = 0.5, 0.5
                H[S == s[1], A == a[1]] += lr * ape[1] / var_ape[1]
                H[S == s[1], A != a[1]] -= lr * ape[1] / var_ape[1]
        elif task == 'accumulation':
            test = False if idx < 10000 else True
            N = 10
            s = npr.choice(S)                                           # true state
            stim = [np.abs(s - bernoulli.rvs(p_n)) for n in range(N)]   # stimulus
            method = 'confirmation' if test else 'bayesian'
            bel = accumulate(stim, s, p_n, Q, H, beta, method=method, adapt=adapt*test)
            bel = bel if s else 1 - bel
            b = discretize(bel)
            p_a = softmax(winner(Q[b[-1]].flatten(), H[b[-1]].flatten()), beta)
            a = np.nonzero(npr.multinomial(1, p_a))[0][0]               # action
            r = R[S == s, A == a][0]                                    # reward
            if not test:
                # calculate prediction errors
                rpe = r - Q[b[-1], A == a][0]
                var_rpe = Q[b[-1], A == a][0] * (1 - Q[b[-1], A == a][0]) + eps
                ape = 1 - H[b[-1], A == a][0]
                var_ape = p_a[a] * (1 - p_a[a]) + eps
                # update Q value
                Q[b[-1], A == a] += lr * rpe  #/ var_rpe
                Q[b[-1], A != a] -= lr * rpe  #/ var_rpe       # comment unless Q_init = 0.5, 0.5
                # update H value
                H[b[-1], A == a] += lr * ape  #/ var_ape
                H[b[-1], A != a] -= lr * ape  #/ var_ape
        # save variables for plotting
        if task == 'accumulation':
            model['stim'].append(stim)
            model['belief'].append(bel[-1])
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
        model['p_c'].append(p_c)
    for key in model.keys():
        model[key] = np.array(model[key], dtype='object')
    model['params'] = params
    return model


def schedule(task='vr', ratio=0.1, interval=0.5, press_rate=1):
    reward_rates = []
    N_a, N_r = 20000, 80000
    for lambda_a in press_rate:
        t_press = np.cumsum(npr.exponential(1 / lambda_a, N_a))
        if task == 'vr':
            reward_rates.append((len(t_press) / t_press[-1]) * ratio)
        elif task == 'vi':
            r = 0
            t_baits = np.cumsum(npr.exponential(interval, N_a))
            for bait in range(N_a-1):
                takes = np.logical_and(t_press > t_baits[bait], t_press < t_baits[bait+1])
                r += 1 if takes.sum() > 0 else 0
            reward_rates.append(r / t_press[-1])
    return reward_rates


# plt.figure()
# for idx in range(5):
#     plt.plot(np.array(model['Qval'])[:,idx,:])
#
# plt.figure()
# for idx in range(5):
#     plt.plot(np.array(model['Hval'])[:,idx,:])