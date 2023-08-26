from matplotlib import pyplot as plt
import numpy as np


def softmax(x, beta, axis=-1):
    kw = dict(axis=axis, keepdims=True)
    # make every value 0 or below, as exp(0) won't overflow
    xrel = x - x.max(**kw)
    exp_xrel = np.exp(beta * xrel)
    return exp_xrel / exp_xrel.sum(**kw)


def winner(Q, H):
    return np.array([Q[a] if np.abs(Q[a] - 0.5) > np.abs(H[a] - 0.5) else H[a] for a in range(len(Q))])


def movmedian(x, winsize, adapt=True, median=True):
    if adapt:
        if median:
            x_mean = [np.median(x[idx - np.min((idx, winsize)):idx]) for idx in np.arange(len(x))]
        else:
            x_mean = [np.mean(x[idx - np.min((idx, winsize)):idx]) for idx in np.arange(len(x))]
    else:
        if median:
            x_mean = [np.median(x[idx-winsize:idx]) for idx in np.arange(winsize, len(x))]
        else:
            x_mean = [np.mean(x[idx - winsize:idx]) for idx in np.arange(winsize, len(x))]
    return x_mean


def plot(type='learning', model=None, smooth=False):
    smoothwin = 50
    if type == 'learning':
        nsims = len(model)
        params = model[0]['params']
        rew = np.array([model[sim]['rew'] for sim in range(nsims)]).mean(axis=0)
        Qval = np.array([model[sim]['Qval'] for sim in range(nsims)]).mean(axis=0)
        Hval = np.array([model[sim]['Hval'] for sim in range(nsims)]).mean(axis=0)
        # p_a = np.array([model[sim]['p_a'] for sim in range(nsims)]).mean(axis=0)
        # rpe = np.array([model[sim]['rpe'] for sim in range(nsims)]).mean(axis=0)
        # ape = np.array([model[sim]['ape'] for sim in range(nsims)]).mean(axis=0)
        # rpe_var = np.array([model[sim]['rpe_var'] for sim in range(nsims)]).mean(axis=0)
        # ape_var = np.array([model[sim]['ape_var'] for sim in range(nsims)]).mean(axis=0)
        Ntrials, Nstates, Nactions = np.shape(Qval)
        plt.figure(figsize=(12, 3.5), dpi=80)
        plt.subplot(1, 4, 1)
        plt.plot(Qval[:, 0, 0], label='s=0, a=0', color='b', alpha=1)
        plt.plot(Qval[:, 0, 1], label='s=0, a=1', color='b', alpha=.3)
        plt.plot(Qval[:, 1, 0], label='s=1, a=0', color='r', alpha=.3)
        plt.plot(Qval[:, 1, 1], label='s=1, a=1', color='r', alpha=1)
        plt.legend()
        plt.ylim((-0.05, 1.05))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('Q value')
        plt.subplot(1, 4, 2)
        plt.plot(Hval[:, 0, 0], label='s=0, a=0', color='b', alpha=1)
        plt.plot(Hval[:, 0, 1], label='s=0, a=1', color='b', alpha=.3)
        plt.plot(Hval[:, 1, 0], label='s=1, a=0', color='r', alpha=.3)
        plt.plot(Hval[:, 1, 1], label='s=1, a=1', color='r', alpha=1)
        # plt.legend()
        plt.ylim((-0.05, 1.05))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('H value')
        plt.subplot(1, 4, 3)
        p_a = np.array([[softmax(winner(Qval[trial, state].flatten(), Hval[trial, state].flatten()), params['beta'])
                         for state in range(Nstates)] for trial in range(Ntrials)])
        plt.plot(np.convolve(p_a[:, 0, 0], np.ones(smoothwin)/smoothwin, mode='valid'), label='P(a=0|s=0)', color='b', alpha=1) \
            if smooth else plt.plot(p_a[:, 0, 0], label='p(a=0|s=0)', color='b', alpha=1)
        plt.plot(np.convolve(p_a[:, 1, 1], np.ones(smoothwin) / smoothwin, mode='valid'), label='P(a=1|s=1)', color='r', alpha=1) \
            if smooth else plt.plot(p_a[:, 1, 1], label='p(a=1|s=1)', color='r', alpha=1)
        plt.legend()
        plt.ylim((-0.05, 1.05))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('Policy')
        plt.subplot(1, 4, 4)
        # plt.plot(np.convolve(rew, np.ones(smoothwin) / smoothwin)) if smooth else plt.plot(rew)
        plt.plot(np.convolve(rew, np.ones(smoothwin) / smoothwin), color='k') \
            if smooth else plt.plot(movmedian(rew, winsize=100, adapt=False, median=False), color='k')
        plt.ylim((-0.05, 1.05))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('Average reward')
        # plt.suptitle(r'$\beta$ = ' + str(params['beta']) + '; $p_n$ = ' + str(params['p_n']), fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    elif type == 'softmax':
        plt.figure(figsize=(4, 4), dpi=80)
        n = 201
        x = np.linspace(0, 1, n)
        y1 = [softmax(np.array([x[idx], 1-x[idx]]), beta=1.5)[0] for idx in range(n)]
        y2 = [softmax(np.array([x[idx], 1 - x[idx]]), beta=3)[0] for idx in range(n)]
        plt.plot(x, x, '--', color='k')
        plt.plot(x, y1, color='grey')
        plt.plot(x, y2, color='grey')
        plt.ylim((-0.05, 1.05))
        plt.xticks([0, 1]), plt.yticks([0, 1])
    plt.show()
