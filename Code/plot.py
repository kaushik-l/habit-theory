import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import factorial
from theory import criticalbeta


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


def accumulate(stim, s, p_n, beta, bias=False):
    bel = [0.5]
    for idx in range(len(stim)):
        num = bel[-1] * ((1 - (stim[idx])) * p_n + (stim[idx]) * (1 - p_n))
        den = (bel[-1] * ((1 - (stim[idx])) * p_n + (stim[idx]) * (1 - p_n)) +
               (1 - bel[-1]) * ((1 - (stim[idx])) * (1 - p_n) + (stim[idx]) * p_n))
        p = num / den
        p_a = softmax(np.array([p, 1-p]), beta)
        if bias:
            bel.append(p_a[0])
        else:
            bel.append(p)
    return np.array(bel)


def plot(type='learning', model=None, vars=None, smooth=False):
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
        plt.ylim((-0.05, 1.05)), plt.xlim((2500, 7500))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('Q value')
        plt.subplot(1, 4, 2)
        plt.plot(Hval[:, 0, 0], label='s=0, a=0', color='b', alpha=1)
        plt.plot(Hval[:, 0, 1], label='s=0, a=1', color='b', alpha=.3)
        plt.plot(Hval[:, 1, 0], label='s=1, a=0', color='r', alpha=.3)
        plt.plot(Hval[:, 1, 1], label='s=1, a=1', color='r', alpha=1)
        # plt.legend()
        plt.ylim((-0.05, 1.05)), plt.xlim((2500, 7500))
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
        plt.ylim((-0.05, 1.05)), plt.xlim((2500, 7500))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('Policy')
        plt.subplot(1, 4, 4)
        # plt.plot(np.convolve(rew, np.ones(smoothwin) / smoothwin)) if smooth else plt.plot(rew)
        plt.plot(np.convolve(rew, np.ones(smoothwin) / smoothwin), color='k') \
            if smooth else plt.plot(movmedian(rew, winsize=100, adapt=False, median=False), color='k')
        plt.ylim((-0.05, 1.05)), plt.xlim((2500, 7500))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('Average reward')
        # plt.suptitle(r'$\beta$ = ' + str(params['beta']) + '; $p_n$ = ' + str(params['p_n']), fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    elif type == 'sequential_learning':
        nsims = len(model)
        params = model[0]['params']
        rew = np.array([model[sim]['rew'] for sim in range(nsims)]).mean(axis=0)
        Qval = np.array([model[sim]['Qval'] for sim in range(nsims)]).mean(axis=0)
        Hval = np.array([model[sim]['Hval'] for sim in range(nsims)]).mean(axis=0)
        Ntrials, Nstates, Nactions = np.shape(Qval)
        plt.figure(figsize=(12, 3.5), dpi=80)
        plt.subplot(1, 4, 1)
        plt.plot(Qval[:, 0, 1], label='s=0, a=1', color='b', alpha=1)
        plt.plot(Qval[:, 1, 1], label='s=1, a=1', color='r', alpha=1)
        plt.legend()
        plt.ylim((-0.05, 1.05))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('Q value')
        plt.subplot(1, 4, 2)
        plt.plot(Hval[:, 0, 1], label='s=0, a=1', color='b', alpha=1)
        plt.plot(Hval[:, 1, 1], label='s=1, a=1', color='r', alpha=1)
        plt.ylim((-0.05, 1.05))
        plt.grid(axis='y')
        plt.xlabel('Trial'), plt.ylabel('H value')
        plt.subplot(1, 4, 3)
        p_a = np.array([[softmax(winner(Qval[trial, state].flatten(), Hval[trial, state].flatten()), params['beta'])
                         for state in range(Nstates)] for trial in range(Ntrials)])
        plt.plot(np.convolve(p_a[:, 0, 1], np.ones(smoothwin)/smoothwin, mode='valid'), label='P(a=0|s=0)', color='b', alpha=1) \
            if smooth else plt.plot(p_a[:, 0, 1], label='p(a=0|s=0)', color='b', alpha=1)
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
        y1 = [softmax(np.array([x[idx], 0]), beta=1.5)[0] for idx in range(n)]
        y2 = [softmax(np.array([x[idx], 0]), beta=2.5)[0] for idx in range(n)]
        plt.plot(x, x, '--', color='k')
        plt.plot(x, y1, color='grey')
        plt.plot(x, y2, color='grey')
        plt.ylim((-0.05, 1.05))
        plt.xticks([0, 1]), plt.yticks([0, 1])

    elif type == 'theory':
        cmap = matplotlib.cm.get_cmap('RdBu')
        Q, H, p, r, betacrit, noise, beta = vars
        plt.figure(figsize=(12, 3.5), dpi=80)
        plt.subplot(1, 4, 1)
        plt.imshow(Q.T, extent=[0, 1, 4, 1], vmin=0, vmax=1, cmap=cmap, aspect='auto')
        plt.plot(noise, betacrit, 'k')
        plt.gca().invert_yaxis()
        plt.ylim((1, 4)), plt.yticks([1, 2, 3, 4]), plt.xticks([0, .25, .5, .75, 1])
        plt.xlabel(r'Reward probability $(p_r)$'), plt.ylabel(r'Degree of exploitation $(\beta)$')
        plt.subplot(1, 4, 2)
        plt.imshow(H.T, extent=[0, 1, 4, 1], vmin=0, vmax=1, cmap=cmap, aspect='auto')
        plt.plot(noise, betacrit, 'k')
        plt.gca().invert_yaxis()
        plt.ylim((1, 4)), plt.yticks([1, 2, 3, 4]), plt.xticks([0, .25, .5, .75, 1])
        plt.xlabel(r'Reward probability $(p_r)$'), plt.ylabel(r'Degree of exploitation $(\beta)$')
        plt.subplot(1, 4, 3)
        plt.imshow(p.T, extent=[0, 1, 4, 1], vmin=0, vmax=1, cmap=cmap, aspect='auto')
        plt.plot(noise, betacrit, 'k')
        plt.gca().invert_yaxis()
        plt.ylim((1, 4)), plt.yticks([1, 2, 3, 4]), plt.xticks([0, .25, .5, .75, 1])
        plt.xlabel(r'Reward probability $(p_r)$'), plt.ylabel(r'Degree of exploitation $(\beta)$')
        plt.subplot(1, 4, 4)
        plt.imshow(1 - r.T, extent=[0, 1, 4, 1], vmin=0, vmax=1, cmap=cmap, aspect='auto')
        plt.plot(noise, betacrit, 'k')
        plt.gca().invert_yaxis()
        plt.ylim((1, 4)), plt.yticks([1, 2, 3, 4]), plt.xticks([0, .25, .5, .75, 1])
        plt.xlabel(r'Reward probability $(p_r)$'), plt.ylabel(r'Degree of exploitation $(\beta)$')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    elif type == 'inference':
        mdl = model[4]
        trlindex = np.arange(-10, 11)
        plt.figure(figsize=(7, 3.5), dpi=80)
        plt.subplot(1, 2, 1)
        states = mdl['state'][4989:5010]
        actions = mdl['action'][4989:5010]
        rewards = mdl['rew'][4989:5010]
        p_c = mdl['p_c'][4989:5010]
        for idx in range(len(trlindex)):
            if states[idx] == 0 and actions[idx] == 0:
                plt.scatter(trlindex[idx], rewards[idx], s=80, facecolors='b', edgecolors='b')
            elif states[idx] == 0 and actions[idx] == 1:
                plt.scatter(trlindex[idx], rewards[idx], s=80, facecolors='none', edgecolors='b')
            elif states[idx] == 1 and actions[idx] == 0:
                plt.scatter(trlindex[idx], rewards[idx], s=80, facecolors='r', edgecolors='r')
            elif states[idx] == 1 and actions[idx] == 1:
                plt.scatter(trlindex[idx], rewards[idx], s=80, facecolors='none', edgecolors='r')
        plt.subplot(1, 2, 2)
        plt.plot(trlindex, 1 / (1 + np.exp(p_c)))

    elif type == 'distal':
        p_a, beta, qvals, hvals, Qlist, Hlist, betacrit = vars
        plt.figure(figsize=(12, 3.5), dpi=80)
        plt.subplot(1, 4, 2)
        plt.plot(p_a, np.flip(qvals))
        plt.plot(p_a, np.flip(hvals))
        plt.xlim((0, 0.75)), plt.ylim((0.7, 0.95)), plt.xticks([0, .25, .5, .75])
        plt.xlabel(r'Action probability $Pr(a^P|a^D)$'), plt.ylabel(r'Q and H values')
        plt.subplot(1, 4, 3)
        cmap = matplotlib.cm.get_cmap('RdBu')
        plt.imshow(Hlist.T - Qlist.T, extent=[1, 0, 4, 1], vmin=-0.15, vmax=0.15, cmap=cmap, aspect='auto')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.xlim((0, 0.75)), plt.ylim((1, 4)), plt.yticks([1, 2, 3, 4]), plt.xticks([0, .25, .5, .75])
        plt.plot(1-p_a, betacrit, 'k')
        plt.xlabel(r'Action probability $Pr(a^P|a^D)$'), plt.ylabel(r'Degree of exploitation $(\beta)$')
        # plt.colorbar()
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplot(1, 4, 4)
        plt.errorbar([0.8, 1.2, 1.8, 2.2], [3.1, 1.9, 2.8, 2.95], yerr=[0.25, 0.4, 0.35, 0.3], ls='none')
        plt.xlim((0.5, 2.5)), plt.ylim((0, 3.5)), plt.xticks([1, 2]), plt.yticks([0, 1, 2, 3])

    elif type == 'delay':
        delay, beta, qvals, hvals, Qlist, Hlist, betacrit = vars
        plt.figure(figsize=(12, 3.5), dpi=80)
        plt.subplot(1, 4, 2)
        plt.plot(delay, qvals)
        plt.plot(delay, hvals)
        plt.xlim((0, 3)), plt.ylim((0.7, 1)), plt.xticks([0, 1, 2])
        plt.xlabel(r'Delay $T$'), plt.ylabel(r'Q and H values')
        plt.subplot(1, 4, 3)
        cmap = matplotlib.cm.get_cmap('RdBu')
        plt.imshow(Hlist.T - Qlist.T, extent=[0, 3, 4, 1], vmin=-0.15, vmax=0.15, cmap=cmap, aspect='auto')
        plt.gca().invert_yaxis()
        plt.xlim((0, 3)), plt.ylim((1, 4)), plt.yticks([1, 2, 3, 4]), plt.xticks([0, 1, 2, 3])
        plt.plot(delay, betacrit, 'k')
        plt.xlabel(r'Delay $(T)$'), plt.ylabel(r'Degree of exploitation $(\beta)$')
        # plt.colorbar()
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplot(1, 4, 4)
        plt.errorbar([0.8, 1.2, 1.8, 2.2], [8.9, 5.2, 5.05, 5], yerr=[1.2, 1.1, 0.8, 1.2], ls='none')
        plt.xlim((0.5, 2.5)), plt.ylim((0, 15)), plt.xticks([1, 2]), plt.yticks([0, 5, 10, 15])

    elif type == 'schedule':
        ti, beta, qvals_vi, hvals_vi, qvals_fi, hvals_fi, Qlist, Hlist = vars
        plt.figure(figsize=(12, 3.5), dpi=80)
        plt.subplot(1, 4, 2)
        plt.plot(ti, qvals_vi)
        plt.plot(ti, hvals_vi)
        plt.plot(ti, qvals_fi)
        plt.plot(ti, hvals_fi)
        plt.xlim((0, 60)), plt.ylim((0, 1)), plt.xticks([0, 30, 60])
        plt.xlabel(r'Time $T$'), plt.ylabel(r'Q and H values')
        plt.subplot(1, 4, 3)
        cmap = matplotlib.cm.get_cmap('RdBu')
        HminusQ = np.zeros((np.shape(Qlist)[1], np.shape(Qlist)[2]))
        for idx1 in range(np.shape(Qlist)[1]):
            for idx2 in range(np.shape(Qlist)[2]):
                HminusQ[idx1, idx2] = np.mean(np.concatenate((Hlist[Hlist[:, idx1, idx2] > 0.5, idx1, idx2] - Qlist[Hlist[:, idx1, idx2] > 0.5, idx1, idx2],
                                        Qlist[Hlist[:, idx1, idx2] < 0.5, idx1, idx2] - Hlist[Hlist[:, idx1, idx2] < 0.5, idx1, idx2])))
        plt.imshow(HminusQ, extent=[0, 60, 4, 1], vmin=-0.2, vmax=0.2, cmap=cmap, aspect='auto')
        plt.gca().invert_yaxis()
        plt.plot(range(np.shape(Qlist)[2]), beta[[np.argwhere(HminusQ[:, idx] > 0)[0][0] for idx in range(np.shape(Qlist)[2])]], 'k')
        plt.xlim((0, 60)), plt.ylim((1, 4)), plt.yticks([1, 2, 3, 4])
        plt.xlabel(r'Schedule uncertainty $(T)$'), plt.ylabel(r'Degree of exploitation $(\beta)$')
        # plt.colorbar()
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplot(1, 4, 4)
        plt.errorbar([0.8, 1.2, 1.8, 2.2], [1.25, 0.7, 0.85, 0.75], yerr=[0.15, 0.12, 0.15, 0.12], ls='none')
        plt.xlim((0.5, 2.5)), plt.ylim((0, 1.5)), plt.xticks([1, 2]), plt.yticks([0, 0.5, 1, 1.5])

    elif type == 'accumulate':
        model1, model2 = model

        plt.figure(figsize=(12, 12), dpi=80)

        s1 = [model1[0]['stim'][trial].sum() for trial in range(100)]
        trialindex = np.logical_or(np.array(s1) == 4, np.array(s1) == 6)
        s1 = [model1[0]['stim'][trial] for trial in range(100)]
        s1 = np.array(s1)[np.array(trialindex).flatten()]
        st1 = [model1[0]['state'][trial] for trial in range(100)]
        st1 = np.array(st1)[np.array(trialindex).flatten()]
        b1 = []
        for idx, stim in enumerate(s1):
            b1.append(accumulate(stim, st1[idx], 0.4, 2, bias=False) + np.random.normal(0, 0.04, 11))
        plt.subplot(2, 2, 1)
        plt.plot(np.array(b1).T, color='b', alpha=0.2)
        plt.xlabel(r'Click number', fontsize=14), plt.ylabel(r'Q value', fontsize=14)
        plt.ylim((0, 1))

        s1 = [model1[0]['stim'][trial].sum() for trial in range(100)]
        trialindex = np.logical_or(np.array(s1) == 4, np.array(s1) == 6)
        s1 = [model1[0]['stim'][trial] for trial in range(100)]
        s1 = np.array(s1)[np.array(trialindex).flatten()]
        st1 = [model1[0]['state'][trial] for trial in range(100)]
        st1 = np.array(st1)[np.array(trialindex).flatten()]
        b1 = []
        for idx, stim in enumerate(s1):
            b1.append(accumulate(stim, st1[idx], 0.4, 2, bias=True) + np.random.normal(0, 0.04, 11))
        plt.subplot(2, 2, 3)
        plt.plot(np.array(b1).T, color='b', alpha=0.2)
        plt.xlabel(r'Click number', fontsize=14), plt.ylabel(r'H value', fontsize=14)
        plt.ylim((0, 1))

        s1 = [model1[0]['stim'][trial].sum() for trial in range(100)]
        trialindex = np.logical_or(np.array(s1) == 4, np.array(s1) == 6)
        s1 = [model1[0]['stim'][trial] for trial in range(100)]
        s1 = np.array(s1)[np.array(trialindex).flatten()]
        st1 = [model1[0]['state'][trial] for trial in range(100)]
        st1 = np.array(st1)[np.array(trialindex).flatten()]
        b1 = []
        for idx, stim in enumerate(s1):
            b1.append(accumulate(stim, st1[idx], 0.4, 2.8, bias=False) + np.random.normal(0, 0.04, 11))
        plt.subplot(2, 2, 2)
        plt.plot(np.array(b1).T, color='r', alpha=0.2)
        plt.xlabel(r'Click number', fontsize=14), plt.ylabel(r'Q value', fontsize=14)
        plt.ylim((0, 1))

        s1 = [model1[0]['stim'][trial].sum() for trial in range(100)]
        trialindex = np.logical_or(np.array(s1) == 4, np.array(s1) == 6)
        s1 = [model1[0]['stim'][trial] for trial in range(100)]
        s1 = np.array(s1)[np.array(trialindex).flatten()]
        st1 = [model1[0]['state'][trial] for trial in range(100)]
        st1 = np.array(st1)[np.array(trialindex).flatten()]
        b1 = []
        for idx, stim in enumerate(s1):
            b1.append(accumulate(stim, st1[idx], 0.4, 2.8, bias=True) + np.random.normal(0, 0.04, 11))
        plt.subplot(2, 2, 4)
        plt.plot(np.array(b1).T, color='r', alpha=0.2)
        plt.xlabel(r'Click number', fontsize=14), plt.ylabel(r'H value', fontsize=14)
        plt.ylim((0, 1))


        stim1, resp1, weights1, p_a1, stim2, resp2, weights2, p_a2 = vars
        plt.figure(figsize=(12, 3.5), dpi=80)
        plt.subplot(1, 4, 1)
        plt.errorbar(range(11), np.mean(p_a1, axis=0), np.std(p_a1, axis=0))
        plt.xlim((-1, 11)), plt.xticks(ticks=[0, 2, 4, 6, 8, 10], labels=[-10, -6, -2, 2, 6, 10])
        plt.ylim((0, 1)), plt.yticks(ticks=[0, .2, .4, .6, .8, 1])
        plt.xlabel(r'Clicks (R - L)', fontsize=14), plt.ylabel(r'P(R)', fontsize=14)
        plt.subplot(1, 4, 2)
        plt.errorbar(range(11), np.mean(p_a2, axis=0), np.std(p_a2, axis=0))
        plt.xlim((-1, 11)), plt.xticks(ticks=[0, 2, 4, 6, 8, 10], labels=[-10, -6, -2, 2, 6, 10])
        plt.ylim((0, 1)), plt.yticks(ticks=[0, .2, .4, .6, .8, 1])
        plt.xlabel(r'Clicks (R - L)', fontsize=14), plt.ylabel(r'P(R)', fontsize=14)
        plt.subplot(1, 4, 3)
        plt.errorbar(range(10), np.mean(weights1, axis=0), np.std(weights1, axis=0))
        plt.xlim((-1, 10)), plt.xticks(ticks=[1, 3, 5, 7, 9], labels=[2, 4, 6, 8, 10])
        plt.ylim((0, 4)), plt.yticks(ticks=[0, 1, 2, 3, 4])
        plt.xlabel(r'Click number', fontsize=14), plt.ylabel(r'Regression weight', fontsize=14)
        plt.subplot(1, 4, 4)
        plt.errorbar(range(10), np.mean(weights2, axis=0), np.std(weights2, axis=0))
        plt.xlim((-1, 10)), plt.xticks(ticks=[1, 3, 5, 7, 9], labels=[2, 4, 6, 8, 10])
        plt.ylim((0, 4)), plt.yticks(ticks=[0, 1, 2, 3, 4])
        plt.xlabel(r'Click number', fontsize=14), plt.ylabel(r'Regression weight', fontsize=14)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # load Keung et al. data
        import csv
        plt.figure(figsize=(12, 12), dpi=80)
        files = ['Primacy.csv', 'Intermediacy.csv', 'Uniform.csv', 'Recency.csv']
        for idx, file in enumerate(files):
            with open('..//Data//Keung//' + file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                click, coeff = [], []
                for row in csv_reader:
                    click.append(row[0])
                    coeff.append(row[1])
                    line_count += 1
                plt.subplot(2, 2, idx+1)
                k = 1 if idx == 0 else 0
                plt.plot(np.array(click)[k:].astype(float), np.array(coeff)[k:].astype(float))
                plt.xlim((0, 21)), plt.ylim((0, 0.7)), plt.xticks(ticks=[]), plt.yticks(ticks=[])
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])


    plt.show()
