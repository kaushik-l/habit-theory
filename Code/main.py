import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli, norm
import numpy.random as npr
from train import train_single, train_dual


def movmean(x, winsize, adapt=True):
    if adapt:
        x_mean = [x[idx - np.min((idx, winsize)):idx].mean() for idx in np.arange(len(x))]
    else:
        x_mean = [x[idx-winsize:idx].mean() for idx in np.arange(winsize, len(x))]
    return x_mean


def plot(learning, system='dual', smooth=False):
    smoothwin = 50
    nsims = len(learning)
    if system == 'single':
        rew = np.array([learning[sim]['rew'] for sim in range(nsims)]).mean(axis=0)
        Gval = np.array([learning[sim]['Gval'] for sim in range(nsims)]).mean(axis=0)
        rpe = np.array([learning[sim]['rpe'] for sim in range(nsims)]).mean(axis=0)
        ape = np.array([learning[sim]['ape'] for sim in range(nsims)]).mean(axis=0)
        rpe_var = np.array([learning[sim]['rpe_var'] for sim in range(nsims)]).mean(axis=0)
        ape_var = np.array([learning[sim]['ape_var'] for sim in range(nsims)]).mean(axis=0)
        pe = np.array([learning[sim]['pe'] for sim in range(nsims)]).mean(axis=0)
        plt.figure(figsize=(12, 12), dpi=80)
        plt.subplot(2, 2, 1)
        plt.plot(Gval[:, 0, 0], label='s=0, a=0')
        plt.plot(Gval[:, 0, 1], label='s=0, a=1')
        plt.plot(Gval[:, 1, 0], label='s=1, a=0')
        plt.plot(Gval[:, 1, 1], label='s=1, a=1')
        plt.legend()
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.xlabel('Trial'), plt.ylabel('G value')
        plt.subplot(2, 2, 2)
        plt.plot(np.convolve(rpe/rpe_var, np.ones(smoothwin)/smoothwin, mode='valid')) if smooth else plt.plot(rpe/rpe_var)
        plt.plot(np.convolve(ape/ape_var, np.ones(smoothwin) / smoothwin, mode='valid')) if smooth else plt.plot(ape/ape_var)
        plt.plot(np.convolve(pe, np.ones(smoothwin) / smoothwin, mode='valid')) if smooth else plt.plot(pe)
        plt.grid()
        plt.xlabel('Trial'), plt.ylabel('Weighted PE')
        plt.subplot(2, 2, 3)
        plt.plot(np.convolve(rpe_var, np.ones(smoothwin) / smoothwin, mode='valid')) if smooth else plt.plot(rpe_var)
        plt.plot(np.convolve(ape_var, np.ones(smoothwin) / smoothwin, mode='valid')) if smooth else plt.plot(ape_var)
        plt.grid()
        plt.xlabel('Trial'), plt.ylabel('Uncertainty in PE')
        plt.subplot(2, 2, 4)
        plt.plot(np.convolve(rew, np.ones(smoothwin) / smoothwin)) if smooth else plt.plot(rew)
        plt.grid()
        plt.xlabel('Trial'), plt.ylabel('Average reward')
        plt.suptitle(r'$\beta$ = ' + str(params['beta']) + '; $p_n$ = ' + str(params['p_n']), fontsize=16)
    elif system == 'dual':
        rew = np.array([learning[sim]['rew'] for sim in range(nsims)]).mean(axis=0)
        Qval = np.array([learning[sim]['Qval'] for sim in range(nsims)]).mean(axis=0)
        Hval = np.array([learning[sim]['Hval'] for sim in range(nsims)]).mean(axis=0)
        rpe = np.array([learning[sim]['rpe'] for sim in range(nsims)]).mean(axis=0)
        ape = np.array([learning[sim]['ape'] for sim in range(nsims)]).mean(axis=0)
        rpe_var = np.array([learning[sim]['rpe_var'] for sim in range(nsims)]).mean(axis=0)
        ape_var = np.array([learning[sim]['ape_var'] for sim in range(nsims)]).mean(axis=0)
        plt.figure(figsize=(12, 12), dpi=80)
        plt.subplot(2, 2, 1)
        plt.plot(Qval[:, 0, 0], label='s=0, a=0')
        plt.plot(Qval[:, 0, 1], label='s=0, a=1')
        plt.plot(Qval[:, 1, 0], label='s=1, a=0')
        plt.plot(Qval[:, 1, 1], label='s=1, a=1')
        plt.legend()
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.xlabel('Trial'), plt.ylabel('Q value')
        plt.subplot(2, 2, 2)
        plt.plot(Hval[:, 0, 0], label='s=0, a=0')
        plt.plot(Hval[:, 0, 1], label='s=0, a=1')
        plt.plot(Hval[:, 1, 0], label='s=1, a=0')
        plt.plot(Hval[:, 1, 1], label='s=1, a=1')
        plt.legend()
        plt.ylim((-0.1, 1.1))
        plt.grid()
        plt.xlabel('Trial'), plt.ylabel('H value')
        plt.subplot(2, 2, 3)
        plt.plot(np.convolve(rpe/rpe_var, np.ones(smoothwin)/smoothwin, mode='valid')) if smooth else plt.plot(rpe/rpe_var)
        plt.plot(np.convolve(ape/ape_var, np.ones(smoothwin) / smoothwin, mode='valid')) if smooth else plt.plot(ape/ape_var)
        plt.grid()
        plt.xscale('log')
        plt.xlabel('Trial'), plt.ylabel('Weighted PE')
        plt.subplot(2, 2, 4)
        plt.plot(np.convolve(rew, np.ones(smoothwin) / smoothwin)) if smooth else plt.plot(rew)
        plt.grid()
        plt.xlabel('Trial'), plt.ylabel('Average reward')
        plt.suptitle(r'$\beta$ = ' + str(params['beta']) + '; $p_n$ = ' + str(params['p_n']), fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# task parameters
S = np.array([0, 1])                                # state space
A = np.array([0, 1])                                # action space
R = np.array(([1, 0], [0, 1]))                      # reward, punishment
params = {'beta': 3, 'p_n': 0.2, 'lr': 2e-4, 'Ntrials': 20000, 'eps': 1e-3}
learning = []
nsims = 1                 # number of simulations
# dual learning system
for idx in range(nsims):
    print('simulation ' + str(idx+1) + '/' + str(nsims), end='\r')
    params['Q_init'] = 0.5 * np.ones((2, 2))
    params['H_init'] = 0.5 * np.ones((2, 2))
    learning.append(train_dual(S, A, R, params, task='learning'))
plot(learning, system='dual', smooth=True)

# single learning system
# for idx in range(nsims):
#     print('simulation ' + str(idx) + '/' + str(nsims), end='\r')
#     params['G_init'] = 0 * np.ones((2, 2))
#     learning.append(train_single(S, A, R, params))
# plot(learning, system='single', smooth=True)
