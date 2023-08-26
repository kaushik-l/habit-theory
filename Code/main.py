import numpy as np
from train import train
from plot import plot
from theory import steadystate
from matplotlib import pyplot as plt

simulate_learning = False
theory_learning = True
plot_learning = False
plot_softmax = False
plot_theory = True

if simulate_learning:
    # task parameters
    S = np.array([0, 1])                                # state space
    A = np.array([0, 1])                                # action space
    R = np.array(([1, 0], [0, 1]))                      # reward, punishment
    params = {'beta': 3, 'p_n': 0.4, 'lr': 1e-3, 'Ntrials': 5000, 'eps': 1e-3}
    model = []
    nsims = 10                 # number of simulations
    # dual learning system
    for idx in range(nsims):
        print('simulation ' + str(idx+1) + '/' + str(nsims), end='\r')
        params['Q_init'] = 0.5 * np.ones((2, 2))
        params['H_init'] = 0.5 * np.ones((2, 2))
        model.append(train(S, A, R, params, task='learning'))

if theory_learning:
    nvals = 56
    noise = np.linspace(0, 0.5, nvals)
    beta = np.linspace(1, 4, nvals)
    Qlist, Hlist, plist, rlist = (np.zeros((nvals, nvals)), np.zeros((nvals, nvals)),
                                  np.zeros((nvals, nvals)), np.zeros((nvals, nvals)))
    for idx1 in range(nvals):
        for idx2 in range(nvals):
            Q, H, p, r = steadystate(noise[idx1], beta[idx2])
            Qlist[idx1, idx2] = Q
            Hlist[idx1, idx2] = H
            plist[idx1, idx2] = p
            rlist[idx1, idx2] = r


if plot_theory:
    plt.figure(figsize=(12, 3.5), dpi=80)
    plt.subplot(1, 4, 1)
    plt.imshow(Qlist.T, extent=[0, 0.5, 4, 1], vmin=0.5, vmax=1, cmap='coolwarm', aspect='auto')
    plt.gca().invert_yaxis()
    plt.subplot(1, 4, 2)
    plt.imshow(Hlist.T, extent=[0, 0.5, 4, 1], vmin=0.5, vmax=1, cmap='coolwarm', aspect='auto')
    plt.gca().invert_yaxis()
    plt.subplot(1, 4, 3)
    plt.imshow(plist.T, extent=[0, 0.5, 4, 1], vmin=0.5, vmax=1, cmap='coolwarm', aspect='auto')
    plt.gca().invert_yaxis()
    plt.subplot(1, 4, 4)
    plt.imshow(rlist.T, extent=[0, 0.5, 4, 1], vmin=0.5, vmax=1, cmap='coolwarm', aspect='auto')
    plt.gca().invert_yaxis()

if plot_learning:
    plot(type='learning', model=model, smooth=False)

if plot_softmax:
    plot(type='softmax')
