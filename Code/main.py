import numpy as np
from train import train, schedule
from plot import plot
import theory
from sklearn.linear_model import LogisticRegression
from scipy.stats import expon, poisson
from matplotlib import pyplot as plt

# simulations
simulate_learning = False
simulate_reversal = False
simulate_sequential_learning = False
simulate_sequential_reversal = False
simulate_inference_reversal = False
simulate_schedule = False
simulate_accumulation = True

# theory
theory_learning = False
theory_distal = False
theory_delay = False
theory_schedule = False

# plots
plot_simulation = False
plot_softmax = False
plot_theory = False
plot_inference = False
plot_distal = False
plot_delay = False
plot_schedule = False
plot_accumulation = True

if simulate_learning:
    # task parameters
    S = np.array([0, 1])                                # state space
    A = np.array([0, 1])                                # action space
    R = np.array(([1, 0], [0, 1]))                      # reward, punishment
    params = {'beta': 1.5, 'p_n': 0.2, 'lr': 1e-3, 'Ntrials': 10000, 'eps': 1e-3}
    model = []
    nsims = 10                 # number of simulations
    # dual learning system
    for idx in range(nsims):
        print('simulation ' + str(idx+1) + '/' + str(nsims), end='\r')
        params['Q_init'] = 0.5 * np.ones((2, 2))
        params['H_init'] = 0.5 * np.ones((2, 2))
        model.append(train(S, A, R, params, task='learning'))

if simulate_reversal:
    # task parameters
    S = np.array([0, 1])                                # state space
    A = np.array([0, 1])                                # action space
    R = np.array(([1, 0], [0, 1]))                      # reward, punishment
    params = {'beta': 3, 'p_n': 0.2, 'lr': 1e-3, 'Ntrials': 10000, 'eps': 1e-3}
    model = []
    nsims = 10                 # number of simulations
    # dual learning system
    for idx in range(nsims):
        print('simulation ' + str(idx+1) + '/' + str(nsims), end='\r')
        params['Q_init'] = 0.5 * np.ones((2, 2))
        params['H_init'] = 0.5 * np.ones((2, 2))
        model.append(train(S, A, R, params, task='reversal'))

if simulate_sequential_learning:
    # task parameters
    S = np.array([0, 1])                                # state space
    A = np.array([0, 1])                                # action space
    R = np.array(([0, 1], [0, 1]))                      # reward, punishment
    params = {'beta': 2, 'p_n': 0.1, 'lr': 1e-3, 'Ntrials': 5000, 'eps': 1e-3}
    model = []
    nsims = 10                 # number of simulations
    # dual learning system
    for idx in range(nsims):
        print('simulation ' + str(idx+1) + '/' + str(nsims), end='\r')
        params['Q_init'] = 0.5 * np.ones((2, 2))
        params['H_init'] = 0.5 * np.ones((2, 2))
        model.append(train(S, A, R, params, task='sequential_learning'))

if simulate_sequential_reversal:
    # task parameters
    S = np.array([0, 1])                                # state space
    A = np.array([0, 1])                                # action space
    R = np.array(([0, 0], [0, 1]))                      # reward, punishment
    params = {'beta': 2, 'p_n': 0.1, 'lr': 1e-3, 'Ntrials': 10000, 'eps': 1e-3}
    model = []
    nsims = 10                 # number of simulations
    # dual learning system
    for idx in range(nsims):
        print('simulation ' + str(idx+1) + '/' + str(nsims), end='\r')
        params['Q_init'] = 0.5 * np.ones((2, 2))
        params['H_init'] = 0.5 * np.ones((2, 2))
        model.append(train(S, A, R, params, task='sequential_reversal'))

if simulate_schedule:
    press_rates = np.logspace(np.log10(.5), np.log10(40), 40)[:, None]  # max 20, spacing 25
    reward_rates_vr = np.zeros((4, 40))
    reward_rates_vi = np.zeros((4, 40))
    print('simulating variable ratio schedule ', end='\r')
    ratios = [1/2, 1/3, 1/5, 1/20]
    for idx, ratio in enumerate(ratios):
        reward_rates_vr[idx, :] = schedule(task='vr', ratio=ratio, press_rate=press_rates)
    print('simulating variable interval schedule ', end='\r')
    intervals = [1, 0.5, 0.25, 0.1]
    for idx, interval in enumerate(intervals):
        reward_rates_vi[idx, :] = schedule(task='vi', interval=interval, press_rate=press_rates)
    model = {'press_rates': press_rates, 'reward_rates_vr': reward_rates_vr, 'reward_rates_vi': reward_rates_vi}

if simulate_inference_reversal:
    # task parameters
    S = np.array([0, 1])                                # state space
    A = np.array([0, 1])                                # action space
    R = np.array(([1, 0], [0, 1]))                      # reward, punishment
    params = {'beta': 3, 'p_n': 0.2, 'lr': 1e-3, 'Ntrials': 10000, 'eps': 1e-3}
    model = []
    nsims = 10                 # number of simulations
    # dual learning system
    for idx in range(nsims):
        print('simulation ' + str(idx+1) + '/' + str(nsims), end='\r')
        params['Q_init'] = 0.5 * np.ones((2, 2))
        params['H_init'] = 0.5 * np.ones((2, 2))
        model.append(train(S, A, R, params, task='inference_reversal'))

if simulate_accumulation:
    S = np.array([0, 1])  # state space
    A = np.array([0, 1])  # action space
    R = np.array(([1, 0], [0, 1]))  # reward, punishment
    params = {'beta': 2.2, 'p_n': 0.4, 'lr': 5*1e-3, 'Ntrials': 20000, 'eps': 1e-3, 'adapt': 0}
    model1, model2, model3, model4 = [], [], [], []
    weights1, weights2, weights3, weights4 = [], [], [], []
    p_a1, p_a2, p_a3, p_a4 = [], [], [], []
    nsims = 10                 # number of simulations
    # dual learning system
    for idx in range(nsims):
        params['adapt'] = 0
        params['beta'] = 2.2
        params['Q_init'] = 0.5 * np.ones((7, 2))
        params['H_init'] = 0.5 * np.ones((7, 2))
        model1.append(train(S, A, R, params, task='accumulation'))
        stim1 = [model1[idx]['stim'][trial].sum() for trial in range(model1[idx]['params']['Ntrials'])]
        resp1 = [model1[idx]['action'][trial] for trial in range(model1[idx]['params']['Ntrials'])]
        trialindex = [np.logical_or(np.array(stim1) == 4, np.array(stim1) == 6)]
        stimlist = model1[idx]['stim'][trialindex]
        resplist = model1[idx]['action'][trialindex]
        regmodel = LogisticRegression(random_state=0).fit(X=stimlist[-2000:].astype(int), y=resplist[-2000:].astype(int))
        weights1.append(regmodel.coef_.flatten())
        stim_temp = np.array(stim1[-10000:])
        resp_temp = np.array(resp1[-10000:])
        p_a1.append(np.array([np.mean(resp_temp[stim_temp == s]) for s in np.unique(stim1)]))

        params['adapt'] = 0
        params['beta'] = 2.8
        params['Q_init'] = 0.5 * np.ones((7, 2))
        params['H_init'] = 0.5 * np.ones((7, 2))
        model2.append(train(S, A, R, params, task='accumulation'))
        stim2 = [model2[idx]['stim'][trial].sum() for trial in range(model2[idx]['params']['Ntrials'])]
        resp2 = [model2[idx]['action'][trial] for trial in range(model2[idx]['params']['Ntrials'])]
        trialindex = [np.logical_or(np.array(stim2) == 4, np.array(stim2) == 6)]
        stimlist = model2[idx]['stim'][trialindex]
        resplist = model2[idx]['action'][trialindex]
        regmodel = LogisticRegression(random_state=0).fit(X=stimlist[-2000:].astype(int), y=resplist[-2000:].astype(int))
        weights2.append(regmodel.coef_.flatten())
        stim_temp = np.array(stim2[-10000:])
        resp_temp = np.array(resp2[-10000:])
        p_a2.append(np.array([np.mean(resp_temp[stim_temp == s]) for s in np.unique(stim2)]))

        params['adapt'] = 1
        params['beta'] = 2.2
        params['Q_init'] = 0.5 * np.ones((7, 2))
        params['H_init'] = 0.5 * np.ones((7, 2))
        model3.append(train(S, A, R, params, task='accumulation'))
        stim3 = [model3[idx]['stim'][trial].sum() for trial in range(model3[idx]['params']['Ntrials'])]
        resp3 = [model3[idx]['action'][trial] for trial in range(model3[idx]['params']['Ntrials'])]
        trialindex = [np.logical_or(np.array(stim3) == 4, np.array(stim3) == 6)]
        stimlist = model3[idx]['stim'][trialindex]
        resplist = model3[idx]['action'][trialindex]
        regmodel = LogisticRegression(random_state=0).fit(X=stimlist[-2000:].astype(int), y=resplist[-2000:].astype(int))
        weights3.append(regmodel.coef_.flatten())
        stim_temp = np.array(stim3[-10000:])
        resp_temp = np.array(resp3[-10000:])
        p_a3.append(np.array([np.mean(resp_temp[stim_temp == s]) for s in np.unique(stim3)]))

        params['adapt'] = 2
        params['beta'] = 2.2
        params['Q_init'] = 0.5 * np.ones((7, 2))
        params['H_init'] = 0.5 * np.ones((7, 2))
        model4.append(train(S, A, R, params, task='accumulation'))
        stim4 = [model4[idx]['stim'][trial].sum() for trial in range(model4[idx]['params']['Ntrials'])]
        resp4 = [model4[idx]['action'][trial] for trial in range(model4[idx]['params']['Ntrials'])]
        trialindex = [np.logical_or(np.array(stim4) == 4, np.array(stim4) == 6)]
        stimlist = model4[idx]['stim'][trialindex]
        resplist = model4[idx]['action'][trialindex]
        regmodel = LogisticRegression(random_state=0).fit(X=stimlist[-2000:].astype(int), y=resplist[-2000:].astype(int))
        weights4.append(regmodel.coef_.flatten())
        stim_temp = np.array(stim4[-10000:])
        resp_temp = np.array(resp4[-10000:])
        p_a4.append(np.array([np.mean(resp_temp[stim_temp == s]) for s in np.unique(stim4)]))

    zz = 1

if theory_learning:
    nvals = 54
    noise = np.linspace(0, 1, nvals)
    beta = np.linspace(1, 4, nvals)
    Qlist, Hlist, plist, rlist = (np.zeros((nvals, nvals)), np.zeros((nvals, nvals)),
                                  np.zeros((nvals, nvals)), np.zeros((nvals, nvals)))
    for idx1 in range(nvals):
        for idx2 in range(nvals):
            Q, H, p, r = theory.steadystate(noise[idx1], beta[idx2])
            Qlist[idx1, idx2] = Q
            Hlist[idx1, idx2] = H
            plist[idx1, idx2] = p
            rlist[idx1, idx2] = r
    betacrit = np.array([theory.criticalbeta(n) for n in noise])
    betacrit[betacrit > 5] = None

if theory_distal:
    nvals = 54
    p_a = np.linspace(0, 1, nvals)      # probability of proximal action following distal
    beta = np.linspace(1, 4, nvals)     # degree of exploitation
    betacrit = np.array(theory.criticalbeta_distal(p1=1, p2=p_a, r=10))
    qhvals = np.array([theory.distal(p1=1, p2=p, r=10, beta=2.6) for p in p_a])
    qvals, hvals = qhvals[:, 0], qhvals[:, 1]
    Qlist, Hlist = (np.zeros((nvals, nvals)), np.zeros((nvals, nvals)))
    for idx1 in range(nvals):
        for idx2 in range(nvals):
            Q, H = theory.distal(p1=1, p2=p_a[idx1], r=10, beta=beta[idx2])
            Qlist[idx1, idx2] = Q
            Hlist[idx1, idx2] = H

if theory_delay:
    nvals = 54
    delay = np.linspace(0, 3, nvals)      # delays
    beta = np.linspace(1, 4, nvals)     # degree of exploitation
    betacrit = np.array(theory.criticalbeta_delay(p=1, r=20, delay=delay, tau=1))
    qhvals = np.array([theory.delay(p=1, r=20, delay=d, tau=1, beta=2.5) for d in delay])
    qvals, hvals = qhvals[:, 0], qhvals[:, 1]
    Qlist, Hlist = (np.zeros((nvals, nvals)), np.zeros((nvals, nvals)))
    for idx1 in range(nvals):
        for idx2 in range(nvals):
            Q, H = theory.delay(p=1, r=20, delay=delay[idx1], tau=1, beta=beta[idx2])
            Qlist[idx1, idx2] = Q
            Hlist[idx1, idx2] = H

if theory_schedule:
    nvals = 122
    ti = np.linspace(1, 122, nvals)      # probability of rewards at different times
    beta = np.linspace(1, 4, nvals)      # degree of exploitation
    p_vi = (1/nvals)*np.ones(nvals)
    qhvals_vi = np.array([theory.schedule(p=p_vi, r=100, t=t, tau=2, dt=1, beta=2.5) for t in ti])
    qvals_vi, hvals_vi = qhvals_vi[:, 0], qhvals_vi[:, 1]
    p_fi = np.concatenate((np.zeros((1, int(nvals/2))), np.ones((1, 1)), np.zeros((1, int(nvals/2)))), axis=1).flatten()
    qhvals_fi = np.array([theory.schedule(p=p_fi, r=100, t=t, tau=2, dt=1, beta=2.5) for t in ti])
    qvals_fi, hvals_fi = qhvals_fi[:, 0], qhvals_fi[:, 1]
    Qlist, Hlist = (np.zeros((nvals, nvals, int(nvals/2)+1)), np.zeros((nvals, nvals, int(nvals/2)+1)))
    for idx1 in range(nvals):
        for idx2 in range(nvals):
            for idx3 in range(int(nvals/2)+1):
                p_i = np.concatenate((np.zeros((1, int(nvals/2)-idx3)), np.ones((1, 1+(2*idx3))) / (1+(2*idx3)), np.zeros((1, int(nvals/2)-idx3))), axis=1).flatten()
                Q, H = theory.schedule(p=p_i, r=100, t=ti[idx1], tau=2, dt=1, beta=beta[idx2])
                Qlist[idx1, idx2, idx3] = Q
                Hlist[idx1, idx2, idx3] = H
    Qlist = np.array(Qlist[:int(nvals/2)+1, :, :])
    Hlist = np.array(Hlist[:int(nvals/2)+1, :, :])


if plot_simulation:
    plot(type='learning', model=model, vars=None, smooth=False)

if plot_softmax:
    plot(type='softmax')

if plot_theory:
    plot(type='theory', vars=[Qlist, Hlist, plist, rlist, betacrit, noise, beta])

if plot_inference:
    plot(type='inference', model=model, vars=None, smooth=False)

if plot_distal:
    plot(type='distal', vars=[p_a, beta, qvals, hvals, Qlist, Hlist, betacrit])

if plot_delay:
    plot(type='delay', vars=[delay, beta, qvals, hvals, Qlist, Hlist, betacrit])

if plot_schedule:
    plot(type='schedule', vars=[ti, beta, qvals_vi, hvals_vi, qvals_fi, hvals_fi, Qlist, Hlist])

if plot_accumulation:
    plot(type='accumulate', model=[model1, model2], vars=[stim1, resp1, weights1, p_a1, stim2, resp2, weights2, p_a2])
