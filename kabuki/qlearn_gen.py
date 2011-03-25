# Test pour simple RL

#import random
#import math
import numpy as np
import matplotlib.pyplot as plt

def qlearn_generate(trials=50, lrate=.1, rew_prob=.7, inv_temp=20., plot=False, subjs=2):
    Q = .5*np.ones((2,2))
    out = np.empty(trials*subjs, dtype=[('subj_idx',np.int), ('stim', np.int), ('action', np.int), ('reward', np.float)])
    Q1 = []
    Q2 = []
    for subj in range(subjs):
        idx = slice(subj*trials, (subj+1)*trials)
        out[idx]['subj_idx'] = subj
        for t in range(trials):
            # Present stimulus 0 or 1 (alternating)
            state = t%2
            # Chose action based on softmax over Q-values
            action = softmax(Q[state], inv_temp)

            # Reward is probabilistic (70% for a1 to s1 and vice versa)
            rnd = np.random.rand()
            if action == state:
                reward = 1 if rnd < rew_prob else 0
            else:
                reward = 0 if rnd < rew_prob else 1
            
            # Update Q-value based on reward received
            Q[state,action] = Q[state,action] + lrate * (reward - Q[state,action])
        
            out[idx]['stim'][t] = state
            out[idx]['action'][t] = action
            out[idx]['reward'][t] = reward

            Q1.append(Q[0,0])
            Q2.append(Q[0,1])

    if plot:
        plt.figure()
        plt.plot(Q1)
        plt.plot(Q2)

    return out


def softmax(Q1, inv_temp):
    p1 = np.exp(Q1[0]*inv_temp)
    p1 = p1 / (p1 + np.exp(Q1[1]*inv_temp))

    rand = np.random.rand()
    
    return 0 if rand < p1 else 1




