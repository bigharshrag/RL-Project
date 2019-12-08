import numpy as np
import gym
import matplotlib.pyplot as plt
from model import RBFVector

def greedy_policy(w, s, done):
    Q = [np.dot(w, X(s, a, done)) for a in range(env.action_space.n)]
    return np.argmax(Q)

def get_demo_action(s):
    curr = X(s, 0)
    minp = np.argmin(np.sum(np.abs(state_feat - curr), axis=1))                                
    return state_feat_ac[minp]


def _eval(w):
    s, done = env.reset(), False
    G = 0.
    while not done:
        # a = greedy_policy(w, s, done)
        a = get_demo_action(s)
        s,r,done,_ = env.step(a)
        G += r
    return G

env = gym.make('MountainCar-v0')
X = RBFVector(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        [10, 10])

demos = np.load("Saved_weights/demos_all.npy")
state_feat = []
state_feat_ac = []
for di, dem in enumerate(demos):
    print(di)
    for ps, pa, _ in dem:
        state_feat.append(X(ps, 0))
        state_feat_ac.append(pa)
state_feat = np.array(state_feat)

Gs_demo = [_eval(0) for _ in  range(100)]
print("Average reward over 100 trials: ", np.mean(Gs_demo))

plt.plot(Gs_demo)
plt.xlabel('Episode')
plt.ylabel('Returns')
plt.show()

# w_sarsa_opt = np.load('sarsa_good.npy')
# w_sarsa_pess = np.load('sarsa_rbf.npy')
# w_tamer_rl = np.load('tanh2.npy')

# Gs_sopt = [_eval(w_sarsa_opt) for _ in  range(100)]
# print("Average reward over 100 trials: ", np.mean(Gs_sopt))
# Gs_spess = [_eval(w_sarsa_pess) for _ in  range(100)]
# print("Average reward over 100 trials: ", np.mean(Gs_spess))
# Gs_tamrl = [_eval(w_tamer_rl) for _ in  range(100)]
# print("Average reward over 100 trials: ", np.mean(Gs_tamrl))

# plt.plot(Gs_sopt)
# plt.plot(Gs_spess)
# plt.plot(Gs_tamrl)
# plt.xlabel('Episode')
# plt.ylabel('Returns')
# plt.show()