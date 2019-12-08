import numpy as np
from model import RBFVector
import gym
import matplotlib.pyplot as plt

def plot_returns(G):
    plt.plot(G)
    plt.xlabel('Episode')
    plt.ylabel('Time to completion')
    plt.show()

env = gym.make('MountainCar-v0')

X = RBFVector(
            env.observation_space.low,
            env.observation_space.high,
            env.action_space.n,
            [10, 10])

alpha = 0.01
gamma = 1
# w = np.load('Saved_weights/tanh2.npy')

# w = np.zeros(X.feature_vector_len())
w = np.full(X.feature_vector_len(), -200.0)
trajs = np.load("Saved_weights/demos.npy")
print("No of demos: ", len(trajs))
epoch = 50

for ep in range(epoch):
    print(ep)
    for idx, traj in enumerate(trajs):
        for i in range(len(traj)):
            s, a, _ = traj[i]

            s[0] += np.random.normal(0, 0.005)
            # s[1] += np.random.normal(0, 0.00009)

            if i == len(traj) - 1:
                w += alpha * (-1 - np.dot(w, X(s, a))) * X(s, a)
                break
            s_dash, a_dash, _ = traj[i+1]
            w += alpha * (-1 + gamma*np.dot(w, X(s_dash, a_dash)) - np.dot(w, X(s, a))) * X(s, a)

# print(w)

def greedy_policy(s, done):
        Q = [np.dot(w, X(s, a)) for a in range(env.action_space.n)]
        # print(Q)
        return np.argmax(Q)

def _eval(render=False):
    s, done = env.reset(), False
    if render: env.render()

    G = 0.
    while not done:
        a = greedy_policy(s, done)
        s, r, done, _ = env.step(a)
        if render: env.render()

        G += r
    return G

print("Evaluating")
Gs = [_eval() for _ in  range(100)]
print("Average time to completion over 100 trials: ", np.mean(Gs))
_eval(True)

plot_returns(Gs)
env.close()
np.save("Saved_weights/q_demos.npy", w)
