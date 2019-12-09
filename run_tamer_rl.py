import numpy as np
import gym
from tamer_rl import TAMER_RL
from model import RBFVector
import argparse
import matplotlib.pyplot as plt

def plot_returns(G):
    plt.plot(G)
    plt.xlabel('Episode')
    plt.ylabel('Returns')
    plt.show()

def run_tamer_rl(args):
    env = gym.make(args.env)
    gamma = 1.

    X = RBFVector(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        [10, 10])

    w_H = np.load(args.load_H)

    if args.load_path == None:
        w = TAMER_RL(env, gamma, 0.8, 0.01, X, w_H, args.iter, args.verbose)
        np.save(args.save_path, w)
    else:
        w = np.load(args.load_path)

    def greedy_policy(s,done):
        Q = [np.dot(w, X(s, a, done)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s, done)
            s,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G

    # MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
    # print("Evaluating")
    # Gs = [_eval() for _ in  range(100)]
    # print("Average reward over 100 trials: ", np.mean(Gs))
    # _eval(True)

    # plot_returns(Gs)


if __name__ == "__main__":
    """
    Example usage 
    python3 run.py --verbose --iter 10 --load_path weights.npy
    """

    parser = argparse.ArgumentParser(description='RL-Project')
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_path', type=str, default='tamer_rl.npy')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_H', type=str, default='Saved_weights/tamer.npy')
    parser.add_argument('--iter', type=int, default=500, help="How many iterations to run the algorithm for")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--tile', action='store_true')
    parser.add_argument('--lambda', type=float, default=0.98, help="lambda")

    args = parser.parse_args()
    run_tamer_rl(args)
