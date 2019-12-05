import numpy as np
import gym
from tamer import TAMER
import argparse
import matplotlib.pyplot as plt
from model import RBFVector

def plot_returns(G):
    plt.plot(G)
    plt.xlabel('Episode')
    plt.ylabel('Time to completion')
    plt.show()

def run_tamer(args):
    env = gym.make(args.env)

    featVec = RBFVector(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        [10, 10])

    if args.load_path == None:
        w = TAMER(env, featVec, args.alpha, args.verbose, args.iter, credit_range=3, time_between_h=10)
        np.save(args.save_path, w)
    else:
        w = np.load(args.load_path)

    def greedy_policy(s, done):
        H = [np.dot(w, featVec(s, a, done)) for a in range(env.action_space.n)]
        return np.argmax(H)

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

    # MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
    print("Evaluating")
    Gs = [_eval() for _ in  range(100)]
    print("Average time to completion over 100 trials: ", np.mean(Gs))
    _eval(True)

    plot_returns(Gs)


if __name__ == "__main__":
    """
    Example usage 
    python3 run_tamer.py --verbose --iter 10 --load_path weights.npy
    """

    parser = argparse.ArgumentParser(description='RL-Project: Training a TAMER agent H(s, a)')
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_path', type=str, default='tamer.npy')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--iter', type=int, default=5, help="How many iterations to run the TAMER for")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.9, help="alpha learning rate for tamer")

    args = parser.parse_args()
    run_tamer(args)
