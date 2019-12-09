import numpy as np
import gym
from model import RBFVector
import argparse
import matplotlib.pyplot as plt
from utils import evaluate

def DEMO_TAMER_RL(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X,
    demos,
    num_episode:int, 
    verbose:bool,
) -> np.array:
    """
    DEMOS + TAMER + RL 
    """

    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, a, done)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    def type6_action(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, a, done))  for a in range(nA)]

        if np.random.rand() < epsilon:
            curr = X(s, 0)
            minp = np.argmin(np.sum(np.abs(state_feat - curr), axis=1)) 
            return state_feat_ac[minp]                              
        else:
            return np.argmax(Q)

    state_feat = []
    state_feat_ac = []
    for di, dem in enumerate(demos):
        for ps, pa, _ in dem:
            state_feat.append(X(ps, 0))
            state_feat_ac.append(pa)
    state_feat = np.array(state_feat)

    R = -1
    
    w = np.full((X.feature_vector_len()), -150.0)
    # demo_epsilon = 0.8

    for i_epi in range(num_episode):
        state, done = env.reset(), False
        a = epsilon_greedy_policy(state, done, w)
        x = X(state, a, done)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0

        ep_len = 0
        # demo_epsilon *= 1
        while not done:
            s_dash, R, done, _ = env.step(a)
            ep_len += 1

            a_dash = type6_action(s_dash, done, w, epsilon=np.random.normal(0.7, 0.3))
            x_dash = X(s_dash, a_dash, done)
            Q = np.dot(w, x)
            Q_dash = np.dot(w, x_dash)

            td_delta = R + gamma*Q_dash - Q 

            z = (gamma * lam * z) + ((1 - (alpha * gamma * lam * np.dot(z, x))) * x)
            w += alpha*(td_delta + Q - Q_old)*z - alpha*(Q - Q_old)*x 

            Q_old = Q_dash
            x = x_dash
            a = a_dash

        if (i_epi+1) % 50 == 0:
            print(i_epi+1, evaluate(env, w, X))

        if verbose:
            print("Episode: ", i_epi, " Len: ", ep_len)

    return w


def run_demo_rl(args):
    env = gym.make(args.env)
    gamma = 1.

    X = RBFVector(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        [10, 10])

    demos = np.load(args.load_demos)

    if args.load_path == None:
        w = DEMO_TAMER_RL(env, gamma, args.lam, 0.01, X, demos, args.iter, args.verbose)
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
    env.close()


if __name__ == "__main__":
    """
    Example usage 
    python3 run.py --verbose --iter 10 --load_path weights.npy
    """

    parser = argparse.ArgumentParser(description='RL-Project')
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_path', type=str, default='demo_tamer_rl4.npy')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_demos', type=str, default='demos_all.npy')
    parser.add_argument('--iter', type=int, default=500, help="How many iterations to run the algorithm for")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--lam', type=float, default=0.95, help="lambda")

    args = parser.parse_args()

    args.save_path = "Saved_weights/" + args.save_path
    args.load_demos = "Saved_weights/" + args.load_demos
    if args.load_path is not None:
        args.load_path = "Saved_weights/" + args.load_path

    run_demo_rl(args)
