import numpy as np
from model import StateActionFeatureVectorWithTile

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X,
    num_episode:int, 
    verbose:bool,
) -> np.array:
    """
    True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, a, done)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for i_epi in range(num_episode):
        state, done = env.reset(), False
        a = epsilon_greedy_policy(state, done, w)
        x = X(state, a, done)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0

        ep_len = 0
        while not done:
            s_dash, R, done, _ = env.step(a)
            ep_len += 1
            # env.render()

            a_dash = epsilon_greedy_policy(s_dash, done, w)
            x_dash = X(s_dash, a_dash, done)
            Q = np.dot(w, x)
            Q_dash = np.dot(w, x_dash)

            td_delta = R + gamma*Q_dash - Q 

            z = (gamma * lam * z) + ((1 - (alpha * gamma * lam * np.dot(z, x))) * x)
            w += alpha*(td_delta + Q - Q_old)*z - alpha*(Q - Q_old)*x 

            Q_old = Q_dash
            x = x_dash
            a = a_dash

        if verbose:
            print("Episode: ", i_epi, " Len: ", ep_len)

    return w

