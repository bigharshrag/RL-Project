import numpy as np
from utils import evaluate

def TAMER_RL(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X,
    w_H,
    num_episode:int, 
    verbose:bool,
) -> np.array:
    """
    TAMER + RL 
    """

    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, a, done)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)


    def type6_action(s, done, w, w_h, h_alpha, epsilon=.0):
        nA = env.action_space.n
        Q = [np.tanh( np.dot(w, X(s, a, done)) ) + (h_alpha * np.dot(w_h, X(s, a, done)))  for a in range(nA)]
        # Q = [np.dot(w, X(s, a, done)) + (h_alpha * np.dot(w_h, X(s, a, done)))  for a in range(nA)]
        # print(Q)

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)


    # w = np.zeros((X.feature_vector_len()))
    w = np.full((X.feature_vector_len()), -10.0)
    h_alpha = 0.98

    for i_epi in range(num_episode):
        state, done = env.reset(), False
        a = epsilon_greedy_policy(state, done, w)
        x = X(state, a, done)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0

        ep_len = 0
        h_alpha *= 0.99
        while not done:
            s_dash, R, done, _ = env.step(a)
            ep_len += 1
            # env.render()

            # a_dash = epsilon_greedy_policy(s_dash, done, w, 0.3)
            a_dash = type6_action(s_dash, done, w, w_H, h_alpha)
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