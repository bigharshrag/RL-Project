import numpy as np
from model import RBFVector

def get_human_feedback():
    inp = input()
    if inp == 'm':
        return 1
    elif inp == 'n':
        return -1
    return 0

def get_feedback_fixed(s, a):
    pos, vel = s
    if pos < -0.3 and vel < 0.0:
        if a == 0:
            return 1
        else:
            return -1
    elif pos < -0.9 and vel > 0.0:
        if a == 2:
            return 1
        else:
            return -1
    elif pos > -0.9 and vel > 0.0:
        if a == 2:
            return 1
        else:
            return -1
    print(s)
    return 0

def TAMER(env, alpha, verbose, num_episode):

    def choose_action(state):
        nA = env.action_space.n
        actions_vals = [np.dot(w, featVec(state, a)) for a in range(nA)]
        print(state, actions_vals)
        return np.argmax(actions_vals)

    featVec = RBFVector(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        [10, 10])
    
    w = np.zeros(featVec.feature_vector_len())
    credit_range = 3
    np.set_printoptions(suppress=True)

    for i_epi in range(num_episode):
        state_prev, done = env.reset(), False
        t = 0
        time_since_h = 0
        history = []

        while not done:
            a = choose_action(state_prev)
            state, _, done, _ = env.step(a)
            env.render()
            
            t += 1
            time_since_h += 1
            history.append((state, a, time_since_h))
            # print("state", state, "t:", t, "a:", a)

            if t % 1 == 0:
                # h = get_human_feedback()
                h  = get_feedback_fixed(state, a)

                if h != 0:
                    # print(history)
                    credFeat = np.zeros(featVec.feature_vector_len())
                    # Most recent credit_range actions get credit
                    for h_s, h_a, h_t in reversed(history[-credit_range:]) :
                        credit = 1/(credit_range+1)
                        # print(h_s, h_a)
                        # print("featvec: ", featVec(h_s, h_a))
                        credFeat += (credit * featVec(h_s, h_a))

                    # print("credFeat ", np.around(credFeat, 2))
                    # print("W", np.around(w, 3))
                    # print("credfeat", np.around(credFeat, 3))
                    proj_rew = np.dot(w, credFeat)

                    proj_rew = np.clip(proj_rew, -1, 1)

                    error = h - proj_rew
                    w += alpha * error * credFeat
                    # np.clip(w, -1, 1, w)
                    # print(np.around(alpha * error * credFeat, 3))
                    # print("pr: ", proj_rew, "error: ", error)
                    # print("---------------------------------------")
                    # print(w)

                    time_since_h = 0
                    history = []

            state_prev = state

        if verbose:
            print("Episode: ", i_epi, " Len: ", t)
            np.save("tamer_1tiling", w)

    return w