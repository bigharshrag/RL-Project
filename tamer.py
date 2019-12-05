import numpy as np

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
    return 0

def TAMER(env, featVec, alpha, verbose, num_episode, credit_range, time_between_h):

    def choose_action(state):
        nA = env.action_space.n
        actions_vals = [np.dot(w, featVec(state, a)) for a in range(nA)]
        print(state, np.around(actions_vals, 3), np.argmax(actions_vals))
        return np.argmax(actions_vals)
    
    w = np.zeros(featVec.feature_vector_len())
    # Hyperparameters
    credit_range = credit_range
    time_between_h = time_between_h

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

            if t % time_between_h == 0:
                h = get_human_feedback()

                if h != 0:
                    credFeat = np.zeros(featVec.feature_vector_len())
                    # Most recent credit_range actions get credit
                    for h_s, h_a, h_t in reversed(history[-credit_range:]) :
                        credit = 1/(credit_range+1)
                        # credit = 1/(time_since_h - h_t + 1)
                        credFeat += (credit * featVec(h_s, h_a))

                    proj_rew = np.dot(w, credFeat)
                    proj_rew = np.clip(proj_rew, -1, 1)

                    error = h - proj_rew
                    w += alpha * error * credFeat

                    time_since_h = 0
                    history = []

            state_prev = state

        if verbose:
            print("Episode: ", i_epi, " Len: ", t)

    return w