import numpy as np
import math
from model import StateFeatureVectorWithTile

def get_human_feedback():
    inp = input()
    if inp == 'm':
        return 1
    elif inp == 'n':
        return -1
    return 0

def TAMER_old(env, alpha, verbose, num_episode):

    def virtual_step(env, state, action):
        assert env.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = state
        velocity += (action-1)*env.force + math.cos(3*position)*(-env.gravity)
        velocity = np.clip(velocity, -env.max_speed, env.max_speed)
        position += velocity
        position = np.clip(position, env.min_position, env.max_position)
        if (position==env.min_position and velocity<0): velocity = 0

        done = bool(position >= env.goal_position and velocity >= env.goal_velocity)
        reward = -1.0

        s_next = (position, velocity)
        return np.array(s_next), reward, done, {}

    def choose_action(state):
        # print(w)
        f_t = featVec(state)

        proj_rew = np.zeros(nA)
        for a in range(nA):
            state_next, _, _, _ = virtual_step(env, state, a)
            f_tplus1 = featVec(state_next)
            delta = f_tplus1 - f_t
            proj_rew[a] = np.dot(w, delta) 

        return np.argmax(proj_rew)

    def update_reward_model(r, w, state_tminus1, state, a, alpha):
        f_tminus1, f_t = featVec(state_tminus1), featVec(state)
        delta_feat = f_t - f_tminus1
        proj_rew = np.dot(w, delta_feat)
        error = r - proj_rew
        print(proj_rew)
        print(error)
        print(state_tminus1, state)
        # print(f_t)
        # print(f_tminus1)
        # print(delta_feat)
        w += alpha * error * delta_feat 
        return w

    featVec = StateFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        num_tilings=1,
        # tile_width=np.array([.45,.035])
        tile_width=np.array([.18,.014])
    )
    
    np.set_printoptions(threshold=np.inf, linewidth=500)

    # w = np.load('tamer_t10.npy')
    # print(w.reshape(10, 11, 11))
    # print(w)
    # quit()

    w = np.zeros(featVec.feature_vector_len())
    nA = env.action_space.n

    for i_epi in range(num_episode):
        state_tminus1, done = env.reset(), False
        t = 0
        a = choose_action(state_tminus1)
        state, _, done, _ = env.step(a)

        while not done:
            t += 1
            if t % 3 == 0:
                r = get_human_feedback()
                if r != 0:
                    w = update_reward_model(r, w, state_tminus1, state, a, alpha)
                    print(w.reshape(11,11))

            a = choose_action(state)
            print(a)
            state_tplus1, _, done, _ = env.step(a)
            
            state_tminus1 = state
            state = state_tplus1
            
            env.render()

        if verbose:
            print("Episode: ", i_epi, " Len: ", t)
            np.save("tamer_1tiling", w)

    return w