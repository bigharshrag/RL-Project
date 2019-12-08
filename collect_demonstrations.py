import numpy as np
import argparse
import gym

def get_action():
    inp = input()
    if inp == 'a': #Left
        return 0
    elif inp == 'd': #Right
        return 2
    return 0

def get_feedback_fixed(s):
    pos, vel = s
    #Pos  -1.2  0.6
    # Vel -0.07 0.07

    if vel == 0:
        return 1

    if pos < -0.50 and vel < 0.008:
        return 0
    if pos < -0.50 and vel > 0.008:
        return 2 
    if pos >= -0.50 and vel > 0.008:
        return 2
    if pos >= -0.50 and vel < 0.008:
        return 0
    
    print(s)
    _ = input()
    
    return 1

    # if pos < -0.9 and vel > 0.0:
    #     if a == 2:
    #         return 1
    #     else:
    #         return -1
    # elif pos > -0.9 and vel > 0.0:
    #     if a == 2:
    #         return 1
    #     else:
    #         return -1
    # return 0

def collect_demos(env, num_demo, save_path):
    demos = list(np.load(save_path))
    # demos = []
    for nd in range(num_demo):
        demos.append([])
        state, done = env.reset(), False
        t = 0
        while not done:
            env.render()
            print(state, t)
            a = get_action()
            a = get_feedback_fixed(state)
            state_dash, R, done, _ =  env.step(a)
            
            demos[-1].append((state, a, state_dash))
            state = state_dash
            t += 1

    np.save(save_path, demos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--num_demo', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='demos.npy')
    args = parser.parse_args()
    
    args.save_path = "Saved_weights/" + args.save_path

    env = gym.make(args.env)
    np.set_printoptions(suppress=True)

    collect_demos(env, args.num_demo, args.save_path)
    # dem = np.load('Saved_weights/demos.npy')
    # print(len(dem[0]))