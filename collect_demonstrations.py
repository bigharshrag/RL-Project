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

def collect_demos(env, num_demo, save_path):
    demos = list(np.load(save_path))
    for nd in range(num_demo):
        demos.append([])
        state, done = env.reset(), False
        t = 0
        while not done:
            env.render()
            print(state, t)
            a = get_action()
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