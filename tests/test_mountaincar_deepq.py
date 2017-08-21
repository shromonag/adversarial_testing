'''
Here we consider a controller trained  for the mountain-car environment in
OpenAI Gym. The controller was taken from the baselines. The controller is
based on deepq.
'''

import gym
import numpy as np
from baselines import deepq


env = gym.make("MountainCar-v0")
# Enabling layer_norm here is import for parameter space noise!
model = deepq.models.mlp([64], layer_norm=True)
act = deepq.learn(
    env,
    q_func=model,
    lr=1e-3,
    max_timesteps=100000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.1,
    print_freq=10,
    param_noise=True
)

def compute_traj(**kwargs):
    env.reset()
    print(env.env.state)
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
        print(env.env.state)
    if 'goal_pos' in kwargs:
        gp = kwargs['goal_pos']
        env.env.goal_position = gp
    if 'max_speed' in kwargs:
        ms = kwargs['max_speed']
        env.env.max_speed = ms
        env.env.low_state = \
            np.array([env.env.min_position, - env.env.max_speed])
        env.env.high_state = \
            np.array([env.env.max_position, env.env.max_speed])
        env.env.observation_space = \
            spaces.Box(env.env.low_state, env.env.high_state)
    if 'power' in kwargs:
        pow = kwargs['power']
        env.env.power = pow
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf

    iter_time = 0
    reward = 0
    done=False
    traj = [ob]
    while done==False:
        iter_time += 1
        action = act(ob[None])[0]
        ob, r, done, _ = env.step(action)
        traj.append(ob)
        reward += r
        done = done or iter_time >= max_steps
        if done:
            break
    return traj, {'reward':reward, 'iter_time': iter_time}

def sut(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_traj(max_steps=max_steps, init_state=x0[0:2],goal_pos=x0[2],
                        max_speed=x0[3], power=x0[4])

from scipy.stats import chi2, norm

def cost_func(X):
    goal_rv = chi2(5, loc=0.3999999, scale=0.05/3.)
    speed_rv = chi2(5, scale=0.005/3.)
    power_rv = norm(0.0015, 0.00075)
    goal_pdf = goal_rv.pdf(X.T[2]) / goal_rv.pdf(0.45)
    speed_pdf = speed_rv.pdf(0.075- X.T[3]) / speed_rv.pdf(0.005)
    power_pdf = power_rv.pdf(X.T[4]) / power_rv.pdf(0.0015)
    goal_pdf.resize(len(goal_pdf), 1)
    speed_pdf.resize(len(speed_pdf), 1)
    power_pdf.resize(len(power_pdf), 1)
    return goal_pdf*speed_pdf*power_pdf

#------------------------------------------------------------------------------------------------------------------
from active_testing import pred_node, max_node, min_node, test_module
from active_testing.utils import sample_from
rand_nums = [3394106029,1371379834,983384216,4249855274,635982059,1931866921,
 4010388016,2936930573,2182696476,666195781]

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic
bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.01, 0.01)) # Bounds on the velocity
bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
bounds.append((0.0005, 0.0025)) # Bounds on the power magnitude

smooth_details_r1 = []
random_details_r1 = []

# This set assumes random sampling and checking
for r in rand_nums[0:1]:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0, with_random = True, init_sample = 50,
                     optimize_restarts=5, exp_weight=10)
    TM.initialize()
    TM.run_BO(200)
    smooth_details_r1.append([TM.smooth_count, TM.smooth_min_x,TM.smooth_min_val])
    random_details_r1.append([TM.rand_count, TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r1[-1], random_details_r1[-1])


# With cost function
for r in rand_nums[0:1]:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0, with_random = True, init_sample = 50,
                     optimize_restarts=5, exp_weight=10, cost_model=cost_func)
    TM.initialize()
    TM.run_BO(200)
    smooth_details_r1.append([TM.smooth_count,TM.smooth_min_x, TM.smooth_min_val])
    random_details_r1.append([TM.rand_count, TM.rand_min_x,TM.rand_min_val])
    print(r, smooth_details_r1[-1], random_details_r1[-1])

# Requirement 2:
