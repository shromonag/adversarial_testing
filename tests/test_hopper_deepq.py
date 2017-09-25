'''
Here we consider a controller trained  for the hopper environment in
OpenAI Gym. The controller was taken from the baselines. The controller is
based on deepq.
'''

import gym
import numpy as np
from baselines import deepq


env = gym.make("Hopper-v1")
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
    # This sets the init_qpos
    if 'init_state' in kwargs:
        env.env.init_qpos = kwargs['init_state']
    # State perturbation
    if 'state_per' in kwargs:
        state_per = kwargs['state_per']
    # Velocity perturbation
    if 'vel_per' in kwargs:
        vel_per = kwargs['vel_per']

    qpos = state_per+env.env.init_qpos
    qvel = vel_per+env.env.init_qvel
    env.env.set_state(qpos,qvel)
    ob = env.env._get_obs()
    traj = [ob]
    reward = 0
    done=False
    iters = 0
    while not done:
        action= act(ob[None])[0]
        ob, r, done, additional_data = env.step(action)
        reward += r
        traj.append(ob)
        iters+=1
    additional_data['reward']=reward
    additional_data['iters'] = iters
    return traj, additional_data

# ------------------------------------------------------------------------------
from active_testing import pred_node, max_node, min_node, test_module
from active_testing.utils import sample_from
rand_nums=[1230597240,
 366379077,
 1450717077,
 4233612701,
 315635237,
 717888137,
 4012326164,
 3986671499,
 1738011324,
 719534766]

bounds = [(-0.005, 0.005)]*7
bounds[0] = (1.23, 1.27) # Bounds on the init_state and velocity perturbation

def sut(x0):
    init_state = x0[0]
    init_qpos = np.zeros(6)
    init_qpos[1] = init_state
    state_per = np.zeros(6)
    vel_per = x0[1:7]
    return compute_traj(init_state=init_qpos, state_per=state_per,
                        vel_per=vel_per)


# Requirement 1: Find the initial state and velocity that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic

smooth_details_r1 = []
random_details_r1 = []


# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=10)
    TM.initialize()
    TM.run_BO(180)
    smooth_details_r1.append([np.sum(TM.f_acqu.GP.Y < -3.75),
                              TM.smooth_min_x,TM.smooth_min_val])
    random_details_r1.append([np.sum(np.array(TM.random_Y) < -3.75),
                              TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r1[-1], random_details_r1[-1])

# Requirement 2: Find the initial state, goal state that minimizes the amount
# of time the robot is able to stay upright
# We need only one node for the time. The time taken is a smooth function
# given that the closed loop system is deterministic.

smooth_details_r2 = []
random_details_r2 = []


# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['iters'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=10)
    TM.initialize()
    TM.run_BO(130)
    smooth_details_r2.append([np.sum(TM.f_acqu.GP.Y < 50),
                              TM.smooth_min_x,TM.smooth_min_val])
    random_details_r2.append([np.sum(np.array(TM.random_Y) < 50),
                              TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r2[-1], random_details_r2[-1])