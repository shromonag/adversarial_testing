'''
Here we consider a controller trained on deepq for the cartpole
environment in OpenAI Gym. The controller was taken from the baselines.
'''


import gym
import numpy as np
from baselines.ddpg.ddpg import DDPG
import tensorflow as tf
from baselines.ddpg.models import Critic, Actor
from baselines.ddpg.memory import Memory
import baselines.common.tf_util as U
import time
from collections import deque
from gym import spaces
from scipy.stats import chi2, norm

from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
    SimpleMonitor
)

from active_testing import pred_node, max_node, min_node, test_module
from active_testing.utils import sample_from
rand_nums = [3394106029,1371379834,983384216,4249855274,635982059,1931866921,
 4010388016,2936930573,2182696476,666195781]

bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.01, 0.01)) # Bounds on the velocity
bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
bounds.append((0.0005, 0.0025)) # Bounds on the power magnitude


env = gym.make("MountainCarContinuous-v0")
action_noise = None
param_noise = None
nb_actions = env.action_space.shape[-1]
nb_epochs = 500
nb_epoch_cycles = 20
nb_rollout_steps = 100
nb_train_steps=50
param_noise_adaption_interval=50


param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.02),
                                     desired_action_stddev=float(0.02))
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)

seed = 581397839
tf.reset_default_graph()
set_global_seeds(seed)
env.seed(seed)

assert (np.abs(env.action_space.low) == env.action_space.high).all()
max_action = env.action_space.high
agent = DDPG(actor, critic, memory, env.observation_space.shape,
             env.action_space.shape, batch_size=64,
             param_noise=param_noise, critic_l2_reg=1e-2,)


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
    q_vals = []
    while done==False:
        iter_time += 1
        action, q = agent.pi(ob, apply_noise=False, compute_Q=True)
        ob, r, done, _ = env.step(action)
        traj.append(ob)
        reward += r
        q_vals.append(q)
        done = done or iter_time >= max_steps
        if done:
            break
    return traj, {'reward':reward, 'iter_time': iter_time, 'q_vals':q_vals}

def sut(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_traj(max_steps=max_steps, init_state=x0[0:2],goal_pos=x0[2],
                        max_speed=x0[3], power=x0[4])


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

# Training
step = 0
episode = 0
eval_episode_rewards_history = deque(maxlen=100)
episode_rewards_history = deque(maxlen=100)
with U.single_threaded_session() as sess:
    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()

    agent.reset()
    obs = env.reset()
    done = False
    episode_reward = 0.
    episode_step = 0
    episodes = 0
    t = 0

    epoch = 0
    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_episode_eval_rewards = []
    epoch_episode_eval_steps = []
    epoch_start_time = time.time()
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                assert action.shape == env.action_space.shape

                # Execute next action.
                assert max_action.shape == action.shape
                new_obs, r, done, info = env.step(max_action * action)
                # scale for execution in env (as far as DDPG is concerned,
                # every action is in [-1, 1])
                t += 1
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs

                if done:
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    episode_reward = 0.
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset()
                    obs = env.reset()

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= agent.batch_size and \
                                        t % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()


#------------------------------------------------------------------------------------------------------------------

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic

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

