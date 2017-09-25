'''
Here we consider a controller trained on nearest neighbor for the pendulum
environment in OpenAI Gym. The controller is taken from baselines ppo.
'''

import gym
import numpy as np
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.noise import *
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from mpi4py import MPI
from collections import deque


def train_return(env, param_noise, actor, critic, memory,nb_epochs=250, nb_epoch_cycles=20, reward_scale=1.,
                 render=False,normalize_returns=False, normalize_observations=True, critic_l2_reg=1e-2, actor_lr=1e-4,
                 critic_lr=1e-3,
          action_noise=None, popart=False, gamma=0.99, clip_norm=None,nb_train_steps=50, nb_rollout_steps=2048,
          batch_size=64,tau=0.01, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)

    # Set up logging stuff only for a single worker.



    episode_rewards_history = deque(maxlen=100)
    #with U.single_threaded_session() as sess:
    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()

    agent.reset()
    obs = env.reset()
    episode_reward = 0.
    episode_step = 0
    episodes = 0
    t = 0

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        print('epoch number:', epoch)
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                assert action.shape == env.action_space.shape

                # Execute next action.
                if rank == 0 and render:
                    env.render()
                assert max_action.shape == action.shape
                new_obs, r, done, info = env.step(
                    max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                t += 1
                if rank == 0 and render:
                    env.render()
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
                if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()
    return agent


env = gym.make("Pendulum-v0")
seed = 9699278477418928551
env.seed(seed)
sess = U.make_session(num_cpu=1).__enter__()
nb_actions = env.action_space.shape[-1]
layer_norm=True
param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)


agent = train_return(env=env,actor=actor, critic=critic, memory=memory, param_noise=param_noise)
max_action = env.action_space.high

from gym import spaces
def compute_traj(max_steps, ead=False, **kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'max_speed' in kwargs:
        env.env.max_speed = kwargs['max_speed']
        high = np.array([1., 1., env.env.max_speed])
        env.env.observation_space = spaces.Box(low=-high, high=high)
    if 'max_torque' in kwargs:
        env.env.max_torque = kwargs['max_torque']
        env.env.action_space = spaces.Box(low=-env.env.max_torque,
                                          high=env.env.max_torque, shape=(1,))
    traj = []
    reward = 0
    ob = env.env._get_obs()
    for _ in range(max_steps):
        action, _ = agent.pi(ob, apply_noise=False, compute_Q=True)
        ob, r, _, _ = env.step(max_action*action)
        reward += r
        traj.append(ob)
    additional_data = {'reward':reward}
    return traj, additional_data

def sut(max_steps,x0,ead=False):
    return compute_traj(max_steps, ead=ead,init_state=x0[0:2], max_speed=x0[2],
                        max_torque=x0[3])

from scipy.stats import norm
def cost_func(X):
    theta_rv = norm(np.pi/2., np.pi/2.)
    torque_rv = norm(2, 1)
    speed_rv = norm(8,2)
    theta_pdf = theta_rv.pdf(np.abs(X.T[0]))/theta_rv.pdf(np.pi/2.)
    torque_pdf = torque_rv.pdf(X.T[3])/torque_rv.pdf(2)
    speed_pdf = speed_rv.pdf(X.T[2])/speed_rv.pdf(8)
    theta_pdf.resize(len(theta_pdf), 1)
    torque_pdf.resize(len(torque_pdf), 1)
    speed_pdf.resize(len(speed_pdf), 1)
    return theta_pdf*torque_pdf*speed_pdf

# ------------------------------------------------------------------------------
from active_testing import pred_node, max_node, min_node, test_module
from active_testing.utils import sample_from

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic
bounds = [(-np.pi, np.pi)] # Bounds on theta
bounds.append((-1., 1.)) # Bounds on theta dot
bounds.append((7., 9.)) # Bounds on the speed
bounds.append((1.5, 2.5)) # Bounds on the torque magnitude

smooth_details_r1 = []
random_details_r1 = []

# This set assumes random sampling and checking
for _ in range(10):
    r = np.random.randint(2**32-1)
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward']/200 )
    TM = test_module(bounds=bounds, sut=lambda x0: sut(500,x0, ead=True),
                     f_tree = node0,init_sample = 70,
                     optimize_restarts=5, exp_weight=10, normalizer=True)
    TM.initialize()
    TM.run_BO(140)
    smooth_details_r1.append([np.sum(TM.f_acqu.GP.Y < -5.),
                              np.sum(TM.f_acqu.GP.Y < -7.5),
                              TM.smooth_min_x,TM.smooth_min_val])


    # With cost function
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward']/200)
    TM = test_module(bounds=bounds, sut=lambda x0: sut(500,x0, ead=True),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(30)
    TM.k = 5
    TM.run_BO(40)
    TM.k = 2
    TM.run_BO(70)
    smooth_details_r1.append([np.sum(TM.f_acqu.GP.Y < -5.),
                              np.sum(TM.f_acqu.GP.Y < -7.5),
                              TM.smooth_min_x, TM.smooth_min_val])
    random_details_r1.append([np.sum(np.array(TM.random_Y) < -5.),
                              np.sum(np.array(TM.random_Y) < -7.5),
                              TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r1[-2], smooth_details_r1[-1], random_details_r1[-1])
    rand_nums.append(r)


# Requirement 2: Find the initial condition such that the pendulum stabilizes to 0

smooth_details_r2 = []
random_details_r2 = []

def pred1(traj, gamma=0.25):
    traj = traj[0]
    cos_thetas = np.array(traj).T[0]
    theta_dots = np.array(traj).T[2]
    stab_vals = 0
    for ct, td in zip(cos_thetas, theta_dots):
        stab_vals = np.abs(np.arccos(ct))**2 + np.abs(td)**2 + stab_vals*gamma
    return -stab_vals


# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: pred1(traj))
    TM = test_module(bounds=bounds, sut=lambda x0: sut(500,x0, ead=True),
                     f_tree = node0,init_sample = 60,
                     optimize_restarts=5, exp_weight=2, normalizer=True)
    TM.initialize()
    TM.run_BO(140)
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r2.append([np.sum(smooth_vals < -1.00),
                              np.sum(smooth_vals < -10.0),
                              TM.smooth_min_x,TM.smooth_min_val,
                              TM.smooth_min_loc])

    np.random.seed(r)
    node0_ns = pred_node(f=lambda traj: pred1(traj))
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(500, x0, ead=True),
                     f_tree=node0_ns, init_sample=60, with_smooth=False,
                     with_ns=True,
                     optimize_restarts=5, exp_weight=10, normalizer=True)
    TM_ns.initialize()
    TM_ns.run_BO(30)
    TM_ns.k = 5
    TM_ns.run_BO(40)
    TM_ns.k = 2
    TM_ns.run_BO(70)
    smooth_details_r2.append([np.sum(TM_ns.ns_GP.Y < -1.00),
                              np.sum(TM_ns.ns_GP.Y < -10.0),
                              TM_ns.ns_min_x, TM_ns.ns_min_val,
                          TM_ns.ns_min_loc])


    # With cost function
    np.random.seed(r)

    node0_rand = pred_node(f=lambda traj: pred1(traj))
    TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(500,x0, ead=True),
                     f_tree = node0_rand, with_random = True, with_smooth=False,
                     init_sample = 60, optimize_restarts=5, exp_weight=10,
                     cost_model = cost_func, normalizer=True)
    TM_rand.initialize()
    TM_rand.run_BO(140)
    random_details_r2.append([np.sum(np.array(TM_rand.random_Y) < -1.0),
                              np.sum(np.array(TM_rand.random_Y) < -10.0),
                              TM_rand.rand_min_x, TM_rand.rand_min_val,
                              TM_rand.rand_min_loc])
    print(r, smooth_details_r2[-2], smooth_details_r2[-1],random_details_r2[-1])


# Requirement 3: Find the initial configuration such that it stabilizies to either
# 0 or to np.pi

smooth_details_r3 = []
ns_details_r3 = []
random_details_r3 = []

def pred1(traj, gamma=0.25):
    traj = traj[0]
    cos_thetas = np.array(traj).T[0]
    theta_dots = np.array(traj).T[2]
    stab_vals = 0
    for ct, td in zip(cos_thetas, theta_dots):
        stab_vals = np.abs(np.arccos(ct))**2 + np.abs(td)**2 + stab_vals*gamma
    return -stab_vals

def pred2(traj, gamma=0.25):
    traj = traj[0]
    cos_thetas = np.array(traj).T[0]
    theta_dots = np.array(traj).T[2]
    stab_vals = 0
    for ct, td in zip(cos_thetas, theta_dots):
        stab_vals = (np.pi - np.abs(np.arccos(ct)))**2 + np.abs(td)**2 + stab_vals*gamma
    return -stab_vals

# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f = lambda traj:pred1(traj))
    node1 = pred_node(f = lambda traj:pred2(traj))
    node2 = max_node(children=[node0, node1])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(500,x0, ead=True),
                     f_tree = node2,init_sample = 60,
                     optimize_restarts=5, exp_weight=10, normalizer=True)
    TM.initialize()
    TM.run_BO(30)
    TM.k  =5
    TM.run_BO(40)
    TM.k = 2
    TM.run_BO(70)
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r3.append([np.sum(smooth_vals < -1.00),
                              np.sum(smooth_vals < -10.0),
                              TM.smooth_min_x,TM.smooth_min_val,
                              TM.smooth_min_loc])

    np.random.seed(r)
    node0_ns = pred_node(f=lambda traj: pred1(traj))
    node1_ns = pred_node(f=lambda traj: pred2(traj))
    node2_ns = max_node(children=[node0_ns, node1_ns])
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(500, x0, ead=True),
                     f_tree=node2_ns, init_sample=60, with_smooth=False,
                     with_ns=True,
                     optimize_restarts=5, exp_weight=10, normalizer=True)
    TM_ns.initialize()
    TM_ns.run_BO(30)
    TM_ns.k = 5
    TM_ns.run_BO(40)
    TM_ns.k = 2
    TM_ns.run_BO(70)
    ns_details_r3.append([np.sum(TM_ns.ns_GP.Y < -1.00),
                        np.sum(TM_ns.ns_GP.Y < -10.0),
                        TM_ns.ns_min_x, TM_ns.ns_min_val,
                        TM_ns.ns_min_loc])


    # With cost function
    '''np.random.seed(r)

    node0_rand = pred_node(f=lambda traj: pred1(traj))
    node1_rand = pred_node(f=lambda traj: pred2(traj))
    node2_rand = max_node(children=[node0_rand, node1_rand])
    TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(500,x0, ead=True),
                     f_tree = node2_rand, with_random = True, with_smooth=False,
                     init_sample = 60, optimize_restarts=5, exp_weight=10,
                     cost_model = cost_func, normalizer=True)
    TM_rand.initialize()
    TM_rand.run_BO(140)
    random_details_r3.append([np.sum(np.array(TM_rand.random_Y) < -1.0),
                              np.sum(np.array(TM_rand.random_Y) < -10.0),
                              TM_rand.rand_min_x, TM_rand.rand_min_val,
                              TM_rand.rand_min_loc])
    '''
    print(r, smooth_details_r3[-1], ns_details_r3[-1],random_details_r3[-1])




