'''
Here we consider a controller trained  for the reacher environment in
OpenAI Gym. The controller was taken from the baselines. The controller is
based on deepq.
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

seed = 2146337346
set_global_seeds(seed)
env = gym.make("Reacher-v1")

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

def compute_traj(max_steps,early=False,done_early=False,**kwargs):
    env.reset()
    # This sets the init_qpos
    if 'init_state' in kwargs:
        env.env.init_qpos = kwargs['init_state']
    # This sets the goal
    if 'goal' in kwargs:
        env.env.goal = kwargs['goal']
    # This is the init_qvel
    if 'init_velocity' in kwargs:
        env.env.init_qvel = kwargs['init_velocity']
    # State perturbation
    if 'state_per' in kwargs:
        state_per = kwargs['state_per']
    # Velocity perturbation
    if 'vel_per' in kwargs:
        vel_per = kwargs['vel_per']

    qpos = state_per+env.env.init_qpos
    qvel = vel_per+env.env.init_qvel
    qpos[-2:] = env.env.goal
    qvel[-2:] = 0.
    env.env.set_state(qpos,qvel)
    ob = env.env._get_obs()
    traj = [ob]
    reward = 0
    iters = 0
    closest = np.inf
    total_theta1 = 0.
    total_theta2 = 0.
    pt1 = np.arccos(ob[0]) if ob[2] > 0 else np.pi + np.arccos(ob[0])
    pt2 = np.arccos(ob[1]) if ob[3] > 0 else np.pi +np.arccos(ob[1])
    for _ in range(max_steps):
        action, _ = agent.pi(ob, apply_noise=False, compute_Q=True)
        ob, r, done, additional_data = env.step(max_action * action)
        if early and np.linalg.norm(env.env.get_body_com("fingertip")\
                                  -env.env.get_body_com("target")) < 0.1:
            break
        nt1 = np.arccos(ob[0]) if ob[2] > 0 else np.pi + np.arccos(ob[0])
        nt2 = np.arccos(ob[1]) if ob[3] > 0 else np.pi + np.arccos(ob[1])
        total_theta1 += nt1 - pt1
        total_theta2 += nt2 - pt2
        pt1 = nt1
        pt2 = nt2
        if -additional_data['reward_dist']< closest:
            closest = -additional_data['reward_dist']
        if done_early and done:
            break
        reward += r
        traj.append(ob)
        iters+=1.

    additional_data = {}
    additional_data['reward']=reward
    additional_data['iters'] = iters
    additional_data['closest'] = closest
    additional_data['tot_theta1'] = np.abs(total_theta1/(2*np.pi))
    additional_data['tot_theta2'] = np.abs(total_theta2/(2*np.pi))
    return traj, additional_data

# ------------------------------------------------------------------------------
from active_testing import pred_node, max_node, min_node, test_module
from active_testing.utils import sample_from

rand_nums = [3547645943,
 3250606528,
 2906727341,
 772456798,
 2103980956,
 2264249721,
 1171067901,
 3643734338,
 854527104,
 260127400,
 578423204,
 3152488971,
 261317259,
 2798623267,
 3165387405]

bounds = [(-0.2, 0.2)] * 2 # Bounds on the goal
bounds.append((-0.1, 0.1)) # Bounds on the state perturbations
bounds.append((-0.1, 0.1)) # Bounds on the state perturbations
bounds.append((-0.005, 0.005)) # Bounds on the velocity perturbations
bounds.append((-0.005, 0.005)) # Bounds on the velocity perturbations

def sut(max_steps,x0,early=False, done_early=False):
    goal = np.array(x0[0:2])
    state_per = np.zeros(4)
    state_per[0:2] += x0[2:4]
    vel_per = np.zeros(4)
    vel_per[0:2] += x0[4:6]
    return compute_traj(max_steps,early, done_early,goal=goal, state_per=state_per,
                        vel_per=vel_per)


# Requirement 1: Find the initial state, goal state that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic

smooth_details_r1 = []
random_details_r1 = []


# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(2048,x0, done_early=True),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=10, seed=r)
    TM.initialize()
    TM.run_BO(30)
    TM.k = 5
    TM.run_BO(50)
    TM.k=2
    TM.run_BO(50)
    smooth_details_r1.append([np.sum(TM.f_acqu.GP.Y < -10.),
                              TM.smooth_min_x,TM.smooth_min_val])
    random_details_r1.append([np.sum(np.array(TM.random_Y) < -10.),
                              TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r1[-1], random_details_r1[-1])

# Requirement 2: Find the initial state, goal state that maximizes the time
# taken to reach near the goal.
# We need only one node for the time. The time taken is a smooth function
# given that the closed loop system is deterministic.

smooth_details_r2 = []
random_details_r2 = []


# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: -traj[1]['iters'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(2048,x0, early=True),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=3, exp_weight=10,normalizer=True)
    TM.initialize()
    TM.run_BO(30)
    TM.k = 5
    TM.run_BO(50)
    TM.k = 2
    TM.run_BO(50)
    smooth_details_r2.append([np.sum(TM.f_acqu.GP.Y < -50),
                              TM.smooth_min_x,TM.smooth_min_val])
    random_details_r2.append([np.sum(np.array(TM.random_Y) < -50),
                              TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r2[-1], random_details_r2[-1])

# Requirement 3: Find the initial state, goal state that maximizes the minimum
# distance the reacher gets to a goal or minimize the number of rotations.

smooth_details_r3 = []
random_details_r3 = []
ns_details_r3 = []

def compare_s_ns(TM, TM_ns, K, num_sample, r, k):
    # This is the same routine as implemented in sample_opt.
    # Here, I re-implement it as I am comparing the different methods, and
    # hence the sampled X needs to remain the same.
    np.random.seed(r)
    for l in range(K):
        print("Iteration:", l)
        X = sample_from(num_sample, bounds)
        smooth_vals = TM.f_acqu.evaluate(X, k=k)
        m_ns, v_ns = TM_ns.ns_GP.predict(X)
        ns_vals = m_ns - k*np.sqrt(v_ns)
        k_smooth = smooth_vals.argmin()
        k_ns = ns_vals.argmin()
        print(TM.f_acqu.eval_robustness(TM.system_under_test(X[k_smooth])), \
              TM.f_acqu.eval_robustness(TM.system_under_test(X[k_ns])))
        TM.f_acqu.update_GPs(np.vstack((TM.f_acqu.children[0].GP.X, \
                            np.atleast_2d(X[k_smooth]))), \
                             [TM.system_under_test(X[k_smooth])])
        TM_ns.ns_GP.set_XY(np.vstack((TM_ns.ns_GP.X, np.atleast_2d(X[k_ns]))), \
                    np.vstack((TM_ns.ns_GP.Y, \
                    TM_ns.f_acqu.eval_robustness(TM_ns.system_under_test(X[k_ns])))))
        TM_ns.ns_GP.optimize()
    return TM, TM_ns

# This set assumes random sampling and checking
for r in rand_nums[2:]:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: -traj[1]['closest'])
    node1 = pred_node(f=lambda traj: -traj[1]['tot_theta1'])
    node2 = max_node(children=[node0, node1])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(2048,x0),
                     f_tree = node2, with_random = True, init_sample = 70,
                     optimize_restarts=3, exp_weight=10)
    TM.initialize()

    np.random.seed(r)
    node0_ns = pred_node(f=lambda traj: -traj[1]['closest'])
    node1_ns = pred_node(f=lambda traj: -traj[1]['tot_theta1'])
    node2_ns = max_node(children=[node0_ns, node1_ns])
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(2048, x0),
                        with_smooth=False, with_ns=True,
                     f_tree=node2_ns, with_random=True, init_sample=70,
                     optimize_restarts=3, exp_weight=10, seed=r)
    TM_ns.initialize()

    TM, TM_ns = compare_s_ns(TM, TM_ns, 30, 250000, r, k=10)
    TM, TM_ns = compare_s_ns(TM, TM_ns, 50, 250000, r, k=5)
    TM, TM_ns = compare_s_ns(TM, TM_ns, 50, 250000, r, k=2)
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r3.append([np.sum(smooth_vals < -0.1),
                              np.sum(smooth_vals < -0.2),
                              TM.f_acqu.children[0].GP.X[smooth_vals.argmin()],
                              smooth_vals.min(),
                              smooth_vals.argmin()])
    ns_details_r3.append([np.sum(TM_ns.ns_GP.Y < -0.1),
                          np.sum(TM_ns.ns_GP.Y < -0.2),
                          TM_ns.ns_GP.X[TM_ns.ns_GP.Y.argmin()],
                          TM_ns.ns_GP.Y.min(),
                          TM_ns.ns_GP.Y.argmin()])

    TM_ns.with_ns =False
    TM_ns.run_BO(130)
    random_details_r3.append([np.sum(TM_ns.random_Y < -0.1), np.sum(TM_ns.random_Y < -0.2),
                              TM_ns.rand_min_x, TM_ns.rand_min_val])
    print(r, smooth_details_r3[-1], ns_details_r3[-1], random_details_r3[-1])