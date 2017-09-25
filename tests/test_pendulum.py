'''
Here we consider a controller trained on nearest neighbor for the pendulum
environment in OpenAI Gym. The controller is taken from baselines ppo.
'''

import gym
import numpy as np
from gym import spaces
from baselines import deepq
from baselines.common import set_global_seeds, tf_util as U
import gym, logging
from baselines import logger
import numpy as np
import tensorflow as tf
from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.ppo1.pposgd_simple import *

def learn_return(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vfloss1 = tf.square(pi.vpred - ret)
    vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
    vfloss2 = tf.square(vpredclipped - ret)
    vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2)) # we do the same clipping-based trust region for the value function
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        print(sum(seg['rew']),seg['rew'], len(seg['rew']))
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far+=1

    return pi

U.make_session(num_cpu=1).__enter__()
env= gym.make('Pendulum-v1')
seed = 9699278477418928551
env.seed(seed)
num_timesteps=5e6

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)

gym.logger.setLevel(logging.WARN)
pi = learn_return(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=2048,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95,
    )


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
        action, vpred = pi.act(False, ob)
        ob, r, done, _ = env.step(action)
        reward += r
        traj.append(ob)
        if done and ead:
            break
    additional_data = {'reward':reward}
    return traj, additional_data

def sut(max_steps,x0,ead=False):
    return compute_traj(max_steps, ead=ead,init_state=x0[0:2], max_speed=x0[2],
                        max_torque=x0[3])

from scipy.stats import norm
def cost_func(X):
    theta_rv = norm(np.pi/2., np.pi/2.)
    torque_rv = norm(2, 0.5)
    speed_rv = norm(8,1)
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
rand_nums = [1161003323,
 415998644,
 4057120664,
 1747557171,
 2890879164,
 2055758971,
 2911473105,
 618390143,
 691777806,
 4168149016,
 1809706292,
 2771371912,
 1956477866,
 2141514268,
 4025209431]

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
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward']/200 )
    TM = test_module(bounds=bounds, sut=lambda x0: sut(500,x0, ead=True),
                     f_tree = node0,init_sample = 60,
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
                     f_tree = node0, with_random = True, init_sample = 60,
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
                     optimize_restarts=5, exp_weight=2, normalizer=True)
    TM.initialize()
    TM.run_BO(140)
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
                     optimize_restarts=5, exp_weight=2, normalizer=True)
    TM_ns.initialize()
    TM_ns.run_BO(140)
    ns_details_r3.append([np.sum(TM_ns.ns_GP.Y < -1.00),
                              np.sum(TM_ns.ns_GP.Y < -10.0),
                              TM_ns.ns_min_x, TM_ns.ns_min_val,
                          TM_ns.ns_min_loc])


    # With cost function
    np.random.seed(r)

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
    print(r, smooth_details_r3[-1], ns_details_r3[-1],random_details_r3[-1])


