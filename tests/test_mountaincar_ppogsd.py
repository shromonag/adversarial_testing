'''
Here we consider a controller trained  for the mountain-car environment in
OpenAI Gym. The controller was taken from the baselines. The controller is
based on ppo.
'''


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
seed = 8902077161928034768
env = gym.make("MountainCarContinuous-v0")
env.seed(seed)
num_timesteps=5e6

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)

env.seed(seed)
gym.logger.setLevel(logging.WARN)
pi = learn_return(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=2048,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95,
    )

from gym import spaces
def compute_traj(**kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
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
        action, vpred = pi.act(False, ob)
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
rand_nums = [3188388221,
 1954593344,
 2154016205,
 3894811078,
 3493033583,
 3248332584,
 1304673443,
 3857496775,
 2668478815,
 278535713,
 1762150547,
 788841329,
 2525132954,
 677754898,
 754758634]

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic
bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.0001, 0.0001)) # Bounds on the velocity
bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
bounds.append((0.0005, 0.0025)) # Bounds on the power magnitude

smooth_details_r1 = []
random_details_r1 = []

# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0, init_sample = 60,
                     optimize_restarts=5, exp_weight=2,
                     normalizer=True)
    TM.initialize()

    TM.run_BO(140)
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r1.append([np.sum(smooth_vals < 75),
                              np.sum(smooth_vals < 0),
                              TM.smooth_min_x,
                              TM.smooth_min_val])




# With cost function
    np.random.seed(r)
    node0_cf = pred_node(f=lambda traj: traj[1]['reward'])
    TM_cf = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0_cf, init_sample = 60,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM_cf.initialize()

    TM_cf.run_BO(30)
    TM_cf.k = 5
    TM_cf.run_BO(40)
    TM_cf.k = 2
    TM_cf.run_BO(70)
    smooth_cf_vals = np.array(TM_cf.f_acqu.find_GP_func())
    smooth_details_r1.append([np.sum(smooth_cf_vals < 75),
                              np.sum(smooth_cf_vals < 0),
                              TM_cf.smooth_min_x,
                              TM_cf.smooth_min_val])
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node0, init_sample=70, with_smooth=False,
                     with_random=True,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(130)
    random_details_r1.append([np.sum(TM.random_Y < 75),
                              np.sum(TM.random_Y < 0),
                              TM.rand_min_x,
                              TM.rand_min_val])
    print(r, smooth_details_r1[-2],  smooth_details_r1[-1], random_details_r1[-1])


# Requirement 2: Find the initial configuration that maximizes the time
# to completion. We need only one node for the time.

smooth_details_r2 = []
random_details_r2 = []

# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: -traj[1]['iter_time'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node0,  init_sample = 60,
                     optimize_restarts=5, exp_weight=2,
                     normalizer=True)
    TM.initialize()

    TM.run_BO(140)
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r2.append([np.sum(smooth_vals < -250),
                              np.sum(smooth_vals < -150),
                              TM.smooth_min_x,
                              TM.smooth_min_val])


    print(smooth_details_r2[-1])
# With cost function
    np.random.seed(r)
    node0_cf = pred_node(f=lambda traj: -traj[1]['iter_time'])
    TM_cf = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node0_cf, init_sample = 60,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM_cf.initialize()

    TM_cf.run_BO(30)
    TM_cf.k = 5
    TM_cf.run_BO(40)
    TM_cf.k = 2
    TM_cf.run_BO(70)
    smooth_cf_vals = np.array(TM_cf.f_acqu.find_GP_func())
    smooth_details_r2.append([np.sum(smooth_cf_vals < -250),
                              np.sum(smooth_cf_vals < -150),
                              TM_cf.smooth_min_x,
                              TM_cf.smooth_min_val])

    np.random.seed(r)
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0, max_steps=350),
                     f_tree=node0, init_sample=70, with_smooth=False,
                     with_random=True,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(130)
    random_details_r2.append([np.sum(TM.random_Y < -250),
                              np.sum(TM.random_Y < -150),
                              TM.rand_min_x,
                              TM.rand_min_val])

    print(r, smooth_details_r2[-2],smooth_details_r2[-1],
          random_details_r2[-1])

# Requirement 3 : Find the initial configuration that minimizes the following
# requirement :
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap

smooth_details_r3 = []
ns_details_r3 = []
random_details_r3 = []

def pred1(traj):
    traj = traj[0]
    x_s = np.array(traj).T[0]
    init_x = x_s[0]
    dev = init_x - x_s
    dev = np.sum(np.abs(dev))
    return -dev/350.

def pred2(traj):
    iters = traj[1]['iter_time']
    return -iters/350.

def pred3(traj):
    traj=traj[0]
    v_s = np.array(traj).T[1]
    return min(0.025 - np.abs(v_s))

for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=pred1)
    node1 = pred_node(f=pred2)
    node2 = pred_node(f=pred3)
    node3= min_node(children=[node0, node2])
    node4= max_node(children=[node3, node1])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node4, init_sample = 60,
                     optimize_restarts=5, exp_weight=2,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(140)
    smooth_vals = TM.find_GP_func()
    smooth_details_r3.append([np.sum(smooth_vals < -0.25),
                              np.sum(smooth_vals < -0.30),
                              TM.smooth_min_x,
                              TM.smooth_min_val,
                              TM.smooth_min_loc])


# With cost function
    np.random.seed(r)
    node0_ns = pred_node(f=pred1)
    node1_ns = pred_node(f=pred2)
    node2_ns = pred_node(f=pred3)
    node3_ns = min_node(children=[node0_ns, node2_ns])
    node4_ns = max_node(children=[node3_ns, node1_ns])
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node4_ns,  with_smooth=False,
                     with_ns = True, init_sample = 60,
                     optimize_restarts=5, exp_weight=2, cost_model=cost_func,
                     normalizer=True)
    TM_ns.initialize()
    TM_ns.run_BO(140)
    ns_details_r3.append([np.sum(TM_ns.ns_GP.Y < -0.25),
                          np.sum(TM_ns.ns_GP.Y < -0.3),
                        TM_ns.ns_min_x,
                        TM_ns.ns_min_val,
                        TM_ns.ns_min_loc])

    np.random.seed(r)
    node0 = pred_node(f=pred1)
    node1 = pred_node(f=pred2)
    node2 = pred_node(f=pred3)
    node3 = min_node(children=[node0, node2])
    node4 = max_node(children=[node3, node1])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0, max_steps=350),
                     f_tree=node4, init_sample=70, with_smooth=False,
                     with_random=True,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(130)

    random_details_r3.append([np.sum(TM.random_Y < -0.25),
                              np.sum(TM.random_Y < -0.3),
                              TM.rand_min_x,
                              TM.rand_min_val,
                              TM.rand_min_loc])
    print(r, smooth_details_r3[-1], ns_details_r3[-1],
          random_details_r3[-1])