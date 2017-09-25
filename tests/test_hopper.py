'''
Here we consider a controller trained on PPO for the hopper
environment in OpenAI Gym. The controller was taken from baselines/ppo
'''


from baselines.common import set_global_seeds, tf_util as U
import gym, logging
from baselines import logger
import numpy as np
import tensorflow as tf
from baselines.pposgd import mlp_policy, pposgd_simple
from baselines.pposgd.pposgd_simple import *

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
seed = np.random.randint(2**32-1)
set_global_seeds(seed)
env_id = 'Hopper-v1'
env = gym.make(env_id)
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
        action, vpred = pi.act(False,ob)
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
for r in rand_nums[0:2]:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['iters'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=10)
    TM.initialize()
    TM.run_BO(180)
    smooth_details_r2.append([np.sum(TM.f_acqu.GP.Y < 50),
                              TM.smooth_min_x,TM.smooth_min_val])
    random_details_r2.append([np.sum(np.array(TM.random_Y) < 50),
                              TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r2[-1], random_details_r2[-1])