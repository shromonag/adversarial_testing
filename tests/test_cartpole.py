'''
Here we consider a controller trained on nearest neighbor for the cartpole
environment in OpenAI Gym. The controller was taken from the website.
'''

""" Quick script for an "Episodic Controller" Agent, i.e. nearest neighbor """

import logging
import numpy as np

import gym

class EpisodicAgent(object):
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """

    def __init__(self, action_space):
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'unsupported action space for now.'

        # options
        self.epsilon = 1.0  # probability of choosing a random action
        self.epsilon_decay = 0.98  # decay of epsilon per episode
        self.epsilon_min = 0
        self.nnfind = 500  # how many nearest neighbors to consider in the policy?
        self.mem_needed = 500  # amount of data to have before we can start exploiting
        self.mem_size = 50000  # maximum size of memory
        self.gamma = 0.95  # discount factor

        # internal vars
        self.iter = 0
        self.mem_pointer = 0  # memory pointer
        self.max_pointer = 0
        self.db = None  # large array of states seen
        self.dba = {}  # actions taken
        self.dbr = {}  # rewards obtained at all steps
        self.dbv = {}  # value function at all steps, computed retrospectively
        self.ep_start_pointer = 0

    def act(self, observation, reward, done):
        assert isinstance(observation, np.ndarray) and observation.ndim == 1, 'unsupported observation type for now.'

        if self.db is None:
            # lazy initialization of memory
            self.db = np.zeros((self.mem_size, observation.size))
            self.mem_pointer = 0
            self.ep_start_pointer = 0

        # we have enough data, we want to explore, and we have seen at least one episode already (so values were computed)
        if self.iter > self.mem_needed and np.random.rand() > self.epsilon and self.dbv:
            # exploit: find the few closest states and pick the action that led to highest rewards
            # 1. find k nearest neighbors
            ds = np.sum((self.db[:self.max_pointer] - observation) ** 2, axis=1)  # L2 distance
            ix = np.argsort(ds)  # sorts ascending by distance
            ix = ix[:min(len(ix), self.nnfind)]  # crop to only some number of nearest neighbors

            # find the action that leads to most success. do a vote among actions
            adict = {}
            ndict = {}
            for i in ix:
                vv = self.dbv[i]
                aa = self.dba[i]
                vnew = adict.get(aa, 0) + vv
                adict[aa] = vnew
                ndict[aa] = ndict.get(aa, 0) + 1

            for a in adict:  # normalize by counts
                adict[a] = adict[a] / ndict[a]

            its = [(y, x) for x, y in adict.items()]
            its.sort(reverse=True)  # descending
            a = its[0][1]

        else:
            # explore: do something random
            a = self.action_space.sample()

        # record move to database
        if self.mem_pointer < self.mem_size:
            self.db[self.mem_pointer] = observation  # save the state
            self.dba[self.mem_pointer] = a  # and the action we took
            self.dbr[self.mem_pointer - 1] = reward  # and the reward we obtained last time step
            self.dbv[self.mem_pointer - 1] = 0
        self.mem_pointer += 1
        self.iter += 1

        if done:  # episode Ended;

            # compute the estimate of the value function based on this rollout
            v = 0
            for t in reversed(range(self.ep_start_pointer, self.mem_pointer)):
                v = self.gamma * v + self.dbr.get(t, 0)
                self.dbv[t] = v

            self.ep_start_pointer = self.mem_pointer
            self.max_pointer = min(max(self.max_pointer, self.mem_pointer), self.mem_size)

            # decay exploration probability
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)  # cap at epsilon_min

            print('memory size: ', self.mem_pointer)

        return a

def controller(observation, agent):
    ds = np.sum((agent.db[:agent.max_pointer] - observation) ** 2, axis=1)  # L2 distance
    ix = np.argsort(ds)  # sorts ascending by distance
    ix = ix[:min(len(ix), agent.nnfind)]  # crop to only some number of nearest neighbors

    # find the action that leads to most success. do a vote among actions
    adict = {}
    ndict = {}
    for i in ix:
        vv = agent.dbv[i]
        aa = agent.dba[i]
        vnew = adict.get(aa, 0) + vv
        adict[aa] = vnew
        ndict[aa] = ndict.get(aa, 0) + 1

    for a in adict:  # normalize by counts
        adict[a] = adict[a] / ndict[a]

    its = [(y, x) for x, y in adict.items()]
    its.sort(reverse=True)  # descending
    a = its[0][1]
    return a

logger = logging.getLogger()
logger.setLevel(logging.INFO)

env = gym.make('CartPole-v0')
seed = 17588724670887928270
env.seed(seed)
agent = EpisodicAgent(env.action_space)

episode_count = 700
max_steps = 200
reward = 0
done = False
sum_reward_running = 0

training_envs = []

for i in range(episode_count):
    ob = env.reset()
    sum_reward = 0
    training_envs.append(ob)
    for j in range(max_steps):
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        sum_reward += reward
        if done:
            break

    sum_reward_running = sum_reward_running * 0.95 + sum_reward * 0.05
    print('%d running reward: %f' % (i, sum_reward_running))

def compute_traj(max_steps,ead=False, **kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'masspole' in kwargs:
        env.env.masspole = kwargs['masspole']
        env.env.total_mass = env.env.masspole + env.env.masscart
        env.env.polemass_length = env.env.masspole * env.env.length
    if 'length' in kwargs:
        env.env.length = kwargs['length']
        env.env.polemass_length = env.env.masspole * env.env.length
    if 'force_mag' in kwargs:
        env.env.force_mag = kwargs['force_mag']
    traj = [ob]
    reward = 0
    iters= 0
    for _ in range(max_steps):
        iters+=1
        action = controller(ob, agent)
        ob, r, done, _ = env.step(action)
        reward += r
        traj.append(ob)
        if ead and done:
            break
    additional_data = {'reward':reward, 'mass':env.env.total_mass, 'iters':iters}
    return traj, additional_data

def sut(max_steps,x0, ead=False):
    return compute_traj(max_steps,init_state=x0[0:4], masspole=x0[4],
                        length=x0[5], force_mag=x0[6], ead=ead)

from scipy.stats import norm
def cost_func(X):
    mass_rv = norm(0.1, 0.05)
    length_rv = norm(0.5, 0.05)
    force_rv = norm(10,2)
    mass_pdf = mass_rv.pdf(X.T[4])/mass_rv.pdf(0.1)
    length_pdf = length_rv.pdf(X.T[5])/length_rv.pdf(0.5)
    force_pdf = force_rv.pdf(X.T[6])/force_rv.pdf(10)
    mass_pdf.resize(len(mass_pdf), 1)
    length_pdf.resize(len(length_pdf), 1)
    force_pdf.resize(len(force_pdf), 1)
    return mass_pdf*length_pdf*force_pdf

# ------------------------------------------------------------------------------
from active_testing import pred_node, max_node, min_node, test_module
from active_testing.utils import sample_from
rand_nums = [2440271967,
 3816968049,
 3160626546,
 636413671,
 3105544786,
 646012482,
 3406852803,
 1769141240,
 109713304,
 3433822084,
 2481885549,
 2630720097,
 1291718590,
 2572110400,
 3580267181]

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic
bounds = [(-0.05, 0.05)] * 4 # Bounds on the state
bounds.append((0.05, 0.15)) # Bounds on the mass of the pole
bounds.append((0.4, 0.6)) # Bounds on the length of the pole
bounds.append((8.00, 12.00)) # Bounds on the force magnitude

smooth_details_r1 = []
random_details_r1 = []


# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(200,x0, ead=True),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=2, seed=r)
    TM.initialize()
    TM.run_BO(130)
    smooth_details_r1.append([np.sum(TM.f_acqu.GP.Y < 100),
                              np.sum(TM.f_acqu.GP.Y < 150),
                              TM.smooth_min_x,TM.smooth_min_val])
    random_details_r1.append([np.sum(np.array(TM.random_Y) < 100),
                              np.sum(np.array(TM.random_Y) < 150),
                              TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r1[-1], random_details_r1[-1])


# With cost function
    np.random.seed(r)
    node0_cf = pred_node(f=lambda traj: traj[1]['reward'])
    TM_cf = test_module(bounds=bounds, sut=lambda x0: sut(200,x0, ead=True),
                     f_tree = node0_cf, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=2, cost_model=cost_func)
    TM_cf.initialize()
    TM_cf.run_BO(130)
    smooth_details_r1.append([np.sum(TM_cf.f_acqu.GP.Y < 100),
                              np.sum(TM_cf.f_acqu.GP.Y < 150),
                              TM_cf.smooth_min_x, TM_cf.smooth_min_val])
    random_details_r1.append([np.sum(np.array(TM_cf.random_Y) < 100),
                              np.sum(np.array(TM_cf.random_Y) < 150),
                              TM_cf.rand_min_x, TM_cf.rand_min_val])
    print(r, smooth_details_r1[-1], random_details_r1[-1])

# Requirement 2: We would like the cartpole to not travel more than a certain
# distance from its original location(2.4) and the pole should remain within
# a certain degree from rest position(0.209)
def compute_Y(init, traj):
    ys = [min(2.4 - np.abs(y[0] - init[0]), 0.209 - np.abs(y[2])) for y in traj]
    return np.array(ys).min()

# The requirement is a smooth function. Hence we need only one node
smooth_details_r2 = []
random_details_r2 = []
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: compute_Y(traj[0][0], traj[0]))
    TM = test_module(bounds=bounds, sut=lambda x0: sut(200,x0, ead=True),
                     f_tree = node0, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=2,seed=r)
    TM.initialize()
    TM.run_BO(130)
    smooth_details_r2.append([TM.smooth_count,TM.smooth_min_x,TM.smooth_min_val])
    random_details_r2.append([TM.rand_count,TM.rand_min_x, TM.rand_min_val])
    print(r, smooth_details_r2[-1], random_details_r2[-1])


    np.random.seed(r)
    node0_cf = pred_node(f=lambda traj: compute_Y(traj[0][0], traj[0]))
    TM_cf = test_module(bounds=bounds, sut=lambda x0: sut(200,x0, ead=True),
                     f_tree = node0_cf, with_random = True, init_sample = 70,
                     optimize_restarts=5, exp_weight=2, cost_model=cost_func)
    TM_cf.initialize()
    TM_cf.run_BO(130)
    smooth_details_r2.append([TM_cf.smooth_count, TM_cf.smooth_min_x,
                              TM_cf.smooth_min_val])
    random_details_r2.append([TM_cf.rand_count, TM_cf.rand_min_x,
                              TM_cf.rand_min_val])
    print(r, smooth_details_r2[-1], random_details_r2[-1])


# Requirement 3: We would like the cartpole to always satisfy atleast one of the
# following conditions:
# 1. Always stay within the region (-2.4, 2.4)
# 2. Maintain a momentum >=-2.0 and <= 2.0
# 3. The angle made by the cartpole should <=0.2 within the rest position


def pred1(traj):
    traj = traj[0]
    x_s = np.array(traj).T[0]
    return min(2.4 - np.abs(x_s))

def pred2(traj):
    traj_ = traj[0]
    mass = traj[1]['mass']
    v_s = np.array(traj_).T[1]
    return min(2. - np.abs(mass*v_s))

def pred3(traj):
    traj=traj[0]
    theta=np.array(traj).T[2]
    return min(0.2 - np.abs(theta))

smooth_details_r3 = []
ns_details_r3 = []
random_details_r3 = []



for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: pred1(traj))
    node1 = pred_node(f=lambda traj: pred2(traj))
    node2 = pred_node(f=lambda traj: pred3(traj))
    node3 = max_node(children= [node0, node1, node2])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(200,x0),
                     f_tree = node3,init_sample = 60,
                     optimize_restarts=3, exp_weight=5, normalizer=True)
    TM.initialize()
    TM.run_BO(140)

    np.random.seed(r)
    node0_ns = pred_node(f=lambda traj: pred1(traj))
    node1_ns = pred_node(f=lambda traj: pred2(traj))
    node2_ns = pred_node(f=lambda traj: pred3(traj))
    node3_ns = max_node(children=[node0_ns, node1_ns, node2_ns])
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(200,x0),
                     f_tree=node3_ns, with_random=True, init_sample=60,
                     with_smooth=False, with_ns=True, seed=r,
                     optimize_restarts=3, exp_weight=5, normalizer=True)
    TM_ns.initialize()
    TM_ns.run_BO(140)

    smooth_details_r3.append([TM.smooth_count, TM.smooth_min_x,
                              TM.smooth_min_val])
    ns_details_r3.append([TM_ns.ns_count, TM_ns.ns_min_x,
                          TM_ns.ns_min_val])

    random_details_r3.append([TM_ns.rand_count, TM_ns.rand_min_x,
                              TM_ns.rand_min_val])
    print(r, smooth_details_r3[-1], ns_details_r3[-1], random_details_r3[-1])
