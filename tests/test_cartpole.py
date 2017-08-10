'''
Here we consider a controller trained on nearest neighbor for the cartpole
example in OpenAI Gym. The controller was taken from the website.
'''

""" Quick script for an "Episodic Controller" Agent, i.e. nearest neighbor """

import logging
import numpy as np
from collections import namedtuple


import gym
elems = namedtuple('elems', 'x y')

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
agent = EpisodicAgent(env.action_space)

np.random.seed(2352454)
episode_count = 200
max_steps = 100
reward = 0
done = False
sum_reward_running = 0

for i in range(episode_count):
    ob = env.reset()
    sum_reward = 0

    for j in range(max_steps):
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        sum_reward += reward
        if done:
            break

    sum_reward_running = sum_reward_running * 0.95 + sum_reward * 0.05
    print('%d running reward: %f' % (i, sum_reward_running))

def compute_traj(max_steps, **kwargs):
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'masspole' in kwargs:
        env.env.masspole = kwargs['masspole']
        env.env.total_mass = env.env.masspole + env.env.masscart
    traj = [ob]
    for _ in range(max_steps):
        action = controller(ob, agent)
        ob, _, _, _ = env.step(action)
        traj.append(ob)
    return traj, env.env.total_mass

# ------------------------------------------------------------------------------
from active_testing import pred_node, max_node, min_node, test_module
rand_nums = [3099588838, 3262578689, 4162876793, 2715705470]

# Requirement 1: We would like the cartpole to not travel more than a certain
# distance from its original location(0.25) and the pole should remain within
# a certain degree from rest position(0.1)
def compute_Y(init, traj):
    ys = [min(0.25 - np.abs(y[0] - init[0]), 0.1 - np.abs(y[2])) for y in traj]
    return np.array(ys).min()

bounds = [(-0.05, 0.05)] * 4

# The requirement is a smooth function. Hence we need only one node
smooth_details_r1 = []
random_details_r1 = []
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: compute_Y(traj[0], traj))
    TM = test_module(bounds=bounds, sut=lambda x0: compute_traj(50, init_state=x0)[0],
                     f_tree = node0, with_random = True, init_sample = 50,
                     optimize_restarts=5, exp_weight=10)
    TM.initialize()
    TM.run_BO(150)
    smooth_details_r1.append([TM.smooth_count, TM.smooth_min_val])
    random_details_r1.append([TM.random_count, TM.random_min_val])

# Requirement 2: Imagine a wall around th cartpole which is at location -0.15, 0.15
# We would like the cartpole to stay within that region, but if not we would like it
# to hit the obstacle with minimum momentum < 0.05.
# This is captured as always(|x| > 0.15 -> |mv| < 0.05) which can also be written as
# always(max(0.15 - |x|, 0.05 - |mv|))

pred1 = lambda x: 0.15 - np.abs(x)
pred2 = lambda m, v: 0.05 - np.abs(m * v)
max_steps = 50
bounds = [(-0.05, 0.05)] * 5
bounds[4] = (0.05, 0.15)
# The requirement is non-smooth, we break the tree down into nodes
smooth_details_r2 = []
ns_details_r2 = []
random_details_r2 = []
for r in rand_nums:
    np.random.seed(r)
    nodes_pred1 = [pred_node(f=lambda t: pred1(t[0][i][0])) for i in\
                   range(max_steps)]
    nodes_pred2 = [pred_node(f=lambda t: pred2(t[1], t[0][i][1])) for i in \
                   range(max_steps)]
    max_nodes = [max_node(children=[nodes_pred1[i], nodes_pred2[i]]) for i in \
                 range(max_steps)]
    min_root_node = min_node(children=max_nodes)
    TM = test_module(bounds=bounds, sut=lambda x0: compute_traj(50,init_state=x0[0:4],
                    masspole =x0[4]), f_tree = min_root_node, with_ns = True,
                     with_random = True, init_sample = 50, optimize_restarts = 5,
                     exp_weight = 10)
    TM.initialize()
    TM.run_BO(150)
    smooth_details_r2.append([TM.smooth_count, TM.smooth_min_val])
    random_details_r2.append([TM.rand_count, TM.rand_min_val])
    ns_details_r2.append([TM.ns_count, TM.ns_min_val])