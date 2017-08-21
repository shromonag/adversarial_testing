'''
Here we consider a controller trained  for the mountain-car environment in
OpenAI Gym. The controller was taken from the website.
Originally this controller is trained to be non-deterministic sampled from a
gaussian distribution, but we make it deterministic by considering the most
likely control, the mean.
'''

import tensorflow as tf
import numpy as np
import os
import gym
import time
import itertools
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

# Training phase
def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func


env = gym.envs.make("MountainCarContinuous-v0")
video_dir = os.path.abspath("./videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
#env = gym.wrappers.Monitor(env, video_dir, force=True)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def process_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]


class PolicyEstimator:
    def __init__(self, env, lamb=1e-5, learning_rate=0.01, scope="policy_estimator"):
        self.env = env
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.train =True

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [400], name="state")

        self.mu = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.mu = tf.squeeze(self.mu)

        self.sigma = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.norm_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0],
                                       self.env.action_space.high[0])

    def _build_train_op(self):
        self.action_train = tf.placeholder(tf.float32, name="action_train")
        self.advantage_train = tf.placeholder(tf.float32, name="advantage_train")

        self.loss = -tf.log(
            self.norm_dist.prob(self.action_train) + 1e-5) * self.advantage_train \
                    - self.lamb * self.norm_dist.entropy()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        feed_dict = {self.state: process_state(state)}
        return sess.run(self.action, feed_dict=feed_dict)

    def run_deterministic(self, state, sess):
        feed_dict= {self.state:process_state(state)}
        return sess.run(self.mu, feed_dict=feed_dict)

    def update(self, state, action, advantage, sess):
        feed_dict = {
            self.state: process_state(state),
            self.action_train: action,
            self.advantage_train: advantage
        }
        sess.run([self.train_op], feed_dict=feed_dict)


class ValueEstimator:
    def __init__(self, env, learning_rate=0.01, scope="value_estimator"):
        self.env = env
        self.learning_rate = learning_rate

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [400], name="state")

        self.value = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.value = tf.squeeze(self.value)

    def _build_train_op(self):
        self.target = tf.placeholder(tf.float32, name="target")
        self.loss = tf.reduce_mean(tf.squared_difference(self.value, self.target))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: process_state(state)})

    def update(self, state, target, sess):
        feed_dict = {
            self.state: process_state(state),
            self.target: target
        }
        sess.run([self.train_op], feed_dict=feed_dict)


@exec_time
def actor_critic(episodes=100, gamma=0.95, display=False, lamb=1e-5,
                 policy_lr=0.001, value_lr=0.1):
    tf.reset_default_graph()
    policy_estimator = PolicyEstimator(env, lamb=lamb, learning_rate=policy_lr)
    value_estimator = ValueEstimator(env, learning_rate=value_lr)
    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    stats = []
    for i_episode in range(episodes):
        state = env.reset()
        reward_total = 0
        for t in itertools.count():
            action = policy_estimator.predict(state, sess)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward

            if display:
                env.render()

            target = reward + gamma * value_estimator.predict(next_state, sess)
            td_error = target - value_estimator.predict(state, sess)

            policy_estimator.update(state, action, advantage=td_error, sess=sess)
            value_estimator.update(state, target, sess=sess)

            if done:
                break
            state = next_state
        stats.append(reward_total)
        if np.mean(stats[-100:]) > 90 and len(stats) >= 101:
            print(np.mean(stats[-100:]))
            print("Solved.")
        print("Episode: {}, reward: {}.".format(i_episode, reward_total))
    return np.mean(stats[-100:]), policy_estimator, sess


def controller_training(episodes=200):
    policy_lr, value_lr, lamb, gamma = [0.0001, 0.00046415888336127773,
                                        2.782559402207126e-05, 0.98999999999999999]
    loss, policy_estimator, sess = actor_critic(episodes=episodes, gamma=gamma,
                                                display=False, lamb=lamb,
                                                policy_lr=policy_lr, value_lr=value_lr)
    print(-loss)
    env.close()
    return policy_estimator, sess


pe, sess= controller_training(500)


from gym import spaces
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
        action = pe.run_deterministic(ob, sess)
        print(action)
        ob, r, done, _ = env.step(np.array([action]))
        print(ob)
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
