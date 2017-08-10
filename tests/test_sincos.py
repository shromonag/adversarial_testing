'''
In this file we test our approach to model smooth vs non smooth components
of a simple function:
max(sin(x), cos(x)
'''
from active_testing import max_node, pred_node
from active_testing import test_module
from active_testing import delta_opt

import numpy as np
from matplotlib import pyplot as plt

bounds = [(0,10)]
opt = delta_opt(bounds=bounds, epsilon=0.001)

random_ints = [456, 789, 356426]
for i in range(len(random_ints)):
    np.random.seed(random_ints[i])
    node1 = pred_node(f=lambda x:np.sin(x))
    node2 = pred_node(f=lambda x:np.cos(x))
    node0 = max_node(children=[node1, node2])

    TM = test_module(bounds=bounds, sut=lambda x:x, f_tree=node0, with_ns=True,
                 with_random=True, init_sample=10, optimizer=opt,
                 optimize_restarts=1, exp_weight=2)
    TM.initialize()
    TM.run_BO(50)

    plt.figure(i+1)
    plt.plot(TM.smooth_X[10:])
    plt.plot(TM.ns_GP.X[10:])
    plt.plot(TM.random_X[10:])
    plt.ylabel('Location of minimum')
    plt.xlabel('Number of samples')
    plt.title('Finding the minimum of max(sin(x), cos(x))')

X = np.array([[3.7716516],[2.43866633],[2.16238525],[8.82162209],[3.66279053],
              [3.78786564],[5.83747688],[0.98622703],[7.86456213],[5.11302023]])
node1 = pred_node(f=lambda x:np.sin(x))
node2 = pred_node(f=lambda x:np.cos(x))
node0 = max_node(children=[node1, node2])
TM = test_module(bounds=bounds, sut=lambda x:x, f_tree=node0, with_ns=True,
                 with_random=True, init_sample=10, optimizer=opt,
                 optimize_restarts=5, exp_weight=2, X=X)
TM.run_BO(50)
plt.figure(len(random_ints)+ 1)
plt.plot(TM.smooth_X[10:])
plt.plot(TM.ns_GP.X[10:])
plt.plot(TM.random_X[10:])
plt.ylabel('Location of minimum')
plt.xlabel('Number of samples')
plt.title('Finding the minimum of max(sin(x), cos(x))')