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
opt = delta_opt(bounds=bounds, epsilon=0.0001)

random_ints = [1414065923,
               2448114469,
               1906628456,
               2644070884,
               24268670,
               2664125290,
               1182137443,
               100813220,
               1822558109,
               2027862653]
i = 0
for r in random_ints:
    np.random.seed(r)
    node1 = pred_node(f=lambda x:np.sin(x)+0.65)
    node2 = pred_node(f=lambda x:np.cos(x)+0.65)
    node0 = max_node(children=[node1, node2])

    TM = test_module(bounds=bounds, sut=lambda x:x, f_tree=node0, with_ns=True,
                 with_random=True, init_sample=5, optimizer=opt,
                 optimize_restarts=3, exp_weight=2)
    TM.initialize()
    TM.run_BO(50)

    plt.figure(i+1)
    plt.plot(TM.smooth_X[5:])
    plt.plot(TM.ns_GP.X[5:])
    plt.plot(TM.random_X[5:])
    plt.ylabel('Sample returned in iteration i')
    plt.xlabel('BO iterations')
    plt.title('Finding the minimum of max(sin(x), cos(x))')
    i = i+1

plt.show()

