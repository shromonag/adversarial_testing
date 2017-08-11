'''
In this file we test a simple car dynamics with a linear controller
The car has to stop before the obstacle, but it gets noisy readings of the
location of the obstacle at every time.
We consider a horizon on length 100 and show that using KernelPCA produces
more counterexample than using random sampling.

Requirement G_[0,H] (x - x_obs) > 0 where x_obs = 5.0
min of this is a smooth function, and so we need only one node to represent it
'''

from active_testing import pred_node
from active_testing import test_module

import numpy as np
import GPy

# Car dynamics + controller
def car_dyn(x, x_obs, delta_t, eig1, eig2):
    a = -(eig1+eig2)*x[1] + (eig1*eig2)*(x_obs - x[0])
    if a < -3:
        a = -3
    if a > 3:
         a = 3
    x_t1 = x[0] + x[1] * delta_t
    v_t1 = x[1] + a * delta_t
    return np.array([x_t1, v_t1])
# Set the eigen values of the closed loop control system to be -0.15, -0.25
eig1 = 0.15
eig2 = 0.25
delta_t = 0.025
def compute_traj(x0, xobs_n):
    if len(xobs_n.shape) > 1:
        xobs_n = xobs_n[0]
    traj = [x0]
    for xo in xobs_n:
        traj.append(car_dyn(traj[-1], xo, delta_t, eig1, eig2))
    return traj

def f_prednode(traj):
    return np.array([5 - t[0] for t in traj]).min()

bounds = [(4.5, 5.5)] * 100
x0 = np.array([0., 3.])

rand_nums = [3099588838, 3262578689, 4162876793, 2715705470]
rand_details = []
smooth_details = []
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=f_prednode)

    TM = test_module(bounds=bounds, sut=lambda x: compute_traj(x0, x), f_tree=node0,
                     with_random = True, init_sample = 100, optimize_restarts = 5,
                     exp_weight = 10, low_dim = 20, kernel_type = GPy.kern.RBF)
    TM.initialize()
    TM.run_BO(150)
    smooth_details.append([TM.smooth_min_val, TM.smooth_count])
    rand_details.append([TM.rand_min_val, TM.rand_count])