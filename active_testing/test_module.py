'''
This file defines the testing module. This needs the following:
1. The system under test
2. The specification or the function which we are trying to minimize
3. Domains of the uncertainities
'''

from .optimizers import *
from .func_tree import *
from .utils import *
from sklearn.decomposition import KernelPCA

import copy
import GPy

class test_module:
    def __init__(self, sut, bounds, spec=None,f_tree=None, optimizer=None, **kwargs):
        self.system_under_test = sut

        # Choosing the optimizer function
        if spec is None:
            self.f_acqu = f_tree
        else:
            self.spec = spec
            # To implement parser to convert from specification to the function f

        self.bounds = bounds

        if 'cost_model' in kwargs:
            self.cost_model = kwargs['cost_model']
        else:
            self.cost_model = lambda x: 1

        # Choosing the optimizers
        if 'opt_name' in kwargs:
            self.optimizer = select_opt(kwargs[opt_name])(bounds, **kwargs)
        elif optimizer is None:
            self.optimizer = sample_opt(bounds=bounds, cost=self.cost_model)
        else:
            self.optimizer = optimizer

        # Number of samples for initializing GPs
        if 'init_sample' in kwargs:
            self.init_sample = kwargs['init_sample']
        else:
            self.init_sample = 2*len(bounds)

        # Model GPs for the smooth functions
        if 'with_smooth' in kwargs:
            self.with_smooth = kwargs['with_smooth']
        else:
            self.with_smooth = True

        # Model GPs for the top level requirement, potentially modeling
        # non-smooth function
        if 'with_ns' in kwargs:
            self.with_ns = kwargs['with_ns']
        else:
            self.with_ns = False

        # Random sampling
        if 'with_random' in kwargs:
            self.with_random = kwargs['with_random']
        else:
            self.with_random = False

        # Exploration weight for GP-LCB
        if 'exp_weight' in kwargs:
            self.k = kwargs['exp_weight']
        else:
            self.k = 10

        # Optimize retsrats for hyper parameter optimization for GPs
        if 'optimize_restarts' in kwargs:
            self.optimize_restarts = kwargs['optimize_restarts']
        else:
            self.optimize_restarts = 1


        # Search in lower dimension
        if 'low_dim' in kwargs:
            self.using_kpca=True
            self.low_dim = kwargs['low_dim']
            if 'kernel_type' in kwargs:
                self.kernel = kwargs['kernel_type'](self.low_dim)
        elif 'kernel' in kwargs:
            self.kernel = kwargs['kernel']
            self.using_kpca = True
            self.low_dim = self.kernel.input_dim
        else:
            self.using_kpca=False
            if 'kernel_type' in kwargs:
                self.kernel = kwargs['kernel_type'](len(bounds))
            else:
                self.kernel = GPy.kern.Matern32(len(bounds), ARD=True)

        if self.using_kpca:
            if isinstance(self.optimizer, lbfgs_opt) or \
                    isinstance(self.optimizer, direct_opt):
                print('Can use only sample_opt or delta_opt!')
                print('Changing optimizer to sample_opt!')
                self.optimizer = sample_opt(bounds, **kwargs)

        # Sending in pre sampled data
        if 'X' in kwargs:
            self.X = kwargs['X']
        else:
            self.X = []

        if 'seed' in kwargs:
            self.seed =kwargs['seed']
        else:
            self.seed =None

    def initialize(self):
        if len(self.X) == 0:
            X = sample_from(self.init_sample, self.bounds)
            self.X = X

        trajs = []
        for x in self.X:
            trajs.append(self.system_under_test(x))
        Y = self.f_acqu.eval_robustness(trajs)
        if self.with_smooth:
            self.smooth_X = copy.deepcopy(self.X)
            if self.using_kpca:
                self.kpca_s = KernelPCA(kernel='rbf', fit_inverse_transform=True,
                          copy_X=True, n_components=self.low_dim)
                X_s = self.kpca_s.fit_transform(self.smooth_X)
            else:
                X_s = self.smooth_X
            self.f_acqu.init_GPs(X_s, trajs,
                                 kernel=copy.deepcopy(self.kernel),
                                 optimize_restarts=self.optimize_restarts)

        if self.with_ns:
            self.ns_X = copy.deepcopy(self.X)
            if self.using_kpca:
                self.kpca_ns = KernelPCA(kernel='rbf', fit_inverse_transform=True,
                          copy_X=True, n_components=self.low_dim)
                X_ns = self.kpca_ns.fit_transform(self.ns_X)
            else:
                X_ns = copy.deepcopy(self.ns_X)
            self.ns_GP = GPy.models.GPRegression(X_ns, Y,
                                        kernel=copy.deepcopy(self.kernel))
            self.ns_GP.optimize_restarts(self.optimize_restarts)


        if self.with_random:
            self.random_X = copy.deepcopy(self.X)
            self.random_Y = Y


    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            print('BO iteration:', ib)
            if self.with_smooth:
                def f(x):
                    if self.using_kpca:
                        x_s = self.kpca_s.transform(x)
                    else:
                        x_s = x
                    if isinstance(self.optimizer, lbfgs_opt):
                        df = self.f_acqu.eval_df(x_s, k = self.k)
                    else:
                        df=None
                    return self.f_acqu.evaluate(x_s, k=self.k), df
                x,f= self.optimizer.optimize(f=lambda x:f(x)[0],
                                             df = lambda x:f(x)[1])
                self.smooth_X = np.vstack((self.smooth_X, np.atleast_2d(x)))
                trajs = [self.system_under_test(x_i) for x_i in x]
                if self.using_kpca:
                    X_s = self.kpca_s.fit_transform(self.smooth_X)
                else:
                    X_s = self.smooth_X
                self.f_acqu.update_GPs(X_s, trajs,
                                       optimize_restarts=self.optimize_restarts)
            if self.with_ns:
                def f(X):
                    if self.using_kpca:
                        X_ns = self.kpca_ns.transform(X)
                    else:
                        X_ns = X
                    m,v = self.ns_GP.predict(X_ns)
                    if isinstance(self.optimizer, lbfgs_opt):
                        dm,dv = self.ns_GP.predictive_gradients(X_ns)
                        dm = dm[:,:,0]
                        df = dm - (self.k/2)*(dv/np.sqrt(v))
                    else:
                        df =None
                    return m - self.k*np.sqrt(v), df
                x,f = self.optimizer.optimize(f=lambda x: f(x)[0],
                                              df = lambda x:f(x)[1])
                trajs = [self.system_under_test(x_i) for x_i in x]
                f_x = self.f_acqu.eval_robustness(trajs)
                self.ns_X = np.vstack((self.ns_X, np.atleast_2d(x)))
                if self.using_kpca:
                    X_ns = self.kpca_ns.fit_transform(self.ns_X)
                else:
                    X_ns = self.ns_X
                self.ns_GP.set_XY(X_ns,
                                  np.vstack((self.ns_GP.Y, np.atleast_2d(f_x))))
                self.ns_GP.optimize_restarts(self.optimize_restarts)
        if self.with_random:
            if self.seed is not None:
                np.random.seed(self.seed)
                sample_from(self.init_sample, self.bounds)
            rand_x = sample_from(iters_BO, self.bounds)
            trajs = []
            for x in rand_x:
                trajs.append(self.system_under_test(x))
            self.random_X = np.vstack((self.random_X, rand_x))
            rand_y = self.f_acqu.eval_robustness(trajs)
            self.random_Y = np.vstack((self.random_Y, rand_y))

        if self.with_smooth:
            vals = self.f_acqu.find_GP_func()

            self.smooth_min_val = np.array(vals).min()
            self.smooth_min_loc = np.array(vals).argmin()
            self.smooth_min_x = self.smooth_X[self.smooth_min_loc]

            self.smooth_count = np.sum(np.array(vals) < 0)
            self.smooth_ce = np.flatnonzero(np.array(vals) < 0)

        if self.with_ns:
            self.ns_min_val = self.ns_GP.Y.min()
            self.ns_min_loc = self.ns_GP.Y.argmin()
            self.ns_min_x = self.ns_GP.X[self.ns_min_loc]

            self.ns_count = np.sum(self.ns_GP.Y < 0)
            self.ns_ce = np.flatnonzero(self.ns_GP.Y < 0)

        if self.with_random:
            self.rand_min_val = self.random_Y.min()
            self.rand_min_loc = self.random_Y.argmin()
            self.rand_min_x = self.random_X[self.rand_min_loc]

            self.rand_count = np.sum(self.random_Y < 0)
            self.rand_ce = np.flatnonzero(self.random_Y < 0)


