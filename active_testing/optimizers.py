'''
This files defines the three kinds of optimization routines we look into
1. DIRECT
2. LBFGS
3. Sample large number of points within the bounds( this can be directed by
a known distribution
'''

# Include packages
import numpy as np
from .utils import sample_from

def select_opt(type):
    if type=='DIRECT' or type=='direct':
        return direct_opt
    elif type=='LBFGS' or type=='lbfgs':
        return lbfgs_opt
    elif type=='SAMPLE_SEARCH' or type=='sample_search':
        return sample_opt
    elif type=='DELTA_SEARCH' or type=='delta_search':
        return delta_opt
    else:
        print('Optimizer not defined, please define your own optimizer')


class optimization_routine():
    def __init__(self, bounds):
        self.bounds = bounds

    def optimize(self,f,  x0=None, df=None, f_df=None):
        return None, None

class direct_opt(optimization_routine):
    def __init__(self, bounds, iters=1000, funcs=20000, cost=None):
        super(direct_opt, self).__init__(bounds)
        self.iters = iters
        self.funcs = funcs
        if cost is None:
            self.cost = lambda x:1
        else:
            self.cost = cost

    def optimize(self,f,  x0=None, df=None, f_df=None):
        from DIRECT import solve

        def f_direct(x, userdata):
            x = np.atleast_2d(x)
            return f(x)*self.cost(x), 0

        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]

        x, fmin, _ = solve(f_direct, lower_bounds, upper_bounds, alg_method=1,
                           maxT = self.iters, maxf=self.funcs)

        return np.atleast_2d(x), np.atleast_1d(fmin)

class lbfgs_opt(optimization_routine):
    def __init__(self, bounds, iters=15000, funcs=15000, epsilon=1e-3,
                 man_grad=True, k = 3, cost=None):
        super(lbfgs_opt, self).__init__(bounds)
        self.iters = iters
        self.funcs = funcs
        self.epsilon = epsilon
        self.compute_man_grad=man_grad
        self.k = k
        if cost is None:
            self.cost = lambda x:1
        else:
            self.cost = cost
        print('Running lbfgs with manual grad computation: ', man_grad)

    def compute_grad(self, f, x):
        from copy import deepcopy
        set_of_points = []
        bounds = self.bounds
        for i in range(len(x)):
            point = []
            if x[i] - self.epsilon/2. > bounds[i][0]:
                x_lower_i = deepcopy(x)
                x_lower_i[i] = x_lower_i[i] - self.epsilon/2.
                point.append(x_lower_i)
            if x[i] + self.epsilon/2. < bounds[i][1]:
                x_upper_i = deepcopy(x)
                x_upper_i[i] = x_upper_i[i] + self.epsilon / 2.
                point.append(x_upper_i)
            set_of_points.append(point)

        f_X = f(np.atleast_2d(set_of_points))
        grad = [f_X[2*i+1] - f_X[2*i] for i in range(len(x))]
        return np.array(grad)*self.cost(x)

    def optimize(self,f,  x0=None, df=None, f_df=None):
        from scipy.optimize import fmin_l_bfgs_b

        if x0 is None:
            x0 = sample_from(self.k, self.bounds)
        min_val = []
        min_x = []
        func = lambda x : f(x)*self.cost(x)
        df_func = lambda x:df(x)*self.cost(x)
        f_df_func = lambda x: f_df(x)*self.cost(x)
        if df is None and f_df is None:
            if self.compute_man_grad:
                fprime = lambda x: self.compute_grad(f, x)
                for x in x0:
                    res = fmin_l_bfgs_b(func=func, x0=x, fprime=fprime,
                                    bounds=self.bounds, maxiter=self.iters,
                                    maxfun=self.funcs)
                    min_val.append(res[1])
                    min_x.append(res[0])

            else:
                for x in x0:
                    res = fmin_l_bfgs_b(func=func, x0=x, approx_grad=True,
                                    bounds=self.bounds,maxiter=self.iters,
                                    maxfun=self.funcs)
                    min_val.append(res[1])
                    min_x.append(res[0])
        elif f_df is not None:
            for x in x0:
                res = fmin_l_bfgs_b(func=f_df_func, x0=x, bounds=self.bounds,
                                maxiter=self.iters, maxfun=self.funcs)
                min_val.append(res[1])
                min_x.append(res[0])
        else:
            for x in x0:
                res = fmin_l_bfgs_b(func=func, fprime=df_func, x0=x,
                                    bounds=self.bounds,maxiter=self.iters,
                                    maxfun=self.funcs)
                min_val.append(res[1])
                min_x.append(res[0])
        loc = np.array(min_val).argmin()
        return np.atleast_2d(min_x[loc]), np.atleast_2d(min_val[loc])



class sample_opt(optimization_routine):
    def __init__(self, bounds, iters=1, num_sample=250000, sampler=None,
                 save_k=1, cost=None):
        super(sample_opt, self).__init__(bounds)
        self.iters = iters
        self.num_sample =num_sample
        self.save_k = save_k
        self.sampler = sampler
        if cost is None:
            self.cost = lambda x:1
        else:
            self.cost = cost
        print('Running random sampling search for ', iters, 'rounds with ',
              num_sample, 'samples in each round!')

    def optimize(self,f, x0=None, df=None, f_df=None):
        x_across_iters = []
        f_across_iters = []
        for it_num in range(self.iters):
            print('Iteration number: ', it_num)
            X = sample_from(self.num_sample, self.bounds, self.sampler)
            vals = self.cost(X)*f(X)

            save_vals = np.partition(vals.reshape(1, len(vals))[0],
                                     self.save_k)[0:self.save_k]
            save_locs = np.argpartition(vals.reshape(1, len(vals))[0],
                                        self.save_k)[0:self.save_k]
            f_across_iters.append(save_vals)
            x_across_iters.append(X[save_locs])
        x_across_iters = np.array(x_across_iters)
        f_across_iters = np.array(f_across_iters)
        if self.iters == 1:
            return np.atleast_2d(x_across_iters[0]), \
                   np.atleast_2d(f_across_iters[0])
        else:
            final_locs=np.argpartition(f_across_iters[0], self.save_k)\
                [0:self.save_k]
            return np.atleast_2d(x_across_iters[0][final_locs]), \
                   np.atleast_2d(f_across_iters[0][final_locs])



class delta_opt(optimization_routine):
    def __init__(self, bounds, num_sample=None, epsilon=None, save_k=1,cost=None):
        # Here num_sample is the num of samples in each dimension
        super(delta_opt, self).__init__(bounds)
        self.num_sample = num_sample
        self.epsilon = epsilon
        self.save_k = save_k
        if num_sample is None and epsilon is None:
            self.num_sample = 250000./len(bounds)
        if cost is None:
            self.cost = lambda x:1
        else:
            self.cost = cost

    def optimize(self,f,  x0=None, df=None, f_df=None):
        if self.num_sample is None:
            x = [np.arange(b[0], b[1], self.epsilon) for b in self.bounds]
        if self.epsilon is None:
            x = [np.linsapce(b[0], b[1], self.num_sample) for b in self.bounds]

        X = np.meshgrid(*x)
        X = np.array(X).reshape(len(self.bounds), -1).T

        f_X = self.cost(X)*f(X)

        final_locs = np.argpartition(f_X.reshape(1,len(f_X))[0],
                                     self.save_k)[0:self.save_k]
        return np.atleast_2d(X[final_locs]), np.atleast_2d(f_X[final_locs])







