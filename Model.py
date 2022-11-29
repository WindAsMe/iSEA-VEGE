from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK
from scipy.optimize import minimize
import numpy as np
from neupy import algorithms


def SurrogateEnsemble(data, label):
    scale = scales(data)
    best_data = data[np.argmin(label)]
    grnn = algorithms.GRNN(std=0.003)  # GRNN model
    grnn.train(data, label)

    gpr = GPR(data, label)  # GPR model

    s1 = Minimization(best_data, gpr, scale)
    s2 = Minimization(best_data, grnn, scale)

    s1 = ModifyScale(scale, s1)
    s2 = ModifyScale(scale, s2)
    return s1, s2


def GPR(data, label):
    mixed_kernel = CK(1.0, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4))
    gpr = GaussianProcessRegressor(n_restarts_optimizer=20, kernel=mixed_kernel)
    gpr.fit(data, label)
    return gpr


def GRNN(data, label):
    grnn = algorithms.GRNN(std=0.003)
    grnn.train(data, label)
    return grnn


def scales(data):
    data = np.array(data)
    limit_scale = []
    for i in range(len(data[0])):
        d = data[:, i]
        limit_scale.append([min(d), max(d)])
    return limit_scale


def ModifyScale(scales, elite):
    for i in range(len(elite)):
        if elite[i] > scales[i][1]:
            elite[i] = scales[i][1]
        if elite[i] < scales[i][0]:
            elite[i] = scales[i][0]
    return elite


class Model:
    def __init__(self, model, dim):
        self.model = model
        self.dim = dim

    def predict(self, x):
        X = np.array([x]).reshape(-1, self.dim)
        label = self.model.predict(X)
        return label


def Minimization(best_data, model, scale_range):
    m = Model(model, len(best_data))
    func = m.predict
    cons = []
    for i in range(len(best_data)):
        cons.append({'type': 'ineq', 'fun': lambda x: x[i] - -scale_range[i][0]})
        cons.append({'type': 'ineq', 'fun': lambda x: -x[i] + scale_range[i][1]})
    res = minimize(func, best_data, constraints=cons)
    return res.x