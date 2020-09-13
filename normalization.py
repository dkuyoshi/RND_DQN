import chainer
import numpy as np


class UpdateMeanStd(object):
    def __init__(self, epsilon=1e-7, shape=(), xp=np):
        self.xp = xp
        self.mean = self.xp.zeros(shape, 'float64')
        self.var = self.xp.ones(shape, 'float64')
        # zero割防止
        self.count = epsilon

    def update(self, x):
        batch_mean = self.xp.mean(x, axis=0)
        batch_var = self.xp.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count

        m = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)

        new_var = m / (self.count + batch_count)
        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class UpdateMeanStdR(object):
    def __init__(self, epsilon=1e-7):
        self.mean = 0
        self.var = 1
        self.count = epsilon

    def update(self, r):
        count = 1
        delta = r - self.mean
        total_count = self.count + count
        new_mean = self.mean + delta * count / total_count
        m_a = self.var * self.count
        m_b = r * count

        m = m_a + m_b + np.square(delta) * self.count * count / (self.count + count)
        new_var = m / (self.count + count)
        new_count = count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count




        



        

