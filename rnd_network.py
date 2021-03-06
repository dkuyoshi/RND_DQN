import chainer
import numpy as np
from chainer import Chain, Sequential, ChainList, Variable
from chainer import links as L
from chainer import functions as F

from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear
from chainer import cuda


class CNN(Chain):
    def __init__(self, n_history=4, n_hidden=512):
        super().__init__()
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv_layers = chainer.Sequential(
                L.Convolution2D(None, 32, 8, stride=4, initialW=initializer),
                F.relu,
                L.Convolution2D(None, 32, 4, stride=2, initialW=initializer),
                F.relu,
                L.Convolution2D(None, 64, 3, stride=1, initialW=initializer),
                F.relu)
            self.l1 = L.Linear(None, n_hidden, initialW=initializer)
            self.l2 = L.Linear(n_hidden, 512)

    def __call__(self, x, ):
        h = self.conv_layers(x)
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return y


class RNDModel(object):
    def __init__(self, n_history=4, n_hidden=512, gpu=-1):
        self.target = CNN(n_history, n_hidden)
        self.predict = CNN(n_history, n_hidden)

        if gpu < 0:
            self.xp = np
        else:
            self.xp = cuda.cupy

    def get_instinct_reward(self, x):
        f_target = self.target(x)
        f_predict = self.predict(x)
        # L2ノルム
        instinct_reward = self.xp.sqrt(self.xp.sum((f_predict.array - f_target.array)**2))

        return cuda.to_cpu(instinct_reward)
