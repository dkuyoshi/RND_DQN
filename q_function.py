import chainer
from chainer import Chain, Sequential, ChainList, Variable
from chainer import links as L
from chainer import functions as F

from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear


class DQNQFunction(Chain):
    def __init__(self, n_action, n_history=4, n_hidden=512):
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
            self.l2 = L.Linear(n_hidden, n_action)

    def __call__(self, x, ):
        h = self.conv_layers(x)
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return DiscreteActionValue(y)


class DuelingQFunction(Chain):
    def __init__(self, n_action, n_history=4, n_hidden=512):
        self.n_action = n_action

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

            self.l = L.Linear(None, n_hidden, initialW=initializer)
            self.a_stream = L.Linear(n_hidden, n_action)
            self.v_stream = L.Linear(n_hidden, 1)

    def __call__(self, x):
        h = self.conv_layers(x)
        h = F.relu(self.l(h))
        
        batch_size = x.shape[0]
        ya = self.a_stream(h)
        mean = F.reshape(F.sum(ya, axis=1) / self.n_action, (batch_size, 1))
        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        ys = self.v_stream(h)
        ya, ys = F.broadcast(ya, ys)
        q = ya + ys
        return DiscreteActionValue(q)