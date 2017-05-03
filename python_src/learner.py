
from LearnerFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createSampleData(samples):
    a = 10.0;
    b = -5.0;
    noise_sd = 0.1

    xs = np.random.rand(1, samples) * 1.0
    ys = xs * a + b + np.random.normal(0.0, noise_sd, samples)

    return xs, ys


def makeBatch(batch_size, data_x, data_y):
    indices = np.random.permutation(data_x.shape[1])[:batch_size]
    return data_x[:,indices], data_y[:,indices]


batch_size = 1000

class Learner(LearnerInstance):
    def __init__(self, networkSpec):
        self.sess = None
        self.data_x, self.data_y = createSampleData(10000)

        self._buildGraph()

    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.av = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
            self.bv = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)

            self.xv = tf.placeholder(tf.float32, shape=(1, batch_size))
            self.yv = tf.placeholder(tf.float32, shape=(1, batch_size))
            self.ypred = tf.add(tf.matmul(self.av, self.xv), self.bv)

            self.loss = tf.reduce_mean(tf.squared_difference(self.ypred, self.yv))
            # self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            self.opt = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
            self.init_op = tf.global_variables_initializer()


    def Learn(self, batch):
        pass

    def UpdateTargetParams(self):
        pass

    def QFunction(self, state):
        return [np.array([[5.0]]), np.array([[7.0]])]
