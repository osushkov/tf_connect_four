
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
# class Learner:
    def __init__(self):
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


    def LearnIterations(self, iters):
        if self.sess is None:
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init_op)

        with self.sess.as_default():
            for i in range(iters):
                batch_x, batch_y = makeBatch(batch_size, self.data_x, self.data_y)
                _, l, a, b = self.sess.run([self.opt, self.loss, self.av, self.bv],
                                           feed_dict={self.xv: batch_x, self.yv: batch_y})
                self.total_iters += 1
                # print("iter: " + str(self.total_iters) + " loss: " + str(l))


    def GetModelParams(self):
        return [np.array([[5.0]]), np.array([[7.0]])]
