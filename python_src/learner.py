
from LearnerFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

batch_size = 1000

class Learner(LearnerInstance):

    def __init__(self, networkSpec):
        self.num_inputs = networkSpec.numInputs
        self.num_outputs = networkSpec.numOutputs
        self.max_batch_size = networkSpec.maxBatchSize

        self.sess = None
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
        print "wooo", batch.initialStates
        print "booo", batch.successorStates

    def UpdateTargetParams(self):
        pass

    def QFunction(self, state):
        # print "qfunction: ", state
        return np.random.rand(self.num_outputs, 1)
