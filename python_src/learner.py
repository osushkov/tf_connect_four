
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

        self.l1_size = self.num_inputs

        self._buildGraph()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.target_input = tf.placeholder(tf.float32, shape=(self.num_inputs, self.max_batch_size))

            # augment the input matrix with 1s so that we don have to use an explicit bias.
            self.target_weights_l1 = tf.Variable(
                tf.truncated_normal([self.l1_size, self.num_inputs], stddev=(1.0/self.num_inputs)))
            self.target_bias_l1 = tf.Variable(
                tf.truncated_normal([self.l1_size], stddev=(1.0/self.num_inputs)))
            self.target_l1 =
                tf.nn.tanh(tf.matmul(self.target_weights_l1, self.target_input) + self.target_bias_l1)

            self.target_weights_output = tf.Variable(
                tf.truncated_normal([self.num_inputs, self.l1_size], stddev=(1.0/self.num_inputs)))
            self.target_bias_output = tf.Variable(
                tf.truncated_normal([self.l1_size], stddev=(1.0/self.num_inputs)))
            self.target_output =
                tf.matmul(self.target_weights_output, self.target_l1) + self.target_bias_output


            # self.learn_input = tf.placeholder(tf.float32, shape=(self.num_inputs, self.max_batch_size))

            # self.av = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
            # self.bv = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
            #
            # self.xv = tf.placeholder(tf.float32, shape=(1, batch_size))
            # self.yv = tf.placeholder(tf.float32, shape=(1, batch_size))
            # self.ypred = tf.add(tf.matmul(self.av, self.xv), self.bv)
            #
            # self.loss = tf.reduce_mean(tf.squared_difference(self.ypred, self.yv))
            # # self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            # self.opt = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

            self.init_op = tf.global_variables_initializer()


    def Learn(self, batch):
        pass

    def UpdateTargetParams(self):
        pass

    def QFunction(self, state):
        if state.shape[0] != self.num_inputs or state.shape[1] > self.max_batch_size:
            raise Exception("Invalid state, shape too large")

        if state.shape[1] < self.max_batch_size:
            padded_state = np.zeros(self.num_inputs, self.max_batch_size)
            padded_state[:, :state.shape[1]] = state
            state = padded_state

        assert (state.shape[0] == self.num_inputs and state.shape[1] == self.max_batch_size)

        with self.sess.as_default():
            feed_dict = {self.target_input: state}
            return self.sess.run([self.target_output], feed_dict=feed_dict)[0][:, :state.]
        # print "qfunction: ", state
        return np.random.rand(self.num_outputs, 1)
