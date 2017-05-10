
from LearnerFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


class NNLayer:
    def __init__(self, num_inputs, layer_size, activation_func, input_tensor):
        self.num_inputs = num_inputs
        self.layer_size = layer_size

        init_range = math.sqrt(1.0/self.num_inputs)

        self.weights = tf.Variable(
            tf.random_uniform([self.num_inputs, self.layer_size], minval=-init_range, maxval=init_range),
            dtype=tf.float32)

        self.bias = tf.Variable(tf.zeros([self.layer_size]), dtype=tf.float32)

        self.layer_output = activation_func(tf.matmul(input_tensor, self.weights) + self.bias)

class Learner(LearnerInstance):

    def __init__(self, networkSpec):
        self.num_inputs = networkSpec.numInputs
        self.num_outputs = networkSpec.numOutputs
        self.max_batch_size = networkSpec.maxBatchSize
        self.reward_discount = 0.9

        self.layer_sizes = [self.num_inputs]#, self.num_inputs]
        self._buildGraph()

        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            self.sess.run([self.init_op])


    def _buildTargetNetwork(self):
        self.target_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))

        self.target_network = []
        for ls in self.layer_sizes:
            if len(self.target_network) == 0:
                self.target_network.append(
                    NNLayer(self.num_inputs, ls, tf.nn.relu6, self.target_network_input))
            else:
                pl = self.target_network[-1]
                self.target_network.append(NNLayer(pl.layer_size, ls, tf.nn.relu6, pl.layer_output))

        pl = self.target_network[-1]
        self.target_network.append(NNLayer(pl.layer_size, self.num_outputs, tf.nn.tanh, pl.layer_output))
        self.target_network_output = self.target_network[-1].layer_output

    def _buildLearnNetwork(self):
        # learning network
        self.learn_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))
        self.learn_network_action_index = tf.placeholder(tf.int32, shape=(self.max_batch_size))
        self.learn_network_terminal_mask = tf.placeholder(tf.bool, shape=(self.max_batch_size))
        self.learn_network_reward = tf.placeholder(tf.float32, shape=(self.max_batch_size))
        self.learn_rate = tf.placeholder(tf.float32)

        self.learn_network = []
        for ls in self.layer_sizes:
            if len(self.learn_network) == 0:
                self.learn_network.append(
                    NNLayer(self.num_inputs, ls, tf.nn.relu6, self.learn_network_input))
            else:
                pl = self.learn_network[-1]
                self.learn_network.append(NNLayer(pl.layer_size, ls, tf.nn.relu6, pl.layer_output))

        pl = self.learn_network[-1]
        self.learn_network.append(NNLayer(pl.layer_size, self.num_outputs, tf.nn.tanh, pl.layer_output))
        self.learn_network_output = self.learn_network[-1].layer_output

        terminating_target = self.learn_network_reward
        intermediate_target = self.learn_network_reward + (tf.reduce_max(self.target_network_output, axis=1) * self.reward_discount)
        self.desired_output = tf.tile(
            tf.reshape(
                tf.where(self.learn_network_terminal_mask, terminating_target, intermediate_target), [-1, 1]),
            [1, self.num_outputs])

        self.filtered_loss = tf.squared_difference(
            self.desired_output, self.learn_network_output) * tf.one_hot(self.learn_network_action_index, self.num_outputs)

        self.learn_loss = tf.reduce_mean(self.filtered_loss)
        self.learn_optimizer = tf.train.AdamOptimizer(self.learn_rate, beta1=0.99).minimize(self.learn_loss)

        self.update_ops = []

        assert (len(self.target_network) == len(self.learn_network))
        for i in range(len(self.target_network)):
            dst_weights = self.target_network[i].weights
            src_weights = self.learn_network[i].weights
            self.update_ops.append(tf.assign(dst_weights, src_weights, validate_shape=True, use_locking=True))

            dst_bias = self.target_network[i].bias
            src_bias = self.learn_network[i].bias
            self.update_ops.append(tf.assign(dst_bias, src_bias, validate_shape=True, use_locking=True))


    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(1)
            self._buildTargetNetwork()
            self._buildLearnNetwork()

            self.init_op = tf.global_variables_initializer()


    def Learn(self, batch):
        assert (batch.initialStates.ndim == 2 and batch.successorStates.ndim == 2)
        assert (batch.initialStates.shape[0] == self.max_batch_size and batch.initialStates.shape[1] == self.num_inputs)
        assert (batch.successorStates.shape[0] == self.max_batch_size and batch.successorStates.shape[1] == self.num_inputs)
        assert (batch.actionsTaken.ndim == 1 and batch.actionsTaken.shape[0] == self.max_batch_size)
        assert (batch.isEndStateTerminal.ndim == 1 and batch.isEndStateTerminal.shape[0] == self.max_batch_size)
        assert (batch.rewardsGained.ndim == 1 and batch.rewardsGained.shape[0] == self.max_batch_size)

        feed_dict = {
            self.target_network_input: batch.successorStates,
            self.learn_network_input: batch.initialStates,
            self.learn_network_action_index: batch.actionsTaken,
            self.learn_network_terminal_mask: batch.isEndStateTerminal,
            self.learn_network_reward: batch.rewardsGained,
            self.learn_rate: batch.learnRate
        }

        # _, l, lo, to, fl = self.sess.run([self.learn_optimizer,
        #                                   self.learn_loss,
        #                                   self.learn_network_output,
        #                                   self.target_output,
        #                                   self.filtered_loss],
        #                      feed_dict=feed_dict)
        # print "learn loss: ", l
        # print "learn output:\n", lo
        # print "target output:\n", to
        # print "filtered loss:\n", fl

        _, l, do, fl = self.sess.run([self.learn_optimizer, self.learn_loss, self.desired_output, self.filtered_loss], feed_dict=feed_dict)
        # print "actions taken: ", batch.actionsTaken
        # print "desired output: ", do
        # print "filtered loss: ", fl
        # raw_input("blah")
        # print "loss: ", l

    def UpdateTargetParams(self):
        # print("pre-update params")
        # with self.sess.as_default():
        #     ps = self.sess.run([self.learn_network[0].weights, self.learn_network[0].bias,
        #                         self.learn_network[1].weights, self.learn_network[1].bias,
        #                         self.target_network[0].weights, self.target_network[0].bias,
        #                         self.target_network[1].weights, self.target_network[1].bias])
        #     for p in ps:
        #         print p
        #
        with self.sess.as_default():
            self.sess.run(self.update_ops)


    def QFunction(self, state):
        assert(state.ndim == 1 or state.ndim == 2)

        if state.ndim == 1:
            assert(state.shape[0] == self.num_inputs)
            state = state.reshape(1, self.num_inputs)

        if state.shape[0] > self.max_batch_size or state.shape[1] != self.num_inputs:
            raise Exception("Invalid state, wrong shape")

        original_input_size = state.shape[0]
        if state.shape[0] < self.max_batch_size:
            padded_state = np.zeros((self.max_batch_size, self.num_inputs), dtype=np.float32)
            padded_state[:state.shape[0], :] = state
            state = padded_state

        assert (state.shape[0] == self.max_batch_size and state.shape[1] == self.num_inputs)


        with self.sess.as_default():
            feed_dict = {self.target_network_input: state}
            r = self.sess.run([self.target_network_output], feed_dict=feed_dict)[0][:original_input_size, :]
            # print r
            # raw_input("Press Enter to continue...")
            if original_input_size < 10:
                print state[0]
                print r
                raw_input("weee")
            return r
