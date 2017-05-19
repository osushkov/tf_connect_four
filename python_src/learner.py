
from LearnerFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.keras as keras
import math


class Learner(LearnerInstance):
    def __init__(self, networkSpec):
        self.num_inputs = networkSpec.numInputs
        self.num_outputs = networkSpec.numOutputs
        self.max_batch_size = networkSpec.maxBatchSize
        self.reward_discount = 1.0

        self.layer_sizes = [self.num_inputs * 2, self.num_inputs, self.num_inputs / 2]
        self._buildGraph()

        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            self.sess.run([self.init_op])
            self.sess.run(self.update_ops)


    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()

        with self.graph.as_default():
            self._buildTargetNetwork()
            self._buildLearnNetwork()
            self.init_op = tf.global_variables_initializer()


    def _buildNetwork(self, out_layers, in_tensor):
        # feed_in_tensor = tf.reshape(in_tensor, [-1, 6, 7, 1])
        #
        # new_layer = keras.layers.Conv2D(
        #     filters=16,
        #     kernel_size=3,
        #     activation=tf.nn.elu)
        # new_layer.build(feed_in_tensor.shape)
        # out_layers.append(new_layer)
        # feed_in_tensor = new_layer.call(feed_in_tensor)
        #
        # new_layer = keras.layers.MaxPool2D()
        # new_layer.build(feed_in_tensor.shape)
        # out_layers.append(new_layer)
        # feed_in_tensor = new_layer.call(feed_in_tensor)
        #
        # # flatten it
        # size = int(feed_in_tensor.shape[1] * feed_in_tensor.shape[2] * feed_in_tensor.shape[3])
        # feed_in_tensor = tf.reshape(feed_in_tensor, [-1, size])

        feed_in_tensor = in_tensor

        for ls in self.layer_sizes:
            new_layer = keras.layers.Dense(units=ls, activation=tf.nn.elu)
            new_layer.build(feed_in_tensor.shape)
            out_layers.append(new_layer)
            feed_in_tensor = new_layer.call(feed_in_tensor)

        new_layer = keras.layers.Dense(units=self.num_outputs, activation=tf.nn.tanh)
        new_layer.build(feed_in_tensor.shape)
        out_layers.append(new_layer)

        return new_layer.call(feed_in_tensor)


    def _buildTargetNetwork(self):
        self.target_network = []
        self.target_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))

        output_tensor = self._buildNetwork(self.target_network, self.target_network_input)
        self.target_network_output = tf.stop_gradient(output_tensor)


    def _buildLearnNetwork(self):
        # learning network
        self.learn_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))
        self.learn_network_action_index = tf.placeholder(tf.int32, shape=(self.max_batch_size))
        self.learn_network_terminal_mask = tf.placeholder(tf.bool, shape=(self.max_batch_size))
        self.learn_network_reward = tf.placeholder(tf.float32, shape=(self.max_batch_size))
        self.learn_rate = tf.placeholder(tf.float32)

        self.learn_network = []
        self.learn_network_output = self._buildNetwork(self.learn_network, self.learn_network_input)

        terminating_target = self.learn_network_reward
        intermediate_target = self.learn_network_reward + (tf.reduce_max(self.target_network_output, axis=1) * self.reward_discount)
        self.desired_output = tf.stop_gradient(
            tf.where(self.learn_network_terminal_mask, terminating_target, intermediate_target))

        index_range = tf.constant(np.arange(self.max_batch_size), dtype=tf.int32)
        action_indices = tf.stack([index_range, self.learn_network_action_index], axis=1)
        self.indexed_output = tf.gather_nd(self.learn_network_output, action_indices)

        self.learn_loss = tf.losses.mean_squared_error(self.desired_output, self.indexed_output)

        opt = tf.train.AdamOptimizer(self.learn_rate)

        vars_to_optimise = []
        for ll in self.learn_network:
            vars_to_optimise.append(ll.trainable_weights)

        self.learn_optimizer = opt.minimize(self.learn_loss, var_list=vars_to_optimise)

        self.update_ops = []
        assert (len(self.target_network) == len(self.learn_network))
        for i in range(len(self.target_network)):
            target_weights = self.target_network[i].trainable_weights
            learn_weights = self.learn_network[i].trainable_weights
            assert (len(target_weights) == len(learn_weights))

            for j in range(len(target_weights)):
                self.update_ops.append(
                    tf.assign(target_weights[j], learn_weights[j], validate_shape=True, use_locking=True))


    def Learn(self, batch):
        assert (batch.initialStates.ndim == 2 and batch.successorStates.ndim == 2)
        assert (batch.initialStates.shape[0] == self.max_batch_size and batch.initialStates.shape[1] == self.num_inputs)
        assert (batch.successorStates.shape[0] == self.max_batch_size and batch.successorStates.shape[1] == self.num_inputs)
        assert (batch.actionsTaken.ndim == 1 and batch.actionsTaken.shape[0] == self.max_batch_size)
        assert (batch.isEndStateTerminal.ndim == 1 and batch.isEndStateTerminal.shape[0] == self.max_batch_size)
        assert (batch.rewardsGained.ndim == 1 and batch.rewardsGained.shape[0] == self.max_batch_size)

        with self.sess.as_default():
            learn_feed_dict = {
                self.learn_network_input: batch.initialStates,
                self.target_network_input: batch.successorStates,
                self.learn_network_action_index: batch.actionsTaken,
                self.learn_network_terminal_mask: batch.isEndStateTerminal,
                self.learn_network_reward: batch.rewardsGained,
                self.learn_rate: batch.learnRate
            }

            self.sess.run([self.learn_optimizer], feed_dict=learn_feed_dict)


    def UpdateTargetParams(self):
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
            return self.sess.run([self.target_network_output], feed_dict=feed_dict)[0][:original_input_size, :]
