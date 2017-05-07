
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
        self.reward_discount = 0.99

        self.l1_size = self.num_inputs

        self._buildGraph()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)


    def _buildTargetNetwork(self):
        self.target_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))

        # augment the input matrix with 1s so that we don have to use an explicit bias.
        self.target_weights_l1 = tf.Variable(
            tf.truncated_normal([self.num_inputs, self.l1_size], stddev=(1.0/self.num_inputs)), dtype=tf.float32)
        self.target_bias_l1 = tf.Variable(
            tf.truncated_normal([self.l1_size], stddev=(1.0/self.num_inputs)), dtype=tf.float32)
        target_l1 = tf.nn.tanh(
            tf.matmul(self.target_network_input, self.target_weights_l1) + self.target_bias_l1)

        self.target_weights_output = tf.Variable(
            tf.truncated_normal([self.l1_size, self.num_outputs], stddev=(1.0/self.l1_size)), dtype=tf.float32)
        self.target_bias_output = tf.Variable(
            tf.truncated_normal([self.num_outputs], stddev=(1.0/self.l1_size)), dtype=tf.float32)

        self.target_network_output = tf.matmul(target_l1, self.target_weights_output) + self.target_bias_output


    def _buildLearnNetwork(self):
        # learning network
        self.learn_network_input = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_inputs))
        self.learn_network_action_mask = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_outputs))
        self.learn_network_terminal_mask = tf.placeholder(tf.bool, shape=(self.max_batch_size))
        self.learn_network_reward = tf.placeholder(tf.float32, shape=(self.max_batch_size, self.num_outputs))

        self.learn_weights_l1 = tf.Variable(
            tf.truncated_normal([self.num_inputs, self.l1_size], stddev=(1.0/self.num_inputs)), dtype=tf.float32)
        self.learn_bias_l1 = tf.Variable(
            tf.truncated_normal([self.l1_size], stddev=(1.0/self.num_inputs)), dtype=tf.float32)
        learn_l1 = tf.nn.tanh(
            tf.matmul(self.learn_network_input, self.learn_weights_l1) + self.learn_bias_l1)

        self.learn_weights_output = tf.Variable(
            tf.truncated_normal([self.l1_size, self.num_outputs], stddev=(1.0/self.l1_size)), dtype=tf.float32)
        self.learn_bias_output = tf.Variable(
            tf.truncated_normal([self.num_outputs], stddev=(1.0/self.l1_size)), dtype=tf.float32)
        learn_network_output = tf.matmul(learn_l1, self.learn_weights_output) + self.learn_bias_output

        terminating_target = self.learn_network_reward
        intermediate_target = self.target_network_output * self.reward_discount + self.learn_network_reward
        target_output = tf.where(self.learn_network_terminal_mask, terminating_target, intermediate_target)

        filtered_loss = tf.squared_difference(target_output, learn_network_output) * self.learn_network_action_mask

        self.learn_loss = tf.reduce_mean(filtered_loss)
        self.learn_optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.learn_loss)

        self.update0 = tf.assign(self.target_weights_l1, self.learn_weights_l1, validate_shape=True, use_locking=True)
        self.update1 = tf.assign(self.target_bias_l1, self.learn_bias_l1, validate_shape=True, use_locking=True)
        self.update2 = tf.assign(self.target_weights_output, self.learn_weights_output, validate_shape=True, use_locking=True)
        self.update3 = tf.assign(self.target_bias_output, self.learn_bias_output, validate_shape=True, use_locking=True)


    def _buildGraph(self):
        self.total_iters = 0
        self.graph = tf.Graph()

        with self.graph.as_default():
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

        batch_indices = np.empty(self.max_batch_size * 2, dtype=np.int32)
        batch_indices[0::2] = np.arange(self.max_batch_size)
        batch_indices[1::2] = batch.actionsTaken
        batch_indices = batch_indices.reshape(self.max_batch_size, 2)

        actions_mask = np.zeros((self.max_batch_size, self.num_outputs), dtype=np.float32)
        actions_mask[batch_indices[:,0], batch_indices[:,1]] = 1.0

        rewards = np.zeros((self.max_batch_size, self.num_outputs), dtype=np.float32)
        rewards[batch_indices[:,0], batch_indices[:,1]] = batch.rewardsGained

        feed_dict = {
            self.target_network_input: batch.initialStates,
            self.learn_network_input: batch.successorStates,
            self.learn_network_action_mask: actions_mask,
            self.learn_network_terminal_mask: batch.isEndStateTerminal,
            self.learn_network_reward: rewards,
        }

        _, l = self.sess.run([self.learn_optimizer, self.learn_loss], feed_dict=feed_dict)
        print "learn loss: ", l


    def UpdateTargetParams(self):
        with self.sess.as_default():
            self.sess.run([self.update0, self.update1, self.update2, self.update3])


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
