from ModelFramework import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Model(ModelInstance):
# class Model:
    def __init__(self, spec):
        self.batch_size = spec.maxBatchSize
        self.sess = None
        self._buildGraph();


    def Inference(self, input):
        print "bleeh: ", input
        return np.array([1,2,3])
        # self._initSession()
        # with self.sess.as_default():
        #     return self.sess.run([self.output], feed_dict={self.input: input.reshape(1, self.batch_size)})[0]


    def SetModelParams(self, params):
        self._initSession()
        self.sess.run([self.av.assign(params[0]), self.bv.assign(params[1])])


    def _buildGraph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.av = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)
            self.bv = tf.Variable(np.random.uniform(-10.0, 10.0, 1).reshape(1, 1), dtype=tf.float32)

            self.input = tf.placeholder(tf.float32, shape=(1, self.batch_size))
            self.output = tf.add(tf.matmul(self.av, self.input), self.bv)

            self.init_op = tf.global_variables_initializer()


    def _initSession(self):
        if self.sess is None:
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init_op)
