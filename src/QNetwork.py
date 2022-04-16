import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import random
from collections import deque
from constructor import State, Action, Item, Obs, PomdpInit


class DRQNAgent:
    def __init__(self, state_size, action_size):
        self.optimizer = None
        self.cost = None
        self.q_value = None
        self.target_q_value = None
        self.state_input = None
        self.action_input = None
        self.target_replace_op = None
        self.y_input = None

        self.state_dim = state_size
        self.action_dim = action_size
        self.replay_buffer = deque()
        self.create_q_network()
        self.create_updating_method()
        self.epsilon = 0.5
        self.final_epsilon = 0.01
        self.gamma = 0.9
        self.session = tf.InteractiveSession()
        self.session.run(tf.globle_variables_initializer())

    def create_q_network(self):
        self.state_input = tf.placeholder('float', [None, self.state_dim], name='inputs')
        with tf.variable_scope('curr_network'):
            w1 = self.weight_variable([self.state_dim, 50])
            b1 = self.bias_variable([50])
            w2 = self.weight_variable([50, 20])
            b2 = self.bias_variable([20])
            w3 = self.weight_variable([20, self.action_dim])
            b3 = self.bias_variable([self.action_dim])
            # set layers
            hidden_layer1 = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w2) + b2)
            self.q_value = tf.matmul(hidden_layer2, w3) + b3

        with tf.variable_scope('target_network'):
            t_hidden_layer1 = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
            t_hidden_layer2 = tf.nn.relu(tf.matmul(t_hidden_layer1, w2) + b2)
            self.target_q_value = tf.matmul(t_hidden_layer2, w3) + b3
            # set parameters
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='curr_network')
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    @staticmethod
    def weight_variable(shape):
        init = tf.truncated_normal(shape)
        return tf.Variable(init)

    @staticmethod
    def bias_variable(shape):
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init)

    def create_updating_method(self):
        self.action_input = tf.placeholder('float', [None, self.action_dim])
        self.y_input = tf.placeholder('float', [None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def policy(self, state):
        q_value = self.q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]

        # explore method
        if random.random() <= self.epsilon:
            # return a random action
            self.epsilon -= (0.5 - self.final_epsilon)/10000
            return random.randint(0, self.action_dim - 1)
        else:
            # return the argmax action
            self.epsilon -= (0.5 - self.final_epsilon)/10000
            return np.argmax(q_value)

    def store_data(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action._a_index] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))

        if len(self.replay_buffer) > 10000:
            self.replay_buffer.popleft()

    def train_network(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        q_value_batch = self.target_q_value.eval(feed_dict={
            self.state_input: next_state_batch
        })

        for i in range(0, batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def update_target_network(self):
        self.session.run(self.target_replace_op)



