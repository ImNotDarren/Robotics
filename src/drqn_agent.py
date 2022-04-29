from src.agent import BaseAgent
from src.replay_memory import DRQNReplayMemory
from src.drqn import DRQN
import numpy as np


class DRQNAgent(BaseAgent):

    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)
        self.replay_memory = DRQNReplayMemory(config)
        self.net = DRQN(len(self.env_wrapper.action), config)
        self.net.build()
        self.net.add_summary(
            ["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game",
             "learning_rate"], ["ep_rewards", "ep_actions"])

    def observe(self, t):
        # TODO
        a = 1

    def policy(self, state):
        self.random = False
        if np.random.rand() < self.epsilon:
            self.random = True  # take random action
            return self.env_wrapper.random_step()
        else:
            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run(
                [self.net.q_action, self.net.state_output_c, self.net.state_output_h], {
                    self.net.state: [[state]],
                    self.net.c_state_train: self.lstm_state_c,
                    self.net.h_state_train: self.lstm_state_h
                })
            return a[0]

    def train(self):
        # TODO
        a = 1

    def play(self):
        # TODO
        a = 1
