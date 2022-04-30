
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from drqn import Q_Network, EpisodeMemory, EpisodeBuffer
from constructor import State, Action, Obs, PomdpInit

class Agent:
    def __init__(self, pomdp):
        # init state
        self.current_obs = pomdp.get_state(False, {'object': '', 'person': ''}, self.get_obj_list(pomdp))
        self.next_obs = None
        self.actionOut = None
        self.rewardOut = None

        self.q_network = Q_Network(len(pomdp._state), len(pomdp._action))
        self.target_q_network = Q_Network(len(pomdp._state), len(pomdp._action))
        self.device = None

        self.hidden_space = 16
        self.batch_size = 4
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.random_update = True
        self.lookup_step = 20
        self.max_epi_len = 20
        self.min_epi_num = 4
        self.max_epi_num = 1000

        self.episode_memory = EpisodeMemory(random_update=self.random_update, max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len, batch_size=self.batch_size, lookup_step=self.lookup_step)

        self.target_update_period = 4
        self.eps_start = 0.1
        self.eps_end = 0.001
        self.eps_decay = 0.999
        self.tau = 1e-2

    @staticmethod
    def get_obj_list(pomdp):
        obj = ''
        for i in range(len(pomdp._known_props)):
            obj += '0'
        return [obj, obj, obj]

    def policy(self, state):
        # TODO
        return

    def train(self):
        # setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

