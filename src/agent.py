
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

from train import Classifier
from oracle import Table


class Agent:
    def __init__(self, pomdp):
        # init state
        self.current_state = pomdp.get_state(False, {'object': self.get_obj(pomdp), 'title': 'u', 'gender': 'u'}, [self.get_obj(pomdp), self.get_obj(pomdp), self.get_obj(pomdp)])
        self.current_obs = None
        self.next_obs = None
        self.actionOut = None
        self.rewardOut = None

        self.hidden_space = 16
        self.batch_size = 4
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.random_update = True
        self.lookup_step = 20
        self.max_epi_len = 20
        self.min_epi_num = 4
        self.max_epi_num = 1000

        self.episode_memory = EpisodeMemory(random_update=self.random_update,
                                            max_epi_num=self.max_epi_num,
                                            max_epi_len=self.max_epi_len,
                                            batch_size=self.batch_size,
                                            lookup_step=self.lookup_step)

        self.target_update_period = 4
        self.eps_start = 0.1
        self.eps_end = 0.001
        self.eps_decay = 0.999
        self.tau = 1e-2

        self.device = torch.device('cpu')
        # self.get_device()

        self.q_network = Q_Network(len(pomdp._state), len(pomdp._action), self.hidden_space).to(self.device)
        self.target_q_network = Q_Network(len(pomdp._state), len(pomdp._action), self.hidden_space).to(self.device)

        # classifier data path
        self.data_path = '../data/cy101/normalized_data_without_noobject/'

    @staticmethod
    def get_obj(pomdp):
        obj = ''
        for i in range(len(pomdp.predicates)):
            obj += '0'
        return obj

    def get_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            return True
        return False

    def policy(self, state):
        # TODO
        return

    def test(self):
        # TODO
        return

    def train(self, person):  # person.name / person.object / person.prop_ground_truth
        # setup device
        self.print_device()

        # print out ground truth
        self.print_ground_truth(person)

        # classifier
        classifier_all = Classifier(self.data_path)
        # print(classifier_all.get_features(table.contexts[0], 'cup_yellow', 2))

    @staticmethod
    def print_ground_truth(person):
        print('--- Ground Truth ---')
        print('Person:')
        print('name: ' + person.name)
        print('title: ' + person.title)
        print('gender: ' + person.gender)
        print('\nObject:')
        print('name: ' + person.object)
        print('properties: ' + person.prop_ground_truth)

    def print_device(self):
        if self.get_device():
            print('Using device: cuda')
        else:
            print('Using device: cpu')

