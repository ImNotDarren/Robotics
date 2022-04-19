# /usr/bin/env python

import numpy as np
from math import log2
import sys
import pickle
from datetime import datetime
from oracle import Table


class State(object):
    def __init__(self, term, s_index, tuple_, item_list):
        self._term = term  # terminal state
        self._s_index = s_index
        self._tuple = tuple_
        # tuple is a dictionary, it's {item: "", person: ""}
        # (3 items + 1 empty) x (5 people + 1 empty) = 24 possibilities
        self._item_list = item_list
        # ['00000', '00100', '00110'] assume there will only be 3 items
        # there will only be 10 attributes for now
        # attributes:
        # blue
        # yellow
        # bottle
        # can
        # empty
        # full
        # soft
        # hard
        # metal
        # plastic
        # 2^5 X 3 = 32 X 3 = 96
        # TODO: implementing time later
        # self._current_time = current_time  # morning 7-11, noon 11-13, afternoon 13-17, night 17-22, midnight 22-7
        self._name = self.get_name()
        # number of states: 24 X 96 = 2304

    def item_to_str(self):
        res = ''
        for item in self._item_list:
            res = res + item + ','
        return res[:-1]  # remove the last ','

    def tuple_to_str(self):
        res = 'i'
        if self._tuple['item'] != None:
            res += self._tuple['item']
        res += 'p'
        if self._tuple['person'] != None:
            res += self._tuple['person']
        return res

    def get_name(self):
        if self._term is True:
            return 'terminal'
        else:
            return 'S' + str(self._s_index) + ',IS' + self.item_to_str() + ',T' + self.tuple_to_str()


class Action(object):
    def __init__(self, term, a_index, name, a_type, prop_values):
        self._term = term  # T/F, deliver action
        self._a_index = a_index
        self._a_type = a_type
        # 'e' as explore action
        # 'p' ask polar question
        # 'wh' ask wh-question
        # 'd' deliver
        self._sentence = None

        if self._a_type == 'd':
            self._prop_values = None
            self._name = name
            self._sentence = self.generate_sentence()

        else:
            self._prop_values = prop_values
            self._name = name
            if self._a_type == 'p' or 'wh':
                self._sentence = self.generate_sentence()

        # prop_values are ['look'] when _a_type is 'e'
        # prop_values is ['item'] or ['person'] when _a_type is 'wh'
        # prop_values is ['item', 'sandwich'] or ['person', 'Alice'] or ['room', 'Office 1'] when _a_type is 'wh'

    def generate_sentence(self):
        if self._a_type == 'wh':
            # if it's a wh-question
            if self._prop_values[0] == 'item':
                qs = 'Which item should be delivered?'
            elif self._prop_values[0] == 'person':
                qs = 'Who should I deliver this item to?'
            else:
                print('Invalid person value! ' + str(self._prop_values[0]))
                exit(1)
        elif self._a_type == 'p':
            if self._prop_values[0] == 'item':
                qs = 'Should I deliver a ' + self._prop_values[1] + '?'
            elif self._prop_values[0] == 'person':
                qs = 'Is this deliver for ' + self._prop_values[1] + '?'
            # elif self._prop_values[0] == 'room':
            #     qs = 'Should I deliver it to ' + self._prop_values[1] + '?'
            else:
                print('Invalid property value!')
                exit(1)
        elif self._a_type == 'd':
            qs = 'Is this what you want?'

        elif self._a_type == 'e':
            qs = 'Exploring...'
        else:
            print('Invalid action type ' + str(self._a_type) + ' !')
            exit(1)

        return qs


class Obs(object):
    def __init__(self, o_type, prop_values):
        self._o_type = o_type
        # na, e, qs

        if o_type == 'e':
            self._prop_values = prop_values
            self.name = 'p' + ','.join(prop_values[0])
        elif o_type == 'qs':
            # 0: item/person, 1: prop_value, 2: T/F
            self._prop_values = prop_values
            self.name = prop_values[0][0] + prop_values[1] + prop_values[2]
        elif o_type == 'na':
            self._prop_values = None
            self._name = 'na'


class PomdpInit:
    def __init__(self, initial_facts):
        self._known_items = ['1', '2', '3']
        # self._known_people = ['Alice', 'Jack', 'Anna', 'Bob', 'Hoy']
        self._known_people = initial_facts
        # self._known_props = ['blue', 'yellow', 'bottle', 'can', 'empty', 'full', 'soft', 'hard', 'metal', 'plastic']
        self._known_props = ['blue', 'yellow', 'empty', 'full', 'soft', 'hard']
        self._state = []
        self._state_item_set = []
        self._num_of_attr = len(self._known_props)
        self._action = []
        self._obs = []
        self._classifiers = []  # TODO: what is this

        self.generate_state_set()
        self.generate_action_set()


        # self._trans = np.zeros((len(self._action), len(self._state), len(self._state)))
        # self._obs_function = np.zeros((len(self._action), len(self._state), len(self._obs)))
        # self._reward_fun = np.zeros((len(self._action), len(self._state)))

        # self.generate_observation_set()
        # self.generate_obs_fun()
        # self.generate_reward_fun()



    def generate_state_set(self):
        # initialized state
        index = 1
        items = ['1', '2', '3', '']
        table = Table()
        people = table.known_professors + table.known_students
        people.append('')

        for item in items:
            for person in people:
                tmp_tuple = {'item': item, 'person': person}
                self.generate_item_set()
                for item_set in self._state_item_set:
                    self._state.append(State(False, index, tmp_tuple, item_set))
                    index += 1
                    if item != '' and person != '':
                        self._state.append(State(True, index, tmp_tuple, item_set))
                        index += 1


        # time = datetime.now()
        # # get current hour
        # curr_hour = time.strftime("%H")
        # curr_time = self.time_translator(int(curr_hour))

        # s_index = len(self._state)

    def generate_item_set(self):
        self.generate_item_set_helper(0, [], self._num_of_attr)


    def generate_item_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._state_item_set.append(path)
            return

        self.generate_item_set_helper(curr_depth + 1, path + ['0'], depth)
        self.generate_item_set_helper(curr_depth + 1, path + ['1'], depth)

    # translate time from hour to time period
    def time_translator(self, curr_hour):
        # morning 7-11, noon 11-13, afternoon 13-17, night 17-22, midnight 22-7
        if 7 <= curr_hour < 11:
            return 'morning'
        elif 11 <= curr_hour < 13:
            return 'noon'
        elif 13 <= curr_hour < 17:
            return 'afternoon'
        elif 17 <= curr_hour < 22:
            return 'night'
        else:
            return 'midnight'

    def generate_action_set(self):
        # ask wh-questions
        self._action.append(Action(False, 0, 'wh-item', 'wh', ['item']))
        self._action.append(Action(False, 1, 'wh-person', 'wh', ['person']))
        self._action.append(Action(False, 2, 'look', 'e', ['look']))
        self._action.append(Action(False, 3, 'grasp', 'e', ['grasp']))
        self._action.append(Action(False, 4, 'lift_slow', 'e', ['lift_slow']))
        self._action.append(Action(False, 5, 'hold', 'e', ['hold']))
        self._action.append(Action(False, 6, 'shake', 'e', ['shake']))
        self._action.append(Action(False, 7, 'low_drop', 'e', ['low_drop']))
        self._action.append(Action(False, 8, 'tap', 'e', ['tap']))
        self._action.append(Action(False, 9, 'push', 'e', ['push']))
        self._action.append(Action(False, 10, 'poke', 'e', ['poke']))
        self._action.append(Action(False, 11, 'crush', 'e', ['crush']))
        self._action.append(Action(False, 12, 'reinit', 'e', ['reinit']))
        action_index_count = 13
        # ask polar questions
        for item in self._known_items:
            self._action.append(Action(False, action_index_count, 'p-item-'+item, 'p', ['item', item]))
            action_index_count += 1
        for person in self._known_people:
            self._action.append(Action(False, action_index_count, 'p-person-'+person, 'p', ['person', person]))
            action_index_count += 1

        self._action.append(Action(True, action_index_count, 'deliver', 'd', ['']))

    def generate_observation_set(self):
        TODO = 1

    def generate_obs_fun(self):
        TODO = 1

    def generate_reward_fun(self):

        for action in self._action:
            for state in self._state:
                if state._term:
                    self._reward_fun[action._a_index, state._s_index] = 0.0
                else:
                    if action._a_type == 'e':
                        ITRS_reward = 0  # TODO: need to get ITRS_reward
                        self._reward_fun[action._a_index, state._s_index] = ITRS_reward

                    elif action._a_type == 'wh' or action._a_type == 'p':

                        self._reward_fun[action._a_index, state._s_index] = -8.0  # TODO: this is only a rough number
                    # TODO: need to add final reward

    @property
    def state(self):
        return self._state
