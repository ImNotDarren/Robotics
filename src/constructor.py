# /usr/bin/env python

import numpy as np
from math import log2
import sys
import pickle
from datetime import datetime
from oracle import Table


class State(object):
    def __init__(self, term, s_index, tuple_, obj_list):
        self._term = term  # terminal state
        self._s_index = s_index
        self._tuple = tuple_
        # tuple is a dictionary, it's {object: "", person: ""}
        self._obj_list = obj_list
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
        for obj in self._obj_list:
            res = res + obj + ','
        return res[:-1]  # remove the last ','

    def tuple_to_str(self):
        res = 'o'
        if self._tuple['object'] != None:
            res += self._tuple['object']
        else:
            res += 'None'
        res += 'p'
        if self._tuple['person'] != None:
            res += self._tuple['person']
        else:
            res += 'None'
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
            if self._prop_values[0] == 'object':
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
            self._name = 'e' + ','.join(prop_values[0])
        elif o_type == 'qs':
            # 0: object/person, 1: prop_value, 2: T/F
            self._prop_values = prop_values
            self._name = prop_values[0][0] + prop_values[1] + self.TF_to_str(prop_values[2])
        elif o_type == 'na':
            self._prop_values = None
            self._name = 'na'

    @staticmethod
    def TF_to_str(p):
        if p:
            return 'T'
        else:
            return 'F'


class PomdpInit:
    def __init__(self, initial_facts):
        self._known_objects = ['1', '2', '3']
        # self._known_people = ['Alice', 'Jack', 'Anna', 'Bob', 'Hoy']
        self._known_people = initial_facts
        # self._known_props = ['blue', 'yellow', 'bottle', 'can', 'empty', 'full', 'soft', 'hard', 'metal', 'plastic']
        self._known_props = ['blue', 'yellow', 'empty', 'full', 'soft', 'hard']
        self._state = []
        self._state_object_set = []
        self._num_of_attr = len(self._known_props)
        self._action = []
        self._obs = []
        self._classifiers = []  # TODO: what is this

        self.generate_state_set()
        self.generate_action_set()

        # self._trans = np.zeros((len(self._action), len(self._state), len(self._state)))
        # self._obs_function = np.zeros((len(self._action), len(self._state), len(self._obs)))
        # self._reward_fun = np.zeros((len(self._action), len(self._state)))

        self.generate_obs_set()
        self.generate_obs_fun()
        # self.generate_reward_fun()

    def generate_state_set(self):
        # only initialize initial state
        self._state.append(State(False, 0, {'object': '', 'person': ''}, self.get_obj_list()))

    def get_obj_list(self):
        obj = ''
        for i in range(len(self._known_props)):
            obj += '0'
        return [obj, obj, obj]

    # def generate_state_set(self):
    #     # initialized state
    #     index = 1
    #     objects = ['1', '2', '3', '']
    #     table = Table()
    #     people = table.known_professors + table.known_students
    #     people.append('')
    #
    #     for obj in objects:
    #         for person in people:
    #             tmp_tuple = {'object': obj, 'person': person}
    #             self.generate_obj_set()
    #             for obj1 in self._state_object_set:
    #                 for obj2 in self._state_object_set:
    #                     for obj3 in self._state_object_set:
    #                         self._state.append(State(False, index, tmp_tuple, [obj1, obj2, obj3]))
    #                         index += 1
    #                         if obj != '' and person != '':
    #                             self._state.append(State(True, index, tmp_tuple, [obj1, obj2, obj3]))
    #                             index += 1

        # time = datetime.now()
        # # get current hour
        # curr_hour = time.strftime("%H")
        # curr_time = self.time_translator(int(curr_hour))

        # s_index = len(self._state)

    # initialize state without knowing the index
    def get_state(self, term, tuple_, obj_list):
        for s in self._state:
            if s._term == term and s._tuple == tuple_ and s._obj_list == obj_list:
                return s
        print("No such state!")
        exit(1)

    # def generate_obj_set(self):
    #     self.generate_obj_set_helper(0, '', self._num_of_attr)
    #
    # def generate_obj_set_helper(self, curr_depth, path, depth):
    #     if len(path) == depth:
    #         self._state_object_set.append(path)
    #         return
    #
    #     self.generate_obj_set_helper(curr_depth + 1, path + '0', depth)
    #     self.generate_obj_set_helper(curr_depth + 1, path + '1', depth)

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
        for obj in self._known_objects:
            self._action.append(Action(False, action_index_count, 'p-obj-' + obj, 'p', ['object', obj]))
            action_index_count += 1
        for person in self._known_people:
            self._action.append(Action(False, action_index_count, 'p-person-' + person, 'p', ['person', person]))
            action_index_count += 1

        self._action.append(Action(True, action_index_count, 'deliver', 'd', ['']))

    def generate_obs_set(self):
        # e
        self.generate_obs_set_helper(0, [], len(self._known_props))

        # qs
        for person in self._known_people:
            self._obs.append(Obs('qs', ['person', person, True]))
            self._obs.append(Obs('qs', ['person', person, False]))

        for prop in self._known_props:
            self._obs.append(Obs('qs', ['object', prop, True]))
            self._obs.append(Obs('qs', ['object', prop, False]))

        # na
        self._obs.append(Obs('na', None))

    def generate_obs_set_helper(self, curr_depth, path, depth):
        if len(path) == depth:
            self._obs.append(Obs('e', [path]))
            return

        self.generate_obs_set_helper(curr_depth + 1, path + ['0'], depth)
        self.generate_obs_set_helper(curr_depth + 1, path + ['1'], depth)

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
