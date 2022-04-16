#! /usr/bin/env python

import numpy as np
import tensorflow as tf
import csv
from constructor import State, Action, Obs, PomdpInit
from oracle import Table

if __name__ == '__main__':
    # initialize known attributes
    table = Table()
    # get known attributes
    items = table.known_items
    people = table.known_professors + table.known_students
    rooms = table.known_rooms
    props = table.known_props
    ka = [items, people, rooms, props]
    print(people)
    # model = PomdpInit([items, people, rooms, props])
    # agent = DQNAgent(len(model._state), len(model._action))
    pomdp = PomdpInit()
    pretraining_data_path = '../data/pretraining.csv'
    # a = State(False, 1, {'item': None, 'person': None, 'room': None}, [], 'morning')
