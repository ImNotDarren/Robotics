#! /usr/bin/env python

import numpy as np
import tensorflow as tf
import csv
from constructor import State, Action, Obs, PomdpInit
from oracle import Table
from human import Human, Person

if __name__ == '__main__':
    # initialize known attributes
    table = Table()
    # get known attributes
    people = table.known_professors + table.known_students

    pomdp = PomdpInit(people)

    for obs in pomdp._obs:
        print(obs._name)

    human = Human(table.training_data)

    for h in human.training_data:
        print(h.name + ',' + h.object)
