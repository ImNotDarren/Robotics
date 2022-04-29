#! /usr/bin/env python

from constructor import PomdpInit
from oracle import Table
from human import Human

if __name__ == '__main__':
    # initialize known attributes
    table = Table()
    # get known attributes
    people = table.known_professors + table.known_students

    pomdp = PomdpInit(people)

    # for obs in pomdp._obs:
    #     print(obs._name)

    human = Human(table.training_data)

    # for h in human.training_data:
    #     print(h.name + ',' + h.object)
