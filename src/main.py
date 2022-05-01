#! /usr/bin/env python

from constructor import PomdpInit
from oracle import Table
from human import Human
from drqn import Q_Network
from agent import Agent
from object import Object

if __name__ == '__main__':
    # initialize known attributes
    table = Table()
    # get known attributes
    people = table.known_professors + table.known_students
    # initialize pomdp module
    pomdp = PomdpInit(people)

    # set up training data
    human = Human(table.training_data)
    for person in human.training_data:
        print(person.name)
        print(person.object)
        obj = Object(person.object)
        print(obj.prop)
        print('')

    agent = Agent(pomdp)

