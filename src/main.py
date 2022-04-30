#! /usr/bin/env python

from constructor import PomdpInit
from oracle import Table
from human import Human
from drqn import Q_Network
from agent import Agent

if __name__ == '__main__':
    # initialize known attributes
    table = Table()
    # get known attributes
    people = table.known_professors + table.known_students

    pomdp = PomdpInit(people)
    human = Human(table.training_data)

    agent = Agent(pomdp)
    agent.train()

