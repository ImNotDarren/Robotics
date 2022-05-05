#! /usr/bin/env python

from constructor import PomdpInit
from oracle import Table
from human import Human
from drqn import Q_Network
from agent import Agent
from object import Object


def main():
    # initialize pomdp module
    table = Table()
    pomdp = PomdpInit()
    # print(pomdp._known_props)
    # set up training data
    human = Human(table.training_data)
    for person in human.training_data:
        agent = Agent(pomdp)
        # print(agent.person.name)
        # print(agent.person.prop_ground_truth)
        agent.train(person)
        exit()


if __name__ == '__main__':
    main()

