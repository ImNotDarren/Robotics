#! /usr/bin/env python3

from constructor import PomdpInit
from oracle import Table
from human import Human
from drqn import Q_Network
from agent import Agent
from object import Object


def main():
    # initialize pomdp module
    pomdp = PomdpInit()
    # set up training data
    table = Table()
    human = Human(table.training_data)
    for person in human.training_data:
        agent = Agent(pomdp)
        agent.train(person)
        exit()

    # after training, start testing
    agent.test()


if __name__ == '__main__':
    main()

