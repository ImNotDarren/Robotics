#! /usr/bin/env python

class Person(object):
    def __init__(self, data):
        self.object = data[0]
        self.name = data[1]


class Human:
    def __init__(self, training_data):
        # training_data is [[object1, person1], [object2, person2], ...]
        self.training_data = []
        self.get_training_data(training_data)

    def get_training_data(self, training_data):
        for data in training_data:
            temp_person = Person(data)
            self.training_data.append(temp_person)
