#! /usr/bin/env python3
from object import Object
from oracle import Table


class Person(object):
    def __init__(self, data):
        self.object = data[0]
        self.name = data[1]
        self.object_set = data[2].split(', ')
        self.object_index = self.object_set.index(self.object)
        self.obj_path = '../data/object_list.csv'
        self.prop_ground_truth = Object(self.object).prop
        self.prop_list = Object(self.object).prop_list
        self.queried_attr = []
        self._known_props = []
        self.get_known_props()

    def get_known_props(self):
        table = Table()
        self._known_props = table.known_props

    # answer questions, return answers as strings
    def answer(self, qs):
        if qs == 'Who should I deliver this object to?':
            return self.name
        elif qs == 'What kind of object should be delivered?':
            # only answer one or two attribute
            i = 0
            while i < len(self.prop_ground_truth):
                index = self.prop_ground_truth.find('1', i)
                if self.queried_attr.count(index) > 0:
                    # if the attribute is already answered
                    i = index + 1
                else:
                    # if the attribute is never answered
                    self.queried_attr.append(index)  # add the attribute index into the list
                    return self._known_props[index]
        elif qs.find('Should I deliver a') != -1:
            i = qs.find(' object?')
            # get the property name
            prop_name = qs[19:i]
            prop_index = self._known_props.index(prop_name)
            if self.prop_ground_truth[prop_index] == '1':
                return 'Yes'
            elif self.prop_ground_truth[prop_index] == '0':
                return 'No'
            else:
                print('There is no ' + prop_name + '!')
                exit(0)

        elif qs.find('Is this deliver for ') != -1:
            i = qs.find('?')
            person_name = qs[20:i]
            if self.name == person_name:
                return 'Yes'
            else:
                return 'No'

        elif qs.find('Is the ') != -1:
            object_index = qs[7:8]
            if object_index == str(self.object_index + 1):
                return 'Yes'
            else:
                return 'No'

        else:
            print('Invalid question!')
            exit(1)


class Human:
    def __init__(self, training_data):
        # training_data is [[object1, person1, object_set1], [object2, person2, object_set2], ...]
        self.training_data = []
        self.get_training_data(training_data)

    def get_training_data(self, training_data):
        for data in training_data:
            temp_person = Person(data)
            self.training_data.append(temp_person)
