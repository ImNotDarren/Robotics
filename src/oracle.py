#! /usr/bin/env python

import csv
import pandas as pd
import re


class Table:
    def __init__(self):
        # item_props_table_path = '../data/cy101/cy101_labels.csv'
        self.initial_facts_path = '../data/initial_facts.txt'
        self.training_data_path = '../data/training.csv'
        self.known_items = []
        self.known_professors = []
        self.known_students = []
        self.known_rooms = []
        self.known_props = ['blue', 'yellow', 'empty', 'full', 'soft', 'hard']
        self.defaults = []
        self.training_data = []
        # self.get_items_and_props(item_props_table_path)
        self.get_people()
        self.get_training_data()

    def get_training_data(self):
        with open(self.training_data_path) as csv_file:
            df = pd.read_csv(csv_file, names=['object', 'person'])
            data_objects = df.object.tolist()[1:]
            data_people = df.person.tolist()[1:]

            if len(data_people) == len(data_objects):
                for i in range(len(data_objects)):
                    self.training_data.append([data_objects[i], data_people[i]])

    def get_people(self):
        with open(self.initial_facts_path, "r") as initial_facts:
            for line in initial_facts:
                fact = line.strip('\n')
                self.defaults.append(fact)
        p = re.compile(r'[(](.*?)[)]', re.S)
        for default in self.defaults:
            if default[0:3] == 'pro':
                self.known_professors.append(re.findall(p, default)[0])
            elif default[0:3] == 'stu':
                self.known_students.append(re.findall(p, default)[0])
            # elif default[0:3] == 'pla':
            #     self.known_rooms.append(re.findall(p, default)[0])
