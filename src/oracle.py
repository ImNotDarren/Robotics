#! /usr/bin/env python

import csv
import pandas as pd
import re


class Table:
    def __init__(self):
        # item_props_table_path = '../data/cy101/cy101_labels.csv'
        self.initial_facts_path = '../data/initial_facts.txt'
        self.training_data_path = '../data/training.csv'
        self.objects = []
        self.professors = []
        self.students = []
        self.females = []
        self.males = []
        self.rooms = []
        self.predicates = ['soft', 'green', 'full', 'empty', 'container', 'plastic', 'hard', 'blue', 'metal', 'toy']
        self.defaults = []
        self.training_data = []
        # self.get_items_and_props(item_props_table_path)

        self.behaviors = ['look', 'grasp', 'lift_slow', 'hold', 'shake', 'low_drop', 'tap', 'push', 'poke', 'crush']
        self.modalities = ['surf', 'color', 'flow', 'audio', 'vibro', 'fingers', 'haptics']
        self.contexts = []

        self.get_contexts()
        self.get_people()
        self.get_training_data()

    def get_training_data(self):
        with open(self.training_data_path) as csv_file:
            df = pd.read_csv(csv_file, names=['object', 'person', 'object_set'])
            data_objects = df.object.tolist()[1:]
            data_people = df.person.tolist()[1:]
            data_object_set = df.object_set.tolist()[1:]

            # get objects
            self.objects = data_objects

            for i in range(len(data_objects)):
                self.training_data.append([data_objects[i], data_people[i], data_object_set[i]])

    def get_people(self):
        with open(self.initial_facts_path, "r") as initial_facts:
            for line in initial_facts:
                fact = line.strip('\n')
                self.defaults.append(fact)
        p = re.compile(r'[(](.*?)[)]', re.S)
        for default in self.defaults:
            if default[0:3] == 'pro':
                self.professors.append(re.findall(p, default)[0])
            elif default[0:3] == 'stu':
                self.students.append(re.findall(p, default)[0])
            elif default[0:3] == 'fem':
                self.females.append(re.findall(p, default)[0])
            elif default[0:3] == 'mal':
                self.males.append(re.findall(p, default)[0])
            # elif default[0:3] == 'pla':
            #     self.known_rooms.append(re.findall(p, default)[0])

    def get_contexts(self):
        for b in self.behaviors:
            for m in self.modalities:
                if self.is_valid_context(b, m):
                    context_bm = b+'_'+m
                    self.contexts.append(context_bm)

    @staticmethod
    def is_valid_context(behavior, modality):
        if behavior == 'look':
            if modality == 'color' or modality == 'surf':
                return True
            else:
                return False
        elif behavior == 'grasp' and modality == 'fingers':
            return True
        elif modality in ['flow', 'surf', 'audio', 'vibro', 'haptics']:
            return True
        else:
            return False
