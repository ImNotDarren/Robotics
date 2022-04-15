#! /usr/bin/env python

import csv
import pandas as pd
import re


class Table:
    def __init__(self):
        item_props_table_path = '../data/cy101/cy101_labels.csv'
        initial_facts_path = '../data/initial_facts.txt'
        self.known_items = []
        self.known_professors = []
        self.known_students = []
        self.known_rooms = []
        self.known_props = []
        self.defaults = []
        self.get_items_and_props(item_props_table_path)
        self.get_people_and_rooms(initial_facts_path)

    def get_items_and_props(self, table_path):
        with open(table_path) as csv_file:
            df = pd.read_csv(csv_file, names=['objects', 'words'])
            data_objects = df.objects.tolist()[1:]
            data_words = df.words.tolist()[1:]

            for row in data_objects:
                if row != 'no_object':
                    self.known_items.append(row)
            temp_props = []
            for i, row in enumerate(data_words):
                if data_objects[i] != 'no_object':
                    if type(row) == str:
                        split_row = row.split(', ')
                        for prop in split_row:
                            temp_props.append(prop)

            self.known_props = list(set(temp_props))

    def get_people_and_rooms(self, initial_facts_path):
        with open(initial_facts_path, "r") as initial_facts:
            for line in initial_facts:
                fact = line.strip('\n')
                self.defaults.append(fact)
        p = re.compile(r'[(](.*?)[)]', re.S)
        for default in self.defaults:
            if default[0:3] == 'pro':
                self.known_professors.append(re.findall(p, default)[0])
            elif default[0:3] == 'stu':
                self.known_students.append(re.findall(p, default)[0])
            elif default[0:3] == 'pla':
                self.known_rooms.append(re.findall(p, default)[0])
