import pandas as pd


class Object:
    def __init__(self, name):
        self.obj_path = '../data/object_list.csv'
        self.name = name
        self.prop_list = self.get_prop()
        self.prop = self.prop_to_str()

    # get object's properties by its name
    # ['blue', 'yellow', 'empty', 'full', 'soft', 'hard']
    def get_prop(self):
        prop_string = None
        with open(self.obj_path) as csv_file:
            df = pd.read_csv(csv_file, names=['object', 'words'])
            data_objects = df.object.tolist()[1:]
            data_words = df.words.tolist()[1:]
            if len(data_objects) == len(data_words):
                for i in range(len(data_objects)):
                    if data_objects[i] == self.name:
                        prop_string = data_words[i]

            if prop_string is None:
                print('No object ' + self.name + '!')
                exit(1)
            else:
                # turn prop_string into list and return
                return prop_string.split(', ')

    def prop_to_str(self):  # turn property list into 000110
        res = ''
        flag = '1'
        prop_set = ['blue', 'yellow', 'empty', 'full', 'soft', 'hard']
        for prop in prop_set:
            for obj_prop in self.prop_list:
                if prop == obj_prop:
                    flag = '2'
                    break
            res += flag
            flag = '1'

        return res
