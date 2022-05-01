import pandas as pd


class Object:
    def __init__(self, name):
        self.obj_path = '../data/object_list.csv'
        self.name = name
        self.prop = self.get_prop()

    # get object's properties by its name
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
