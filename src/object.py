import pandas as pd


class Object:
    def __init__(self, name):
        self.obj_path = '../data/object_list.csv'
        self.name = name
        self.pred_list = self.get_pred()
        self.pred = self.pred_to_str()

    # get object's properties by its name
    # ['soft', 'green', 'full', 'empty', 'container', 'plastic', 'hard', 'blue', 'metal', 'toy']
    def get_pred(self):
        pred_string = None
        with open(self.obj_path) as csv_file:
            df = pd.read_csv(csv_file, names=['object', 'words'])
            data_objects = df.object.tolist()[1:]
            data_words = df.words.tolist()[1:]
            if len(data_objects) == len(data_words):
                for i in range(len(data_objects)):
                    if data_objects[i] == self.name:
                        pred_string = data_words[i]

            if pred_string is None:
                print('No object ' + self.name + '!')
                exit(1)
            else:
                # turn prop_string into list and return
                return pred_string.split(', ')

    def pred_to_str(self):  # turn property list into 000110
        res = ''
        flag = '1'
        pred_set = ['soft', 'green', 'full', 'empty', 'container', 'plastic', 'hard', 'blue', 'metal', 'toy']
        for pred in pred_set:
            for obj_pred in self.pred_list:
                if pred == obj_pred:
                    flag = '2'
                    break
            res += flag
            flag = '1'

        return res
