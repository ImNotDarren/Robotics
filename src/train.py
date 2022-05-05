import csv


class Classifier:
    def __init__(self, data_path, table, objects, predicates):
        self._objects = objects
        self._path = data_path
        self._behaviors = table.behaviors
        self._modalities = table.modalities
        self._table = table
        self._predicates = predicates

        # predicates for which we have classifiers
        self._learned_predicates = []

        # some constants
        self._num_trials_per_object = 5
        # self._train_test_split_fraction = 2/3 # what percentage of data is used for training when doing internal cross validation on training data

        # compute lists of contexts
        self._contexts = table.contexts

        # dictionary that holds context specific weights for each predicate
        self._pred_context_weights_dict = dict()

        # dictionary holding all data for a given context (the data is a dictionary itself)
        self._context_db_dict = dict()
        self.get_context_db_dict()

        # initialize classifier dict for each predicate, each test object
        # at training stage, key will be behavior-modality pair
        self.classifier = dict()
        self.get_classifier()

        self._CM_p_dict = dict()
        self._CM_p_b_dict = dict()
        self._CM_p_c_dict = dict()

    def get_context_db_dict(self):
        for context in self._contexts:
            context_filename = self._path + "/" + context + ".txt"
            data_dict = dict()
            with open(context_filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    features = row[1:len(row)]
                    key = row[0]
                    # print(key)
                    data_dict[key] = features
            self._context_db_dict[context] = data_dict

    def get_classifier(self):
        for p in self._predicates:
            self.classifier[p] = dict()
            for test_object in self._objects:
                self.classifier[p][test_object] = dict()
