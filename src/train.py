import csv
import numpy as np
import sys
from oracle import Table

class Classifier:
    def __init__(self, data_path):
        self._table = Table()
        self._objects = self._table.objects
        self._path = data_path
        self._behaviors = self._table.behaviors
        self._modalities = self._table.modalities
        self._predicates = self._table.predicates

        # predicates for which we have classifiers
        self._learned_predicates = []

        # some constants
        self._num_trials_per_object = 5
        # self._train_test_split_fraction = 2/3 # what percentage of data is used for training when doing internal cross validation on training data

        # compute lists of contexts
        self._contexts = self._table.contexts

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

    def get_features(self, context, object_name, trial_number):
        key = str(object_name) + "_t" + str(trial_number)
        return self._context_db_dict[context][key]

    # TODO: modify this
    def retrain_classifier(self, train_x, train_y):
        # split group cross validation
        # num_group = 50
        num_group = dict()
        for p in self._predicates:
            num_group[p] = dict()
            for c in self._contexts:
                num_group[p][c] = len(train_y[p][c])

        for p in self._predicates:
            for b in self._behaviors:
                for c in self._contexts:
                    if b in c:
                        num_group[p][b] = num_group[p][c]
                        break

        CM_p_c_dict = dict()
        prob_p_c_i = dict()
        gt_p_c_i = dict()

        for p in self._predicates:
            CM_p_c_dict[p] = dict()
            prob_p_c_i[p] = dict()
            gt_p_c_i[p] = dict()
            print("Retrain classifier for predicate: " + p)

            for c in self._contexts:

                # create and train classifier, save it in the dictionary
                # deal with dataset: split to negatives and positives (using index)
                CM_p_c_dict[p][c] = np.zeros((2, 2))
                prob_p_c_i[p][c] = dict()
                gt_p_c_i[p][c] = dict()

                # for every group, train classifier and do behaivor evaluation
                for i in range(0, num_group[p][c]):
                    test_idx = [i]  # index list
                    train_idx = []
                    for j in range(len(train_y[p][c])):
                        if j not in test_idx:
                            train_idx.append(j)

                    x_train = []
                    Y_train = []
                    x_test = []
                    Y_test = []
                    for t in train_idx:
                        x_train.append(train_x[p][c][t])
                        Y_train.append(train_y[p][c][t])
                    for t in test_idx:
                        x_test.append(train_x[p][c][t])
                        Y_test.append(train_y[p][c][t])

                    classifier_t = self.createScikitClassifier(True)
                    classifier_t.fit(x_train, Y_train)

                    # evaluating the classifier using test data
                    # set ground truth
                    gt_p_c_i[p][c][i] = Y_test

                    # get prediction labels
                    # prediction = classifier_t.predict(x_test)
                    prediction = []
                    # get prob distribution
                    prob_p_c_i[p][c][i] = classifier_t.predict_proba(x_test)
                    prob = prob_p_c_i[p][c][i]
                    for pb in prob:
                        if float(pb[0]) >= float(pb[1]):
                            prediction.append(0)
                        else:
                            prediction.append(1)

                    # if (c == 'look_color'):
                    # print (prob)
                    # print (prediction)
                    # print (Y_test)

                    for j in range(len(gt_p_c_i[p][c][i])):
                        CM_p_c_dict[p][c][prediction[j]][gt_p_c_i[p][c][i][j]] += 1

        # print (prob_p_c_i['soft']['look_color']

        self._CM_p_c_dict = CM_p_c_dict
        self.set_weights()

        CM_p_b_dict = dict()

        for p in self._predicates:
            CM_p_b_dict[p] = dict()

            for b in self._behaviors:

                CM_p_b_dict[p][b] = np.zeros((2, 2))

                # overfitting issue
                CM_p_b_dict[p][b][0][0] += 5
                CM_p_b_dict[p][b][0][1] += 5
                CM_p_b_dict[p][b][1][0] += 5
                CM_p_b_dict[p][b][1][1] += 5

                contexts_b = []
                for c in self._contexts:
                    if b in c:
                        contexts_b.append(c)

                for i in range(0, num_group[p][b]):

                    prob = np.zeros(2)

                    # give gt an inital value
                    gt = []
                    for c in contexts_b:
                        gt = gt_p_c_i[p][c][i]
                        break

                    for c in contexts_b:

                        gt_i = gt_p_c_i[p][c][i]  # set ground truth

                        if gt != gt_i:  # check if all gt are the same
                            print("something wrong with the ground truth label. ")
                            sys.exit()

                        prob_i = prob_p_c_i[p][c][i]
                        if (self._weights[p][c] >= 0):
                            prob[0] += self._weights[p][c] * prob_i[0][0]  # add weight
                            prob[1] += self._weights[p][c] * prob_i[0][1]  # add weight
                            # prob[0] += prob_i[0][0] #add weight
                            # prob[1] += prob_i[0][1] #add weight

                    # normalize

                    # prob[0] = prob[0]/(prob[0] + prob[1])
                    # prob[1] = 1 - prob[0]
                    if (prob[0] >= prob[1]):
                        prediction = 0
                    else:
                        prediction = 1
                    CM_p_b_dict[p][b][prediction][gt[0]] += 1

        self._CM_p_b_dict = CM_p_b_dict
        # self.print_results()

