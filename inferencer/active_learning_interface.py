import numpy as np
import re

from collections import defaultdict as dd

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.svm import LinearSVC

from Inferencer import Inferencer
from algorithm.active_learning import active_learning

def get_name_features(names):

        name = []
        for i in names:
            s = re.findall('(?i)[a-z]{2,}',i)
            name.append(' '.join(s))

        cv = CV(analyzer='char_wb', ngram_range=(3,4))
        fn = cv.fit_transform(name).toarray()

        return fn

class active_learning_interface(Inferencer):

    def __init__(self,
	target_building,
        fold,
        rounds
	):

	super(active_learning_interface, self).__init__(
            target_building='rice'
        )

        #Merged Initializations
        self.fold = fold
        self.rounds = rounds
        self.acc_sum = [[] for i in xrange(self.rounds)] #acc per iter for each fold

        #Added target building as a parameter to get raw_pt and tmp
        raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../data/' + target_building + '_pt').readlines()]
        tmp = np.genfromtxt('../data/' + target_building + '_hour', delimiter=',')
        self.fn = get_name_features(raw_pt)
        self.label = tmp[:,-1]

        self.learner = active_learning(
            self.fold,
            self.rounds,
            self.fn,
            self.label
        )


    def select_example():

        idx, c_idx = self.learner.select_example()

        return idx


    def update_model():

        pass


    def predict(self, target_srcids):

        return self.learner.clf.predict(target_srcids)


    def run_auto(self):

        self.learner.run_CV()

