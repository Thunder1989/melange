import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl

from collections import defaultdict as dd
from collections import Counter as ct

from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

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

        self.tao = 0
        self.alpha_ = 1

        self.clf = LinearSVC()
        self.ex_id = dd(list)

        self.learner = active_learning(
            self.fold,
            self.rounds,
            self.fn,
            self.label
        )


    def update_model():

        pass


    def predict(self, target_srcids):

        return self.learner.clf.predict(target_srcids)


    def run_auto(self):

        self.learner.run_CV()


if __name__ == "__main__":

    mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}

    fold = 10
    rounds = 100
    al = active_learning_interface(
        target_building='rice',
        fold=fold,
        rounds=rounds
        )

    al.run_auto()

