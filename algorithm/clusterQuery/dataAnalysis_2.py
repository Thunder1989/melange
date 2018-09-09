"""
active transfer learning we use the lower bound to estimate whether we should trust the classifer or not.  
"""
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
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

def get_name_features(names):

	name = []
	for i in names:
		s = re.findall('(?i)[a-z]{2,}',i)
		name.append(' '.join(s))

	cv = CV(analyzer='char_wb', ngram_range=(3,4))
	fn = cv.fit_transform(name).toarray()

	return fn

class transferActiveLearning:

	def __init__(self, fold, rounds, source_fd, source_label, target_fd, target_label, target_fn):

		self.fold = fold
		self.rounds = rounds
		self.acc_sum = [[] for i in xrange(self.rounds)] #acc per iter for each fold

		self.m_source_fd = source_fd
		self.m_source_label = source_label

		self.m_target_fd = target_fd

		self.m_target_fn = target_fn
		self.m_target_label = target_label
		# self.fn = fn
		# self.m_target_label = label

		self.bl = []

		self.tao = 0
		self.alpha_ = 1

		self.clf = LinearSVC()
		self.ex_id = dd(list)

		self.judgeClassifier = LR()
		self.m_cbRate = 0.01


	def run_CV(self):

		totalInstanceNum = len(self.m_target_label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		# np.random.shuffle(indexList)
		kf = KFold(totalInstanceNum, n_folds=self.fold, shuffle=True)
		cvIter = 0
		totalAccList = []
		for train, test in kf:

			# np.random.shuffle(indexList)
			self.judgeClassifier = LinearSVC()
			print("cvIter...\t",cvIter)
			trainNum = int(totalInstanceNum*0.9)
			# train = indexList[:trainNum]
			# test = indexList[trainNum:]

			# train = indexList
			# test = indexList

			# for train, test in kf:
			fn_train = self.m_target_fn[train]
			label_train = self.m_target_label[train]
			fd_train = self.m_target_fd[train]

			fn_test = self.m_target_fn[test]
			label_test = self.m_target_label[test]
			fd_test = self.m_target_fd[test]

			self.judgeClassifier.fit(fn_train, label_train)
			fn_preds = self.judgeClassifier.predict(fn_test)
			print("+++++++")
			print(fn_preds)
			print(label_test)
			print("======")
			acc = accuracy_score(label_test, fn_preds)
			print("acc\t", acc)
			totalAccList.append(acc)
			###transfer learning

			# self.get_base_learners()

			# print(debug)
			cvIter += 1
		print("mean+/-variance\t",np.mean(totalAccList), np.sqrt(np.var(totalAccList)))
		# f = open("al_tl_judge_6.txt", "w")
		# for i in range(10):
		# 	totalAlNum = len(totalAccList[i])
		# 	for j in range(totalAlNum):
		# 		f.write(str(totalAccList[i][j])+"\t")
		# 	f.write("\n")
		# f.close()

if __name__ == "__main__":
	mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}


	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../data/rice_pt_sdh').readlines()]
	tmp = np.genfromtxt('../data/rice_hour_sdh', delimiter=',')
	target_label = tmp[:,-1]
	print 'class count of true labels of all ex:\n', ct(target_label)

	target_fn = get_name_features(raw_pt)
	fold = 10
	rounds = 100

	input1 = np.genfromtxt("../data/rice_hour_sdh", delimiter=",")
	fd1 = input1[:, 0:-1]
	target_fd = fd1
	target_label2 = input1[:,-1]


	input2 = np.genfromtxt("../data/keti_hour_sum", delimiter=",")
	input3 = np.genfromtxt("../data/sdh_hour_rice", delimiter=",")
	input2 = np.vstack((input2, input3))
	fd2 = input2[:, 0:-1]
	source_fd = fd2
	source_label = input2[:,-1]


	al = transferActiveLearning(fold, rounds, source_fd, source_label, target_fd, target_label, target_fn)

	al.run_CV()