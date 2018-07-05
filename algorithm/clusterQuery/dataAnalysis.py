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

	def update_tao(self, al_tl_fn_train, al_tl_label_train):

		dist_inter = []
		fn_train_tao = np.array(al_tl_fn_train)
		label_train_tao = np.array(al_tl_label_train)

		indexList_tao = [i for i in range(len(label_train_tao))]

		pair = list(itertools.combinations(indexList_tao,2))

		for p in pair:
			if label_train_tao[p[0]] != label_train_tao[p[1]]:
				d = np.linalg.norm(fn_train_tao[p[0]]-fn_train_tao[p[1]])
				dist_inter.append(d)

		try:
			self.tao = self.alpha_*min(dist_inter)/2 #set tao be the min(inter-class pair dist)/2
		except Exception as e:
			self.tao = self.tao

	def update_pseudo_set(self, new_ex_id, new_ex_label, cluster_id, p_idx, p_label, p_dist):

		tmp = []
		idx_tmp=[]
		label_tmp=[]

		#re-visit exs removed on previous itr with the new tao
		for i,j in zip(p_idx,p_label):

			if p_dist[i] < self.tao:
				idx_tmp.append(i)
				label_tmp.append(j)
			else:
				p_dist.pop(i)
				tmp.append(i)

		p_idx = idx_tmp
		p_label = label_tmp

		#added exs to pseudo set
		for ex in self.ex_id[cluster_id]:

			if ex == new_ex_id:
				continue
			d = np.linalg.norm(self.m_target_fn[ex]-self.m_target_fn[new_ex_id])

			if d < self.tao:
				p_dist[ex] = d
				p_idx.append(ex)
				# p_label.append(self.m_target_label[new_ex_id])
				p_label.append(new_ex_label)
			else:
				tmp.append(ex)

		if not tmp:
			self.ex_id.pop(cluster_id)
		else:
			self.ex_id[cluster_id] = tmp

		return p_idx, p_label, p_dist
	
	def select_example(self, labeled_set):

		sub_pred = dd(list) #Mn predicted labels for each cluster
		idx = 0

		for k,v in self.ex_id.items():
			sub_pred[k] = self.clf.predict(self.m_target_fn[v]) #predict labels for cluster learning set

		#entropy-based cluster selection
		rank = []
		for k,v in sub_pred.items():
			count = ct(v).values()
			count[:] = [i/float(max(count)) for i in count]
			H = np.sum(-p*math.log(p,2) for p in count if p!=0)
			rank.append([k,len(v),H])
		rank = sorted(rank, key=lambda x: x[-1], reverse=True)

		if not rank:
			raise ValueError('no clusters found in this iteration!')

		c_idx = rank[0][0] #pick the 1st cluster on the rank, ordered by label entropy
		c_ex_id = self.ex_id[c_idx] #examples in the cluster picked
		sub_label = sub_pred[c_idx] #used when choosing cluster by H
		sub_fn = self.m_target_fn[c_ex_id]

		#sub-cluster the cluster
		c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
		c_.fit(sub_fn)
		dist = np.sort(c_.transform(sub_fn))

		ex_ = dd(list)
		for i,j,k,l in zip(c_.labels_, c_ex_id, dist, sub_label):
			ex_[i].append([j,l,k[0]])
		for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
			ex_[i] = sorted(j, key=lambda x: x[-1])
		for k,v in ex_.items():

			if v[0][0] not in labeled_set: #find the first unlabeled ex

				idx = v[0][0]
				c_ex_id.remove(idx) #update the training set by removing selected ex id

				if len(c_ex_id) == 0:
					self.ex_id.pop(c_idx)
				else:
					self.ex_id[c_idx] = c_ex_id
				break

		return idx, c_idx

	def get_pred_acc(self, fn_test, label_test, al_tl_fn_train, al_tl_label_train, pseudo_set, pseudo_label):

		fn_train_pred = []
		label_train_pred = []

		if not pseudo_set:
			# print(active_fn_train.shape)
			# print(transfer_fn_train.shape)
			# print(active_label_train.shape)
			# print(transfer_label_train.shape)

			fn_train_pred = np.array(al_tl_fn_train)
			label_train_pred = np.array(al_tl_label_train)
		else:

			# print(fn_train_pred.shape)
			# print(transfer_fn_train.shape)
			# print(label_train_pred.shape)
			# print(transfer_label_train.shape)

			fn_train_pred = self.m_target_fn[pseudo_set]
			fn_train_pred = np.vstack((fn_train_pred, al_tl_fn_train))

			# print("pseudo_label\t", pseudo_label, al_tl_label_train)
			label_train_pred = np.hstack((pseudo_label, al_tl_label_train))

		self.clf.fit(fn_train_pred, label_train_pred)
		fn_preds = self.clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

	def get_base_learners(self):
		rf = RFC(n_estimators=100, criterion='entropy')
		svm = SVC(kernel='rbf', probability=True)
		lr = LR()
		self.bl = [rf, lr, svm] #set of base learners
		for b in self.bl:
			b.fit(self.m_source_fd, self.m_source_label) #train each base classifier
	
	def initConfidenceBound(self, _lambda, featureDim):
		self.m_A = _lambda*np.identity(featureDim)

		self.m_AInv = np.linalg.inv(self.m_A)

	def updateConfidenceBound(self, idx):
		self.m_A += np.outer(self.m_target_fn[idx], self.m_target_fn[idx])
		self.m_AInv = np.linalg.inv(self.m_A)

	def getConfidenceBound(self, idx):

		CB = np.sqrt(np.dot(np.dot(self.m_target_fn[idx], self.m_AInv), self.m_target_fn[idx]))

		return CB

	def transferOrNot(self, transferFeatureList, transferFlagList, idx):
		transferThreshold = 0.8

		predLabel = self.bl[0].predict(self.m_target_fd[idx].reshape(1, -1))[0]

		if len(np.unique(transferFlagList)) > 1:
			self.judgeClassifier.fit(np.array(transferFeatureList), np.array(transferFlagList))
		else:
			return False, predLabel
		###judge correct prob 
		transferProb = self.judgeClassifier.predict_proba(self.m_target_fn[idx].reshape(1, -1))[0][1]
		# print("transferProb\t", transferProb)
		transferFlag = self.judgeClassifier.predict(self.m_target_fn[idx].reshape(1, -1))

##upper bound
		UCB = self.getConfidenceBound(idx)
		UCB = transferProb + self.m_cbRate*UCB
##low bound
		LCB = self.getConfidenceBound(idx)
		LCB = transferProb - self.m_cbRate*LCB

		if UCB >= transferThreshold:
			return True, predLabel
		else:
			return False, predLabel

	def run_CV(self):

		totalInstanceNum = len(self.m_target_label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]

		# np.random.shuffle(indexList)
		kf = KFold(totalInstanceNum, n_folds=self.fold, shuffle=True)
		cvIter = 0
		totalAccList = [0.0 for i in range(10)]
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