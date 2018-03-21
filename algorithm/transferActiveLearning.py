import itertools

from sklearn.mixture import DPGMM
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from collections import defaultdict as DD
from collections import Counter as CT
from matplotlib import cm as Color


import numpy as np
import re
import math
import random
import itertools
import pylab as pl
import matplotlib.pyplot as plt


def get_name_features(names):

	name = []
	for i in names:

		s = re.findall('(?i)[a-z]{2,}',i)
		name.append(' '.join(s))

	cv = CV(analyzer='char_wb', ngram_range=(3,4))
	fn = cv.fit_transform(name).toarray()

	return fn


class transferActiveLearning:
	def __init__(self, source_fd, source_label, target_fd, target_label, target_fn, switch=False):

		self.m_source_fd = source_fd
		self.m_source_label = source_label

		self.m_target_fd = target_fd
		self.m_target_label = target_label

		self.m_target_fn = target_fn

		self.bl = []

		self.tao = 0
		self.alpha_ = 1

		self.rounds = 100
		self.acc_sum = [0 for i in xrange(self.rounds)] #acc per iter for each fold


		self.clf = SVC()
		self.ex_id = DD(list)

		if switch==True:
			fd_tmp = self.m_source_fd
			self.m_source_fd = self.m_target_fd
			self.m_target_fd = fd_tmp

			l_tmp = self.m_source_label
			self.m_source_label = self.m_target_label
			self.m_target_label = l_tmp
	
	def update_pseudo_set(self, new_ex_id, cluster_id, p_idx, p_label, p_dist):

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
				p_label.append(self.m_target_label[new_ex_id])
			else:
				tmp.append(ex)

		if not tmp:
			self.ex_id.pop(cluster_id)
		else:
			self.ex_id[cluster_id] = tmp

		return p_idx, p_label, p_dist


	def get_pred_acc(self, fn_test, label_test, al_labeled_idList, pseudo_set, pseudo_label):
			
		if not pseudo_set:
			fn_train = self.m_target_fn[al_labeled_idList]
			label_train = self.m_target_label[al_labeled_idList]
		else:
			fn_train = self.m_target_fn[np.hstack((al_labeled_idList, pseudo_set))]
			label_train = np.hstack((self.m_target_label[al_labeled_idList], pseudo_label))

		self.clf.fit(fn_train, label_train)
		fn_preds = self.clf.predict(fn_test)

		acc = ACC(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc

		# fn_train = []
		# label_train = []

		# if not pseudo_set:
		# 	labeledList = []
		# 	# labeledList.extend(tl_labeled_idList)
		# 	labeledList.extend(al_labeled_idList)

		# 	# print("label list\t", labeledList)

		# 	fn_train = self.m_target_fn[labeledList]
		# 	label_train = self.m_target_label[labeledList]
		# else:
		# 	labeledList = []
		# 	# labeledList.extend(tl_labeled_idList)
		# 	labeledList.extend(al_labeled_idList)

		# 	label_train = []
		# 	label_train.extend(self.m_target_label[labeledList])

		# 	# print("label list\t", labeledList)

		# 	labeledList.extend(pseudo_set)
		# 	fn_train = self.m_target_fn[labeledList]
			
		# 	label_train.extend(pseudo_label)

		# self.clf.fit(fn_train, label_train)
		# fn_preds = self.clf.predict(fn_test)

		# acc = ACC(label_test, fn_preds)

		# return acc

	def update_tao(self, labeled_set):
		dist_inter = []
		pair = list(itertools.combinations(labeled_set,2))

		for p in pair:

			d = np.linalg.norm(self.m_target_fn[p[0]]-self.m_target_fn[p[1]])
			if self.m_target_label[p[0]] != self.m_target_label[p[1]]:
				dist_inter.append(d)
		try:
			self.tao = self.alpha_*min(dist_inter)/2 #set tao be the min(inter-class pair dist)/2
		except Exception as e:
			self.tao = self.tao
	
	def select_example(self, labeled_set):
	
		sub_pred = DD(list) #Mn predicted labels for each cluster
		idx = 0

		for k,v in self.ex_id.items():
			sub_pred[k] = self.clf.predict(self.m_target_fn[v]) #predict labels for cluster learning set

		#entropy-based cluster selection
		rank = []
		for k,v in sub_pred.items():
			count = CT(v).values()
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

		ex_ = DD(list)
		for i,j,k,l in zip(c_.labels_, c_ex_id, dist, sub_label):
			ex_[i].append([j,l,k[0]])
		for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
			ex_[i] = sorted(j, key=lambda x: x[-1])
		for k,v in ex_.items():

			if v[0][0] not in labeled_set: #find the first unlabeled ex

				idx = v[0][0]
				break
		
		return idx, c_idx

	def get_base_learners(self):
		rf = RFC(n_estimators=100, criterion='entropy')

		svm = SVC(kernel='rbf', probability=True)
		lr = LR()

		self.bl = [rf, lr, svm]

		for b in self.bl:
			b.fit(self.m_source_fd, self.m_source_label)

	def run(self):
		rf = RFC(n_estimators=100, criterion='entropy')
		rf.fit(self.m_source_fd, self.m_source_label)

		pred = rf.predict(self.m_target_fd)
		print('data feature transfer learning acc:', ACC(pred, self.m_target_label))

		self.get_base_learners()

		label = self.m_target_label
		class_ = np.unique(self.m_source_label)

		for b in self.bl:
			print("base classifier rf, lr, svm")
			print(b.score(self.m_target_fd, label))

	##cluster based on name features
		n_class = 32/2
		c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
		c.fit(self.m_target_fn)
		dist = np.sort(c.transform(self.m_target_fn))
		ex_id = DD(list)
		for i,j,k in zip(c.labels_, xrange(len(self.m_target_fn)), dist):
			ex_id[i].append(int(j))

### get the neighbors for each example, nb_c from the clustering results
		nb_c = DD()
		for exx in ex_id.values():
			exx = np.asarray(exx)
			for e in exx:
				nb_c[e] = exx[exx!=e]

###get the neighbors from classification
		nb_f = [DD(), DD(), DD()]

		for b, n in zip(self.bl, nb_f):
			preds = b.predict(self.m_target_fd)
			ex_ = DD(list)
			for i,j in zip(preds, xrange(len(self.m_target_fd))):
				ex_[i].append(int(j))

			for exx in ex_.values():
				exx = np.asarray(exx)
				for e in exx:
					n[e] = exx[exx!=e]

		preds = np.array([999 for i in xrange(len(self.m_target_fd))])

		acc_ = []
		# for delta in np.linspace(0.1, 0.5, 5):
		threshold4TL = 0.5
		print("transfer learning agreement threshold\t", threshold4TL)

		pred = []
		l_idList = []
		correct = 0.0
		predNum = 0.0

		output = DD()
		print("len self.m_target_fn\t",len(self.m_target_fn))
		for i in xrange(len(self.m_target_fn)):
			w = []
			v_c = set(nb_c[i])
			for n in nb_f:
				v_f = set(n[i])
				cns = len(v_c&v_f)/float(len(v_c|v_f))

				inter = v_c & v_f
				union = v_c | v_f

				d_i = 0
				d_u = 0

				for it in inter:
					d_i += np.linalg.norm(self.m_target_fn[i]-self.m_target_fn[it])

				for u in union:
					# print(len(self.m_target_fn[i]))
					# print(len(self.m_target_fn[u]))
					d_u += np.linalg.norm(self.m_target_fn[i]-self.m_target_fn[u])

				if len(inter) != 0:
					sim = 1-(d_i/d_u)/cns

				w.append(sim)

			if np.mean(w) >= threshold4TL:
				w[:] = [float(j)/sum(w) for j in w]
				pred_pr = np.zeros(len(class_))
				for wi, b in zip(w, self.bl):
					pr = b.predict_proba(self.m_target_fd[i].reshape(1, -1))

					pred_pr = pred_pr + wi*pr

				preds[i] = class_[np.argmax(pred_pr)]
				pred.append(preds[i])
				l_idList.append(i)

				predNum += 1.0

				if preds[i] == label[i]:
					correct += 1.0

		print("accuracy\t", correct/predNum)

		print("accurate prediction num\t", len(l_idList))		

		al_Target_fn = []
		al_Target_label = []

		l_idList = []

		totalInstanceNum = len(self.m_target_fn)
		print("totalInstanceNum\t", totalInstanceNum)

		al_Target_index = []
		al_Target_index = [i for i in range(totalInstanceNum)]

		# for i in range(len(self.m_target_fn)):
		# 	if i not in l_idList:
				# al_Target_fn.append(self.m_target_fn[i])
				# al_Target_label.append(self.m_target_label[i])
				# al_Target_index.append(i)

		np.random.shuffle(al_Target_index)
		al_Target_num = len(al_Target_index)
		print("active learning example num\t", al_Target_num)
		trainNum = int(al_Target_num*0.9)
		print("training num\t", trainNum)

		al_Target_trainIndex = al_Target_index[:trainNum]
		al_Target_testIndex = al_Target_index[trainNum:]

		al_Target_train_fn = self.m_target_fn[al_Target_trainIndex]
		al_Target_train_label = self.m_target_label[al_Target_trainIndex]

		al_Target_test_fn = self.m_target_fn[al_Target_testIndex]
		al_Target_test_label = self.m_target_label[al_Target_testIndex]
		
		print("train label set\t", set(al_Target_train_label))
		print("test label set\t", set(al_Target_test_label))

		c = KMeans(init='k-means++', n_clusters=28, n_init=10)
		c.fit(al_Target_train_fn)
		dist = np.sort(c.transform(al_Target_train_fn))

## i is the cluster id, j is the id of the instance, k is the distance
#### ex: {clusterID:[instanceID, distance]}
####ex_id: {clusterID:instanceID}
###ex_Ns: {clusterID:number of instances}

		ex = DD(list)
		self.ex_id = DD(list)
		ex_N = [] # num of examples in each C
		for i,j,k in zip(c.labels_, al_Target_trainIndex, dist):
			ex[i].append([j,k[0]])
			self.ex_id[i].append(int(j))
		for i,j in ex.items():
			ex[i] = sorted(j, key=lambda x: x[-1])
			ex_N.append([i,len(ex[i])])
		ex_N = sorted(ex_N, key=lambda x: x[-1],reverse=True)

		km_idx = []
		p_idx = []
		p_label = []
		p_dist = DD()

		ctr = 0
		for ee in ex_N:
			c_idx = ee[0]
			idx = ex[c_idx][0][0]
			km_idx.append(idx)
			ctr += 1

			if ctr < 3:
				continue

			self.update_tao(km_idx)
			p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist)

			# acc = self.get_pred_acc(al_Target_test_fn, al_Target_test_label, l_idList, km_idx, p_idx, p_label)
			acc = self.get_pred_acc(al_Target_test_fn, al_Target_test_label, km_idx, p_idx, p_label)
			print acc
			# print("acc\t", acc)
			self.acc_sum[ctr-1] = acc

		cl_id = []
		ex_al = []

		rounds = self.rounds

		# tl_fn = self.m_target_fn[l_idList]
		# tl_label = self.m_target_label[l_idList]

		for iterIndex in range(ctr, rounds):
			# if not p_idx:
			# 	fn_train = self.fn[km_idx]
			if not p_idx:
				tmpList = []
				tmpList.extend(km_idx)
				# tmpList.extend(l_idList)
				fn_train = self.m_target_fn[tmpList]
				label_train = self.m_target_label[tmpList]
			else:
				tmpList = []
				tmpList.extend(km_idx)
				# tmpList.extend(l_idList)

				label_train = np.hstack((self.m_target_label[tmpList], p_label))
				tmpList.extend(p_idx)
				fn_train = self.m_target_fn[tmpList]
			# fn_train = np.concatenate(fn_train, tl_fn)
			# label_train = np.concatenate(label_train, tl_label)
			# fn_train.append(tl_fn, axis=0)
			# label_train.append(tl_label, axis=0)

			self.clf.fit(fn_train, label_train)                        

			idx, c_idx, = self.select_example(km_idx)
			km_idx.append(idx)
			cl_id.append(c_idx)

			self.update_tao(km_idx)
			p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist)

			# acc = self.get_pred_acc(al_Target_test_fn, al_Target_test_label, l_idList, km_idx, p_idx, p_label)
			acc = self.get_pred_acc(al_Target_test_fn, al_Target_test_label, km_idx, p_idx, p_label)

			self.acc_sum[iterIndex] = acc
			print acc
			# print("acc\t", acc)
		# print("acc\t", self.acc_sum)

if __name__ == '__main__':
	input1 = np.genfromtxt("../data/rice_hour_sdh", delimiter=",")
	
	input2 = np.genfromtxt("../data/keti_hour_sum", delimiter=",")
	input3 = np.genfromtxt("../data/sdh_hour_rice", delimiter=",")

	input2 = np.vstack((input2, input3))

	fd1 = input1[:, 0:-1]
	fd2 = input2[:, 0:-1]

	source_fd = fd1
	target_fd = fd2

	source_label = input1[:,-1]
	target_label = input2[:,-1]

	ptn = [i.strip().split("\\")[-1][:-5] for i in open("../data/rice_pt_sdh").readlines()]

	target_fn = get_name_features(ptn)
	print("len(target_fn)\t", len(target_fn))

	tl = transferActiveLearning(source_fd, source_label, target_fd, target_label, target_fn, True)
	tl.run()
