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
				break

		return idx, c_idx

		
	def get_pred_acc(self, fn_test, label_test, labeled_set, pseudo_set, pseudo_label):

		if not pseudo_set:
			fn_train = self.m_target_fn[labeled_set]
			label_train = self.m_target_label[labeled_set]
		else:
			fn_train = self.m_target_fn[np.hstack((labeled_set, pseudo_set))]
			label_train = np.hstack((self.m_target_label[labeled_set], pseudo_label))

		self.clf.fit(fn_train, label_train)
		fn_preds = self.clf.predict(fn_test)

		acc = accuracy_score(label_test, fn_preds)
		# print("acc\t", acc)
		# print debug
		return acc


	def plot_confusion_matrix(self, label_test, fn_test):

		fn_preds = self.clf.predict(fn_test)
		acc = accuracy_score(label_test, fn_preds)

		cm_ = CM(label_test, fn_preds)
		cm = normalize(cm_.astype(np.float), axis=1, norm='l1')

		fig = pl.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		fig.colorbar(cax)
		for x in xrange(len(cm)):
			for y in xrange(len(cm)):
				ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
							horizontalalignment='center',
							verticalalignment='center',
							fontsize=10)
		cm_cls =np.unique(np.hstack((label_test,fn_preds)))

		cls = []
		for c in cm_cls:
			cls.append(mapping[c])
		pl.yticks(range(len(cls)), cls)
		pl.ylabel('True label')
		pl.xticks(range(len(cls)), cls)
		pl.xlabel('Predicted label')
		pl.title('Mn Confusion matrix (%.3f)'%acc)

		pl.show()
	
	def get_base_learners(self):
		rf = RFC(n_estimators=100, criterion='entropy')
		svm = SVC(kernel='rbf', probability=True)
		lr = LR()
		self.bl = [rf, lr, svm] #set of base learners
		for b in self.bl:
			b.fit(self.m_source_fd, self.m_source_label) #train each base classifier


	def run_CV(self):

		totalInstanceNum = len(self.m_target_label)
		print("totalInstanceNum\t", totalInstanceNum)
		indexList = [i for i in range(totalInstanceNum)]
		np.random.shuffle(indexList)

		trainNum = int(totalInstanceNum*0.9)
		train = indexList[:trainNum]
		test = indexList[trainNum:]

		# for train, test in kf:
		fn_train = self.m_target_fn[train]
		label_train = self.m_target_label[train]
		fd_train = self.m_target_fd[train]

		fn_test = self.m_target_fn[test]
		label_test = self.m_target_label[test]
		fd_test = self.m_target_fd[test]

		###transfer learning

		self.get_base_learners()

		n_class = 32/2
		c_transfer = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
		c_transfer.fit(fn_train)
		dist_transfer = np.sort(c_transfer.transform(fn_train))
		ex_id_transfer = dd(list) #example id for each C
		for i,j,k in zip(c_transfer.labels_, xrange(len(fn_train)), dist_transfer):
			ex_id_transfer[i].append(int(j))

		nb_c_transfer = dd() #nb from clustering results
		for exx in ex_id_transfer.values():
			exx = np.asarray(exx)
			for e in exx:
				nb_c_transfer[e] = exx[exx!=e]

		nb_f_transfer = [dd(), dd(), dd()] #nb from classification results
		for b,n in zip(self.bl, nb_f_transfer):
			preds = b.predict(fd_train)
			ex_transfer = dd(list)
			for i,j in zip(preds, xrange(len(fd_train))):
				ex_transfer[i].append(int(j))
			for exx in ex_transfer.values():
				exx = np.asarray(exx)
				for e in exx:
					n[e] = exx[exx!=e]

		preds = np.array([999 for i in xrange(len(fn_train))])
		acc_ = []
		cov_ = []

		pred_transfer = 0
		correct_pred_transfer = 0
		transfer_idList = []

		class_ = np.unique(self.m_source_label)

		delta = 0.5
		for i in xrange(len(fn_train)):
					#getting C v.s. F similiarity
			w = []
			v_c = set(nb_c_transfer[i])
			for n in nb_f_transfer:
				v_f = set(n[i])
				cns = len(v_c & v_f) / float(len(v_c | v_f)) #original count based weight
				inter = v_c & v_f
				union = v_c | v_f
				d_i = 0
				d_u = 0
				for it in inter:
					d_i += np.linalg.norm(fn_train[i]-fn_train[it])
				for u in union:
					d_u += np.linalg.norm(fn_train[i]-fn_train[u])
				if len(inter) != 0:
					sim = 1 - (d_i/d_u)/cns

				w.append(sim)

			if np.mean(w) >= delta:
				w[:] = [float(j)/sum(w) for j in w]
				pred_pr = np.zeros(len(class_))
				for wi, b in zip(w,self.bl):
					pr = b.predict_proba(fd_train[i].reshape(1,-1))
					pred_pr = pred_pr + wi*pr
				preds[i] = class_[np.argmax(pred_pr)]
			   
				pred_transfer+=1.0
				transfer_idList.append(i)
				if preds[i]==label_train[i]:
					correct_pred_transfer += 1.0

		print("acc transfer\t", correct_pred_transfer/pred_transfer)

		transfer_idList = []
		train_al = []
		for i in range(len(train)):
			# if train[i] not in transfer_idList:
			train_al.append(train[i])

		# train_al = train
		fn_train = self.m_target_fn[train_al]
		c = KMeans(init='k-means++', n_clusters=28, n_init=10)
		c.fit(fn_train)
		dist = np.sort(c.transform(fn_train))

		ex = dd(list) #example id, distance to centroid
		self.ex_id = dd(list) #example id for each C
		ex_N = [] # num of examples in each C
		for i,j,k in zip(c.labels_, train, dist):
			ex[i].append([j,k[0]])
			self.ex_id[i].append(int(j))
		for i,j in ex.items():
			ex[i] = sorted(j, key=lambda x: x[-1])
			ex_N.append([i,len(ex[i])])
		ex_N = sorted(ex_N, key=lambda x: x[-1],reverse=True)

		km_idx = []
		p_idx = []
		p_label = []
		p_dist = dd()
		#first batch of exs: pick centroid of each cluster, and cluster visited based on its size
		ctr = 0
		for ee in ex_N:

			c_idx = ee[0] #cluster id
			idx = ex[c_idx][0][0] #id of ex closest to centroid of cluster
			km_idx.append(idx)
			ctr+=1

			if ctr<3:
				continue

			self.update_tao(km_idx)

			p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist)

			acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label)
			self.acc_sum[ctr-1] = (acc)
			print acc
			# print("acc\t", acc)

		cl_id = [] #track cluster id on each iter
		ex_al = [] #track ex added on each iter
		fn_test = self.m_target_fn[test]
		label_test = self.m_target_label[test]
		for rr in range(ctr, rounds):

			if not p_idx:
				fn_train = self.m_target_fn[km_idx]
				label_train = self.m_target_label[km_idx]
			else:
				fn_train = self.m_target_fn[np.hstack((km_idx, p_idx))]
				label_train = np.hstack((self.m_target_label[km_idx], p_label))

			self.clf.fit(fn_train, label_train)                        

			idx, c_idx, = self.select_example(km_idx)                
			km_idx.append(idx)
			cl_id.append(c_idx) #track picked cluster id on each iteration
			# ex_al.append([rr,key,v[0][-2],self.m_target_label[idx],raw_pt[idx]]) #for debugging

			self.update_tao(km_idx)
			p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist)
			
			acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label)
			self.acc_sum[rr] = (acc)

			print acc
		print debug
			# print '# of p label', len(p_label)
			# print cl_id
			# if not p_label:
			#     print 'p label acc', 0
			#     p_acc.append(0)
			# else:
			#     print 'p label acc', sum(self.m_target_label[p_idx]==p_label)/float(len(p_label))
			#     p_acc.append(sum(self.m_target_label[p_idx]==p_label)/float(len(p_label)))
			# print '----------------------------------------------------'
			# print '----------------------------------------------------'

		print 'class count of clf training ex:', ct(label_train)
		self.acc_sum = [i for i in self.acc_sum if i]
		print 'average acc:', [np.mean(i) for i in self.acc_sum]
		print 'average p label acc:', np.mean(p_acc)

		self.plot_confusion_matrix(label_test, fn_test)


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
	source_fd = fd1
	source_label = input1[:,-1]

	input2 = np.genfromtxt("../data/keti_hour_sum", delimiter=",")
	input3 = np.genfromtxt("../data/sdh_hour_rice", delimiter=",")
	input2 = np.vstack((input2, input3))
	fd2 = input2[:, 0:-1]
	target_fd = fd2


	al = transferActiveLearning(fold, rounds, source_fd, source_label, target_fd, target_label, target_fn)

	al.run_CV()

