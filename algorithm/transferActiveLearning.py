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
	def __init__(self, source_fd, source_label, target_fd, target_label, target_fn):

		self.m_source_fd = source_fd
		self.m_source_label = source_label

		self.m_target_fd = target_fd
		self.m_target_label = target_label

		self.m_target_fn = target_fn

		self.bl = []


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
		for delta in np.linspace(0.1, 0.5, 5):
			print("transfer learning agreement threshold\t", delta)

			ct = 0
			t = 0
			pred = []
			l_id = []

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
						print(len(self.m_target_fn[i]))
						print(len(self.m_target_fn[u]))
						d_u += np.linalg.norm(self.m_target_fn[i]-self.m_target_fn[u])

					if len(inter) != 0:
						sim = 1-(d_i/d_u)/cns

					w.append(sim)

				if np.mean(w) >= delta:
					w[:] = [float(j)/sum(w) for j in w]
					pred_pr = np.zeros(len(class_))
					for wi, b in zip(w, self.bl):
						pr = b.predict_proba(self.m_target_fd[i].reshape(1, -1))

						pred_pr = pred_pr + wi*pr

					preds[i] = class_[np.argmax(pred_pr)]
					pred.append(preds[i])
					l_id.append(i)

		print("accurate prediction\t", l_id)				

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

	tl = transferActiveLearning(source_fd, source_label, target_fd, target_label, target_fn)
	tl.run()
