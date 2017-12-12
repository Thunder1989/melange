import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl

from scikits.statsmodels.tools.tools import ECDF
from scipy import stats
from scipy.optimize import curve_fit
from time import time
from collections import defaultdict as dd
from collections import Counter as ct

from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as FS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

input3 = [i.strip().split('\\')[-1][:-5] for i in open('rice_pt_soda').readlines()]
input4 = np.genfromtxt('rice_hour_soda', delimiter=',')
label = input4[:,-1]
name = []
for i in input3:
    s = re.findall('(?i)[a-z]{2,}',i)
    name.append(' '.join(s))

cv = CV(analyzer='char_wb', ngram_range=(3,4))
fn = cv.fit_transform(name).toarray()
fd = input4[:,[0,1,2,3,5,6,7]]
print 'class count of true labels of all ex:\n', ct(label)

fold = 10
rounds = 100
clf = LinearSVC()
#clf = RFC(n_estimators=100, criterion='entropy')

kf = KFold(len(label), n_folds=fold, shuffle=True)
mapping = {1:'co2',2:'humidity',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu'}
p_acc = [] #pseudo label acc
acc_sum = [[] for i in xrange(rounds)]
acc_ave = dd(list)
tao = 0
alpha_ = 1

run = 0
for train, test in kf:
    run += 1
    #print 'class count of true labels on cluster training ex:\n', ct(label[train])
    train_fd = fn[train]
    #n_class = len(np.unique(label[train]))
    c = KMeans(init='k-means++', n_clusters=28, n_init=10)
    c.fit(train_fd)
    '''
    c = DPGMM(n_components=50, covariance_type='diag', alpha=1)
    c.fit(train_fd)
    c_labels = c.predict(train_fd)
    print '# of GMM', len(np.unique(c_labels))
    mu = c.means_
    cov = c._get_covars()
    c_inv = []
    for co in cov:
        c_inv.append(np.linalg.inv(co))
    e_pr = np.sort(c.predict_proba(train_fd))
    '''
    dist = np.sort(c.transform(train_fd))
    ex = dd(list) #example id, distance to centroid
    ex_id = dd(list) #example id for each C
    ex_N = [] #example id for each C
    #for i,j,k in zip(c_labels, train, e_pr):
    for i,j,k in zip(c.labels_, train, dist):
        ex[i].append([j,k[0]])
        ex_id[i].append(int(j))
    for i,j in ex.items():
        ex[i] = sorted(j, key=lambda x: x[-1])
        ex_N.append([i,len(ex[i])])
    ex_N = sorted(ex_N, key=lambda x: x[-1],reverse=True)
    km_idx = []
    p_idx = []
    p_label = []
    p_dist = dd()
    #print 'initial exs from k clusters centroid=============================='

    #'''
    #ordered by density on the first batch of exs
    ctr = 0
    for ee in ex_N:
        key = ee[0]
        idx = ex[key][0][0]
        km_idx.append(idx)
        ctr+=1
        if ctr<3:
            continue
        #'''
        fit_dist = []
        fit_same = []
        fit_diff = []
        pair = list(itertools.combinations(km_idx,2))
        for p in pair:
            d = np.linalg.norm(fn[p[0]]-fn[p[1]])
            fit_dist.append(d)
            if label[p[0]] == label[p[1]]:
                fit_same.append(d)
            else:
                fit_diff.append(d)
        src = fit_diff
        tao = alpha_*min(src)/2

        #exclude exs
        tmp = []
        #re-visit exs removed on previous itr with the new tao
        idx_tmp=[]
        label_tmp=[]
        for i,j in zip(p_idx,p_label):
            if p_dist[i]<tao:
                idx_tmp.append(i)
                label_tmp.append(j)
            else:
                p_dist.pop(i)
                tmp.append(i)
        p_idx = idx_tmp
        p_label = label_tmp

        if ctr==3:
            #make up for p_labels for the first 2 itrs
            #TBD
            pass

        for e in ex_id[key]:
            if e == idx:
                continue
            d = np.linalg.norm(fn[e]-fn[idx])
            if d<tao:
                p_dist[e] = d
                p_idx.append(e)
                p_label.append(label[idx])
            else:
                tmp.append(e)
        if not tmp:
            ex_id.pop(key)
        else:
            ex_id[key] = tmp
        #'''
        test_fn = fn[test]
        test_label = label[test]
        if not p_idx:
            train_fn = fn[km_idx]
            train_label = label[km_idx]
        else:
            train_fn = fn[np.hstack((km_idx, p_idx))]
            train_label = np.hstack((label[km_idx], p_label))
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        acc_sum[ctr-3].append(acc)

    #'''

    cl_id = []
    ex_al = [] #the ex added in each itr
    test_fn = fn[test]
    test_label = label[test]
    for rr in range(ctr, rounds):
    #for rr in range(rounds):
        if not p_idx:
            train_fn = fn[km_idx]
            train_label = label[km_idx]
        else:
            train_fn = fn[np.hstack((km_idx, p_idx))]
            train_label = np.hstack((label[km_idx], p_label))
        #print 'ct on traing label', ct(train_label)
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        sub_pred = dd(list) #Mn predicted labels for each cluster
        uc = dd()
        for k,v in ex_id.items():
            sub_pred[k] = clf.predict(fn[v]) #predict labels for cluster learning set
            df = np.sort(clf.decision_function(fn[v]))
            for vv,pp in zip(v,df):
                uc[vv] = pp[-1]

        '''
        for k in ex.keys():
            prev = ex_cur[k]
            nb = neighbor[prev]
            for n in nb:
                if n[0] not in km_idx:
                    km_idx.append(n[0])
                    ex_cur[k] = n[0]
                    ex_al.append([rr,k,label[n[0]],input3[n[0]]])
                    break
        '''

        #the original H based cluster selection
        rank = []
        for k,v in sub_pred.items():
            count = ct(v).values()
            count[:] = [i/float(max(count)) for i in count]
            H = np.sum(-p*math.log(p,2) for p in count if p!=0)
            #H /= len(v)/float(len(train))
            rank.append([k,len(v),H])
            #if rr+1 == 3*n_class:
            #print k,'---',len(v), H
        rank = sorted(rank, key=lambda x: x[-1], reverse=True)
        if not rank:
            break
        idx = rank[0][0] #pick the id of the 1st cluster on the rank
        cl_id.append(idx) #track cluster id on each iteration

        #for cc,ll in sub_pred.items(): #if commented out, the following out also
            #print 'cluster',cc,'# of ex.', len(ll),'# predicted L', len(np.unique(ll))
        cc = idx #id of the cluster picked by H
        c_id = ex_id[cc] #example id of the cluster picked
        sub_label = sub_pred[idx]#used when choosing cluster by H
        #sub_label = ll
        sub_fn = fn[c_id]

        #sub-clustering the cluster
        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)
        dist = np.sort(c_.transform(sub_fn))
        ex_ = dd(list)
        for i,j,k,l in zip(c_.labels_, c_id, dist, sub_label):
            ex_[i].append([j,l,k[0]])
            #ex_[i].append([j,l,k[0]*uc[j]]) #combine p(x)*U(x)
        for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
            ex_[i] = sorted(j, key=lambda x: x[-1])
        for k,v in ex_.items():
            if v[0][0] not in km_idx:
                idx = v[0][0]
                km_idx.append(idx)
                #update tao then remove ex<tao
                fit_dist = []
                fit_same = []
                fit_diff = []
                pair = list(itertools.combinations(km_idx,2))
                for p in pair:
                    d = np.linalg.norm(fn[p[0]]-fn[p[1]])
                    fit_dist.append(d)
                    if label[p[0]] == label[p[1]]:
                        fit_same.append(d)
                    else:
                        fit_diff.append(d)
                src = fit_dist
                src = fit_diff #set tao be the min(inter-class pair dist)/2
                tao = alpha_*min(src)/2

                tmp = []
                #re-visit exs removed on previous itr with the new tao
                idx_tmp=[]
                label_tmp=[]
                for i,j in zip(p_idx,p_label):
                    if p_dist[i]<tao:
                        idx_tmp.append(i)
                        label_tmp.append(j)
                    else:
                        p_dist.pop(i)
                        tmp.append(i)
                p_idx = idx_tmp
                p_label = label_tmp

                #tmp = []
                for e in ex_id[cc]:
                    if e == idx:
                        continue
                    d = np.linalg.norm(fn[e]-fn[idx])
                    #a = fn[e] - mu[k]
                    #b = c_inv[k]
                    #d = np.abs(np.dot(np.dot(a,b),a.T))
                    if d<tao:
                        #print 'added ex with d',d
                        p_dist[e] = d
                        p_idx.append(e)
                        p_label.append(label[idx])
                        #if label[idx]!=label[e]:
                        #    print input3[e],label[idx], label[e],d
                    else:
                        tmp.append(e)
                #print '# of p label before clean', len(p_label)
                #print '# of p label after clean', len(p_label)
                if not tmp:
                    ex_id.pop(cc)
                else:
                    ex_id[cc] = tmp
                #ex_cur[k] = idx
                ex_al.append([rr,cc,v[0][-2],label[idx],input3[idx]])
                #print cc,label[idx],input3[idx]
                break

        #print len(km_idx), 'training examples'
        if not p_idx:
            train_fn = fn[km_idx]
            train_label = label[km_idx]
        else:
            train_fn = fn[np.hstack((km_idx, p_idx))]
            train_label = np.hstack((label[km_idx], p_label))
        clf.fit(train_fn, train_label)
        preds_fn = clf.predict(test_fn)
        acc = accuracy_score(test_label, preds_fn)
        acc_sum[rr].append(acc)

    print '# of p label', len(p_label)
    print cl_id
    if not p_label:
        print 'psudo label acc', 0
        p_acc.append(0)
    else:
        print 'psudo label acc', sum(label[p_idx]==p_label)/float(len(p_label))
        p_acc.append(sum(label[p_idx]==p_label)/float(len(p_label)))
    print '----------------------------------------------------'
    print '----------------------------------------------------'

print 'class count of clf training ex:', ct(train_label)
print 'average acc:', [np.mean(i) for i in acc_sum]
print 'average p label acc:', np.mean(p_acc)

tmp = []
for i,j in acc_ave.items():
    tmp.append([i,np.mean(j)])
tmp = sorted(tmp, key=lambda x: x[0])
x = [i[0] for i in tmp]
y = [i[1] for i in tmp]

cm_ = CM(test_label, preds_fn)
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
cm_cls =np.unique(np.hstack((test_label,preds_fn)))
cls = []
for c in cm_cls:
    cls.append(mapping[c])
pl.yticks(range(len(cls)), cls)
pl.ylabel('True label')
pl.xticks(range(len(cls)), cls)
pl.xlabel('Predicted label')
pl.title('Mn Confusion matrix (%.3f)'%acc)
pl.show()

