import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as ACC


input1 = np.genfromtxt('./rice_hour_sdh', delimiter=',')
pn1 = [i.strip().split('\\')[-1][:-5] for i in open('./rice_pt_sdh').readlines()]

input2 = np.genfromtxt('./keti_hour_sum', delimiter=',')
input3 = np.genfromtxt('./sdh_hour_rice', delimiter=',')
input2 = np.vstack((input2,input3))
pn2 = [i.strip().split('+')[-1][:-5] for i in open('./sdh_pt_rice').readlines()]

input3 = np.genfromtxt('./soda_hour_sum', delimiter=',')
pn3 = [i.strip().split('+')[-1][:-5] for i in open('./soda_pt').readlines()]

X_fd = [input1, input2, input3]
X_fn = [pn1, pn2, pn3]

Y = [np.unique(X[:,-1]) for X in X_fd]
common = set(Y[0]) & set(Y[1])
common &= set(Y[2])
#common = np.array(common)

fd = []
fn = []
for d,n in zip(X_fd,X_fn):
    #fd_tmp.append( np.array(list(filter(lambda x: x[-1] in common, X_fd))) )
    fd_tmp = []
    fn_tmp = []
    for fd_,fn_ in zip(d,n):
        if fd_[-1] in common:
            fd_tmp.append(fd_)
            fn_tmp.append(fn_)

    fd.append(np.array(fd_tmp))
    fn.append(fn_tmp)

X_fd = fd
X_fn = fn

rf = RFC(n_estimators=100, criterion='entropy')
bldg = ['rice','sdh','soda']
for i in range(len(X_fd)):
    source = [X_fd[j] for j in range(len(X_fd)) if j!=i]
    train = np.vstack(source)
    train_fd = train[:,:-1]
    train_label = train[:, -1]
    test_fd, test_label = X_fd[i][:,:-1], X_fd[i][:,-1]
    #print (train_fd.shape, train_label.shape, test_fd.shape, test_label.shape)

    rf.fit(train_fd, train_label)
    preds = rf.predict(test_fd)

    print (ACC(preds, test_label))
    assert(len(test_label) == len(X_fn[i]))


    df = pd.DataFrame( np.vstack( (np.array(preds == test_label).astype(int), preds, test_label) ).T )
    df.to_csv('%s_labels.csv'%bldg[i])

    with open('%s_names'%bldg[i], 'w') as outfile:
            outfile.write('\n'.join(X_fn[i]) + '\n')

    #ptn = [i.strip().split('\\')[-1][:-5] for i in open('./soda_pt_sdh').readlines()]
    #test_fn = get_name_features(ptn)
