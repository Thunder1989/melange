from ..algorithm.transfer_learning import transfer_learning
from .Inferencer import Inferencer

if __name__ == "__main__":

    input1 = np.genfromtxt('../data/rice_hour_sdh', delimiter=',')
    #input1 = np.genfromtxt('sdh_hour_soda', delimiter=',')
    #input1 = np.genfromtxt('soda_hour_rice', delimiter=',')
    input2 = np.genfromtxt('../data/keti_hour_sum', delimiter=',')
    #input21 = np.genfromtxt('rice_hour_soda', delimiter=',')
    input3 = np.genfromtxt('../data/sdh_hour_rice', delimiter=',')
    input2 = np.vstack((input2,input3))
    fd1 = input1[:, 0:-1]
    fd2 = input2[:, 0:-1]

    #self.train_fd = np.hstack((fd1,fd2))
    train_fd = fd1
    test_fd = fd2
    train_label = input1[:, -1]
    #self.test_label = np.hstack((input2[:,-1],input3[:,-1]))
    test_label = input2[:,-1]

    ptn = [i.strip().split('\\')[-1][:-5] for i in open('../data/rice_pt_sdh').readlines()]
    #test_fn = get_name_features(ptn)

