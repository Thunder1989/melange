#various statistical metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def getAccuracy(pred, true):

    return accuracy_score(pred, true)


def getF1(pred, true, method):

    return f1_score(pred, true, average=method)

