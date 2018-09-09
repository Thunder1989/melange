import numpy as np
import csv
import collections
import random
###generate features for n samples 

random.seed(100)

featureDim = 100 ##20

featureMeanVec = np.zeros(featureDim)
featureVariance = np.diag([1 for i in range(featureDim)])

sampleNum = 500

featureMatrix = np.random.multivariate_normal(featureMeanVec, featureVariance, sampleNum)

X = featureMatrix
###generate classifier

classifierNum = 2 ###5

classifierParam = np.random.random_sample((classifierNum, featureDim))

#### save classifier parameter
classifierParamFile = "simulatedClassifierParam_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"

f = open(classifierParamFile, "w")

for classifierIndex in range(classifierNum):
	for featureIndex in range(featureDim):
		f.write(str(classifierParam[classifierIndex][featureIndex])+"\t")
	f.write(str(classifierIndex)+"\n")

f.close()

# print(classifierParam)

#### generate label for each sample
uniqueLabelList = [i for i in range(classifierNum)]
y = [-1 for i in range(sampleNum)]
for sampleIndex in range(sampleNum):
	sampleFeature = featureMatrix[sampleIndex]

	labelRank = classifierParam.dot(sampleFeature)

	# print(labelRank)

	labelIndex = np.argmax(labelRank)

	y[sampleIndex] = labelIndex

counter = collections.Counter(y)
print("y distribution", counter)

###save generated features and labels into a txt
featureLabelFile = "simulatedFeatureLabel_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"

f = open(featureLabelFile, "w")

for sampleIndex in range(sampleNum):
	for featureIndex in range(featureDim):
		# print(X[sampleIndex][featureIndex])
		f.write(str(X[sampleIndex][featureIndex])+"\t")

	f.write(str(y[sampleIndex])+"\n")

f.close()

### generate transferred label
transferY = [-1 for i in range(sampleNum)]

## generate auditor
auditorParam = np.random.random_sample(featureDim)
for sampleIndex in range(sampleNum):
	sampleFeature = featureMatrix[sampleIndex]

	auditorLabel = auditorParam.dot(sampleFeature)

	if auditorLabel > 0:
		transferY[sampleIndex] = y[sampleIndex]
	else:

		leftLabelList = uniqueLabelList[:y[sampleIndex]]+uniqueLabelList[y[sampleIndex]+1:]
		# print(leftLabelList)
		transferY[sampleIndex] = random.sample(leftLabelList, 1)[0]

### save auditor parameter
auditorParamFile = "auditorParam_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"

f = open(auditorParamFile, "w")

for featureIndex in range(featureDim):
	f.write(str(auditorParam[featureIndex])+"\t")
	f.write("\n")
f.close()

### save transferred label
transferLabelFile = "simulatedTransferLabel_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"

f = open(transferLabelFile, "w")
f.write("transferlabel\t truelabel\n")
for sampleIndex in range(sampleNum):
	if transferY[sampleIndex] == y[sampleIndex]:	
		f.write(str(1.0)+"\t"+str(transferY[sampleIndex])+"\t"+str(y[sampleIndex])+"\n")
	else:
		f.write(str(0.0)+"\t"+str(transferY[sampleIndex])+"\t"+str(y[sampleIndex])+"\n")

f.close()

