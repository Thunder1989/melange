#### transform multi class into binary class
import numpy as np

def transformMultiClass2BinaryClass(transferMultiClass, targetMultiClass, specificClass):

	targetBinaryClass = (targetMultiClass == specificClass)*1.0
	transferBinaryClass = (transferMultiClass == specificClass)*1.0

	return transferBinaryClass, targetBinaryClass

def readTransferLabel(transferLabelFile):
	f = open(transferLabelFile)

	transferLabelList = []
	targetLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		transferLabelList.append(int(line[1]))
		targetLabelList.append(int(line[2]))

	f.close()

	return transferLabelList, targetLabelList

sampleNum = 500
featureDim = 20
classifierNum = 5

transferLabelFile = "./simulatedTransferLabel_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+".txt"	

transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)

transferLabelArray = np.array(transferLabelList)
targetLabelArray = np.array(targetLabelList)

specificClass = 2
transferBinaryClass, targetBinaryClass = transformMultiClass2BinaryClass(transferLabelArray, targetLabelArray, specificClass)

transferLabelFile = "simulatedTransferLabel_"+str(sampleNum)+"_"+str(featureDim)+"_"+str(classifierNum)+"_"+str(specificClass)+".txt"

f = open(transferLabelFile, "w")
f.write("transferlabel\t truelabel\n")
for sampleIndex in range(sampleNum):
	if transferBinaryClass[sampleIndex] == targetBinaryClass[sampleIndex]:	
		if transferBinaryClass[sampleIndex] == 1.0:
			f.write(str(1.0)+"\t"+str(transferBinaryClass[sampleIndex])+"\t"+str(targetBinaryClass[sampleIndex])+"\n")
		else:
			f.write(str(0.0)+"\t"+str(transferBinaryClass[sampleIndex])+"\t"+str(targetBinaryClass[sampleIndex])+"\n")
	else:
		f.write(str(0.0)+"\t"+str(transferBinaryClass[sampleIndex])+"\t"+str(targetBinaryClass[sampleIndex])+"\n")


f.close()



