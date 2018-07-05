import numpy as np

from sklearn.feature_selection import SelectKBest, chi2

class _Vocab:
	def __init__(self):
		self.m_vocSize = 0
		###wordStr: wordID
		self.m_word2IDMap = {}
		###wordStr: wordDF
		self.m_wordDFMap = {}

		self.m_ID2wordMap = {}

	def addWordDF(self, wordStr):
		if wordStr not in self.m_wordDFMap:
			self.m_wordDFMap.setdefault(wordStr, 0.0)

		self.m_wordDFMap[wordStr] += 1.0

class _Doc:
	def __init__(self):
		###True:pos; False:neg
		self.m_posNeg = True
		self.m_label = ""
		##wordStr: wordTF
		self.m_wordMap = {}
		self.m_wordList = []

	def addWordTF(self, wordStr, wordTF):
		if wordStr not in self.m_wordMap.keys():
			self.m_wordMap.setdefault(wordStr, wordTF)
		else:
			print("error existing word")

def readReviewFile(fileName, vocabObj, docList, label, sentiment):
	f = open(fileName)

	for rawLine in f:
		splittedLine = rawLine.strip().split(" ")
		lineLen = len(splittedLine)
		# print(splittedLine, lineLen)
		docObj = _Doc()
		docList.append(docObj)

		docObj.m_label = label
		docObj.m_posNeg = sentiment

		for unitIndex in range(lineLen-1):
			wordUnit = splittedLine[unitIndex]
			wordUnitSplitted = wordUnit.split(":")

			wordStr = wordUnitSplitted[0]
			wordTF = float(wordUnitSplitted[1])

			vocabObj.addWordDF(wordStr)

			docObj.addWordTF(wordStr, wordTF)


def filterByChiSquare(vocabObj, docList):
	for wordStr in vocabObj.m_wordDFMap:
		if wordStr not in vocabObj.m_word2IDMap:
			wordIndex = len(vocabObj.m_word2IDMap)
			vocabObj.m_word2IDMap.setdefault(wordStr, wordIndex)
			vocabObj.m_ID2wordMap.setdefault(wordIndex, wordStr)

	vocabObj.m_vocSize = len(vocabObj.m_word2IDMap)

	featureMatrix = []
	labelMatrix = []

	docNum = len(docList)

	for docIndex in range(docNum):
		docObj = docList[docIndex]
		docObj.m_wordList = [0.0 for i in range(vocabObj.m_vocSize)]
		for wordStr in docObj.m_wordMap:
			wordTF = docObj.m_wordMap[wordStr]
			if wordStr in vocabObj.m_word2IDMap:
				wordIndex = vocabObj.m_word2IDMap[wordStr]
				docObj.m_wordList[wordIndex] = wordTF

		featureMatrix.append(docObj.m_wordList)
		labelMatrix.append(docObj.m_posNeg)

	featureMatrix = np.array(featureMatrix)
	labelMatrix = np.array(labelMatrix)

	ch2 = SelectKBest(chi2, k=50)
	selectedFeatureMatrix = ch2.fit_transform(featureMatrix, labelMatrix)

	selectedFeatureIndexList = ch2.get_support(indices=True)
	vocabObj.m_word2IDMap = {}
	# vocabObj.m_ID2wordMap = {}

	for featureIndex in selectedFeatureIndexList:
		wordStr = vocabObj.m_ID2wordMap[featureIndex]
		wordIndex = len(vocabObj.m_word2IDMap)
		vocabObj.m_word2IDMap.setdefault(wordStr, wordIndex)
		vocabObj.m_ID2wordMap.setdefault(wordIndex, wordStr)

	vocabObj.m_vocSize = len(vocabObj.m_word2IDMap)
	print("vocab size", vocabObj.m_vocSize)

def filterWordByDF(vocabObj, DFthreshold = 50):
	print("before filtering", len(vocabObj.m_wordDFMap))

	vocThreshold = 50
	vocIndex = 0

	for wordStr in vocabObj.m_wordDFMap:
		wordDF = vocabObj.m_wordDFMap[wordStr]

		# if vocIndex < vocThreshold:
		# print(wordStr, vocabObj.m_wordDFMap[wordStr])
		# print("wordDF", wordDF)
		if wordDF >= 3:
			if wordDF <= 52: 
				print(wordStr, " :wordDF ", wordDF)
				vocabObj.m_word2IDMap.setdefault(wordStr, len(vocabObj.m_word2IDMap))

	vocabObj.m_vocSize = len(vocabObj.m_word2IDMap)
	print("vocab size", vocabObj.m_vocSize)

def saveReview(fileName, vocabObj, docList, label):
	f = open(fileName, "w")

	print("docNum", len(docList))

	for docObj in docList:
		docObj.m_wordList = [0.0 for i in range(vocabObj.m_vocSize)]
		for wordStr in docObj.m_wordMap:
			wordTF = docObj.m_wordMap[wordStr]
			if wordStr in vocabObj.m_word2IDMap:
				wordIndex = vocabObj.m_word2IDMap[wordStr]
				docObj.m_wordList[wordIndex] = wordTF

	wordNum = vocabObj.m_vocSize
	for docObj in docList:
		if docObj.m_label != label:
			continue
		for wordIndex in range(wordNum):
			f.write(str(docObj.m_wordList[wordIndex])+"\t")

		if docObj.m_posNeg == True:
			f.write(str(1.0))
		else:
			f.write(str(0.0))
		f.write("\n")

	f.close()

	f = open("vocab", "w")
	for wordStr in vocabObj.m_word2IDMap:
		f.write(wordStr+":"+str(vocabObj.m_word2IDMap[wordStr]))
		f.write("\n")

	f.close()

docList = []
vocabObj = _Vocab()

inputFile_pos = "../processed_acl/kitchen/positive.review"
inputFile_neg = "../processed_acl/kitchen/negative.review"

inputFile_pos = "../processed_acl/electronics/positive.review"
inputFile_neg = "../processed_acl/electronics/negative.review"

inputFile_pos = "../processed_acl/dvd/positive.review"
inputFile_neg = "../processed_acl/dvd/negative.review"

inputFile_pos = "../processed_acl/books/positive.review"
inputFile_neg = "../processed_acl/books/negative.review"

outputFile = "./kitchenReview"

readReviewFile(inputFile_pos, vocabObj, docList, "kitchen", True)
readReviewFile(inputFile_neg, vocabObj, docList, "kitchen", False)

filterByChiSquare(vocabObj, docList)
# readReviewFile(inputFile_pos, vocabObj, docList, "electronics", True)
# readReviewFile(inputFile_neg, vocabObj, docList, "electronics", False)

# readReviewFile(inputFile_pos, vocabObj, docList, "dvd", True)
# readReviewFile(inputFile_neg, vocabObj, docList, "dvd", False)

# readReviewFile(inputFile_pos, vocabObj, docList, "books", True)
# readReviewFile(inputFile_neg, vocabObj, docList, "books", False)

# filterWordByDF(vocabObj)

saveReview(outputFile, vocabObj, docList, "kitchen")


