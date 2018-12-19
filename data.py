import os
import random
from copy import deepcopy

from utils import Triple

# Change the head of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_head_raw(quadruple, entityTotal):
	newQuadruple = deepcopy(quadruple)
	oldHead = quadruple.s
	while True:
		newHead = random.randrange(entityTotal)
		if newHead != oldHead:
			break
	newQuadruple.s = newHead
	return newQuadruple

# Change the tail of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_tail_raw(quadruple, entityTotal):
	newQuadruple = deepcopy(quadruple)
	oldTail = newQuadruple.o
	while True:
		newTail = random.randrange(entityTotal)
		if newTail != oldTail:
			break
	newQuadruple.o = newTail
	return newQuadruple

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter(quadruple, entityTotal, quadrupleDict):
	newQuadruple = deepcopy(quadruple)
	while True:
		newHead = random.randrange(entityTotal)
		if (newHead, newQuadruple.o, newQuadruple.r, newQuadruple.t) not in quadrupleDict:
			break
	newQuadruple.s = newHead
	return newQuadruple

# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter(quadruple, entityTotal, quadrupleDict):
	newQuadruple = deepcopy(quadruple)
	while True:
		newTail = random.randrange(entityTotal)
		if (newQuadruple.s, newTail, newQuadruple.r, newQuadruple.t) not in quadrupleDict:
			break
	newQuadruple.o = newTail
	return newQuadruple

# Split the tripleList into #num_batches batches
def getBatchList(tripleList, num_batches):
	batchSize = len(tripleList) // num_batches
	batchList = [0] * num_batches
	for i in range(num_batches - 1):
		batchList[i] = tripleList[i * batchSize : (i + 1) * batchSize]
	batchList[num_batches - 1] = tripleList[(num_batches - 1) * batchSize : ]
	return batchList

def getFourElements(quadrupleList):
	headList = [quadruple.s for quadruple in quadrupleList]
	tailList = [quadruple.o for quadruple in quadrupleList]
	relList = [quadruple.r for quadruple in quadrupleList]
	timeList = [quadruple.t for quadruple in quadrupleList]
	return headList, tailList, relList, timeList

def getThreeElements(tripleList):
	headList = [triple.s for triple in tripleList]
	tailList = [triple.o for triple in tripleList]
	relList = [triple.r for triple in tripleList]
	return headList, tailList, relList

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_all(quadrupleList, entityTotal):
	newQuadrupleList = [corrupt_head_raw(quadruple, entityTotal) if random.random() < 0.5
		else corrupt_tail_raw(quadruple, entityTotal) for quadruple in quadrupleList]
	ps, po, pr, pt = getFourElements(quadrupleList)
	ns, no, nr, nt = getFourElements(newQuadrupleList)
	return ps, po, pr, pt, ns, no, nr, nt

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_all(quadrupleList, entityTotal, quadrupleDict):
	newQuadrupleList = [corrupt_head_filter(quadruple, entityTotal, quadrupleDict) if random.random() < 0.5
		else corrupt_tail_filter(quadruple, entityTotal, quadrupleDict) for quadruple in quadrupleList]
	ps, po, pr, pt = getFourElements(quadrupleList)
	ns, no, nr, nt = getFourElements(newQuadrupleList)
	return ps, po, pr, pt, ns, no, nr, nt
