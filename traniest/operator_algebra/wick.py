import numpy as np
import itertools

def SignOfContraction(Contraction):
	Hits = 0
	for i in range(len(Contraction)):
		Min = min(Contraction[i])
		Max = max(Contraction[i])
		for j in range(i + 1, len(Contraction)):
			if Contraction[j][0] < Max and Contraction[j][0] > Min:
				Hits += 1
			if Contraction[j][1] < Max and Contraction[j][1] > Min:
				Hits += 1
	Sign = 1
	if Hits % 2 == 1:
		Sign = -1
	return Sign
			

def WickContraction(CrIdx, AnIdx, OrbList = None):
	#idx = range(len(OrbList))
	#AnIdx = [i for i in idx if i not in CrIdx]
	assert len(AnIdx) == len(CrIdx)
	Contractions = []
	PorQ = []
	Signs = []
	if len(CrIdx) == 0:
		return Contractions, PorQ, Signs
	
	CrPermutations = itertools.permutations(CrIdx, len(AnIdx))
	for each_permutation in CrPermutations:
		zipped = zip(each_permutation, AnIdx)
		Contractions.append(list(zipped))
	for Contraction in Contractions:
		if len(Contractions) == 1:
			break
		doRemove = True
		for Pair in Contraction:
			OtherPairs = Contraction.copy()
			OtherPairs.remove(Pair)
			for OtherPair in OtherPairs:
				if (OtherPair[0] > min(Pair[0], Pair[1]) and OtherPair[0] < max(Pair[0], Pair[0])) or (OtherPair[1] > min(Pair[0], Pair[1]) and OtherPair[1] < max(Pair[0], Pair[1])):
					doRemove = False
				if not doRemove:
					break
			if not doRemove:
				break
		if doRemove:
			Contractions.remove(Contraction)
			break

	for Contraction in Contractions:
		PorQTerm = []
		for Pair in Contraction:
			if Pair[0] < Pair[1]:
				PorQTerm.append('P')
			else:
				PorQTerm.append('Q')
		PorQ.append(PorQTerm)

		Sign = SignOfContraction(Contraction)
		Signs.append(Sign)

	if OrbList is not None:
		OrbContractions = []
		for Contraction in Contractions:
			OrbContraction = []
			for Pair in Contraction:
				OrbContraction.append((OrbList[Pair[0]], OrbList[Pair[1]]))
			OrbContractions.append(OrbContraction)
		Contractions = OrbContractions

	return Contractions, PorQ, Signs

def ContractionIndexToOrbitals(Contractions, OrbList):
	OrbContractions = []
	for Contraction in Contractions:
		OrbContraction = []
		for Pair in Contraction:
			OrbContraction.append((OrbList[Pair[0]], OrbList[Pair[1]]))
		OrbContractions.append(OrbContraction)
	return OrbContractions	

if __name__ == "__main__":
	OrbCons, PQ, Signs = WickContraction([0, 1], [2, 3])
	print(OrbCons)
	print(PQ)
	print(Signs)
