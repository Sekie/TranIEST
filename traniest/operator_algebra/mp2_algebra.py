from itertools import combinations
import numpy as np
from wick import WickContraction, ContractionIndexToOrbitals

'''
Takes a list of indices corresponding to position of operations within one 
subspace and calculates the sign of moving the operators together, assuming
there are only two subspaces
'''
def SignOfSeparation(Indices):
	IndicesSort = sorted(Indices)
	List = list(range(len(IndicesSort)))
	Moves = sum([ai - bi for ai, bi in zip(IndicesSort, List)])
	Sign = 1
	if Moves % 2 == 1:
		Sign = -1
	return Sign

'''
Takes a string of creation and annihilation indices acting on two subspaces
and divides them into a list of operator indices on each subspace.
We assume that each list of indices are ordered.
'''
def SeparateOperatorIndices(CrIndexA, AnIndexA, CrIndexB, AnIndexB):
	Max = max(max(CrIndexA), max(AnIndexA), max(CrIndexB), max(AnIndexB))
	NewCrIndexA = []
	NewAnIndexA = []
	NewCrIndexB = []
	NewAnIndexB = []
	
	AIndex = 0
	BIndex = 0
	for i in range(Max + 1):
		if i in CrIndexA:
			NewCrIndexA.append(AIndex)
			AIndex += 1
		if i in AnIndexA:
			NewAnIndexA.append(AIndex)
			AIndex += 1
		if i in CrIndexB:
			NewCrIndexB.append(BIndex)
			BIndex += 1
		if i in AnIndexB:
			NewAnIndexB.append(BIndex)
			BIndex += 1
	return NewCrIndexA, NewAnIndexA, NewCrIndexB, NewAnIndexB

'''
Takes a list of creation and annihilation operator indices and determines all
non-zero cases (which subspace) that will survive the expectation value. Returns
a list of cases. Each element is the indices of the creation on subspace A,
annihilation on A, creation on B, annihilation on B
'''
def GenerateSubspaceCases(CrIndex, AnIndex, FixedCrA = None, FixedAnA = None, FixedCrB = None, FixedAnB = None):
	IndicesA = []
	IndicesB = []
	for Num in range(len(CrIndex) + 1):
		CrA = list(combinations(CrIndex, Num))
		AnA = list(combinations(AnIndex, Num))
		for i in range(len(CrA)):
			CrA[i] = list(CrA[i])
			AnA[i] = list(AnA[i])

		CrB = []
		AnB = []
		for x in CrA:
			CrB.append([idx for idx in CrIndex if idx not in x])
		for x in AnA:
			AnB.append([idx for idx in AnIndex if idx not in x])

		Skip = False
		for i in range(len(CrA)):
			if FixedCrA is not None:
				for a in FixedCrA:
					if a not in CrA[i]:
						Skip = True
			if FixedCrB is not None and not Skip:
				for a in FixedCrB:
					if a not in CrB[i]:
						Skip = True
			if Skip:
				Skip = False
				continue
			for j in range(len(AnA)):
				if FixedAnA is not None and not Skip:
					for a in FixedAnA:
						if a not in AnA[j]:
							Skip = True
				if FixedAnB is not None and not Skip:
					for a in FixedAnB:
						if a not in AnB[j]:
							Skip = True
				if Skip:
					Skip = False
					continue

				IndicesA.append([CrA[i], AnA[j]])
				IndicesB.append([CrB[i], AnB[j]])
	return IndicesA, IndicesB

'''
For an MP2 bath, we need to sum over indices pqrstuvw. This takes a case generated
by GenerateSubspaceCases and returns the list of indices to be summed over.
'''
def tCaseIndices(AllS, AllE, FIndex, BIndex, EIndex):
	ps, qs, rs, ss, ts, us, vs, ws = []
	SIndex = FIndex + BIndex
	if 'p' in AllS[0] or 'p' in AllS[1]:
		ps = SIndex
	else:
		ps = EIndex
	if 'q' in AllS[0] or 'q' in AllS[1]:
		qs = SIndex
	else:
		qs = EIndex
	if 'r' in AllS[0] or 'r' in AllS[1]:
		rs = SIndex
	else:
		rs = EIndex
	if 's' in AllS[0] or 's' in AllS[1]:
		ss = SIndex
	else:
		ss = EIndex
	if 't' in AllS[0] or 't' in AllS[1]:
		ts = SIndex
	else:
		ts = EIndex
	if 'u' in AllS[0] or 'u' in AllS[1]:
		us = SIndex
	else:
		us = EIndex
	if 'v' in AllS[0] or 'v' in AllS[1]:
		vs = SIndex
	else:
		vs = EIndex
	if 'w' in AllS[0] or 'w' in AllS[1]:
		ws = SIndex
	else:
		ws = EIndex
	return ps, qs, rs, ss, ts, us, vs, ws

'''
Given a list of symbolic names for operators in S and operators in E,
and a normal ordering of the names, we covert the list of indices into
integers representing the position of the operators in the string
Example:
	For IndexS = [['p', 't'], ['q', 'w']]
	    IndexE = [['u'], ['v']]
	and NormalOrder = ['p', 'q', 'v', 'w', 't', 'u']
	we return
	    PositionS = [[0, 4], [1, 3]]
	    PositionE = [[5], [2]]

'''
def CaseToOperatorPositions(IndexS, IndexE, NormalOrder, NormalOrderOrbitals = None):
	PositionSCr = []
	PositionSAn = []
	PositionECr = []
	PositionEAn = []
	OrbSCr = []
	OrbSAn = []
	OrbECr = []
	OrbEAn = []
	for a in IndexS[0]:
		PositionSCr.append(NormalOrder.index(a))
		if NormalOrderOrbitals is not None:
			OrbSCr.append(NormalOrderOrbitals[NormalOrder.index(a)])
	for a in IndexS[1]:
		PositionSAn.append(NormalOrder.index(a))
		if NormalOrderOrbitals is not None:
			OrbSAn.append(NormalOrderOrbitals[NormalOrder.index(a)])
	for a in IndexE[0]:
		PositionECr.append(NormalOrder.index(a))
		if NormalOrderOrbitals is not None:
			OrbECr.append(NormalOrderOrbitals[NormalOrder.index(a)])
	for a in IndexE[1]:
		PositionEAn.append(NormalOrder.index(a))
		if NormalOrderOrbitals is not None:
			OrbEAn.append(NormalOrderOrbitals[NormalOrder.index(a)])
	PositionS = [PositionSCr, PositionSAn]
	PositionE = [PositionECr, PositionEAn]

	return PositionS, PositionE #, [OrbSCr, OrbSAn], [OrbECr, OrbEAn]

def CalcWickTerms(Contractions, PorQ, Signs, P, Q):
		Value = 0.0
		if len(Contractions) == 0:
			return 1.0
		for i in range(len(Signs)):
			iValue = float(Signs[i])
			for n in range(len(Contractions[i])):
				if PorQ[i][n] == 'P':
					iValue *= P[Contractions[i][n][0], Contractions[i][n][1]]
				else:
					iValue *= Q[Contractions[i][n][0], Contractions[i][n][1]]
			Value += iValue
		return Value

def MakeProjectorCases(ijklBath, NormalOrder, OrbitalListNoT, Containskl):
	TranslateToSymbols = ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l'] # Takes index to 2i and 2i+1 as symbols to remove
	ExtraOrbitalLists1 = []
	ExtraNormalOrders1 = []
	ExtraOrbitalLists2 = []
	ExtraNormalOrders2 = []
	ExtraOrbitalLists3 = []
	ExtraNormalOrders3 = []
	ExtraOrbitalLists4 = []
	ExtraNormalOrders4 = []
	RemovedSymbols1 = []; RemovedSymbols2 = []; RemovedSymbols3 = []; RemovedSymbols4 = []
	for n in range(len(ijklBath)):
		if ijklBath[n] == 1:
			ExtraOrbList1 = OrbitalListNoT.copy()
			del ExtraOrbList1[2*n:(2*n+2)]
			ExtraNormalOrder1 = NormalOrder.copy()
			ExtraNormalOrder1.remove(TranslateToSymbols[2 * n])
			ExtraNormalOrder1.remove(TranslateToSymbols[2 * n + 1])
			ExtraOrbitalLists1.append(ExtraOrbList1)
			ExtraNormalOrders1.append(ExtraNormalOrder1)
			RemovedSymbols1.append([TranslateToSymbols[2 * n], TranslateToSymbols[2 * n + 1]])	
			for m in range(len(ijklBath) - n - 1):
				if ijklBath[n + m + 1] == 1:
					ExtraOrbList2 = ExtraOrbList1.copy()
					del ExtraOrbList2[(2*m):(2*m+2)]
					ExtraNormalOrder2 = ExtraNormalOrder1.copy()
					ExtraNormalOrder2.remove(TranslateToSymbols[2 * (n + m + 1)])
					ExtraNormalOrder2.remove(TranslateToSymbols[2 * (n + m + 1) + 1])
					ExtraOrbitalLists2.append(ExtraOrbList2)
					ExtraNormalOrders2.append(ExtraNormalOrder2)
					RemovedSymbols2.append([TranslateToSymbols[2 * n], TranslateToSymbols[2 * n + 1], TranslateToSymbols[2 * (n + m + 1)], TranslateToSymbols[2 * (n + m + 1) + 1]])
					if Containskl:
						for o in range(len(ijklBath) - n - m - 1):
							if ijklBath[n + m + o + 1] == 1:
								ExtraOrbList3 = ExtraOrbList2.copy()
								del ExtraOrbList3[(2*o):(2*o+2)]
								ExtraNormalOrder3 = ExtraNormalOrder2.copy()
								ExtraNormalOrder3.remove(TranslateToSymbols[2 * (n + m + o + 1)])
								ExtraNormalOrder3.remove(TranslateToSymbols[2 * (n + m + o + 1) + 1])
								ExtraOrbitalLists3.append(ExtraOrbList3)
								ExtraNormalOrders3.append(ExtraNormalOrder3)
								RemovedSymbols3.append([TranslateToSymbols[2 * n], TranslateToSymbols[2 * n + 1], TranslateToSymbols[2 * (n + m + 1)], TranslateToSymbols[2 * (n + m + 1) + 1], TranslateToSymbols[2 * (n + m + o + 1)], TranslateToSymbols[2 * (n + m + o + 1) + 1]])
								for p in range(len(ijklBath) - n - m - o - 1):
									if ijklBath[n + m + o + p + 1] == 1:
										ExtraOrbList4 = ExtraOrbList3.copy()
										del ExtraOrbList4[(2*p):(2*p+2)]
										ExtraNormalOrder4 = ExtraNormalOrder3.copy()
										ExtraNormalOrder4.remove(TranslateToSymbols[2 * (n + m + o + p + 1)])
										ExtraNormalOrder4.remove(TranslateToSymbols[2 * (n + m + o + p + 1) + 1])
										ExtraOrbitalLists4.append(ExtraOrbList4)
										ExtraNormalOrders4.append(ExtraNormalOrder4)
										RemovedSymbols4 = [TranslateToSymbols]
	ijklBathNum = 0
	for x in ijklBath:
		ijklBathNum = ijklBathNum + x
	ijklBathNum = ijklBathNum % 2

	return ExtraNormalOrders1, ExtraNormalOrders2, ExtraNormalOrders3, ExtraNormalOrders4, ExtraOrbitalLists1, ExtraOrbitalLists2, ExtraOrbitalLists3, ExtraOrbitalLists4, RemovedSymbols1, RemovedSymbols2, RemovedSymbols3, RemovedSymbols4, ijklBathNum

def RemoveFromLists(Lists, Remove):
	for List in Lists:
		for x in Remove:
			if x in List:
				List.remove(x)

def MakeRemovedCrAnLists(CrSym, AnSym, Remove1, Remove2, Remove3, Remove4):
	Cr1 = []; Cr2 = []; Cr3 = []; Cr4 = []
	An1 = []; An2 = []; An3 = []; An4 = []
	for Rem in Remove1:
		C = CrSym.copy()
		A = AnSym.copy()
		for x in Rem:
			if x in C:
				C.remove(x)
			if x in A:
				A.remove(x)
		Cr1.append(C)
		An1.append(A)
	for Rem in Remove2:
		C = CrSym.copy()
		A = AnSym.copy()
		for x in Rem:
			if x in C:
				C.remove(x)
			if x in A:
				A.remove(x)
		Cr2.append(C)
		An2.append(A)
	for Rem in Remove3:
		C = CrSym.copy()
		A = AnSym.copy()
		for x in Rem:
			if x in C:
				C.remove(x)
			if x in A:
				A.remove(x)
		Cr3.append(C)
		An3.append(A)
	for Rem in Remove4:
		C = CrSym.copy()
		A = AnSym.copy()
		for x in Rem:
			if x in C:
				C.remove(x)
			if x in A:
				A.remove(x)
		Cr4.append(C)
		An4.append(A)
	return Cr1, Cr2, Cr3, Cr4, An1, An2, An3, An4

class MP2Bath:
	def __init__(self, t, FIndex, BIndex, EIndex, PS, PE, h, V):
		self.t = t
		self.FIndex = FIndex
		self.BIndex = BIndex
		self.SIndex = FIndex + BIndex
		self.EIndex = EIndex
		self.PS = PS
		self.PE = PE
		self.QS = np.eye(PS.shape[0]) - PS
		self.QE = np.eye(PE.shape[0]) - PE
		self.h = h
		self.V = V
	
	def GetAmplitudeIndices(self, SymbolS):
		tIndices = []
		uIndices = [] 
		vIndices = []
		wIndices = []
		SymS = SymbolS[0] + SymbolS[1]
		if 't' in SymS:
			tIndices = self.SIndex
		else:
			tIndices = self.EIndex
		if 'u' in SymS:
			uIndices = self.SIndex
		else:
			uIndices = self.EIndex
		if 'v' in SymS:
			vIndices = self.SIndex
		else:
			vIndices = self.EIndex
		if 'w' in SymS:
			wIndices = self.SIndex
		else:
			wIndices = self.EIndex
		return tIndices, uIndices, vIndices, wIndices

	def GetPQIndices(self, SymbolS):
		pIndices = []
		qIndices = []
		SymS = SymbolS[0] + SymbolS[1]
		if 'p' in SymS:
			pIndices = self.SIndex
		else:
			pIndices = self.EIndex
		if 'q' in SymS:
			qIndices = self.SIndex
		else:
			qIndices = self.EIndex
		return pIndices, qIndices

	def GetRSIndices(self, SymbolS):
		rIndices = []
		sIndices = []
		SymS = SymbolS[0] + SymbolS[1]
		if 'r' in SymS:
			rIndices = self.SIndex
		else:
			rIndices = self.EIndex
		if 's' in SymS:
			sIndices = self.SIndex
		else:
			sIndices = self.EIndex
		return rIndices, sIndices

	def CalcExpValue(self, SymbolS, SymbolE, NormalOrder, OrbitalList):
		PosS, PosE = CaseToOperatorPositions(SymbolS, SymbolE, NormalOrder)
		Sign = SignOfSeparation(PosE[0] + PosE[1])
		ConS, PorQS, SignsS = WickContraction(PosS[0], PosS[1], OrbList = OrbitalList)
		ConE, PorQE, SignsE = WickContraction(PosE[0], PosE[1], OrbList = OrbitalList)
		ExpS = CalcWickTerms(ConS, PorQS, SignsS, self.PS, self.QS)
		ExpE = CalcWickTerms(ConE, PorQE, SignsE, self.PE, self.QE)
		return float(Sign) * ExpS * ExpE

	def CombinedIndex(self, Indices):
		i = 0
		NS = len(self.SIndex)
		if len(Indices) == 2:
			i = Indices[0] + NS * Indices[1]
			i = i + 1
		if len(Indices) == 4:
			i = Indices[0] + NS * Indices[1] + NS * NS * Indices[2] + NS * NS * NS * Indices[3]
			i = i + 1 + NS * NS
		return i

	'''
	Calculates A for given indices ijkl, pqrs
	For the cases where ijkl are in B, we assume that OrbitalListNoT is in the order [i, i, j, j, k, k, l, l, ...]
	'''
	def CalcAElements(self, CrSymbols, AnSymbols, NormalOrder, OrbitalListNoT, FixedCrS = None, FixedAnS = None, Case = 'MF'):
		SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrA = FixedCrS, FixedAnA = FixedAnS)
		AElement = 0.0

		if len(OrbitalListNoT) == 0 and Case == 'MF':
			return 1.0

		Containsij = ('i' in NormalOrder)
		Containskl = ('k' in NormalOrder)
		ijklBath = []
		ijkl = []
		if Containsij:
			ijkl.append(OrbitalListNoT[0])
			ijkl.append(OrbitalListNoT[2])
			if Containskl:
				ijkl.append(OrbitalList[4])
				ijkl.append(OrbitalList[6])
				ijklBath = [0, 0, 0, 0]
			else:
				ijklBath = [0, 0]
		for n in range(len(ijklBath)):
			if ijkl[n] in self.BIndex:
				ijklBath[n] = 1
				OrbitalListNoT[2 * n] = ijkl[n] - len(self.SIndex)
				OrbitalListNoT[2 * n + 1] = ijkl[n] - len(self.SIndex)

		ExtraNormalOrders1, ExtraNormalOrders2, ExtraNormalOrders3, ExtraNormalOrders4, ExtraOrbitalLists1, ExtraOrbitalLists2, ExtraOrbitalLists3, ExtraOrbitalLists4, RemovedSymbols1, RemovedSymbols2, RemovedSymbols3, RemovedSymbols4, ijklBathNum = MakeProjectorCases(ijklBath, NormalOrder, OrbitalListNoT, Containskl)

		for n in range(len(SymbolsS)):
			if Case == 'MF':
				ExpSE = self.CalcExpValue(SymbolsS[n], SymbolsE[n], NormalOrder, OrbitalListNoT)
				ExpSE1 = []; ExpSE2 = []; ExpSE3 = []; ExpSE4 = [];
				for a in range(len(ExtraNormalOrders1)):
					SymS = SymbolsS[n].copy()
					for x in RemovedSymbols1[a]:
						if x in SymS[0]:
							SymS[0].remove(x)
						if x in SymS[1]:
							SymS[1].remove(x)
					aExpSE = self.CalcExpValue(SymS, SymbolsE[n], ExtraNormalOrders1[a], OrbitalListNoT)
					ExpSE1.append(aExpSE)
				for a in range(len(ExtraNormalOrders2)):
					SymS = SymbolsS[n].copy()
					for x in RemovedSymbols2[a]:
						if x in SymS[0]:
							SymS[0].remove(x)
						if x in SymS[1]:
							SymS[1].remove(x)
					aExpSE = self.CalcExpValue(SymS, SymbolsE[n], ExtraNormalOrders2[a], OrbitalListNoT)
					ExpSE2.append(aExpSE)
				for a in range(len(ExtraNormalOrders3)):
					SymS = SymbolsS[n].copy()
					for x in RemovedSymbols3[a]:
						if x in SymS[0]:
							SymS[0].remove(x)
						if x in SymS[1]:
							SymS[1].remove(x)
					aExpSE = self.CalcExpValue(SymS, SymbolsE[n], ExtraNormalOrders3[a], OrbitalListNoT)
					ExpSE3.append(aExpSE)
	
				for a in range(len(ExtraNormalOrders4)):
					SymS = SymbolsS[n].copy()
					for x in RemovedSymbols4[a]:
						if x in SymS[0]:
							SymS[0].remove(x)
						if x in SymS[1]:
							SymS[1].remove(x)
					aExpSE = self.CalcExpValue(SymS, SymbolsE[n], ExtraNormalOrders4[a], OrbitalListNoT)
					ExpSE4.append(aExpSE)
				if ijklBathNum == 1:
					ExpSE = -1.0 * ExpSE
				Parity1 = 1.0; Parity2 = 1.0; Parity3 = 1.0; Parity4 = 1.0
				if ijkBathNum == 1:
					Parity2 = -1.0; Parity4 = -1.0
				else:
					Parity1 = -1.0; Parity3 = -1.0
				for a in ExpSE1:
					ExpSE = ExpSE + Parity1 * a
				for a in ExpSE2:
					ExpSE = ExpSE + Parity2 * a
				for a in ExpSE3:
					ExpSE = ExpSE + Parity3 * a
				for a in ExpSE4:
					ExpSE = ExpSE + Parity4 * a
			
				AElement += ExpSE
				continue
			PosS, PosE = CaseToOperatorPositions(SymbolsS[n], SymbolsE[n], NormalOrder)
			Sign = SignOfSeparation(PosE[0] + PosE[1])
			ConS, PorQS, SignsS = WickContraction(PosS[0], PosS[1])
			ConE, PorQE, SignsE = WickContraction(PosE[0], PosE[1])
	
			ConS1 = []; ConS2 = []; ConS3 = []; ConS4 = []
			ConE1 = []; ConE2 = []; ConE3 = []; ConE4 = []
			PorQS1 = []; PorQS2 = []; PorQS3 = []; PorQS4 = []
			PorQE1 = []; PorQE2 = []; PorQE3 = []; PorQE4 = []
			SignsS1 = []; SignsS2 = []; SignsS3 = []; SignsS4 = []
			SignsE1 = []; SignsE2 = []; SignsE3 = []; SignsE4 = []
			Sign1 = []; Sign2 = []; Sign3 = []; Sign4 = []
			for a in range(len(ExtraNormalOrders1)):
				SymS = SymbolsS[n].copy()
				for x in RemovedSymbols1[a]:
					if x in SymS[0]:
						SymS[0].remove(x)
					if x in SymS[1]:
						SymS[1].remove(x)
				aPosS, aPosE = CaseToOperatorPositions(SymS, SymbolsE[n], ExtraNormalOrders1[a])
				aSign = SignOfSeparation(aPosE[0] + aPosE[1])
				aConS, aPorQS, aSignsS = WickContraction(aPosS[0], aPosS[1])
				aConE, aPorQE, aSignsE = WickContraction(aPosE[0], aPosE[1])
				ConS1.append(aConS)
				ConE1.append(aConE)
				PorQS1.append(aPorQS)
				PorQE1.append(aPorQE)
				SignsS1.append(aSignsS)
				SignsE1.append(aSignsE)
				Sign1.append(aSign)
			for a in range(len(ExtraNormalOrders2)):
				SymS = SymbolsS[n].copy()
				for x in RemovedSymbols2[a]:
					if x in SymS[0]:
						SymS[0].remove(x)
					if x in SymS[1]:
						SymS[1].remove(x)
				aPosS, aPosE = CaseToOperatorPositions(SymS, SymbolsE[n], ExtraNormalOrders2[a])
				aSign = SignOfSeparation(aPosE[0] + aPosE[1])
				aConS, aPorQS, aSignsS = WickContraction(aPosS[0], aPosS[1])
				aConE, aPorQE, aSignsE = WickContraction(aPosE[0], aPosE[1])
				ConS2.append(aConS)
				ConE2.append(aConE)
				PorQS2.append(aPorQS)
				PorQE2.append(aPorQE)
				SignsS2.append(aSignsS)
				SignsE2.append(aSignsE)
				Sign2.append(aSign)
			for a in range(len(ExtraNormalOrders3)):
				SymS = SymbolsS[n].copy()
				for x in RemovedSymbols3[a]:
					if x in SymS[0]:
						SymS[0].remove(x)
					if x in SymS[1]:
						SymS[1].remove(x)
				aPosS, aPosE = CaseToOperatorPositions(SymS, SymbolsE[n], ExtraNormalOrders3[a])
				aSign = SignOfSeparation(aPosE[0] + aPosE[1])
				aConS, aPorQS, aSignsS = WickContraction(aPosS[0], aPosS[1])
				aConE, aPorQE, aSignsE = WickContraction(aPosE[0], aPosE[1])
				ConS3.append(aConS)
				ConE3.append(aConE)
				PorQS3.append(aPorQS)
				PorQE3.append(aPorQE)
				SignsS3.append(aSignsS)
				SignsE3.append(aSignsE)
				Sign3.append(aSign)
			for a in range(len(ExtraNormalOrders4)):
				SymS = SymbolsS[n].copy()
				for x in RemovedSymbols4[a]:
					if x in SymS[0]:
						SymS[0].remove(x)
					if x in SymS[1]:
						SymS[1].remove(x)
				aPosS, aPosE = CaseToOperatorPositions(SymS, SymbolsE[n], ExtraNormalOrders4[a])
				aSign = SignOfSeparation(aPosE[0] + aPosE[1])
				aConS, aPorQS, aSignsS = WickContraction(aPosS[0], aPosS[1])
				aConE, aPorQE, aSignsE = WickContraction(aPosE[0], aPosE[1])
				ConS4.append(aConS)
				ConE4.append(aConE)
				PorQS4.append(aPorQS)
				PorQE4.append(aPorQE)
				SignsS4.append(aSignsS)
				SignsE4.append(aSignsE)
				Sign4.append(aSign)

			tIndices, uIndices, vIndices, wIndices = self.GetAmplitudeIndices(SymbolsS[n])
			for t in tIndices:
				for u in uIndices:
					for v in vIndices:
						for w in wIndices:
							OrbitalList = OrbitalListNoT.copy()
							if Case == 'Right':
								OrbitalList = OrbitalList + [v, w, u, t]
							if Case == 'Left':
								OrbitalList = [t, u, w, v] + OrbitalList
							assert len(OrbitalList) == len(NormalOrder)
							ConOrbsS = ContractionIndexToOrbitals(ConS, OrbitalList)
							ConOrbsE = ContractionIndexToOrbitals(ConE, OrbitalList)
							ExpS = CalcWickTerms(ConOrbsS, PorQS, SignsS, self.PS, self.QS)
							ExpE = CalcWickTerms(ConOrbsE, PorQE, SignsE, self.PE, self.QE)
							ExpSE = float(Sign) * ExpS * ExpE

							ConOrbsS1 = []; ConOrbsS2 = []; ConOrbsS3 = []; ConOrbsS4 = []
							ExpSE1 = []; ExpSE2 = []; ExpSE3 = []; ExpSE4 = []
							for a in range(len(ExtraOrbitalLists1)):
								if Case == 'Right':
									ExtraOrbitalLists1[a] = ExtraOrbitalLists1[a] + [v, w, u, t]
								if Case == 'Left':
									ExtraOrbitalLists1[a] = [t, u, w, v] + ExtraOrbitalLists1[a]
								aConOrbsS = ContractionIndexToOrbitals(ConS1[a], ExtraOrbitalList1[a])
								aConOrbsE = ContractionIndexToOrbitals(ConE1[a], ExtraOrbitalList1[a])
								aExpS = CalcWickTerms(aConOrbsS, PorQS1[a], SignsS1[a], self.PS, self.QS)
								aExpE = CalcWickTerms(aconOrbsE, PorQE1[a], SIgnsE1[a], self.PE, self.QE)
								aExpSE = float(Sign1[a]) * aExpS * aExpE
								ExpSE1.append(aExpSE)
							for a in range(len(ExtraOrbitalLists2)):
								if Case == 'Right':
									ExtraOrbitalLists2[a] = ExtraOrbitalLists2[a] + [v, w, u, t]
								if Case == 'Left':
									ExtraOrbitalLists2[a] = [t, u, w, v] + ExtraOrbitalLists2[a]
								aConOrbsS = ContractionIndexToOrbitals(ConS2[a], ExtraOrbitalList2[a])
								aConOrbsE = ContractionIndexToOrbitals(ConE2[a], ExtraOrbitalList2[a])
								aExpS = CalcWickTerms(aConOrbsS, PorQS2[a], SignsS2[a], self.PS, self.QS)
								aExpE = CalcWickTerms(aconOrbsE, PorQE2[a], SIgnsE2[a], self.PE, self.QE)
								aExpSE = float(Sign2[a]) * aExpS * aExpE
								ExpSE2.append(aExpSE)

							for a in range(len(ExtraOrbitalLists3)):
								if Case == 'Right':
									ExtraOrbitalLists3[a] = ExtraOrbitalLists3[a] + [v, w, u, t]
								if Case == 'Left':
									ExtraOrbitalLists3[a] = [t, u, w, v] + ExtraOrbitalLists3[a]
								aConOrbsS = ContractionIndexToOrbitals(ConS3[a], ExtraOrbitalList3[a])
								aConOrbsE = ContractionIndexToOrbitals(ConE3[a], ExtraOrbitalList3[a])
								aExpS = CalcWickTerms(aConOrbsS, PorQS3[a], SignsS3[a], self.PS, self.QS)
								aExpE = CalcWickTerms(aconOrbsE, PorQE3[a], SIgnsE3[a], self.PE, self.QE)
								aExpSE = float(Sign3[a]) * aExpS * aExpE
								ExpSE3.append(aExpSE)

							for a in range(len(ExtraOrbitalLists1)):
								if Case == 'Right':
									ExtraOrbitalLists4[a] = ExtraOrbitalLists4[a] + [v, w, u, t]
								if Case == 'Left':
									ExtraOrbitalLists4[a] = [t, u, w, v] + ExtraOrbitalLists4[a]
								aConOrbsS = ContractionIndexToOrbitals(ConS4[a], ExtraOrbitalList4[a])
								aConOrbsE = ContractionIndexToOrbitals(ConE4[a], ExtraOrbitalList4[a])
								aExpS = CalcWickTerms(aConOrbsS, PorQS4[a], SignsS4[a], self.PS, self.QS)
								aExpE = CalcWickTerms(aconOrbsE, PorQE4[a], SIgnsE4[a], self.PE, self.QE)
								aExpSE = float(Sign4[a]) * aExpS * aExpE
								ExpSE4.append(aExpSE)
							
							# Add all projection cases
							if ijklBathNum == 1:
								ExpSE = -1.0 * ExpSE
							Parity1 = 1.0; Parity2 = 1.0; Parity3 = 1.0; Parity4 = 1.0
							if ijklBathNum == 1:
								Parity2 = -1.0; Parity4 = -1.0
							else:
								Parity1 = -1.0; Parity3 = -1.0
							for a in ExpSE1:
								ExpSE = ExpSE + Parity1 * a
							for a in ExpSE2:
								ExpSE = ExpSE + Parity2 * a
							for a in ExpSE3:
								ExpSE = ExpSE + Parity3 * a
							for a in ExpSE4:
								ExpSE = ExpSE + Parity4 * a	

							AElement += self.t[t, u, v, w] * ExpSE
		return AElement

	def CalcA(self):
		NS = len(self.SIndex)
		DimA = 1 + NS * NS + NS * NS * NS * NS
		A = np.zeros((DimA, DimA))

		# ijkl - Case 0
		# pqrs - Case 0
		A[0, 0] = 1.0 # Might not be normalized. Need to check on this.
		# pqrs - Case 1
		for p in self.SIndex:
			for q in self.SIndex:
				pq = self.CombinedIndex([p, q])
				A[0, pq] = self.CalcAElements(['p'], ['q'], ['p', 'q'], [p, q], FixedCrS = ['p'], FixedAnS = ['q'], Case = 'MF')
				A[0, pq] += self.CalcAElements(['p', 'v', 'w'], ['q', 'u', 't'], ['p', 'q', 'v', 'w', 'u', 't'], [p, q], FixedCrS = ['p'], FixedAnS = ['q'], Case = 'Right')
				A[0, pq] += self.CalcAElements(['p', 't', 'u'], ['q', 'v', 'w'], ['t', 'u', 'w', 'v', 'p', 'q'], [p, q], FixedCrS = ['p'], FixedAnS = ['q'], Case = 'Left')
				# pqrs - Case 2
				for r in self.SIndex:
					for s in self.SIndex:
						pqrs = self.CombinedIndex([p, q, r, s])
						A[0, pqrs] = self.CalcAElements(['p', 'r'], ['s', 'q'], ['p', 'r', 's', 'q'], [p, r, s, q], FixedCrS = ['p', 'r'], FixedAnS = ['s', 'q'], Case = 'MF')
						A[0, pqrs] += self.CalcAElements(['p', 'r', 'v', 'w'], ['s', 'q', 'u', 't'], ['p', 'r', 's', 'q', 'v', 'w', 'u', 't'], [p, r, s, q], FixedCrS = ['p', 'r'], FixedAnS = ['s', 'q'], Case = 'Right')
						A[0, pqrs] += self.CalcAElements(['p', 'r', 't', 'u'], ['s', 'q', 'v', 'w'], ['t', 'u', 'w', 'v', 'p', 'r', 's', 'q'], [p, r, s, q], FixedCrS = ['p', 'r'], FixedAnS = ['s', 'q'], Case = 'Left')
		# ijkl - Case 1
		for i in self.SIndex:
			for j in self.SIndex:
				ij = self.CombinedIndex([i, j])
				# pqrs - Case 0
				A[ij, 0] = self.CalcAElements(['id', 'jd'], ['i', 'j'], ['id', 'i', 'jd', 'j'], [i, i, j, j], FixedCrS = ['id', 'jd'], FixedAnS = ['i', 'j'], Case = 'MF')
				A[ij, 0] += self.CalcAElements(['id', 'jd', 'v', 'w'], ['i', 'j', 'u', 't'], ['id', 'i', 'jd', 'j', 'v', 'w', 'u', 't'], [i, i, j, j], FixedCrS = ['id', 'jd'], FixedAnS = ['i', 'j'], Case = 'Right')
				A[ij, 0] += self.CalcAElements(['t', 'u', 'id', 'jd'], ['w', 'v', 'i', 'j'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j'], [i, i, j, j], FixedCrS = ['id', 'jd'], FixedAnS = ['i', 'j'], Case = 'Left')
				# pqrs - Case 1
				for p in self.SIndex:
					for q in self.SIndex:
						pq = self.CombinedIndex([p, q])
						A[ij, pq] = self.CalcAElements(['id', 'jd', 'p'], ['i', 'j', 'q'], ['id', 'i', 'jd', 'j', 'p', 'q'], [i, i, j, j, p, q], FixedCrS = ['id', 'jd', 'p'], FixedAnS = ['i', 'j', 'q'], Case = 'MF')
						A[ij, pq] += self.CalcAElements(['id', 'jd', 'p', 'v', 'w'], ['i', 'j', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'p', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, p, q], FixedCrS = ['id', 'jd', 'p'], FixedAnS = ['i', 'j', 'q'], Case = 'Right')
						A[ij, pq] += self.CalcAElements(['t', 'u', 'id', 'jd', 'p'], ['w', 'v', 'i', 'j', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'p', 'q'], [i, i, j, j, p, q], FixedCrS = ['id', 'jd', 'p'], FixedAnS = ['i', 'j', 'q'], Case = 'Left')
						# pqrs - Case 2
						for r in self.SIndex:
							for s in self.SIndex:
								pqrs = self.CombinedIndex([p, q, r, s])
								A[ij, pqrs] = self.CalcAElements(['id', 'jd', 'p', 'r'], ['i', 'j', 's', 'q'], ['id', 'i', 'jd', 'j', 'p', 'r', 's', 'q'], [i, i, j, j, p, r, s, q], FixedCrS = ['id', 'jd', 'p', 'r'], FixedAnS = ['i', 'j', 's', 'q'], Case = 'MF')
								A[ij, pqrs] += self.CalcAElements(['id', 'jd', 'p', 'r', 'v', 'w'], ['i', 'j', 's', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'p', 'r', 's', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, p, r, s, q], FixedCrS = ['id', 'jd', 'p', 'r'], FixedAnS = ['i', 'j', 's', 'q'], Case = 'Right')
								A[ij, pqrs] += self.CalcAElements(['t', 'u', 'id', 'jd', 'p', 'r'], ['w', 'v', 'i', 'j', 's', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'p', 'r', 's', 'q'], [i, i, j, j, p, r, s, q], FixedCrS = ['id', 'jd', 'p', 'r'], FixedAnS = ['i', 'j', 's', 'q'], Case = 'Left')
		#ijkl - Case 2
		for i in self.SIndex:
			for j in self.SIndex:
				for k in self.SIndex:
					for l in self.SIndex:
						ijkl = self.CombinedIndex([i, j, k, l])
						print("doing ijkl", i, j, k, l)
						# pqrs - Case 0
						A[ijkl, 0] = self.CalcAElements(['id', 'jd', 'kd', 'ld'], ['i', 'j', 'k', 'l'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l'], [i, i, j, j, k, k, l, l], FixedCrS = ['id', 'jd', 'kd', 'ld'], FixedAnS = ['i', 'j', 'k', 'l'], Case = 'MF')
						A[ijkl, 0] += self.CalcAElements(['id', 'jd', 'kd', 'ld', 'v', 'w'], ['i', 'j', 'k', 'l', 'u', 't'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'v', 'w', 'u', 't'], [i, i, j, j, k, k, l, l], FixedCrS = ['id', 'jd', 'kd', 'ld'], FixedAnS = ['i', 'j', 'k', 'l'], Case = 'Right')
						A[ijkl, 0] += self.CalcAElements(['t', 'u', 'id', 'jd', 'kd', 'ld'], ['w', 'v', 'i', 'j', 'k', 'l'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l'], [i, i, j, j, k, k, l, l], FixedCrS = ['id', 'jd', 'kd', 'ld'], FixedAnS = ['i', 'j', 'k', 'l'], Case = 'Left')
						# pqrs - Case 1
						for p in self.SIndex:
							for q in self.SIndex:
								pq = self.CombinedIndex([p, q])
								A[ijkl, pq] = self.CalcAElements(['id', 'jd', 'kd', 'ld', 'p'], ['i', 'j', 'k', 'l', 'q'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'q'], [i, i, j, j, k, k, l, l, p, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p'], FixedAnS = ['i', 'j', 'k', 'l', 'q'], Case = 'MF')
								A[ijkl, pq] += self.CalcAElements(['id', 'jd', 'kd', 'ld', 'p', 'v', 'w'], ['i', 'j', 'k', 'l', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, k, k, l, l, p, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p'], FixedAnS = ['i', 'j', 'k', 'l', 'q'], Case = 'Right')
								A[ijkl, pq] += self.CalcAElements(['t', 'u', 'id', 'jd', 'kd', 'ld', 'p'], ['w', 'v', 'i', 'j', 'k', 'l', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'q'], [i, i, j, j, k, k, l, l, p, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p'], FixedAnS = ['i', 'j', 'k', 'l', 'q'], Case = 'Left')
								# pqrs - Case 2
								for r in self.SIndex:
									for s in self.SIndex:	
										pqrs = self.CombinedIndex([p, q, r, s])
										A[ijkl, pqrs] = self.CalcAElements(['id', 'jd', 'kd', 'ld', 'p', 'r'], ['i', 'j', 'k', 'l', 's', 'q'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'r', 's', 'q'], [i, i, j, j, k, k, l, l, p, r, s, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p', 'r'], FixedAnS = ['i', 'j', 'k', 'l', 's', 'q'], Case = 'MF')
										A[ijkl, pq] += self.CalcAElements(['id', 'jd', 'kd', 'ld', 'p', 'r', 'v', 'w'], ['i', 'j', 'k', 'l', 's', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'r', 's', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, k, k, l, l, p, r, s, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p', 'r'], FixedAnS = ['i', 'j', 'k', 'l', 's', 'q'], Case = 'Right')
										A[ijkl, pq] += self.CalcAElements(['t', 'u', 'id', 'jd', 'kd', 'ld', 'p', 'r'], ['w', 'v', 'i', 'j', 'k', 'l', 's', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'r', 's', 'q'], [i, i, j, j, k, k, l, l, p, r, s, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p', 'r'], FixedAnS = ['i', 'j', 'k', 'l', 's', 'q'], Case = 'Left')
		return A

	def CalcYElements(self, NormalOrder1, CrSymbols1, AnSymbols1, ProjectorOrbitalList):
		SymbolsS1, SymbolsE1 = GenerateSubspaceCases(CrSymbols1 + ['p'], AnSymbols1 + ['q'], FixedCrA = CrSymbols1, FixedAnA = AnSymbols1)
		YElement = 0.0

		for n in range(len(SymbolsS1)):
			NormalOrder1MF = NormalOrder1 + ['p', 'q']
			PosSMF, PosEMF = CaseToOperatorPositions(SymbolsS1[n], SymbolsE1[n], NormalOrder1MF)
			SignMF = SignOfSeparation(PosEMF[0] + PosEMF[1])
			ConSMF, PorQSMF, SignsSMF = WickContraction(PosSMF[0], PosSMF[1])
			ConEMF, PorQEMF, SignsEMF = WickContraction(PosEMF[0], PosEMF[1])

			NormalOrder1R  = NormalOrder1 + ['p', 'q', 'v', 'w', 'u', 't']	
			PosSR, PosER = CaseToOperatorPositions(SymbolsS1[n], SymbolsE1[n], NormalOrder1R)
			SignR = SignOfSeparation(PosER[0] + PosER[1])
			ConSR, PorQSR, SignsSR = WickContraction(PosSR[0], PosSR[1])
			ConER, PorQER, SignsER = WickContraction(PosER[0], PosER[1])

			NormalOrder1L  = ['t', 'u', 'w', 'v'] + NormalOrder1 + ['p', 'q']
			PosSL, PosEL = CaseToOperatorPositions(SymbolsS1[n], SymbolsE1[n], NormalOrder1L)
			SignL = SignOfSeparation(PosEL[0] + PosEL[1])
			ConSL, PorQSL, SignsSL = WickContraction(PosSL[0], PosSL[1])
			ConEL, PorQEL, SignsEL = WickContraction(PosEL[0], PosEL[1])

			pIndices, qIndices = self.GetPQIndices(SymbolsS1[n])
			tIndices, uIndices, vIndices, wIndices = self.GetAmplitudeIndices(SymbolsS1[n])
			for p in pIndices:
				for q in qIndices:
					Ypq = 0.0
					# MF Case
					OrbitalList1MF = ProjectorOrbitalList + [p, q]
					ConOrbsSMF = ContractionIndexToOrbitals(ConSMF, OrbitalList1MF)
					ConOrbsEMF = ContractionIndexToOrbitals(ConEMF, OrbitalList1MF)
					ExpSMF = CalcWickTerms(ConOrbsSMF, PorQSMF, SignsSMF, self.PS, self.QS)
					ExpEMF = CalcWickTerms(ConOrbsEMF, PorQEMF, SignsEMF, self.PE, self.QE)
					ExpSEMF = float(SignMF) * ExpSMF * ExpEMF
					Ypq = ExpSEMF
					for t in tIndices:
						for u in uIndices:
							for v in vIndices:
								for w in wIndices:
									OrbitalList1R = OrbitalList1MF + [v, w, u, t]
									ConOrbsSR = ContractionIndexToOrbitals(ConSR, OrbitalList1R)
									ConOrbsER = ContractionIndexToOrbitals(ConER, OrbitalList1R)
									ExpSR = CalcWickTerms(ConOrbsSR, PorQSR, SignsSR, self.PS, self.QS)
									ExpER = CalcWickTerms(ConOrbsER, PorQER, SignsER, self.PE, self.QE)
									ExpSER = float(SignR) * ExpSR * ExpER

									OrbitalList1L = [t, u, w, v] + OrbitalList1MF
									ConOrbsSL = ContractionIndexToOrbitals(ConSL, OrbitalList1L)
									ConOrbsEL = ContractionIndexToOrbitals(ConEL, OrbitalList1L)
									ExpSL = CalcWickTerms(ConOrbsSL, PorQSL, SignsSL, self.PS, self.QS)
									ExpEL = CalcWickTerms(ConOrbsEL, PorQEL, SignsEL, self.PE, self.QE)
									ExpSEL = float(SignL) * ExpSR * ExpER								
									
									Ypq += self.t[t, u, v, w] * (ExpSER + ExpSEL)
					Ypq *= self.h[p, q]
					YElement += Ypq

		SymbolsS2, SymbolsE2 = GenerateSubspaceCases(CrSymbols1 + ['p', 'r'], AnSymbols1 + ['s', 'q'], FixedCrA = CrSymbols1, FixedAnA = AnSymbols1)
		for n in range(len(SymbolsS2)):
			NormalOrder2MF = NormalOrder1 + ['p', 'r', 's', 'q']
			PosSMF, PosEMF = CaseToOperatorPositions(SymbolsS2[n], SymbolsE2[n], NormalOrder2MF)
			SignMF = SignOfSeparation(PosEMF[0] + PosEMF[1])
			ConSMF, PorQSMF, SignsSMF = WickContraction(PosSMF[0], PosSMF[1])
			ConEMF, PorQEMF, SignsEMF = WickContraction(PosEMF[0], PosEMF[1])

			NormalOrder2R  = NormalOrder2MF + ['v', 'w', 'u', 't']	
			PosSR, PosER = CaseToOperatorPositions(SymbolsS2[n], SymbolsE2[n], NormalOrder2R)
			SignR = SignOfSeparation(PosER[0] + PosER[1])
			ConSR, PorQSR, SignsSR = WickContraction(PosSR[0], PosSR[1])
			ConER, PorQER, SignsER = WickContraction(PosER[0], PosER[1])

			NormalOrder2L  = ['t', 'u', 'w', 'v'] + NormalOrder2MF
			PosSL, PosEL = CaseToOperatorPositions(SymbolsS2[n], SymbolsE2[n], NormalOrder2L)
			SignL = SignOfSeparation(PosEL[0] + PosEL[1])
			ConSL, PorQSL, SignsSL = WickContraction(PosSL[0], PosSL[1])
			ConEL, PorQEL, SignsEL = WickContraction(PosEL[0], PosEL[1])

			pIndices, qIndices = self.GetPQIndices(SymbolsS2[n])
			rIndices, sIndices = self.GetRSIndices(SymbolsS2[n])
			tIndices, uIndices, vIndices, wIndices = self.GetAmplitudeIndices(SymbolsS2[n])
			for p in pIndices:
				for q in qIndices:
					for r in rIndices:
						for s in sIndices:
							Ypqrs = 0.0
							# MF Case
							OrbitalList2MF = ProjectorOrbitalList + [p, r, s, q]
							ConOrbsSMF = ContractionIndexToOrbitals(ConSMF, OrbitalList2MF)
							ConOrbsEMF = ContractionIndexToOrbitals(ConEMF, OrbitalList2MF)
							ExpSMF = CalcWickTerms(ConOrbsSMF, PorQSMF, SignsSMF, self.PS, self.QS)
							ExpEMF = CalcWickTerms(ConOrbsEMF, PorQEMF, SignsEMF, self.PE, self.QE)
							ExpSEMF = float(SignMF) * ExpSMF * ExpEMF
							Ypqrs = ExpSEMF

							for t in tIndices:
								for u in uIndices:
									for v in vIndices:
										for w in wIndices:
											OrbitalList2R = OrbitalList2MF + [v, w, u, t]
											ConOrbsSR = ContractionIndexToOrbitals(ConSR, OrbitalList2R)
											ConOrbsER = ContractionIndexToOrbitals(ConER, OrbitalList2R)
											ExpSR = CalcWickTerms(ConOrbsSR, PorQSR, SignsSR, self.PS, self.QS)
											ExpER = CalcWickTerms(ConOrbsER, PorQER, SignsER, self.PE, self.QE)
											ExpSER = float(SignR) * ExpSR * ExpER

											OrbitalList2L = [t, u, w, v] + OrbitalList2MF
											ConOrbsSL = ContractionIndexToOrbitals(ConSL, OrbitalList2L)
											ConOrbsEL = ContractionIndexToOrbitals(ConEL, OrbitalList2L)
											ExpSL = CalcWickTerms(ConOrbsSL, PorQSL, SignsSL, self.PS, self.QS)
											ExpEL = CalcWickTerms(ConOrbsEL, PorQEL, SignsEL, self.PE, self.QE)
											ExpSEL = float(SignL) * ExpSR * ExpER											

											Ypqrs += self.t[t, u, v, w] * (ExpSER + ExpSEL)
							Ypqrs *= self.V[p, q, r, s]
							YElement += Ypqrs
		return YElement

	def CalcY(self):
		NS = len(self.SIndex)
		DimY = 1 + NS * NS + NS * NS * NS * NS
		Y = np.zeros((DimY))
		Y[0] = self.CalcYElements([], [], [], [])
		for i in self.SIndex:
			for j in self.SIndex:
				ij = self.CombinedIndex([i, j])

				ival = i
				jval = j
				ijBath = [0, 0]
				if i in self.BIndex:
					ijBath[0] = 1
					ival = ival - len(self.BIndex)
				if j in self.BIndex:
					ijBath[1] = 1
					jval = jval - len(self.BIndex)

				ijNormalOrders1, ijNormalOrders2, ijNormalOrders3, ijNormalOrders4, ijOrbitalLists1, ijOrbitalLists2, ijOrbitalLists3, ijOrbitalLists4, ijRemovedSymbols1, ijRemovedSymbols2, ijRemovedSymbols3, ijRemovedSymbols4, ijBathNum = MakeProjectorCases(ijBath, ['id', 'i', 'jd', 'j'], [ival, ival, jval, jval], False)
				ijCrLists1, ijCrLists2, ijCrLists3, ijCrLists4, ijAnLists1, ijAnLists2, ijAnLists3, ijAnLists4 = MakeRemovedCrAnLists(['id', 'jd'], ['i', 'j'], ijRemovedSymbols1, ijRemovedSymbols2, ijRemovedSymbols3, ijRemovedSymbols4)

				Yij = self.CalcYElements(['id', 'i', 'jd', 'j'], ['id', 'jd'], ['i', 'j'], [ival, ival, jval, jval])
				Yij1 = []; Yij2 = []
				for a in range(len(ijNormalOrders1)):
					aYij = self.CalcYElements(ijNormalOrders1[a], ijCrLists1[a], ijAnLists1[a], ijOrbitalLists1[a])
					Yij1.append(aYij)
				for a in range(len(ijNormalOrders2)):
					aYij = self.CalcYElements(ijNormalOrders2[a], ijCrLists2[a], ijAnLists2[a], ijOrbitalLists2[a])
					Yij2.append(aYij)

				if ijBathNum == 1:
					Yij = -1.0 * Yij
				Parity1 = 1.0; Parity2 = 1.0; Parity3 = 1.0; Parity4 = 1.0
				if ijBathNum == 1:
					Parity2 = -1.0; Parity4 = -1.0
				else:
					Parity1 = -1.0; Parity3 = -1.0
				for a in Yij1:
					Yij = Yij + Parity1 * a
				for a in Yij2:
					Yij = Yij + Parity2 * a
				Y[ij] = Yij
			
				for k in self.SIndex:
					for l in self.SIndex:
						ijkl = self.CombinedIndex([i, j, k, l])
						kval = k
						lval = l
						ijklBath = ijBath.copy()
						ijklBath = ijklBath + [0, 0]
						if k in self.BIndex:
							kval = kval - len(self.SIndex)
							ijklBath[2] = 1
						if l in self.BIndex:
							lval = lval - len(self.SIndex)
							ijklBath[3] = 1

						NormalOrders1, NormalOrders2, NormalOrders3, NormalOrders4, OrbitalLists1, OrbitalLists2, OrbitalLists3, OrbitalLists4, RemovedSymbols1, RemovedSymbols2, RemovedSymbols3, RemovedSymbols4, ijklBathNum = MakeProjectorCases(ijklBath, ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l'], [ival, ival, jval, jval, kval, kval, lval, lval], True)
						CrLists1, CrLists2, CrLists3, CrLists4, AnLists1, AnLists2, AnLists3, AnLists4 = MakeRemovedCrAnLists(['id', 'jd', 'kd', 'ld'], ['i', 'j', 'k', 'l'], RemovedSymbols1, RemovedSymbols2, RemovedSymbols3, RemovedSymbols4)

						Yijkl = self.CalcYElements(['id', 'i', 'jd', 'j'], ['id', 'jd'], ['i', 'j'], [ival, ival, jval, jval])
						Yijkl1 = []; Yijkl2 = []; Yijkl3 = []; Yijkl4 = []
						for a in range(len(NormalOrders1)):
							aYijkl = self.CalcYElements(NormalOrders1[a], CrLists1[a], AnLists1[a], OrbitalLists1[a])
							Yijkl1.append(aYijkl)
						for a in range(len(NormalOrders2)):
							aYijkl = self.CalcYElements(NormalOrders2[a], CrLists2[a], AnLists2[a], OrbitalLists2[a])
							Yijkl2.append(aYijkl)
						for a in range(len(NormalOrders3)):
							aYijkl = self.CalcYElements(NormalOrders3[a], CrLists3[a], AnLists3[a], OrbitalLists3[a])
							Yijkl3.append(aYijkl)
						for a in range(len(NormalOrders4)):
							aYijkl = self.CalcYElements(NormalOrders4[a], CrLists4[a], AnLists4[a], OrbitalLists4[a])
							Yijkl4.append(aYijkl)

						if ijklBathNum == 1:
							Yijkl = -1.0 * Yijkl
						Parity1 = 1.0; Parity2 = 1.0; Parity3 = 1.0; Parity4 = 1.0
						if ijBathNum == 1:
							Parity2 = -1.0; Parity4 = -1.0
						else:
							Parity1 = -1.0; Parity3 = -1.0
						for a in Yijkl1:
							Yijkl = Yijkl + Parity1 * a
						for a in Yijkl2:
							Yijkl = Yijkl + Parity2 * a
						for a in Yijkl3:
							Yijkl = Yijkl + Parity3 * a
						for a in Yijkl4:
							Yijkl = Yijkl + Parity4 * a
						Y[ijkl] = Yijkl
		return Y

	def CalcH(self):
		Y = self.CalcY()
		A = self.CalcA()
		H = np.linalg.solve(A, Y)
		NS = len(self.SIndex)
		H0 = 0.0
		H1 = np.zeros((NS, NS))
		H2 = np.zeros((NS, NS, NS, NS))
		H0 = H[0]
		for p in range(NS):
			for q in range(NS):
				pq = self.CombinedIndex([p, q])
				H1[p, q] = H[pq]
				for r in range(NS):
					for s in range(NS):
						pqrs = self.CombinedIndex([p, q, r, s])
						H2[p, q, r, s] = H[pqrs]
		return H0, H1, H2

if __name__ == '__main__':
	#CrA, AnA, CrB, AnB = SeparateOperatorIndices([0, 2, 4], [1, 3, 7], [5], [6])
	#print(CrA)
	#print(AnA)
	#print(CrB)
	#print(AnB)

	#CrIndex = ['p', 'r', 'v', 'w']
	#AnIndex = ['q', 's', 't', 'u']
	#IA, IB = GenerateSubspaceCases(CrIndex, AnIndex, FixedCrA = ['p'], FixedAnA = ['q'], FixedCrB = ['r'], FixedAnB = ['s'])
	#for i in range(len(IA)):
	#	print(IA[i])
	#	print(IB[i])


	#IndexS = [['p'], ['q']]
	#IndexE = [['r'], ['s']]
	#NormalOrder = ['p', 'r', 's', 'q']
	#NormalOrderOrb = [1, 2, 3, 4]
	#PositionS, PositionE = CaseToOperatorPositions(IndexS, IndexE, NormalOrder, NormalOrderOrb)
	#print(PositionS)
	#print(PositionE)
	#Sign = SignOfSeparation(PositionE[0] + PositionE[1])
	#ContractionsS, PorQS, SignsS = WickContraction(PositionS[0], PositionS[1], OrbList = NormalOrderOrb)
	#print(ContractionsS)
	#print(PorQS)
	#print(SignsS)

	from functools import reduce
	from pyscf import gto, scf, mp, lo, ao2mo
	from frankenstein.tools.tensor_utils import get_symm_mat_pow
	N = 6
	nocc = int(N / 2)
	r = 1.0
	mol = gto.Mole()
	mol.atom = []
	for i in range(N):
		angle = i / N * 2.0 * np.pi
		mol.atom.append(('H', (r * np.sin(angle), r * np.cos(angle), 0)))
	mol.basis = 'sto-3g'
	mol.build(verbose = 0)
	mf = scf.RHF(mol).run()
	
	S = mol.intor_symmetric("int1e_ovlp")
	mo_coeff = mf.mo_coeff
	StoOrth = get_symm_mat_pow(S, 0.50)
	StoOrig = get_symm_mat_pow(S, -0.5)
	mo_occ = mo_coeff[:, :nocc]
	mo_occ = np.dot(StoOrth.T, mo_occ)
	P = np.dot(mo_occ, mo_occ.T)
	Nf = 2
	PFrag = P[:Nf, :Nf]
	PEnvBath = P[Nf:, Nf:]
	eEnv, vEnv = np.linalg.eigh(PEnvBath)
	thresh = 1.e-9
	BathOrbs = [i for i, v in enumerate(eEnv) if v > thresh and v < 1.0 - thresh]
	EnvOrbs  = [i for i, v in enumerate(eEnv) if v < thresh or v > 1.0 - thresh]
	TBath = vEnv[:,BathOrbs]
	TBath = np.concatenate((np.zeros((Nf, TBath.shape[1])), TBath))
	TEnv  = vEnv[:,EnvOrbs]
	TEnv  = np.concatenate((np.zeros((Nf, TEnv.shape[1])), TEnv))
	TFrag = np.eye(TBath.shape[0], Nf)
	TSch = np.concatenate((TFrag, TBath), axis = 1)
	T = np.concatenate((TSch, TEnv), axis = 1)
	BathOrbs = [x + Nf for x in BathOrbs]
	SchOrbs = [0] + BathOrbs
	EnvOrbs = [x + Nf for x in EnvOrbs]
	PEnv = reduce(np.dot, (TEnv.T, P, TEnv))
	PSch = reduce(np.dot, (TSch.T, P, TSch))
	PEnv[PEnv < thresh] = 0.0
	PEnv[PEnv > 1.0 - thresh] = 1.0
	print(PSch)
	print(PEnv)

	TTotal = np.dot(StoOrth, T)
	hSO = reduce(np.dot, (TTotal.T, mf.get_hcore(), TTotal))
	VSO = ao2mo.kernel(mol, TTotal)
	VSO = ao2mo.restore(1, VSO, hSO.shape[0])

	mp2 = mp.MP2(mf)
	E, t2 = mp2.kernel()

	Nocc = t2.shape[0]
	Nvir = t2.shape[2]
	Norb = Nocc + Nvir
	tMO = np.zeros((Norb, Norb, Norb, Norb))
	tMO[:Nocc, :Nocc, Nocc:, Nocc:] = t2

	#tLO = np.einsum('ia,jb,kc,ld,ijkl->abcd', StoOrth, StoOrth, StoOrth, StoOrth, tMO)
	#tSO = np.einsum('ap,bq,cr,ds,abcd->pqrs', T, T, T, T, tLO)
	tSO = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, tMO)

	SIndex = list(range(PSch.shape[0]))
	FIndex = SIndex[:int(len(SIndex)/2)]
	BIndex = SIndex[int(len(SIndex)/2):]
	EIndex = list(range(PEnv.shape[0]))
	
	#tZero = np.zeros((Norb, Norb, Norb, Norb))
	#testMFBath = MP2Bath(tZero, SIndex, EIndex, PSch, PEnv, hSO, VSO)
	#mf0, mf1, mf2 = testMFBath.CalcH()
	#mf0.tofile("mf0")
	#mf1.tofile("mf1")
	#mf2.tofile("mf2")

	myMP2Bath = MP2Bath(tSO, FIndex, BIndex, EIndex, PSch, PEnv, hSO, VSO)
	H0, H1, H2 = myMP2Bath.CalcH()
	H0.tofile("H0")
	H1.tofile("H1")
	H2.tofile("H2")
