from itertools import combinations
import numpy as np
from wick import WickContraction

'''
Takes a list of indices corresponding to position of operations within one 
subspace and calculates the sign of moving the operators together, assuming
there are only two subspaces
'''
def SignOfSeparation(Indices):
	IndicesSort = sorted(Indices)
	List = list(range(len(IndicesSort)))
	Moves = sum([ai - bi for ai, bi in zip(IndicesSort, List)])
	print(Moves)
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
		for i in range(len(Signs)):
			iValue = float(Signs[i])
			for n in range(len(Contractions[i])):
				if PorQ[i][n] == 'P':
					iValue *= P[Contractions[i][n][0], Contractions[i][n][1]]
				else:
					iValue *= Q[Contractions[i][n][0], Contractions[i][n][1]]
				Value += iValue
		return Value


class MP2Bath:
	def __init__(self, t, FIndex, BIndex, EIndex, PS, PE):
		self.t = t
		self.FIndex = FIndex
		self.BIndex = BIndex
		self.SIndex = FIndex + BIndex
		self.EIndex = EIndex
		self.PS = PS
		self.PE = PE
		self.QS = np.eye(PS.shape[0]) - PS
		self.QE = np.eye(PE.shape[0]) - PE
	
	def CalcAijpq_MF(self, i, j, p, q):
		CrSymbols = ['id', 'jd', 'p']
		AnSymbols = ['i', 'j', 'q']
		FixedCrS = ['id', 'jd', 'p']
		FixedAnS = ['i', 'j', 'q']
		SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrS = FixedCrS, FixedAnS = FixedAnS)

	def GetAmplitudeIndices(self, SymbolS):
		tIndices, uIndices, vIndices, wIndices = []
		if 't' in SymbolS[0]:
			tIndices = self.SIndex
		else:
			tIndices = self.EIndex
		if 'u' in SymbolS[0]:
			uIndices = self.SIndex
		else:
			uIndices = self.EIndex
		if 'v' in SymbolS[1]:
			vIndices = self.SIndex
		else:
			vIndices = self.EIndex
		if 'w' in SymbolS[1]:
			wIndices = self.SIndex
		else:
			wIndices = self.EIndex
		return tIndices, uIndices, vIndices, wIndices

	def CalcExpValue(self, SymbolS, SymbolE, NormalOrder, OrbitalList):
		PosS, PosE = CaseToOperatorPosition(SymbolS, SymbolE, NormalOrder)
		Sign = SignOfSeparation(PosE[0] + PosE[1])
		ConS, PorQS, SignsS = WickContraction(PosS[0], PosS[1], OrbList = OrbitalList)
		ConE, PorQE, SignsE = WickContraction(PosE[0], PosE[1], OrbList = OrbitalList)
		ExpS = CalcWickTerms(ConS, PorQS, SignsS, self.PS, self.QS)
		ExpE = CalcWickTerms(ConE, PorQE, SignsE, self.PE, self.QE)
		return float(sign) * ExpS * ExpE

	'''
	Calculates A for given indices pqrsijkl
	'''
	def CalcAElements(self, CrSymbols, AnSymbols, NormalOrder, OrbitalListNoT, FixedCrS = None, FixedAnS = None, Case = 'MF'):
		SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrS = FixedCrS, FixedAnS = FixedAnS)
		A = 0.0
		if len(OrbitalListNoT) == 0 and Case == 'MF':
			return 1.0
		for n in range(len(SymbolsS)):
			if Case == 'MF':
				ExpSE = self.CalcExpValue(SymbolsS[n], SymbolsE[n], NormalOrder, OrbitalListNoT)
				A += ExpSE
				continue
			tIndices, uIndices, vIndices, wIndices = self.GetAmplitudeIndices(SymbolsS[n])
			for t in tIndices:
				for u in uIndices:
					for v in vIndices:
						for w in wIndices:
							OrbitalList = OrbitalListNoT.copy()
							if Case = 'Right':
								OrbitalList = OrbitalList + [v, w, u, t]
							if Case = 'Left':
								OrbitalList = [t, u, w, v] + OrbitalList
							ExpSE = self.CalcExpValue(SymbolsS[n], SymbolsE[n], NormalOrder, OrbitalList)
							A += self.t[t, u, v, w] * ExpSE
		return A
	
	def CalcAijpq_R(self, i, j, p, q):
		CrSymbols = ['id', 'jd', 'p', 'v', 'w']
		AnSymbols = ['i', 'j', 'q', 'u', 't']
		FixedCrS = ['id', 'jd', 'p']
		FixedAnS = ['i', 'j', 'q']
		SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrS = FixedCrS, FixedAnS = FixedAnS)
		Aijpq_R = 0.0
		for a in range(len(SymbolsS)):
			tIndices, uIndices, vIndices, wIndices = GetAmplitudeIndices(SymbolsS[a])
			for t in tIndices:
				for u in uIndices:
					for v in vIndices:
						for w in wIndices:
							NormalOrder = ['id', 'i', 'jd', 'j', 'p', 'q', 'v', 'w', 'u', 't']
							OrbitalList = [i, i, j, j, p, q, v, w, u, t]
							ExpSE = self.CalcExpValue(SymbolsS[a], SymbolsE[a], NormalOrder, OrbitalList)
							#ExpSEj, ExpSEi, ExpSE1 = 0.0
							#if i in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('id')
							#	NormOrder.remove('i')
							#	OrbList = [j, j, p, q, v, w, u, t]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('id')
							#	SymS[1].remove('i')
							#	ExpSEj = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	if j not in self.BIndex:
							#		ExpSE = ExpSEj - ExpSE
							#if j in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('jd')
							#	NormOrder.remove('j')
							#	OrbList = [i, i, p, q, v, w, u, t]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('jd')
							#	SymS[1].remove('j')
							#	ExpSEi = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	if i not in self.BIndex:
							#		ExpSE = ExpSEi - ExpSE
							#if i in self.BIndex and j in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('id')
							#	NormOrder.remove('i')
							#	NormOrder.remove('jd')
							#	NormOrder.remove('j')
							#	OrbList = [p, q, v, w, u, t]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('id')
							#	SymS[0].remove('jd')
							#	SymS[1].remove('i')
							#	SymS[1].remove('j')
							#	ExpSE1 = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	ExpSE = ExpSE1 - ExpSEi - ExpSEj + ExpSE

							Aijpq_R += self.t[t, u, v, w] * ExpSE


	def CalcAijpq_L(self, i, j, p, q):
		CrSymbols = ['id', 'jd', 'p', 't', 'u']
		AnSymbols = ['i', 'j', 'q', 'w', 'v']
		FixedCrS = ['id', 'jd', 'p']
		FixedAnS = ['i', 'j', 'q']
		#SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrS = FixedCrS, FixedAnS = FixedAnS)
		Aijpq_L = 0.0
		for a in range(len(SymbolsS)):
			tIndices, uIndices, vIndices, wIndices = GetAmplitudeIndices(SymbolsS[a])
			for t in tIndices:
				for u in uIndices:
					for v in vIndices:
						for w in wIndices:
							NormalOrder = ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'p', 'q']
							OrbitalList = [t, u, w, v, i, i, j, j, p, q]
							ExpSE = self.CalcExpValue(SymbolsS[a], SymbolsE[a], NormalOrder, OrbitalList)
							#ExpSEj, ExpSEi, ExpSE1 = 0.0
							#if i in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('id')
							#	NormOrder.remove('i')
							#	OrbList = [t, u, w, v, j, j, p, q]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('id')
							#	SymS[1].remove('i')
							#	ExpSEj = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	if j not in self.BIndex:
							#		ExpSE = ExpSEj - ExpSE
							#if j in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('jd')
							#	NormOrder.remove('j')
							#	OrbList = [t, u, w, v, i, i, p, q]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('jd')
							#	SymS[1].remove('j')
							#	ExpSEi = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	if i not in self.BIndex:
							#		ExpSE = ExpSEi - ExpSE
							#if i in self.BIndex and j in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('id')
							#	NormOrder.remove('i')
							#	NormOrder.remove('jd')
							#	NormOrder.remove('j')
							#	OrbList = [t, u, w, v, p, q]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('id')
							#	SymS[0].remove('jd')
							#	SymS[1].remove('i')
							#	SymS[1].remove('j')
							#	ExpSE1 = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	ExpSE = ExpSE1 - ExpSEi - ExpSEj + ExpSE

							Aijpq_L += self.t[t, u, v, w] * ExpSE

	def CalcAijpq_MF(self, i, j, p, q):
		CrSymbols = ['id', 'jd', 'p']
		AnSymbols = ['i', 'j', 'q']
		FixedCrS = ['id', 'jd', 'p']
		FixedAnS = ['i', 'j', 'q']
		SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrS = FixedCrS, FixedAnS = FixedAnS)
		Aijpq_MF = 0.0
		for a in range(len(SymbolsS)):
			tIndices, uIndices, vIndices, wIndices = GetAmplitudeIndices(SymbolsS[a])
			for t in tIndices:
				for u in uIndices:
					for v in vIndices:
						for w in wIndices:
							NormalOrder = ['id', 'i', 'jd', 'j', 'p', 'q']
							OrbitalList = [i, i, j, j, p, q]
							ExpSE = self.CalcExpValue(SymbolsS[a], SymbolsE[a], NormalOrder, OrbitalList)
							#ExpSEj, ExpSEi, ExpSE1 = 0.0
							#if i in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('id')
							#	NormOrder.remove('i')
							#	OrbList = [j, j, p, q]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('id')
							#	SymS[1].remove('i')
							#	ExpSEj = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	if j not in self.BIndex:
							#		ExpSE = ExpSEj - ExpSE
							#if j in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('jd')
							#	NormOrder.remove('j')
							#	OrbList = [i, i, p, q]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('jd')
							#	SymS[1].remove('j')
							#	ExpSEi = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	if i not in self.BIndex:
							#		ExpSE = ExpSEi - ExpSE
							#if i in self.BIndex and j in self.BIndex:
							#	NormOrder = NormalOrder.copy()
							#	NormOrder.remove('id')
							#	NormOrder.remove('i')
							#	NormOrder.remove('jd')
							#	NormOrder.remove('j')
							#	OrbList = [p, q]
							#	SymS = SymbolsS[a].copy()
							#	SymE = SymbolsE[a].copy()
							#	SymS[0].remove('id')
							#	SymS[0].remove('jd')
							#	SymS[1].remove('i')
							#	SymS[1].remove('j')
							#	ExpSE1 = self.CalcExpVal(SymS, SymE, NormOrder, OrbList)
							#	ExpSE = ExpSE1 - ExpSEi - ExpSEj + ExpSE

							Aijpq_MF += self.t[t, u, v, w] * ExpSE	

	def MP2Expectation_ijpq(self, i, j, p, q, t, u, v, w, IndexS, IndexE, Case):
		NormalOrder = []
		OrbitalList = []
		IdxS = IndexS
		IdxE = IndexE
		if Case == 'Right':
			IdxS[0] = ['id', 'jd'] + IndexS[0]
			IdxS[1] = ['i', 'j'] + IndexS[1]
			NormalOrder = ['id', 'i', 'jd', 'j', 'p', 'q', 'v', 'w', 't', 'u']
			OrbitalList = [i, i, j, j, p, q, v, w, t, u]
			PosS, PosE, OrbS, OrbE = CaseToOperatorPositions(IndexS, IndexE, NormalOrder)
			Sign = SignOfSeparation(PosE[0] + PosE[1])
			ConS, PorQS, SignsS = WickContraction(PosS[0], PosS[1], OrbList = OrbitalList)
			ConE, PorQE, SignsE = WickContraction(PosE[0], PosE[1], OrbList = OrbitalList)
			ExpS = CalcWickTerms(ConS, PorQS, SignsS, self.PS, self.QS)
			ExpE = CalcWickTerms(ConE, PorQE, SignsE, self.PE, self.QE)
			
		

	def CalcXij(self, i, j, p, q):
		# MP2 correction on the right
		CrIndex = ['p', 'v', 'w']
		AnIndex = ['q', 't', 'u']
		FixedCrS = None
		FixedAnS = None
		FixedCrE = None
		FoxedAnE = None
		if p in self.SIndex:
			FixedCrS = ['p']
		else:
			FixedCrE = ['p']
		if q in self.SIndex:
			FixedAnS = ['q']
		else:
			FixedAnE = ['q']

		IndicesS, IndicesE = GenerateSubspaceCases(CrIndex, AnIndex, FixedCrA = FixedCrS, FixedAnA = FixedAnS, FixedCrB = FixedCrE, FixedAnB = FixedAnE)

		# MP2 Wavefunction on the Right
		Xijpq = 0.0
		for a in range(len(IndicesS)):
			tIndices, uIndices, vIndices, wIndices = []
			if 't' in IndicesS[a][0]:
				tIndices = self.SIndex
			else:
				tIndices = self.EIndex
			if 'u' in IndicesS[a][0]:
				uIndices = self.SIndex
			else:
				uIndices = self.EIndex
			if 'v' in IndicesS[a][1]:
				vIndices = self.SIndex
			else:
				vIndices = self.EIndex
			if 'w' in IndicesS[a][1]:
				wIndices = self.SIndex
			else:
				wIndices = self.EIndex
				
			IndexE = IndicesE[a]
			IndexS = IndicesS[a]
			IndexS[0] = ['id', 'jd'] + IndicesS[a][0]
			IndexS[1] = ['i', 'j'] + IndicesS[a][1]
			#PosS, PosE = CaseToOperatorPositions(IndexS, IndexE, ['id', 'i', 'jd', 'j', 'p', 'q', 'v', 'w', 't', 'u'])

			for t in tIndices:
				for u in uIndices:
					for v in vIndices:
						for w in wIndices:
							Xijpq += self.t[t, u, v, w]# * MP2ExpVal(IndicesS[a], IndicesE[a])
			
			

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


	IndexS = [['p'], ['q']]
	IndexE = [['r'], ['s']]
	NormalOrder = ['p', 'r', 's', 'q']
	NormalOrderOrb = [1, 2, 3, 4]
	PositionS, PositionE, OrbS, OrbE = CaseToOperatorPositions(IndexS, IndexE, NormalOrder, NormalOrderOrb)
	print(PositionS)
	print(PositionE)
	print(OrbS)
	print(OrbE)
	Sign = SignOfSeparation(PositionE[0] + PositionE[1])
	ContractionsS, PorQS, SignsS = WickContraction(PositionS[0], PositionS[1], OrbList = NormalOrderOrb)
	print(ContractionsS)
	print(PorQS)
	print(SignsS)
