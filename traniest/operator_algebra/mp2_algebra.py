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
	def __init__(self, t, FIndex, BIndex, EIndex, PS, PE, h, V):
		self.t = t
		self.FIndex = FIndex
		self.BIndex = BIndex
		self.SIndex = FIndex + BIndex
		self.EIndex = EIndex
		self.AllIndex = self.SIndex + EIndex
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

	def GetPQIndices(self, SymbolS):
		pIndices = []
		qIndices = []
		if 'p' in SymbolS[0]:
			pIndices = self.SIndex
		else:
			pIndices = self.EIndex
		if 'q' in SymbolsS[0]:
			qIndices = self.SIndex
		else:
			qIndices = self.EIndex
		return pIndices, qIndices

	def GetRSIndices(self, SymbolS):
		rIndices = []
		sIndices = []
		if 'r' in SymbolS[0]:
			rIndices = self.SIndex
		else:
			rIndices = self.EIndex
		if 's' in SymbolS[0]:
			sIndices = self.SIndex
		else:
			sIndices = self.EIndex
		return rIndices, sIndices

	def CalcExpValue(self, SymbolS, SymbolE, NormalOrder, OrbitalList):
		PosS, PosE = CaseToOperatorPosition(SymbolS, SymbolE, NormalOrder)
		Sign = SignOfSeparation(PosE[0] + PosE[1])
		ConS, PorQS, SignsS = WickContraction(PosS[0], PosS[1], OrbList = OrbitalList)
		ConE, PorQE, SignsE = WickContraction(PosE[0], PosE[1], OrbList = OrbitalList)
		ExpS = CalcWickTerms(ConS, PorQS, SignsS, self.PS, self.QS)
		ExpE = CalcWickTerms(ConE, PorQE, SignsE, self.PE, self.QE)
		return float(sign) * ExpS * ExpE

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
	'''
	def CalcAElements(self, CrSymbols, AnSymbols, NormalOrder, OrbitalListNoT, FixedCrS = None, FixedAnS = None, Case = 'MF'):
		SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrA = FixedCrS, FixedAnA = FixedAnS)
		AElement = 0.0
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
							assert len(OrbitalList) == len(NormalOrder)
							ExpSE = self.CalcExpValue(SymbolsS[n], SymbolsE[n], NormalOrder, OrbitalList)
							A += self.t[t, u, v, w] * ExpSE
		return A

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
				A[0, pq] =  CalcAElements(['p'], ['q'], ['p', 'q'], [p, q], FixedCrS = ['p'], FixedAnS = ['q'], Case = 'MF')
				A[0, pq] += CalcAElements(['p', 'v', 'w'], ['q', 'u', 't'], ['p', 'q', 'v', 'w', 'u', 't'], [p, q], FixedCrS = ['p'], FixedAnS = ['q'], Case = 'Right')
				A[0, pq] += CalcAElements(['p', 't', 'u'], ['q', 'v', 'w'], ['t', 'u', 'w', 'v', 'p', 'q'], [p, q], FixedCrS = ['p'], FixedAnS = ['q'], Case = 'Left')
				# pqrs - Case 2
				for r in self.SIndex:
					for s in self.SIndex:
						pqrs = self.CombinedIndex([p, q, r, s])
						A[0, pqrs] = CalcAElements(['p', 'r'], ['s', 'q'], ['p', 'r', 's', 'q'], [p, r, s, q], FixedCrS = ['p', 'r'], FixedAnS = ['s', 'q'], Case = 'MF')
						A[0, pqrs] += CalcAElements(['p', 'r', 'v', 'w'], ['s', 'q', 'u', 't'], ['p', 'r', 's', 'q', 'v', 'w', 'u', 't'], [p, r, s, q], FixedCrS = ['p', 'r'], FixedAnS = ['s', 'q'], Case = 'Right')
						A[0, pqrs] += CalcAElements(['p', 'r', 't', 'u'], ['s', 'q', 'v', 'w'], ['t', 'u', 'w', 'v', 'p', 'r', 's', 'q'], [p, r, s, q], FixedCrS = ['p', 'r'], FixedAnS = ['s', 'q'], Case = 'Left')
		# ijkl - Case 1
		for i in self.SIndex:
			for j in self.SIndex:
				ij = CombinedIndex([i, j])
				# pqrs - Case 0
				A[ij, 0] = CalcAElements(['id', 'jd'], ['i', 'j'], ['id', 'i', 'jd', 'j'], [i, i, j, j], FixedCrS = ['id', 'jd'], FixedAnS = ['i', 'j'], Case = 'MF')
				A[ij, 0] += CalcAElements(['id', 'jd', 'v', 'w'], ['i', 'j', 'u', 't'], ['id', 'i', 'jd', 'j', 'v', 'w', 'u', 't'], [i, i, j, j], FixedCrS = ['id', 'jd'], FixedAnS = ['i', 'j'], Case = 'Right')
				A[ij, 0] += CalcAElements(['t', 'u', 'id', 'jd'], ['w'. 'v', 'i', 'j'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j'], [i, i, j, j], FixedCrS = ['id', 'jd'], FixedAnS = ['i', 'j'], Case = 'Left')
				# pqrs - Case 1
				for p in self.SIndex:
					for q in self.SIndex:
						pq = CombinedIndex([p, q])
						A[ij, pq] = CalcAElements(['id', 'jd', 'p'], ['i', 'j', 'q'], ['id', 'i', 'jd', 'j', 'p', 'q'], [i, i, j, j, p, q], FixedCrS = ['id', 'jd', 'p'], FixedAnS = ['i', 'j', 'q'], Case = 'MF')
						A[ij, pq] += CalcAElements(['id', 'jd', 'p', 'v'. 'w'], ['i', 'j', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'p', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, p, q], FixedCrS = ['id', 'jd', 'p'], FixedAnS = ['i', 'j', 'q'], Case = 'Right')
						A[ij, pq] += CalcAElements(['t', 'u', 'id', 'jd', 'p'], ['w', 'v', 'i', 'j', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'p', 'q'], [i, i, j, j, p, q], FixedCrS = ['id', 'jd', 'p'], FixedAnS = ['i', 'j', 'q'], Case = 'Left')
						# pqrs - Case 2
						for r in self.SIndex:
							for s in self.SIndex:
								pqrs = self.CombinedIndex([p, q, r, s])
								A[ij, pqrs] = CalcAElements(['id', 'jd', 'p', 'r'], ['i', 'j', 's', 'q'], ['id', 'i', 'jd', 'j', 'p', 'r', 's', 'q'], [i, i, j, j, p, r, s, q], FixedCrS = ['id', 'jd', 'p', 'r'], FixedAnS = ['i', 'j', 's', 'q'], Case = 'MF')
								A[ij, pqrs] += CalcAElements(['id', 'jd', 'p', 'r', 'v', 'w'], ['i', 'j', 's', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'p', 'r', 's', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, p, r, s, q], FixedCrS = ['id', 'jd', 'p', 'r'], FixedAnS = ['i', 'j', 's', 'q'], Case = 'Right')
								A[ij, pqrs] += CalcAElements(['t', 'u', 'id', 'jd', 'p', 'r'], ['w', 'v', 'i', 'j', 's', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'p', 'r', 's', 'q'], [i, i, j, j, p, r, s, q], FixedCrS = ['id', 'jd', 'p', 'r'], FixedAnS = ['i', 'j', 's', 'q'], Case = 'Left')
		#ijkl - Case 2
		for i in self.SIndex:
			for j in self.SIndex:
				for k in self.SIndex:
					for l in self.SIndex:
						ijkl = self.CombinedIndex([i, j, k, l])
						# pqrs - Case 0
						A[ijkl, 0] = CalcAElements(['id', 'jd', 'kd', 'ld'], ['i', 'j', 'k', 'l'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l'], [i, i, j, j, k, k, l, l], FixedCrS = ['id', 'jd', 'kd', 'ld'], FixedAnS = ['i', 'j', 'k', 'l'], Case = 'MF')
						A[ijkl, 0] += CalcAElements(['id', 'jd', 'kd', 'ld', 'v', 'w'], ['i', 'j', 'k', 'l', 'u', 't'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'v', 'w', 'u', 't'], [i, i, j, j, k, k, l, l], FixedCrS = ['id', 'jd', 'kd', 'ld'], FixedAnS = ['i', 'j', 'k', 'l'], Case = 'Right')
						A[ijkl, 0] += CalcAElements(['t'. 'u', 'id', 'jd', 'kd', 'ld'], ['w', 'v', 'i', 'j', 'k', 'l'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l'], [i, i, j, j, k, k, l, l], FixedCrS = ['id', 'jd', 'kd', 'ld'], FixedAnS = ['i', 'j', 'k', 'l'], Case = 'Left')
						# pqrs - Case 1
						for p in self.SIndex:
							for q in self.SIndex:
								pq = self.CombinedIndex([p, q])
							A[ijkl, pq] = CalcAElements(['id', 'jd', 'kd', 'ld', 'p'], ['i', 'j', 'k', 'l', 'q'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'q'], [i, i, j, j, k, k, l, l, p, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p'], FixedAnS = ['i', 'j', 'k', 'l', 'q'], Case = 'MF')
							A[ijkl, pq] += CalcAElements(['id', 'jd', 'kd', 'ld', 'p', 'v', 'w'], ['i', 'j', 'k', 'l', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, k, k, l, l, p, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p'], FixedAnS = ['i', 'j', 'k', 'l', 'q'], Case = 'Right')
							A[ijkl, pq] += CalcAElements(['t'. 'u', 'id', 'jd', 'kd', 'ld', 'p'], ['w', 'v', 'i', 'j', 'k', 'l', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'q'], [i, i, j, j, k, k, l, l, p, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p'], FixedAnS = ['i', 'j', 'k', 'l', 'q'], Case = 'Left')
								# pqrs - Case 2
								for r in self.SIndex:
									for s in self.SIndex:	
										pqrs = self.CombinedIndex([p, q, r, s])
										A[ijkl, pqrs] = CalcAElements(['id', 'jd', 'kd', 'ld', 'p', 'r'], ['i', 'j', 'k', 'l', 's', 'q'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'r', 's', 'q'], [i, i, j, j, k, k, l, l, p, r, s, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p', 'r'], FixedAnS = ['i', 'j', 'k', 'l', 's', 'q'], Case = 'MF')
										A[ijkl, pq] += CalcAElements(['id', 'jd', 'kd', 'ld', 'p', 'r', 'v', 'w'], ['i', 'j', 'k', 'l', 's', 'q', 'u', 't'], ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'r', 's', 'q', 'v', 'w', 'u', 't'], [i, i, j, j, k, k, l, l, p, r, s, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p', 'r'], FixedAnS = ['i', 'j', 'k', 'l', 's', 'q'], Case = 'Right')
										A[ijkl, pq] += CalcAElements(['t'. 'u', 'id', 'jd', 'kd', 'ld', 'p'. 'r'], ['w', 'v', 'i', 'j', 'k', 'l', 's', 'q'], ['t', 'u', 'w', 'v', 'id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l', 'p', 'r', 's', 'q'], [i, i, j, j, k, k, l, l, p, r, s, q], FixedCrS = ['id', 'jd', 'kd', 'ld', 'p', 'r'], FixedAnS = ['i', 'j', 'k', 'l', 's', 'q'], Case = 'Left')
		return A

	def CalcYElements(self, ProjectorOrbitalList):
		NumP = len(ProjectorOrbitalList) # 0 for Case 0, 4 for Case 1, 8 for Case 2
		NormalOrder1 = []
		CrSymbols1 = []
		AnSymbols1 = []
		if NumP == 4:
			NormalOrder1 = ['id', 'i', 'jd', 'j']
			CrSymbols1 = ['id', 'jd']
			AnSymbols1 = ['i', 'j']
		if NumP == 8:
			NormalOrder1 = ['id', 'i', 'jd', 'j', 'kd', 'k', 'ld', 'l']
			CrSymbols1 = ['id', 'jd', 'kd', 'ld']
			AnSymbols1 = ['i', 'j', 'k', 'l']
		SymbolsS1, SymbolsE1 = GenerateSubspaceCases(CrSymbols1 + ['p'], AnSymbols1 + ['q'], FixedCrA = CrSymbols1, FixedAnA = AnSymbols1)
		YElement = 0.0

		for n in range(len(SymbolsS1)):
			pIndices, qIndices = self.GetPQIndices(SymbolsS1[n])
			tIndices, uIndices, vIndices, wIndices = GetAmplitudeIndices(SymbolsS1[n])
			for p in pIndices:
				for q in qIndices:
					Ypq = 0.0
					# MF Case
					Ypq = self.CalcExpValue(SymbolsS1[n], SymbolsE1[n], NormalOrder1 + ['p', 'q'], ProjectorOrbitalList + [p, q])
					for t in tIndices:
						for u in uIndices:
							for v in vIndices:
								for w in wIndices:
									ExpR = self.CalcExpValue(SymbolsS1[n], SymbolsE1[n], NormalOrder1 + ['p', 'q', 'v', 'w', 'u', 't'], ProjectorOrbitalList + [p, q, v, w, u ,t])
									ExpL = self.CalcExpValue(SymbolsS1[n], SymbolsE1[n], ['t', 'u', 'w', 'v'] + NormalOrder1 + ['p', 'q'], [t, u, w, v] + ProjectorOrbitalList + [p, q])
									Ypq += self.t[t, u, v, w] * (ExpR + ExpL)
					Ypq *= self.h[p, q]
					YElement += Ypq
		SymbolsS2, SymbolsE2 = GenerateSubspaceCases(CrSymbols1 + ['p', 'r'], AnSymbols1 + ['s', 'q'], FixedCrA = CrSymbols1, FixedAnA = AnSymbols1)
		for n in range(len(SymbolsS2)):
			pIndices, qIndices = self.GetPQIndices(SymbolsS2[n])
			rIndices, sIndices = self.getRSIndices(SymbolsS2[n])
			tIndices, uIndices, vIndices, wIndices = GetAmplitudeIndices(SymbolsS2[n])
			for p in pIndices:
				for q in qIndices:
					for r in rIndices:
						for s in sIndices:
							Ypqrs = 0.0
							# MF Case
							Ypqrs = self.CalcExpValue(SymbolsS2[n], SymbolsE2[n], NormalOrder1 + ['p', 'r', 's', 'q'], ProjectorOrbitalList + [p, q, r, s])
							for t in tIndices:
								for u in uIndices:
									for v in vIndices:
										for w in wIndices:
											ExpR = self.CalcExpValue(SymbolsS2[n], SymbolsE2[n], NormalOrder1 + ['p', 'r', 's', 'q', 'v', 'w', 'u', 't'], ProjectorOrbitalList + [p, r, s, q, v, w, u, t])
											ExpL = self.CalcExpValue(SymbolsS2[n], SymbolsE2[n], ['t', 'u', 'w', 'v'] + NormalOrder1 + ['p', 'r', 's', 'q'], [t, u, w, v] + ProjectorOrbitalList + [p, r, s, q])
											Ypqrs += self.t[t, u, v, w] * (ExpR + ExpL)
							Ypqrs *= self.V[p, q, r, s]
							YElement += Ypqrs
		return YElement

	def CalcY(self):
		NS = len(self.SIndex)
		DimY = 1 + NS * NS + NS * NS * NS * NS
		Y = np.zeros((DimY, 1))
		Y[0, 0] = CalcYElements([])
		for i in self.SIndex:
			for j in self.SIndex:
				ij = self.CombinedIndex([i, j])
				Y[ij, 0] = CalcYElement([i, i, j, j])
				for k in self.SIndex:
					for l in self.SIndex:
						ijkl = self.CombinedIndex([i, j, k, l])
						Y[ijkl, 0] = CalcYElement([i, i, j, j, k, k, l, l])
		return Y

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
