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
	def __init__(self, t, SIndex, EIndex, PS, PE, h, V):
		self.t = t
		#self.FIndex = FIndex
		#self.BIndex = BIndex
		self.SIndex = SIndex #FIndex + BIndex
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
	'''
	def CalcAElements(self, CrSymbols, AnSymbols, NormalOrder, OrbitalListNoT, FixedCrS = None, FixedAnS = None, Case = 'MF'):
		SymbolsS, SymbolsE = GenerateSubspaceCases(CrSymbols, AnSymbols, FixedCrA = FixedCrS, FixedAnA = FixedAnS)
		AElement = 0.0
		if len(OrbitalListNoT) == 0 and Case == 'MF':
			return 1.0
		for n in range(len(SymbolsS)):
			if Case == 'MF':
				ExpSE = self.CalcExpValue(SymbolsS[n], SymbolsE[n], NormalOrder, OrbitalListNoT)
				AElement += ExpSE
				continue
			PosS, PosE = CaseToOperatorPositions(SymbolsS[n], SymbolsE[n], NormalOrder)
			Sign = SignOfSeparation(PosE[0] + PosE[1])
			ConS, PorQS, SignsS = WickContraction(PosS[0], PosS[1])
			ConE, PorQE, SignsE = WickContraction(PosE[0], PosE[1])
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
			tIndices, uIndices, vIndices, wIndices = GetAmplitudeIndices(SymbolsS1[n])
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
			rIndices, sIndices = self.getRSIndices(SymbolsS2[n])
			tIndices, uIndices, vIndices, wIndices = GetAmplitudeIndices(SymbolsS2[n])
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
		Y[0] = self.CalcYElements([])
		for i in self.SIndex:
			for j in self.SIndex:
				ij = self.CombinedIndex([i, j])
				Y[ij] = self.CalcYElement([i, i, j, j])
				for k in self.SIndex:
					for l in self.SIndex:
						ijkl = self.CombinedIndex([i, j, k, l])
						Y[ijkl] = self.CalcYElement([i, i, j, j, k, k, l, l])
		return Y

	def CalcH(self):
		A = self.CalcA()
		Y = self.CalcY()
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
	N = 10
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
	mo_occ = mo_coeff[:, :5]
	mo_occ = np.dot(StoOrth.T, mo_occ)
	P = np.dot(mo_occ, mo_occ.T)
	Nf = 1
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
	EIndex = list(range(PEnv.shape[0]))
	
	tZero = np.zeros((Norb, Norb, Norb, Norb))
	testMFBath = MP2Bath(tZero, SIndex, EIndex, PSch, PEnv, hSO, VSO)
	mf0, mf1, mf2 = testMFBath.CalcH()
	mf0.tofile("mf0")
	mf1.tofile("mf1")
	mf2.tofile("mf2")

	myMP2Bath = MP2Bath(tSO, SIndex, EIndex, PSch, PEnv, hSO, VSO)
	H0, H1, H2 = myMP2Bath.CalcH()
	H0.tofile("H0")
	H1.tofile("H1")
	H2.tofile("H2")
