import numpy as np
from scipy.optimize import newton
import math

def CheckSymmetry(ERI):
	def CheckCong(V):
		n = V.shape[0]
		for i in range(n):
			for j in range(n):
				V1 = V[i, j]
				V2 = V1.T
				T1 = np.isclose(V1, V2).all()
				V1 = V[j, i]
				T2 = np.isclose(V1, V2).all()

				V1 = V[:, :, i, j]
				V2 = V1.T
				T3 = np.isclose(V1, V2).all()
				V1 = V[:, :, j, i]
				T4 = np.isclose(V1, V2).all()
				if not T1 or not T2 or not T3 or not T4:
					return False
		return True
	Sym = CheckCong(ERI)
	if not Sym:
		return Sym
	ERI2 = np.swapaxes(ERI, 0, 2)
	ERI2 = np.swapaxes(ERI2, 1, 3)
	Sym = CheckCong(ERI2)
	if Sym:
		Sym = CheckSymmetryTwoFold(ERI)
	return Sym

def CheckSymmetryTwoFold(ERI):
	for i in range(ERI.shape[0]):
		for j in range(ERI.shape[1]):
			for k in range(ERI.shape[2]):
				for l in range(ERI.shape[3]):
					if not np.isclose(ERI[i, j, k, l], ERI[k, l, i, j]):
						print(i, j, k, l)
						return False
	return True
	

def MP2Corr(V, g, nOcc):
	ECorr = 0.0
	nVir = V.shape[0] - nOcc
	for i in range(nOcc):
		for j in range(nOcc):
			for a in range(nVir):
				for b in range(nVir):
					ECorr += (2. * V[i, nOcc + a, j, nOcc + b] - V[i, nOcc + b, j, nOcc + a]) * V[i, nOcc + a, j, nOcc + b]# / (g[nOcc + a] + g[nOcc + b] - g[i] - g[j])
	return ECorr

def PartialMP2Corr(V):
	ECorr = 0.0
	n = V.shape[0]
	for i in range(n):
		for j in range(n):
			for a in range(n):
				for b in range(n):
					ECorr += (2. * V[i, a, j, b] - V[i, b, j, a]) * V[i, a, j, b]
	return ECorr

def FragMP2Corr(V, FIndices):
	ECorr = 0.0
	n = V.shape[0]
	print(FIndices)
	for i in FIndices:
		for j in range(n):
			for a in range(n):
				for b in range(n):
					ECorr += (2. * V[i, a, j, b] - V[i, b, j, a]) * V[i, a, j, b]
					#ECorr += V[i, a, j, b]**2.
	return ECorr

def DoubleDoink(V):
	DD = 0.0
	for i in range(V.shape[0]):
		DD += V[i, i, i, i]**2.
	return DD

def FragH(h, FIndicies):
	E = 0.0
	n = h.shape[1]
	for i in FIndices:
		for j in range(n):
			E += h[i, j]**2.
	return E

def SumSqMatrix(V):
	return np.square(V).sum()

def CrossSqMatrix(V1, V2):
	s = 0.0
	for i in range(V1.shape[0]):
		for a in range(V1.shape[1]):
			for j in range(V1.shape[2]):
				for b in range(V1.shape[3]):
					s += V1[i, a, j, b] * V2[i, b, j, a]
	return s


def CalcFragEnergy(h, hcore, V, P, G, CenterIndices):
	FragE = 0.0
	n = h.shape[0]
	E1 = 0.0
	E2 = 0.0
	for i in CenterIndices:
		for j in range(n):
			FragE += 0.5 * (h[i, j] + hcore[i, j]) * P[i, j]
			E1 += 0.5 * (h[i, j] + hcore[i, j]) * P[i, j]
			for k in range(n):
				for l in range(n):
					FragE += 0.5 * G[i, j, k, l] * V[i, j, k, l]
					E2 += 0.5 * G[i, j, k, l] * V[i, j, k, l]
	return FragE


def FragmentRMP2(h, hcore, V, CenterIndices, S = None):
	if S is None:
		S = np.eye(h.shape[0])
	mol = gto.M()
	n = h.shape[0]
	mol.nelectron = n

	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: hcore
	mf.get_ovlp = lambda *args: S
	mf._eri = ao2mo.restore(8, V, n)
	mf.max_cycle = 1000

	mf.kernel()
	C = mf.mo_coeff
	
	mp2 = mp.MP2(mf)
	E, T2 = mp2.kernel()
	P = mp2.make_rdm1()
	P = C @ P @ C.T
	print(P)
	G = mp2.make_rdm2()
	G = np.einsum('ijkl,ip,jq,kr,ls->pqrs', G, C, C, C, C)
	FragE = CalcFragEnergy(h, hcore, V, P, G, CenterIndices)
	return FragE

def CustomRMP2(h, V, S = None):
	if S is None:
		S = np.eye(h.shape[0])
	mol = gto.M()
	n = h.shape[0]
	mol.nelectron = n

	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: h
	mf.get_ovlp = lambda *args: S
	mf._eri = ao2mo.restore(8, V, n)
	mf.max_cycle = 1000

	mf.kernel()
	C = mf.mo_coeff
	
	mp2 = mp.MP2(mf)
	E, T2 = mp2.kernel()
	print(E, mf.e_tot)

def CompileFullh(hFF, hFB, hBB, Index = None):
	nF = hFF.shape[0]
	h = np.zeros((2 * nF, 2 * nF))

	if Index is None:
		Fs = list(range(nF))
		Bs = list(range(nF, 2 * nF))
	else:
		Fs = Index[0]
		Bs = Index[1]

	h[np.ix_(Fs, Fs)] = hFF
	h[np.ix_(Fs, Bs)] = hFB
	h[np.ix_(Bs, Fs)] = hFB.T
	h[np.ix_(Bs, Bs)] = hBB

	return h

def CompileFullV(VFFFF, VFFFB, VFFBB, VFBFB, VFBBB, VBBBB):
	nF = VFFFF.shape[0]
	nBE = VBBBB.shape[0]
	n = nF + nBE
	V = np.zeros((n, n, n, n))
	Fs = list(range(nF))
	Bs = list(range(nF, n))
	
	V[np.ix_(Fs, Fs, Fs, Fs)] = VFFFF

	V[np.ix_(Fs, Fs, Fs, Bs)] = VFFFB
	V[np.ix_(Fs, Fs, Bs, Fs)] = np.swapaxes(VFFFB, 2, 3)
	VFBFF = np.swapaxes(VFFFB, 1, 3)
	VFBFF = np.swapaxes(VFBFF, 0, 2)
	V[np.ix_(Fs, Bs, Fs, Fs)] = VFBFF
	V[np.ix_(Bs, Fs, Fs, Fs)] = np.swapaxes(VFBFF, 0, 1)

	V[np.ix_(Fs, Fs, Bs, Bs)] = VFFBB
	VBBFF = np.swapaxes(VFFBB, 0, 2)
	VBBFF = np.swapaxes(VBBFF, 1, 3)
	V[np.ix_(Bs, Bs, Fs, Fs)] = VBBFF

	V[np.ix_(Fs, Bs, Fs, Bs)] = VFBFB
	V[np.ix_(Bs, Fs, Fs, Bs)] = np.swapaxes(VFBFB, 0, 1)
	V[np.ix_(Fs, Bs, Bs, Fs)] = np.swapaxes(VFBFB, 2, 3)
	VBFBF = np.swapaxes(VFBFB, 0, 1)
	VBFBF = np.swapaxes(VBFBF, 2, 3)
	V[np.ix_(Bs, Fs, Bs, Fs)] = VBFBF

	V[np.ix_(Fs, Bs, Bs, Bs)] = VFBBB
	V[np.ix_(Bs, Fs, Bs, Bs)] = np.swapaxes(VFBBB, 0, 1)
	VBBFB = np.swapaxes(VFBBB, 0, 2)
	VBBFB = np.swapaxes(VBBFB, 1, 3)
	V[np.ix_(Bs, Bs, Fs, Bs)] = VBBFB
	V[np.ix_(Bs, Bs, Bs, Fs)] = np.swapaxes(VBBFB, 2, 3)

	V[np.ix_(Bs, Bs, Bs, Bs)] = VBBBB

	return V

def BuildVTwoFold(VFFFF, VFFFB, VFFBF, VFFBB, VFBBF, VFBFB, VBFBF, VFBBB, VBFBB, VBBBB):
	nF = VFFFF.shape[0]
	nBE = VBBBB.shape[0]
	n = nF + nBE
	V = np.zeros((n, n, n, n))
	Fs = list(range(nF))
	Bs = list(range(nF, n))
	
	V[np.ix_(Fs, Fs, Fs, Fs)] = VFFFF

	V[np.ix_(Fs, Fs, Fs, Bs)] = VFFFB
	V[np.ix_(Fs, Fs, Bs, Fs)] = VFFBF
	VFBFF = np.swapaxes(VFFFB, 1, 3)
	VFBFF = np.swapaxes(VFBFF, 0, 2)
	V[np.ix_(Fs, Bs, Fs, Fs)] = VFBFF
	VBFFF = np.swapaxes(VFFBF, 1, 3)
	VBFFF = np.swapaxes(VBFFF, 0, 2)
	V[np.ix_(Bs, Fs, Fs, Fs)] = VBFFF

	V[np.ix_(Fs, Fs, Bs, Bs)] = VFFBB
	VBBFF = np.swapaxes(VFFBB, 0, 2)
	VBBFF = np.swapaxes(VBBFF, 1, 3)
	V[np.ix_(Bs, Bs, Fs, Fs)] = VBBFF

	V[np.ix_(Fs, Bs, Fs, Bs)] = VFBFB

	V[np.ix_(Fs, Bs, Bs, Fs)] = VFBBF
	VBFFB = np.swapaxes(VFBBF, 1, 3)
	VBFFB = np.swapaxes(VBFFB, 0, 2)
	V[np.ix_(Bs, Fs, Fs, Bs)] = VBFFB

	V[np.ix_(Bs, Fs, Bs, Fs)] = VBFBF

	V[np.ix_(Fs, Bs, Bs, Bs)] = VFBBB
	V[np.ix_(Bs, Fs, Bs, Bs)] = VBFBB
	VBBFB = np.swapaxes(VFBBB, 0, 2)
	VBBFB = np.swapaxes(VBBFB, 1, 3)
	V[np.ix_(Bs, Bs, Fs, Bs)] = VBBFB
	VBBBF = np.swapaxes(VBFBB, 0, 2)
	VBBBF = np.swapaxes(VBBBF, 1, 3)
	V[np.ix_(Bs, Bs, Bs, Fs)] = VBBBF

	V[np.ix_(Bs, Bs, Bs, Bs)] = VBBBB

	return V

def SVDOEI(h):
	nF = h.shape[0]
	U, S, T = np.linalg.svd(h)
	print(T)
	B = h @ T.T
	print(B)
	return B[:, :nF]

# Assumes V is given as VFFFA in chemist notation
def OneExternal(V, ReturnFull = False):
	nF = V.shape[0]
	B = V.copy()
	for i in range(nF):
		for j in range(nF):
			U, S, T = np.linalg.svd(V[i, j])
			B[i, j] = V[i, j] @ T.T
			print(i, j, "\n", B[i, j])
	if ReturnFull:
		return B, T
	return B[:, :, :, :nF], T

def ReshapeTwo(V):
	if len(V.shape) == 4:
		nA = V.shape[0]
		nB = V.shape[2]
		VExtended = V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
		VCondense = np.zeros((int(nA * (nA + 1) / 2), int(nB * (nB + 1) / 2)))
		ij = -1
		for i in range(nA):
			for j in range(i, nA):
				ij += 1
				kl = -1
				for k in range(nB):
					for l in range(k, nB):
						kl += 1
						VCondense[ij, kl] = V[i, j, k, l]
		return VCondense
	else:
		UA = V.shape[0]
		UB = V.shape[1]
		nA = int((-1 + math.sqrt(1 + 8 * UA)) / 2.0)
		nB = int((-1 + math.sqrt(1 + 8 * UB)) / 2.0)
		VExpand = np.zeros((nA, nA, nB, nB))
		ij = -1
		for i in range(nA):
			for j in range(i, nA):
				ij += 1
				kl = -1
				for k in range(nB):
					for l in range(k, nB):
						kl += 1
						VExpand[i, j, k, l] = V[ij, kl]
						VExpand[j, i, k, l] = V[ij, kl]
						VExpand[i, j, l, k] = V[ij, kl]
						VExpand[j, i, l, k] = V[ij, kl]
		return VExpand

def TwoExternal(V, VbbAA = None, ReturnFull = False, UpperTriangleSVD = False, ReturnT = False):
	OrigDim = V.shape
	nF = OrigDim[0]
	if UpperTriangleSVD:
		VExtended = ReshapeTwo(V)
	else:
		VExtended = V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
	U, S, T = np.linalg.svd(VExtended)
	VExtended = VExtended @ T.T
	if ReturnFull:
		return ReshapeTwo(VExtended) #VExtended.reshape(OrigDim)
	print(VExtended.shape)
	VExtended = VExtended[:, :(S.shape[0])] #[:, :(nF * nF)]
	if VbbAA is not None:
		#Idx = list(range(nF)) + list(range(V.shape[2], V.shape[2] + nF))
		TCut = T[:(S.shape[0]), :] #T[Idx, :]
		VBathExtended = ReshapeTwo(VbbAA) #VbbAA.reshape(VbbAA.shape[0] * VbbAA.shape[1], VbbAA.shape[2] * VbbAA.shape[3])
		VBathExtended = VBathExtended @ TCut.T
		#print(VBathExtended)
		UB, SB, TB = np.linalg.svd(VBathExtended)
		VBathExtended = VBathExtended @ TB.T
		#print(VBathExtended)
		VExtended = VExtended @ TB.T
		return ReshapeTwo(VExtended) #VExtended.reshape(nF, nF, nF, nF)
	if UpperTriangleSVD:
		if ReturnT:
			return ReshapeTwo(VExtended), T
		return ReshapeTwo(VExtended)
	else:
		if ReturnT:
			return VExtended.reshape(nF, nF, nF, nF), T
		return VExtended.reshape(nF, nF, nF, nF)

def MakeVFFBB(VFFAA):
	nF = VFFAA.shape[0]
	nBE = VFFAA.shape[2]

	# These are the transformation matrices, one for fixeg fp and one for fixed gq in (fg|pq)
	Tfp = np.zeros((nF, nBE, nBE, nF)) # Index: fpqy
	Tgq = np.zeros((nF, nBE, nBE, nF)) # Index: gqpx

	for f in range(nF):
		for p in range(nBE):
			Vpq = VFFAA[f, :, p, :]
			U, S, T = np.linalg.svd(Vpq)
			Tfp[f, p] = T.T[:, :nF]
			Vpq = VFFAA[:, f, :, p]
			U, S, T = np.linalg.svd(Vpq)
			Tgq[f, p] = T.T[:, :nF]
	VFFBB = np.einsum('fgpq,fpqy,gqpx->fgxy', VFFAA, Tfp, Tgq)
	return VFFBB

def MakeVFFBB_PostSVD(VFFAA):
	VFFBB = TwoExternal(VFFAA, UpperTriangleSVD = False)
	for f in range(VFFBB.shape[0]):
		for g in range(VFFBB.shape[1]):
			U, S, T = np.linalg.svd(VFFBB[f, g])
			VFFBB[f, g] = np.zeros(VFFBB[f, g].shape)
			np.fill_diagonal(VFFBB[f, g], S)
	return VFFBB
			

def ThreeExternalOld(V, ReturnFull = False):
	OrigDim = V.shape
	nF = OrigDim[0]
	VExtended = V.reshape(V.shape[0], V.shape[1] * V.shape[2] * V.shape[3])
	U, S, T = np.linalg.svd(VExtended)
	VExtended = VExtended @ T.T
	Idx = list(range(nF))
	if ReturnFull:
		return VExtended.reshape(OrigDim)
	return VExtended.reshape(OrigDim)[np.ix_(Idx, Idx, Idx, Idx)]

def ThreeExternal(V, ReturnFull = False):
	#VFABB = TwoExternal(V, ReturnFull = ReturnFull)
	#VFABB = ThreeExternalOld(V, ReturnFull = True)

	# Turn VFAAA into VFBAA
	VFBAA = V.copy() #np.zeros((V.shape[0], V.shape[0], V.shape[2], V.shape[3]))
	for p in range(V.shape[2]):
		for q in range(V.shape[3]):
			A = V[:, :, p, q]
			U, S, T = np.linalg.svd(A)
			B = A @ T.T
			VFBAA[:, :, p, q] = B
	VFBAA = VFBAA[:, :(V.shape[0]), :, :]

	VFBBB = MakeVFFBB(VFBAA) #TwoExternal(VFBAA, ReturnFull = ReturnFull)
	return VFBBB

if __name__ == '__main__':
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
	#	if i == 1:
	#		angle = angle + 0.8 * angle
		mol.atom.append(('H', (r * np.sin(angle), r * np.cos(angle), 0)))
	mol.basis = 'sto-3g'
	mol.build()
	mf = scf.RHF(mol)
	mf.kernel()
	print("E_elec(HF) =", mf.e_tot - mf.energy_nuc())

	nocc = int(np.sum(mf.mo_occ) / 2)

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
	CoreOrbs = [i for i, v in enumerate(eEnv) if v > 1.0 - thresh]
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

	FIndices = list(range(Nf)) # In the Schmidt space
	BIndices = list(range(Nf, 2 * Nf)) # In the Schmidt space
	BEIndices = list(range(Nf, N))
	SIndices = FIndices + BIndices
	CoreIdx = [i + Nf for i in CoreOrbs]

	TTotal = np.dot(StoOrig, T)
	TMOtoAO = np.linalg.inv(mo_coeff)
	TMOtoSO = np.dot(TMOtoAO, TTotal)
	TFragMOtoSO = TMOtoSO[:, FIndices]
	TFragOccMOtoSO = TFragMOtoSO[:nocc, :]
	TFragVirMOtoSO = TFragMOtoSO[nocc:, :]

	hSO = reduce(np.dot, (TTotal.T, mf.get_hcore(), TTotal))
	VSO = ao2mo.kernel(mol, TTotal)
	VSO = ao2mo.restore(1, VSO, hSO.shape[0])
	hMO = reduce(np.dot, (mo_coeff.T, mf.get_hcore(), mo_coeff))
	VMO = ao2mo.kernel(mol, mo_coeff) #np.eye(TTotal.shape[0]))
	VMO = ao2mo.restore(1, VMO, hMO.shape[0])
	hLO = reduce(np.dot, (StoOrig.T, mf.get_hcore(), StoOrig))
	VLO = ao2mo.kernel(mol, StoOrig)
	VLO = ao2mo.restore(1, VLO, hLO.shape[0])
	hAO = mf.get_hcore()
	VAO = ao2mo.kernel(mol, np.eye(StoOrig.shape[0]))
	VAO = ao2mo.restore(1, VAO, hAO.shape[0])

	VFrag = VSO[SIndices, :, :, :][:, SIndices, :, :][:, :, SIndices, :][:, :, :, SIndices]
	hFrag = hSO[SIndices, :][:, SIndices]
	hNoCore = hFrag.copy()
	for i in SIndices:
		for j in SIndices:
			CoreContribution = 0.0
			for c in CoreIdx:
				CoreContribution += (2.0 * VSO[i, j, c, c] - VSO[i, c, c, j])
			hFrag[i, j] += CoreContribution

	mp2 = mp.MP2(mf)
	E, T2 = mp2.kernel()
	print("E_elec(MP2) =", mf.e_tot + E - mf.energy_nuc())
	EMP2 = mf.e_tot + E - mf.energy_nuc()	

	Nocc = T2.shape[0]
	Nvir = T2.shape[2]
	Norb = Nocc + Nvir
	print(T2.shape)
	t2 = np.zeros((Nocc, Nocc, Nvir, Nvir))
	for i in range(Nocc):
		for j in range(Nocc):
			for a in range(Nvir):
				for b in range(Nvir):
					t2[i, j, a, b] = 2.0 * T2[i, j, a, b] - T2[i, j, b, a]

	TMOtoLO = np.dot(TMOtoAO, StoOrig)
	RDM2 = mp2.make_rdm2()
	RDM2LO = np.einsum('ijkl,ip,jq,kr,ls->pqrs', RDM2, TMOtoLO, TMOtoLO, TMOtoLO, TMOtoLO)
	np.save("2RDM", RDM2LO)

	ETestCorr = MP2Corr(VMO, mf.mo_energy, int(N / 2))
	print("mp2 partial corr", ETestCorr)
	VMO_OVOV = np.zeros(VMO.shape)
	TMO = np.zeros(VMO.shape)
	tMO = np.zeros(VMO.shape)
	for i in range(Nocc):
		for j in range(Nocc):
			for a in range(Nvir):
				for b in range(Nvir):
					VMO_OVOV[i, Nocc + a, j, Nocc + b] = VMO[i, Nocc + a, j, Nocc + b]
					TMO[i, j, Nocc + a, Nocc + b] = T2[i, j, a, b]
					tMO[i, j, Nocc + a, Nocc + b] = t2[i, j, a, b]
	VLO_OVOV = VLO #np.einsum('ijkl,ip,jq,kr,ls->pqrs', VMO_OVOV, TMOtoLO, TMOtoLO, TMOtoLO, TMOtoLO)
	TLO = np.einsum('ijab,ip,jq,ar,bs->pqrs', TMO, TMOtoLO, TMOtoLO, TMOtoLO, TMOtoLO)
	tLO = np.einsum('ijab,ip,jq,ar,bs->pqrs', tMO, TMOtoLO, TMOtoLO, TMOtoLO, TMOtoLO)
	np.save("TLO", TLO)
	np.save("tLO", tLO)
	ETestCorr = PartialMP2Corr(VLO_OVOV)
	print("test corr", ETestCorr)

	sym = CheckSymmetryTwoFold(VLO_OVOV)
	print(sym)

	VFFFA = VLO_OVOV[np.ix_(FIndices, FIndices, FIndices, BEIndices)] #VLO[FIndices, :, :, :][:, FIndices, :, :][:, :, FIndices, :][:, :, :, :]
	VFFFB, TOneIdx = OneExternal(VFFFA)
	VFFAA = VLO_OVOV[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]#[FIndices, :, :, :][:, FIndices, :, :][:, :, BEIndices, :][:, :, :, BEIndices]
	print(VFFAA.shape)
	np.save("VFFAA", VFFAA)
	#VbbAA = np.einsum('ijkl,ip,jq->pqkl', VLO, TBath, TBath)
	#VbbAA = VbbAA[np.ix_(list(range(Nf)), list(range(Nf)), BEIndices, BEIndices)]
	VFFBB = MakeVFFBB(VFFAA) #TwoExternal(VFFAA)
	print(VFFBB)
	#print((VFFAA*VFFAA).sum(), (VFFBB*VFFBB).sum())
	VFAFA = VLO_OVOV[np.ix_(FIndices, BEIndices, FIndices, BEIndices)]
	#VbAbA = np.einsum('ijkl,ip,kq->pjql', VLO, TBath, TBath)
	#VbAbA = VbAbA[np.ix_(list(range(Nf)), BEIndices, list(range(Nf)), BEIndices)]
	VFAFA_Phys = np.swapaxes(VFAFA, 1, 2)
	#VbAbA_Phys = np.swapaxes(VbAbA, 1, 2)
	VFBFB_Phys = MakeVFFBB(VFAFA_Phys) #TwoExternal(VFAFA_Phys)
	VFBFB = np.swapaxes(VFBFB_Phys, 1, 2)
	#print(VFBFB)
	VFAAA = VLO_OVOV[np.ix_(FIndices, BEIndices, BEIndices, BEIndices)]
	VFBBB = ThreeExternal(VFAAA)
	#print(VFBBB)

	VFFFF = VLO_OVOV[np.ix_(FIndices, FIndices, FIndices, FIndices)]
	VBBBB = VFrag[np.ix_(BIndices, BIndices, BIndices, BIndices)]

	#VFFBB = VFrag[np.ix_(FIndices, FIndices, BIndices, BIndices)]
	#VFBFB = VFrag[np.ix_(FIndices, BIndices, FIndices, BIndices)]
	#VFBBB = VFrag[np.ix_(FIndices, BIndices, BIndices, BIndices)]
	#VFFFB = VFrag[np.ix_(FIndices, FIndices, FIndices, BIndices)]

	#VFFFF = np.zeros(VFFFF.shape)
	#VFFFB = np.zeros(VFFFB.shape)
	#VFFBB = np.zeros(VFFBB.shape)
	#VFBBB = np.zeros(VFBBB.shape)
	#VBBBB = np.zeros(VBBBB.shape)

	VDO = CompileFullV(VFFFF, VFFFB, VFFBB, VFBFB, VFBBB, VBBBB)
	Sym = CheckSymmetry(VDO)
	print("sym", Sym)

	# These are not formed from 2 fold symmetry
	VFFAF = VLO_OVOV[np.ix_(FIndices, FIndices, BEIndices, FIndices)]
	VFFAF_FFFA = np.swapaxes(VFFAF, 2, 3)
	VFFBF_FFFB, TOneIdx = OneExternal(VFFAF_FFFA)
	VFFBF = np.swapaxes(VFFBF_FFFB, 2, 3)

	VFAAF = VLO_OVOV[np.ix_(FIndices, BEIndices, BEIndices, FIndices)]
	VFAAF_FAFA_FFAA = np.swapaxes(VFAAF, 2, 3)
	VFAAF_FAFA_FFAA = np.swapaxes(VFAAF_FAFA_FFAA, 1, 2)
	VFBBF_FBFB_FFBB = TwoExternal(VFAAF_FAFA_FFAA)
	VFBBF = np.swapaxes(VFBBF_FBFB_FFBB, 1, 2)
	VFBBF = np.swapaxes(VFBBF, 2, 3)

	VAFAF = VLO_OVOV[np.ix_(BEIndices, FIndices, BEIndices, FIndices)]
	VAFAF_FAFA = np.swapaxes(VAFAF, 0, 1)
	VAFAF_FAFA = np.swapaxes(VAFAF_FAFA, 2, 3)
	VAFAF_FAFA_FFAA = np.swapaxes(VAFAF_FAFA, 1, 2)
	VBFBF_FBFB_FFBB = MakeVFFBB(VAFAF_FAFA_FFAA) #TwoExternal(VAFAF_FAFA_FFAA)
	VBFBF_FBFB = np.swapaxes(VBFBF_FBFB_FFBB, 1, 2)
	VBFBF = np.swapaxes(VBFBF_FBFB, 0, 1)
	VBFBF = np.swapaxes(VBFBF, 2, 3)

	VAFAA = VLO_OVOV[np.ix_(BEIndices, FIndices, BEIndices, BEIndices)]
	VAFAA_FAAA = np.swapaxes(VAFAA, 0, 1)
	VBFBB_FBBB = ThreeExternal(VAFAA_FAAA)
	VBFBB = np.swapaxes(VBFBB_FBBB, 0, 1)

	VDO = BuildVTwoFold(VFFFF, VFFFB, VFFBF, VFFBB, VFBBF, VFBFB, VBFBF, VFBBB, VBFBB, VBBBB)
	Sym = CheckSymmetryTwoFold(VDO)
	print("sym", Sym)

	hFA = hLO[np.ix_(FIndices, BEIndices)]
	hFB = SVDOEI(hFA)
	#print(hFB)

	hFF = hLO[np.ix_(FIndices, FIndices)]
	hBB = hLO[np.ix_(BIndices, BIndices)]
	hDO = CompileFullh(hFF, hFB, hBB)

	ELO = FragMP2Corr(VLO_OVOV, FIndices)
	EDO = FragMP2Corr(VDO, FIndices)
	#ESO = FragMP2Corr(VSO, FIndices)
	#EFr = FragMP2Corr(VFrag, FIndices)
	print("corr calc", ELO, EDO)
	print("sq V", (VLO_OVOV*VLO_OVOV).sum(), (VDO*VDO).sum())

	#VLOFFFB = VLO[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]
	#VSOFFFB = VSO[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]
	#VFragFFFB = VLO[np.ix_(FIndices, FIndices, BIndices, BIndices)]

	#V2LO = SumSqMatrix(VLOFFFB)
	#V2SO = SumSqMatrix(VSOFFFB)
	#V2Frag = SumSqMatrix(VFragFFFB)
	#V2DO = SumSqMatrix(VFFBB)
	#print("v2 calc", V2LO, V2DO, V2SO, V2Frag)
	#print(np.sqrt(V2DO/V2LO))

	#VLOFFFB = VLO[np.ix_(FIndices, BEIndices, BEIndices, BEIndices)]
	#VSOFFFB = VSO[np.ix_(FIndices, BEIndices, BEIndices, BEIndices)]
	#VFragFFFB = VLO[np.ix_(FIndices, BIndices, BIndices, BIndices)]

	#V2LO = SumSqMatrix(VLOFFFB)
	#V2SO = SumSqMatrix(VSOFFFB)
	#V2Frag = SumSqMatrix(VFragFFFB)
	#V2DO = SumSqMatrix(VFBBB)
	#print("v2 calc", V2LO, V2DO, V2SO, V2Frag)
	#print(np.sqrt(V2DO/V2LO))

	print("FFFF")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(FIndices, FIndices, FIndices, FIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VFFFF)
	print("v2 calc", V2LO, V2DO)
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX)
	VXDO = CrossSqMatrix(VFFFF, VFFFF)
	print("vx calc", VXLO, VXDO)

	print("FFFB")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(FIndices, FIndices, FIndices, BEIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VFFFB)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(FIndices, BEIndices, FIndices, FIndices)]
	VFBFF = np.swapaxes(VFFFB, 1, 3)
	VFBFF = np.swapaxes(VFBFF, 0, 2)
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VXDO = CrossSqMatrix(VFFFB, VFBFF)
	print("vx calc", VXLO, VXDO)

	print("FFBB")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VFFBB)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(FIndices, BEIndices, BEIndices, FIndices)]
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VXDO = CrossSqMatrix(VFFBB, VFBBF)
	print("vx calc", VXLO, VXDO)


	print("FBFB")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(FIndices, BEIndices, FIndices, BEIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VFBFB)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(FIndices, BEIndices, FIndices, BEIndices)]
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VXDO = CrossSqMatrix(VFBFB, VFBFB)
	print("vx calc", VXLO, VXDO)


	print("FBBB")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(FIndices, BEIndices, BEIndices, BEIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VFBBB)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(FIndices, BEIndices, BEIndices, BEIndices)]
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VXDO = CrossSqMatrix(VFBBB, VFBBB)
	print("vx calc", VXLO, VXDO)


	print("FFBF")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(FIndices, FIndices, BEIndices, FIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VFFBF)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(FIndices, FIndices, BEIndices, FIndices)]
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VXDO = CrossSqMatrix(VFFBF, VFFBF)
	print("vx calc", VXLO, VXDO)


	print("FBBF")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(FIndices, BEIndices, BEIndices, FIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VFBBF)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VXDO = CrossSqMatrix(VFBBF, VFFBB)
	print("vx calc", VXLO, VXDO)


	print("BFBF")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(BEIndices, FIndices, BEIndices, FIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VBFBF)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(BEIndices, FIndices, BEIndices, FIndices)]
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VXDO = CrossSqMatrix(VBFBF, VBFBF)
	print("vx calc", VXLO, VXDO)


	print("BFBB")
	VLO_OVOV_XXXX = VLO_OVOV[np.ix_(BEIndices, FIndices, BEIndices, BEIndices)]
	V2LO = SumSqMatrix(VLO_OVOV_XXXX)
	V2DO = SumSqMatrix(VBFBB)
	print("v2 calc", V2LO, V2DO)
	VLO_OVOV_XXXX2 = VLO_OVOV[np.ix_(BEIndices, BEIndices, BEIndices, FIndices)]
	VXLO = CrossSqMatrix(VLO_OVOV_XXXX, VLO_OVOV_XXXX2)
	VBBBF = np.swapaxes(VBFBB, 0, 2)
	VBBBF = np.swapaxes(VBBBF, 1, 3)
	VXDO = CrossSqMatrix(VBFBB, VBBBF)
	print("vx calc", VXLO, VXDO)




	#eLO = FragH(hLO, FIndices)
	#eDO = FragH(hDO, FIndices)
	#eSO = FragH(hSO, FIndices)
	#print("h calc:", eLO, eDO, eSO)

	#h0 = np.zeros(hLO.shape)
	#CustomRMP2(h0, VLO)
	#h0 = np.zeros((2 * Nf, 2 * Nf))
	##from mp2_ml import FragmentRFCI
	#EFrag = FragmentRMP2(h0, h0, VTest, FIndices)
	#print(EFrag)
	#print(EFrag * N / Nf)
