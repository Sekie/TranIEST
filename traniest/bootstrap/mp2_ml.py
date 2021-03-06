import numpy as np
from scipy.optimize import newton

def GetFock(h, V):
	J = np.einsum('iijj->ij', V)
	K = np.einsum('ijij->ij', V)
	F = h + 2.0 * J - K
	return F

def GetG(F):
	e, v = np.linalg.eigh(F)
	return e

def PermuteABBB(V, AIndex, BIndex):
	VABBB = V[np.ix_(AIndex, BIndex, BIndex, BIndex)]
	V[np.ix_(BIndex, AIndex, BIndex, BIndex)] = VABBB
	V[np.ix_(BIndex, BIndex, AIndex, BIndex)] = VABBB
	V[np.ix_(BIndex, BIndex, BIndex, AIndex)] = VABBB

def PermuteABAB(V, AIndex, BIndex):
	VABAB = V[np.ix_(AIndex, BIndex, AIndex, BIndex)]
	V[np.ix_(BIndex, AIndex, BIndex, AIndex)] = VABAB
	V[np.ix_(BIndex, AIndex, AIndex, BIndex)] = VABAB
	V[np.ix_(AIndex, BIndex, BIndex, AIndex)] = VABAB

def PermuteAABB(V, AIndex, BIndex):
	VAABB = V[np.ix_(AIndex, AIndex, BIndex, BIndex)]
	V[np.ix_(BIndex, BIndex, AIndex, AIndex)] = VAABB

def make_rhf_rdm2(P):
	n = P.shape[0]
	G = np.zeros((n, n, n, n))
	for i in range(n):
		for j in range(n):
			G[i, j] = (2.0 * P[i, j] * P - np.outer(P[i], P[j]))[:n, :n]
			G[i, j][np.diag_indices(n)] *= 0.5
			G[i, j] += G[i, j].T
	return G

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


def FragmentRFCI(h, hcore, V, CenterIndices, S = None):
	from pyscf import fci
	n = hcore.shape[0]
	nocc = int(n/2)
	cisolver = fci.direct_spin0.FCI()
	cisolver.verbose = 5
	e, fcivec = cisolver.kernel(hcore, V, hcore.shape[0], (nocc, nocc))
	P, G = cisolver.make_rdm12(fcivec, n, (nocc, nocc))
	FragE = CalcFragEnergy(h, hcore, V, P, G, CenterIndices)
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
	G = mp2.make_rdm2()
	G = np.einsum('ijkl,ip,jq,kr,ls->pqrs', G, C, C, C, C)
	FragE = CalcFragEnergy(h, hcore, V, P, G, CenterIndices)
	return FragE


def FragmentRHF(hEff, VEff, FIndices, ReturnMO = False):
	mol = gto.M()
	nElec = hEff.shape[0] # Half occupied fragment
	mol.nelectron = nElec

	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: hEff
	mf.get_ovlp = lambda *args: np.eye(nElec)
	mf._eri = ao2mo.restore(8, VEff, nElec)
	mf.max_cycle = 1000

	#from frankenstein.sgscf.sgopt import doRCA
	mf.kernel()
	#try:
	#	mf.kernel()
	#except:
	#	mf.diis = False
	#	mf.kernel()
	#try:
	#	assert(mf.converged)
	#except:
	#	print("DIIS did not converge, trying RCA")
	#	e, P = doRCA(mf, rca_space = 2)
	#	mf.kernel(P)

	VMOEff = np.einsum('ijkl,ip,jq,kr,ls->pqrs', VEff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)
	TFrag = mf.mo_coeff.T[:, FIndices]
	if ReturnMO:
		return VMOEff, mf.mo_energy, TFrag, mf.mo_coeff
	return VMOEff, mf.mo_energy, TFrag

# Takes VEff in the fragment MO basis, and assumes half filling. Returns a four index array where the first two indices
# are the occupied orbitals and the last two indices are the virtual orbitals
def MaketEff(VEff, g):
	nOcc = int(g.shape[0] / 2)
	n = g.shape[0]
	tEff = np.zeros((nOcc, nOcc, n - nOcc, n - nOcc))
	for i in range(nOcc):
		for j in range(nOcc):
			for a in range(n - nOcc):
				for b in range(n - nOcc):
					tEff[i, j, a, b] = (2.0 * VEff[i, nOcc + a, j, nOcc + b] - VEff[i, nOcc + b, j, nOcc + a]) / (g[i] + g[j] - g[nOcc + a] - g[nOcc + b])
	return tEff
	
def ERIToVector(V):
	n = V.shape[0]
	#hVec = h.reshape(n * n)
	VVec = V.reshape(n * n * n * n)
	#ERIVec = np.vstack((hVec, VVec))
	return VVec

def VectorToERI(VVec):
	#n2 = int(n * n)
	#hVec = ERIVec[:n2]
	#VVec = ERIVec[n2:]
	#h = hVec.reshape(n, n)
	n = int(VVec.shape[0]**0.25)
	V = VVec.reshape(n, n, n, n)
	return V

# Assumes Vector is fed in the order of BFFF, FBFB, FFBB, FBBB
def VectorToVSymm(VVec, FIndices, BIndices):
	nFB = 2 * len(FIndices)
	nVec1 = int(VVec.shape[0] / 4)
	nVec2 = 2 * nVec1
	nVec3 = 3 * nVec1
	VBFFF = VectorToERI(VVec[:nVec1])
	VFBFB = VectorToERI(VVec[nVec1:nVec2])
	VFFBB = VectorToERI(VVec[nVec2:nVec3])
	VFBBB = VectorToERI(VVec[nVec3:])
	V = np.zeros((nFB, nFB, nFB, nFB))
	V[np.ix_(BIndices, FIndices, FIndices, FIndices)] = VBFFF
	V[np.ix_(FIndices, BIndices, FIndices, BIndices)] = VFBFB
	V[np.ix_(FIndices, FIndices, BIndices, BIndices)] = VFFBB
	V[np.ix_(FIndices, BIndices, BIndices, BIndices)] = VFBBB
	PermuteABBB(V, BIndices, FIndices)
	PermuteABAB(V, FIndices, BIndices)
	PermuteAABB(V, FIndices, BIndices)
	PermuteABBB(V, FIndices, BIndices)
	return V

def dbgA(T, g, nOcc, nVir, SIndices):
	nFB = len(SIndices)
	TOcc = T[:nOcc, :]
	TVir = T[nOcc:, :]
	gTensor = np.zeros((nOcc, nOcc, nVir, nVir))
	for i in range(nOcc):
		for j in range(nOcc):
			for a in range(nVir):
				for b in range(nVir):
					gTensor[i, j, a, b] = 1.0 / (g[i] + g[j] - g[nOcc + a] - g[nOcc + b])
	A = np.zeros((nOcc, nOcc, nVir, nVir, nFB, nFB, nFB, nFB, nFB, nFB, nFB, nFB))
	for i in range(nOcc):
		for j in range(nOcc):
			for a in range(nVir):
				for b in range(nVir):
					for p in range(nFB):
						for q in range(nFB):
							for r in range(nFB):
								for s in range(nFB):
									for t in range(nFB):
										for u in range(nFB):
											for v in range(nFB):
												for w in range(nFB):
													tmp1 = np.einsum('cp,dq,cv,dw,ijcd->ijpqvw', TVir, TVir, TVir, TVir, gTensor)
													tmp2 = np.einsum('cp,dq,dv,cw,ijcd->ijpqvw', TVir, TVir, TVir, TVir, gTensor)
													A[i, j, a, b, p, q, r, s, t, u, v, w] = 2.0 * tmp1[i, j, p, q, v, w] * TVir[a, r] * TVir[b, s] * TOcc[i, t] * TOcc[j, u] - tmp2[i, j, p, q, v, w] * TVir[a, r] * TVir[b, s] * TOcc[i, t] * TOcc[j, u]
	print(A)
	input()
	return

def MakeLossVector(Losses):
	LossesVec = np.zeros(0)
	for Loss in Losses:
		n1 = Loss.shape[0]
		n2 = Loss.shape[2]
		Dim = n1 * n1 * n2 * n2
		LossVec = np.zeros(Dim)
		for i in range(Loss.shape[0]):
			for j in range(Loss.shape[1]):
				for k in range(Loss.shape[2]):
					for l in range(Loss.shape[3]):
						LossVec[i + j * Loss.shape[0] + k * Loss.shape[0] * Loss.shape[1] + l * Loss.shape[0] * Loss.shape[1] * Loss.shape[3]] = Loss[i, j, k, l]
		LossesVec = np.concatenate((LossesVec, LossVec))
	return LossesVec

# Antisymmetrizes TEI in chemists notation
def AntiSymV(V):
	V2 = np.swapaxes(V, 1, 3)
	return 2.0 * V - V2
			
def TwoConditionsOOVV(VMO_VVVV, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_VVVV)
	CondMO = np.einsum('acbd,ijcd->ijab', VMO, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
	return Cond

def TwoConditionsOOOO(VMO_OVOV, tMO, TFragOcc):
	VMO = AntiSymV(VMO_OVOV)
	CondMO = np.einsum('kcld,ijcd->ijkl', VMO, tMO)
	Cond = np.einsum('ijkl,ip,jq,kr,ls->prqs', CondMO, TFragOcc, TFragOcc, TFragOcc, TFragOcc)
	return Cond

def TwoConditionsVVVV(VMO_OVOV, tMO, TFragVir):
	VMO = AntiSymV(VMO_OVOV)
	CondMO = np.einsum('kcld,klab->abcd', VMO, tMO)
	Cond = np.einsum('abcd,ap,bq,cr,ds->prqs', CondMO, TFragVir, TFragVir, TFragVir, TFragVir)
	return Cond

def TwoConditionsOOVVMix(VMO_OVOV, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_OVOV)
	CondMO = np.einsum('iakc,kjcb->ijab', VMO, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
	return Cond

def TwoConditionsVVVV1e(VMO_VVVV, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_VVVV)
	CondMO = np.einsum('abcd,ijcd->ijab', VMO, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
	return Cond

def TwoConditionsOOOO2e(VMO_OOOO, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_OOOO)
	CondMO = np.einsum('ikjl,klab->ijab', VMO, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
	return Cond

def TwoConditionsOOOO1e(VMO_OOOO, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_OOOO)
	CondMO = np.einsum('ijkl,klab->ijab', VMO, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
	return Cond

def TwoConditionsOOOV2e(VMO_OOOV, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_OOOV)
	CondMO = np.einsum('kila,klbc->iabc', VMO, tMO)
	Cond = np.einsum('iabc,ip,aq,br,cs->prqs', CondMO, TFragOcc, TFragVir, TFragVir, TFragVir)
	return Cond

def TwoConditionsOVVV2e(VMO_OVVV, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_OVVV)
	CondMO = np.einsum('icad,jkcd->ijka', VMO, tMO)
	Cond = np.einsum('ijka,ip,jq,kr,as->prqs', CondMO, TFragOcc, TFragOcc, TFragOcc, TFragVir)
	return Cond

def TwoConditionsMP3VVVV(VMO_VVVV, tMO, TFragOcc):
	VMO = AntiSymV(VMO_VVVV)
	CondMO = np.einsum('acbd,ijab,klcd->ijkl', VMO, tMO, tMO)
	Cond = np.einsum('ijkl,ip,jq,kr,ls->pqrs', CondMO, TFragOcc, TFragOcc, TFragOcc, TFragOcc)
	return Cond

def TwoConditionsMP3OOOO(VMO_OOOO, tMO, TFragVir):
	VMO = AntiSymV(VMO_OOOO)
	CondMO = np.einsum('ikjl,ijab,klcd->abcd', VMO, tMO, tMO)
	Cond = np.einsum('abcd,ap,bq,cr,ds->pqrs', CondMO, TFragVir, TFragVir, TFragVir, TFragVir)
	return Cond

def TwoConditionsMP3OOVV(VMO_OOVV, tMO, TFragOcc, TFragVir):
	VMO = AntiSymV(VMO_OOVV)
	CondMO = np.einsum('klcd,ikac,jlbd->ijab', VMO, tMO, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->pqrs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
	return Cond

def OneConditionsOO(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('ikcd,jkcd->ij', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :][:, SIndices]
	return CondS

def OneConditionsVV(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('klad,klbd->ab', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :][:, SIndices]
	return CondS

def ERIEffectiveToFragMO(hEff, VEff, FIndices, g = None):
	if g is None:
		VMOEff, g, TFrag = FragmentRHF(hEff, VEff, FIndices)
	else:
		VMOEff, gTMP, TFrag = FragmentRHF(hEff, VEff, FIndices)
	tEff = MaketEff(VMOEff, g)
	#nOcc = tEff.shape[0]
	#nVir = tEff.shape[2]
	#VMOEff_VVVV = VMOEff[nOcc:, nOcc:, nOcc:, nOcc:]
	#Unkn = TwoConditionsOOVV(VMOEff_VVVV, tEff, TFrag)
	return VMOEff, tEff, TFrag

# Contains variables required for Loss calculation that only need to be calculated once
def GetConditions(VEff, hEff, VMO, tMO, TFragOcc, TFragVir, FIndices):
	nOcc = tMO.shape[0]
	nVir = tMO.shape[2]
	VMO_VVVV = VMO[nOcc:, nOcc:, nOcc:, nOcc:]
	VMO_OVOV = VMO[:nOcc, nOcc:, :nOcc, nOcc:]
	VMO_OOOO = VMO[:nOcc, :nOcc, :nOcc, :nOcc]
	VMO_OOOV = VMO[:nOcc, :nOcc, :nOcc, nOcc:]
	VMO_OVVV = VMO[:nOcc, nOcc:, nOcc:, nOcc:]
	VMO_OOVV = VMO[:nOcc, :nOcc, nOcc:, nOcc:]
	
	# These give the FF block of the conditions
	CondOOVV = TwoConditionsOOVV(VMO_VVVV, tMO, TFragOcc, TFragVir)
	CondOOOO = TwoConditionsOOOO(VMO_OVOV, tMO, TFragOcc)
	CondVVVV = TwoConditionsVVVV(VMO_OVOV, tMO, TFragVir)
	CondOOVVMix = TwoConditionsOOVVMix(VMO_OVOV, tMO, TFragOcc, TFragVir)
	CondVVVV1e = TwoConditionsVVVV1e(VMO_VVVV, tMO, TFragOcc, TFragVir)
	CondOOOO2e = TwoConditionsOOOO2e(VMO_OOOO, tMO, TFragOcc, TFragVir)
	CondOOOO1e = TwoConditionsOOOO1e(VMO_OOOO, tMO, TFragOcc, TFragVir)

	CondOOOV2e = TwoConditionsOOOV2e(VMO_OOOV, tMO, TFragOcc, TFragVir)
	CondOVVV2e = TwoConditionsOVVV2e(VMO_OVVV, tMO, TFragOcc, TFragVir)

	CondMP3VVVV = TwoConditionsMP3VVVV(VMO_VVVV, tMO, TFragOcc)
	CondMP3OOOO = TwoConditionsMP3OOOO(VMO_OOOO, tMO, TFragVir)
	CondMP3OOVV = TwoConditionsMP3OOVV(VMO_OOVV, tMO, TFragOcc, TFragVir)

	#return [CondOOVV, CondVVVV1e, CondOOOO2e, CondOOOO1e] 
	#return [CondOOVV, CondVVVV, CondOOOO2e, CondMP3OOVV]#, CondMP3OOOO, CondMP3OOVV]#, CondOOOO2e, CondOOOO1e]
	return [CondMP3VVVV, CondMP3OOOO, CondMP3OOVV, CondOOOO2e]
	#return [CondOOVV, CondOOOO, CondVVVV, CondOOVVMix, CondVVVV1e, CondOOOO2e, CondOOOO1e]

def LossPacked(VEff, hEff, FIndices, Conds, gAndMO = None):
	#nOcc = tMO.shape[0]
	#nVir = tMO.shape[2]
	#VMO_VVVV = VMO[nOcc:, nOcc:, nOcc:, nOcc:]
	##VMO_OOVV = VMO[:nOcc, :nOcc, nOcc:, nOcc:]
	#VMO_OVOV = VMO[:nOcc, nOcc:, :nOcc, nOcc:]
	
	## These give the FF block of the conditions
	#CondOOVV = TwoConditionsOOVV(VMO_VVVV, tMO, TFragOcc, TFragVir)
	#CondOOOO = TwoConditionsOOOO(VMO_OVOV, tMO, TFragOcc)
	#CondVVVV = TwoConditionsVVVV(VMO_OVOV, tMO, TFragVir)
	#CondOOVVMix = TwoConditionsOOVVMix(VMO_OVOV, tMO, TFragOcc, TFragVir)
	
	# These give the unknowns
	FixT = True
	if gAndMO is None:
		VMOEff, tEff, TFragEff = ERIEffectiveToFragMO(hEff, VEff, FIndices)
	else:
		# The first part is for fixed T and g. Second part is for fixed g only.
		if FixT:
			VMOEff = np.einsum('ijkl,ip,jq,kr,ls->pqrs', VEff, gAndMO[1], gAndMO[1], gAndMO[1], gAndMO[1])
			TFragEff = gAndMO[1].T[:, FIndices]
			tEff = MaketEff(VMOEff, gAndMO[0])
		else:
			VMOEff, tEff, TFragEff = ERIEffectiveToFragMO(hEff, VEff, FIndices, g = gAndMO[0])
	nOccEff = tEff.shape[0]
	nVirEff = tEff.shape[2]
	TFragEffOcc = TFragEff[:nOccEff, :]
	TFragEffVir = TFragEff[nOccEff:, :]
	#dbgA(gAndMO[1].T, gAndMO[0], nOccEff, nVirEff, [0, 1])
	VMOEff_VVVV = VMOEff[nOccEff:, nOccEff:, nOccEff:, nOccEff:]
	VMOEff_OVOV = VMOEff[:nOccEff, nOccEff:, :nOccEff, nOccEff:]
	VMOEff_OOOO = VMOEff[:nOccEff, :nOccEff, :nOccEff, :nOccEff]
	VMOEff_OOOV = VMOEff[:nOccEff, :nOccEff, :nOccEff, nOccEff:]
	VMOEff_OVVV = VMOEff[:nOccEff, nOccEff:, nOccEff:, nOccEff:]
	VMOEff_OOVV = VMOEff[:nOccEff, :nOccEff, nOccEff:, nOccEff:]

	UnknOOVV = TwoConditionsOOVV(VMOEff_VVVV, tEff, TFragEffOcc, TFragEffVir)
	UnknOOOO = TwoConditionsOOOO(VMOEff_OVOV, tEff, TFragEffOcc)
	UnknVVVV = TwoConditionsVVVV(VMOEff_OVOV, tEff, TFragEffVir)
	UnknOOVVMix = TwoConditionsOOVVMix(VMOEff_OVOV, tEff, TFragEffOcc, TFragEffVir)
	UnknVVVV1e = TwoConditionsVVVV1e(VMOEff_VVVV, tEff, TFragEffOcc, TFragEffVir)
	UnknOOOO2e = TwoConditionsOOOO2e(VMOEff_OOOO, tEff, TFragEffOcc, TFragEffVir)
	UnknOOOO1e = TwoConditionsOOOO1e(VMOEff_OOOO, tEff, TFragEffOcc, TFragEffVir)

	UnknOOOV2e = TwoConditionsOOOV2e(VMOEff_OOOV, tEff, TFragEffOcc, TFragEffVir)
	UnknOVVV2e = TwoConditionsOVVV2e(VMOEff_OVVV, tEff, TFragEffOcc, TFragEffVir)

	UnknMP3VVVV = TwoConditionsMP3VVVV(VMOEff_VVVV, tEff, TFragEffOcc)
	UnknMP3OOOO = TwoConditionsMP3OOOO(VMOEff_OOOO, tEff, TFragEffVir)
	UnknMP3OOVV = TwoConditionsMP3OOVV(VMOEff_OOVV, tEff, TFragEffOcc, TFragEffVir)

	#print(Conds)
	#print(UnknOOVV, UnknOOOO, UnknVVVV, UnknOOVVMix)
	#Loss = [UnknOOVV - Conds[0], UnknVVVV1e - Conds[1], UnknOOOO2e - Conds[2], UnknOOOO1e - Conds[3]]
	#Loss = [UnknOOVV - Conds[0], UnknVVVV - Conds[1], UnknOOOO2e - Conds[2], UnknMP3OOVV - Conds[3]]# UnknMP3OOOO - Conds[4], UnknMP3OOVV - Conds[5]]#, UnknOOOO2e - Conds[4], UnknOOOO1e - Conds[5]]
	Loss = [UnknMP3VVVV - Conds[0], UnknMP3OOOO - Conds[1], UnknMP3OOVV - Conds[2], UnknOOOO2e - Conds[3]]
	#Loss = [UnknOOVV - Conds[0], UnknOOOO - Conds[1], UnknVVVV - Conds[2], UnknOOVVMix - Conds[3], UnknVVVV1e - Conds[4], UnknOOOO2e - Conds[5], UnknOOOO1e - Conds[6]]
	return Loss
	
def Loss(VEffVec, hEff, FIndices, BIndices, VUnmatched, Conds, gAndMO = None):
	VEff = VectorToVSymm(VEffVec, FIndices, BIndices)
	VEff[np.ix_(FIndices, FIndices, FIndices, FIndices)] = VUnmatched[0]
	VEff[np.ix_(BIndices, BIndices, BIndices, BIndices)] = VUnmatched[1]
	#print(VEff)
	Losses = LossPacked(VEff, hEff, FIndices, Conds, gAndMO = gAndMO)
	LossesVec = MakeLossVector(Losses)
	#print(LossesVec)
	#print(np.inner(LossesVec, LossesVec))
	return LossesVec

def dLoss(VEffVec, hEff, FIndices, BIndices, VUnmatched, Conds, gAndMO):
	dV = 0.0001
	Loss0 = Loss(VEffVec, hEff, FIndices, BIndices, VUnmatched, Conds, gAndMO)
	J = np.zeros((Loss0.shape[0], VEffVec.shape[0]))
	J = J.T
	for i in range(VEffVec.shape[0]):
		VEffVecPlusdV = VEffVec.copy()
		VEffVecMinsdV = VEffVec.copy()
		VEffVecPlusdV[i] += dV
		VEffVecMinsdV[i] -= dV
		LossPlusdV = Loss(VEffVecPlusdV, hEff, FIndices, BIndices, VUnmatched, Conds, gAndMO)
		LossMinsdV = Loss(VEffVecMinsdV, hEff, FIndices, BIndices, VUnmatched, Conds, gAndMO)
		dLoss = (LossPlusdV - LossMinsdV) / (dV + dV)
		#print(i)
		#print(LossPlusdV)
		#print(LossMinsdV)
		J[i] = dLoss
	J = J.T
	return J

def BacktrackLineSearch(f, x, dx, args, beta = 0.5, y0 = None):
	alph = 1.0
	xTest = x + alph * dx
	if y0 is None:
		yInit = np.linalg.norm(f(x, *args))
	else:
		yInit = np.linalg.norm(y0)
	y = f(xTest, *args)
	yTest = np.linalg.norm(y)
	while(yTest > yInit):
		alph = beta * alph
		xTest = x + alph * dx
		y = f(xTest, *args)
		yTest = np.linalg.norm(y)
		if alph < 1e-20:
			print("Backtrack line search has failed")
			break
	print("Linesearch finds alph =", alph)
	return xTest, y
	

def NewtonRaphson(f, x0, df, args, tol = 1e-6, eps = 1e-6):
	F = f(x0, *args)
	x = x0.copy()
	while not all([abs(x) < tol for x in F]):
		J = df(x, *args)
		print("L =", F)
		print("J\n", J)
		try:
			JInv = np.linalg.inv(J)
		except:
			e, V = np.linalg.eig(J)
			JMod = np.zeros(J.shape)
			np.fill_diagonal(JMod, e + eps)
			JMod = V @ JMod @ V.T
			JInv = np.linalg.inv(JMod)
		dx = -1.0 * JInv @ F
		print("dx =", dx)
		x, F = BacktrackLineSearch(f, x, dx, args, y0 = F) #x - alp * JInv @ F
		#F = f(x, *args)
		print("x =", x)
		print("L =", F)
		print((F**2.0).sum())
		input()
	return x

def GaussNewton(f, x0, df, args, tol = 1e-12):
	F = f(x0, *args)
	x = x0.copy()
	while (F**2.0).sum() > tol:
		J = df(x, *args)
		x = x - np.linalg.inv(J.T @ J) @ J.T @ F
		F = f(x, *args)
		print(F)
		print((F**2.0).sum())
		input()

def LevenbergMarquardt(f, x0, df, args, tol = 1e-20, lamb = 1e-6):
	F = f(x0, *args)
	x = x0.copy()
	while (F**2.0).sum() > tol:
		J = df(x, *args)
		JTJ = J.T @ J
		Norm = np.zeros(JTJ.shape)
		np.fill_diagonal(Norm, np.diag(JTJ))
		A = (JTJ + lamb * Norm)
		delta = np.linalg.solve(A, J.T @ F)
		x = x - delta
		F = f(x, *args)
		print(x)
		print(F)
		print((F**2.0).sum())
		#input()
	return x


def MP2MLEmbedding(hEff, VMO, tMO, TFragOcc, TFragVir, FIndices, VEff0 = None, gFixed = False):
	N = 2 * int(len(FIndices))
	nFrag = len(FIndices)
	BIndices = [i + nFrag for i in FIndices]
	if VEff0 is None:
		VEffVec = np.zeros(4 * nFrag**4)
		VFFFF = np.zeros((nFrag, nFrag, nFrag, nFrag))
		VBBBB = np.zeros((nFrag, nFrag, nFrag, nFrag))
	else:
		VFFFF = VEff0[np.ix_(FIndices, FIndices, FIndices, FIndices)]
		VBBBB = VEff0[np.ix_(BIndices, BIndices, BIndices, BIndices)]

		VBFFF = VEff0[np.ix_(BIndices, FIndices, FIndices, FIndices)]
		VFBFB = VEff0[np.ix_(FIndices, BIndices, FIndices, BIndices)]
		VFFBB = VEff0[np.ix_(FIndices, FIndices, BIndices, BIndices)]
		VFBBB = VEff0[np.ix_(FIndices, BIndices, BIndices, BIndices)]
		VBFFFVec = ERIToVector(VBFFF)
		VFBFBVec = ERIToVector(VFBFB)
		VFFBBVec = ERIToVector(VFFBB)
		VFBBBVec = ERIToVector(VFBBB)
		VEffVec = np.concatenate((VBFFFVec, VFBFBVec, VFFBBVec, VFBBBVec))
	print("VEff0\n", VEff0)

	VEff = VectorToVSymm(VEffVec, FIndices, BIndices)
	VEff[np.ix_(FIndices, FIndices, FIndices, FIndices)] = VFFFF
	VEff[np.ix_(BIndices, BIndices, BIndices, BIndices)] = VBBBB

	# Get Conditions which are fixed.
	Conds = GetConditions(VEff, hEff, VMO, tMO, TFragOcc, TFragVir, FIndices)

	# Do RHF to get a fixed g, if desired.
	if gFixed:
		VMOEff, g, TFragEff, CMOEff = FragmentRHF(hEff, VEff, FIndices, ReturnMO = True)
		gAndMO = [g, CMOEff]
	else:
		gAndMO = None

	#VEffVec = np.random.rand(VEffVec.shape[0],)
	#scan_start = -2.
	#scan_end = 2.
	#step_size = 0.25
	#steps = int((scan_end - scan_start)/step_size) + 1
	#f = open('scan.txt', 'w')
	#for i in range(steps):
	#	L0 = Loss(np.asarray([0, 0, 0, scan_start + i * step_size]), hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO)
	#	f.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (scan_start + i * step_size, L0[0], L0[1], L0[2], L0[3], L0[4], L0[5], L0[6]))
	#	f.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (scan_start + i * step_size, L0[0], L0[1], L0[2], L0[3], L0[4], L0[5]))
	#	for j in range(steps):
	#		for k in range(steps):
	#			for l in range(steps):
	#				L0 = Loss(np.asarray([scan_start + i * step_size, scan_start + j * step_size, scan_start + k * step_size, scan_start + l * step_size]), hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO)
	#				f.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (scan_start + i * step_size, scan_start + j * step_size, scan_start + k * step_size, scan_start + l * step_size, L0[0], L0[1], L0[2], L0[3]))
	#L1 = Loss(np.asarray([1., 0., 0., 0.]), hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO)
	#L2 = Loss(np.asarray([0., 1., 0., 0.]), hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO)
	#L3 = Loss(np.asarray([0., 0., 1., 0.]), hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO)
	#L4 = Loss(np.asarray([0., 0., 0., 1.]), hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO)
	#print(L1)
	#print(L2)
	#print(L3)
	#print(L4)
	#VEffVec = np.zeros(VEffVec.shape)
	#VEffFinal = NewtonRaphson(Loss, VEffVec, dLoss, [hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO])
	#VEffFinal = GaussNewton(Loss, VEffVec, dLoss, [hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO])
	VEffVecFinal = LevenbergMarquardt(Loss, VEffVec, dLoss, [hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO], tol = 1e-30, lamb = 1e-10)
	VEffFinal = VectorToVSymm(VEffVecFinal, FIndices, BIndices)
	VEffFinal[np.ix_(FIndices, FIndices, FIndices, FIndices)] = VFFFF
	VEffFinal[np.ix_(BIndices, BIndices, BIndices, BIndices)] = VBBBB
	#VEffFinal = newton(Loss, VEffVec, args = [hEff, FIndices, BIndices, [VFFFF, VBBBB], Conds, gAndMO], fprime = dLoss, maxiter = 50)
	return VEffFinal

def MP2EnergyEmbedding(hNoCore, hEff, FIndices, ETarget, VEff0 = None):
	N = 2 * int(len(FIndices))
	nFrag = len(FIndices)
	BIndices = [i + nFrag for i in FIndices]
	if VEff0 is None:
		VEffVec = np.zeros(4 * nFrag**4)
		VFFFF = np.zeros((nFrag, nFrag, nFrag, nFrag))
		VBBBB = np.zeros((nFrag, nFrag, nFrag, nFrag))
	else:
		VFFFF = VEff0[np.ix_(FIndices, FIndices, FIndices, FIndices)]
		VBBBB = VEff0[np.ix_(BIndices, BIndices, BIndices, BIndices)]

		VBFFF = VEff0[np.ix_(BIndices, FIndices, FIndices, FIndices)]
		VFBFB = VEff0[np.ix_(FIndices, BIndices, FIndices, BIndices)]
		VFFBB = VEff0[np.ix_(FIndices, FIndices, BIndices, BIndices)]
		VFBBB = VEff0[np.ix_(FIndices, BIndices, BIndices, BIndices)]
		VBFFFVec = ERIToVector(VBFFF)
		VFBFBVec = ERIToVector(VFBFB)
		VFFBBVec = ERIToVector(VFFBB)
		VFBBBVec = ERIToVector(VFBBB)
		VEffVec = VFFBBVec
	print(VEffVec)
	# Get Conditions which are fixed.
	Conds = np.asarray([ETarget])
	print("ETarget", ETarget)

	# Energy derivative wrt VFFBB
	def dEMP2(VFFBB, hNoCore, h, V, CenterIndices):
		dV = 0.01
		VFFBBPlus = VFFBB + dV
		VFFBBMins = VFFBB - dV
		VPlus = V.copy()
		VMins = V.copy()
		VPlus[np.ix_(FIndices, FIndices, BIndices, BIndices)] = VFFBBPlus
		VMins[np.ix_(FIndices, FIndices, BIndices, BIndices)] = VFFBBMins
		EPlus = FragmentRMP2(hNoCore, h, VPlus, CenterIndices)
		EMins = FragmentRMP2(hNoCore, h, VMins, CenterIndices)
		dE = (EPlus - EMins) / (2.0 * dV)
		J = np.zeros((1,1))
		J[0, 0] = dE
		return J
	def ELoss(VFFBB, hNoCore, h, V, CenterIndices):
		VLocal = V.copy()
		VLocal[np.ix_(FIndices, FIndices, BIndices, BIndices)] = VFFBB
		E = FragmentRMP2(hNoCore, h, VLocal, CenterIndices)
		print("EFrag", E)
		return np.asarray([E]) - Conds
	# Do RHF to get a fixed g, if desired.

	VEffVecFinal = LevenbergMarquardt(ELoss, VEffVec, dEMP2, [hNoCore, hEff, VEff0, FIndices], tol = 1e-30, lamb = 1e-10)
	print(VEffVecFinal)
	return VEffVecFinal


def CombinedIndex(Indices, nFB):
	if len(Indices) == 2:
		i = Indices[0] + nFB * Indices[1]
	if len(Indices) == 4:
		i = Indices[0] + nFB * Indices[1] + nFB * nFB * Indices[2] + nFB * nFB * nFB * Indices[3]
	return i
	

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
	Nf = 1
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
	#t2 = T2

	#VMO_VVVV = np.zeros((Norb, Norb, Norb, Norb))
	#VMO_VVVV[Nocc:, Nocc:, Nocc:, Nocc:] = VMO[Nocc:, Nocc:, Nocc:, Nocc:]
	#VSO_VVVV = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, VMO_VVVV)

	#VMO_OOVV = np.zeros((Norb, Norb, Norb, Norb))
	#VMO_OOVV[:Nocc, :Nocc, Nocc:, Nocc:] = VMO[:Nocc, :Nocc, Nocc:, Nocc:]
	#VSO_OOVV = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, VMO_OOVV)

	#tMO_Occ = np.zeros((Norb, Norb, Norb, Norb))
	#tMO_Vir = np.zeros((Norb, Norb, Norb, Norb))

	#MP2EnergyEmbedding(hNoCore, hFrag, FIndices, EMP2 / N, VEff0 = VFrag)
	VML = MP2MLEmbedding(hFrag, VMO, t2, TFragOccMOtoSO, TFragVirMOtoSO, FIndices, VEff0 = VFrag, gFixed = True)

	FragE = FragmentRMP2(hNoCore, hFrag, VML, FIndices)
	print(FragE*N)
	print(EMP2)
	#print(int(N / Nf) * FragE + mol.energy_nuc())
	#print(int(N / Nf) * FragE)
	
