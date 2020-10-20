import numpy as np
from scipy.optimize import newton

def GetFock(h, V):
	J = np.einsum('iijj->ij', V)
	K = np.einsum('ijij->ij', V)
	F = h + 2.0 * J - K
	return F

def GetG(F):
	e, v = np.linalg.eigh(F)

def ERIToVector(h, V):
	n = h.shape[0]
	hVec = h.reshape(n * n)
	VVec = V.reshape(n * n * n * n)
	ERIVec = np.vstack((hVec, VVec))
	return ERIVec

def VectorToERI(ERIVec, n):
	hVec = ERIVec[:(n * n)]
	VVec = ERIVec[(n * n):]
	h = hVec.reshape(n, n)
	V = VVec.reshape(n, n, n, n)
	return h, V
	

def TWoConditionsOOVV(VSO_VVVV, tSO, SIndices):
	Cond = np.einsum('abcd,ijcd->ijab', VSO_VVVV, tSO)
	CondS = Cond[SIndices, :, :, :][:, SIndices, :, :][:, :, SIndices, :][:, :, :, SIndices]
	return CondS

def TwoConditionsOOOO(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('klcd,ijcd->ijkl', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :, :, :][:, SIndices, :, :][:, :, SIndices, :][:, :, :, SIndices]
	return CondS

def TwoConditionsVVVV(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('klcd,klab->abcd', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :, :, :][:, SIndices, :, :][:, :, SIndices, :][:, :, :, SIndices]
	return CondS

def TwoConditionsOOVVMix(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('ikac,kjcb->ijab', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :, :, :][:, SIndices, :, :][:, :, SIndices, :][:, :, :, SIndices]
	return CondS

def OneConditionsOO(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('ikcd,jkcd->ij', VSO_OOVV, tSO)
	CodeS = Cond[SIndices, :][:, SIndices]
	return CondS

def OneConditionsVV(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('klad,klbd->ab', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :][:, SIndices]
	return CondS

def TwoUnknown(hEff, VEff, Nocc):
	F = GetFock(hEff, VEff)
	g = GetG(F)
	Unk = np.einsum('abcd,ijcd->ijab', VEff, VEff)
	for i in range(Unk.shape[0]):
		for j in range(Unk.shape[1]):
			for a in range(Unk.shape[2]):
				for b in range(Unk.shape[3]):
					Unk[i, j, a, b] = Unk[i, j, a, b] / (g[i] + g[j] - g[Nocc + a] - g[Nocc + b])
	return Unk

def LossPacked(hEff, VEff, VSO_VVVV, VSO_OOVV, tSO, SIndices):
	OneCond = OneConditionsOO(VSO_OOVV, tSO, SIndices)
	TwoCond = TwoConditionsOOVV(VSO_VVVV, tSO, SIndices)
	Nocc = TwoCond.shape[0]
	Nvir = TwoCond.shape[2]
	#OneCondVec = OneCond.reshape(Nocc * Nocc)
	#TWoCondVec = OneCond.reshape(Nocc * Nocc * Nvirt * Nvirt)
	TwoUnkn = TwoUnknown(hEff, VEff, Nocc)
	OneUnkn = np.zeros((Nocc, Nocc))
	return OneUnkn - OneCond, TwoUnkn - TwoCond
	
def Loss(ERIVec, VSO_VVVV, VSO_OOVV, tSO, SIndices):
	h, V = VectorToERI(ERIVec, VSO_VVVV.shape[0] / 2)
	OneLoss, TwoLoss = LossPacked(h, V, VSO_VVVV, VSO_OOVV, tSO, SIndices)
	LossUnpacked = ERIToVector(OneLoss, TwoLoss)
	return LossUnpacked


def MP2MLEmbedding(tSO, VSO, g, FBIndices, EIndices):
	nFBE = tSO.shape[0]
	nFB = int(len(FBIndices))
	nE = int(len(EIndices))
	AllIndices = FBIndices + EIndices
	Veff = np.zeros((nFB, nFB, nFB, nFB))
	tV = np.einsum('abcd,ijcd->abij', (VSO, tSO))
	A = np.zeros((nFB, nFB, nFB, nFB))
	for i in range(nFB):
		for j in range(nFB):
			for a in range(nFBE):
				for b in range(nFBE):
					A[i, j, a, b] = 1.0 / (g[i] + g[j] - g[a] - g[b])
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
	N = 4
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
	mol.build(verbose = 0)
	mf = scf.RHF(mol).run()

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
	hMO = reduce(np.dot, (mo_coeff.T, mf.get_hcore(), mo_coeff))
	VMO = ao2mo.kernel(mol, mo_coeff) #np.eye(TTotal.shape[0]))
	VMO = ao2mo.restore(1, VMO, hMO.shape[0])

	F = mf.get_fock()
	FSO = reduce(np.dot, (TTotal.T, mf.get_fock(), TTotal))
	g = np.diag(FSO)

	mp2 = mp.MP2(mf)
	E, T2 = mp2.kernel()
	print("E(MP2) = ", mf.energy_elec()[0], E)

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
	tMO = np.zeros((Norb, Norb, Norb, Norb))
	tMO[:Nocc, :Nocc, Nocc:, Nocc:] = t2

	tSO = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, tMO)

	VMO_VVVV = np.zeros((Norb, Norb, Norb, Norb))
	VMO_VVVV[Nocc:, Nocc:, Nocc:, Nocc:] = VMO[Nocc:, Nocc:, Nocc:, Nocc:]

	print(np.dot(T.T, T))

	#tMO_Occ = np.zeros((Norb, Norb, Norb, Norb))
	#tMO_Vir = np.zeros((Norb, Norb, Norb, Norb))
			
	SIndex = list(range(PSch.shape[0]))
	FIndex = SIndex[:int(len(SIndex)/2)]
	BIndex = SIndex[int(len(SIndex)/2):]
	EIndex = list(range(PEnv.shape[0]))
	
	#tZero = np.zeros((Norb, Norb, Norb, Norb))
	#testMFBath = MP2Bath(tZero, FIndex, BIndex, EIndex, PSch, PEnv, hSO, VSO)
	#mf0, mf1, mf2 = testMFBath.CalcH()
	#mf0.tofile("mf0")
	#mf1.tofile("mf1")
	#mf2.tofile("mf2")
	
