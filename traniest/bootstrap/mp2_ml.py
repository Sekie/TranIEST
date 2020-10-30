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

def FragmentRHF(hEff, VEff, FIndices):
	mol = gto.M()
	nElec = hEff.shape[0] # Half occupied fragment
	mol.nelectron = nElec

	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: hEff
	mf.get_ovlp = lambda *args: np.eye(nElec)
	mf._eri = ao2mo.restore(8, VEff, nElec)
	mf.max_cycle = 1000

	mf.kernel()
	#try:
	#	mf.kernel()
	#except:
	#	mf.diis = False
	#	mf.kernel()

	VMOEff = np.einsum('ijkl,ip,jq,kr,ls->pqrs', VEff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)
	TFrag = mf.mo_coeff.T[:, FIndices]
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
			
def TwoConditionsOOVV(VMO_VVVV, tMO, TFragOcc, TFragVir):
	CondMO = np.einsum('acbd,ijcd->ijab', VMO_VVVV, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
	return Cond

def TwoConditionsOOOO(VMO_OVOV, tMO, TFragOcc):
	CondMO = np.einsum('kcld,ijcd->ijkl', VMO_OVOV, tMO)
	Cond = np.einsum('ijkl,ip,jq,kr,ls->prqs', CondMO, TFragOcc, TFragOcc, TFragOcc, TFragOcc)
	return Cond

def TwoConditionsVVVV(VMO_OVOV, tMO, TFragVir):
	CondMO = np.einsum('kcld,klab->abcd', VMO_OVOV, tMO)
	Cond = np.einsum('abcd,ap,bq,cr,ds->prqs', CondMO, TFragVir, TFragVir, TFragVir, TFragVir)
	return Cond

def TwoConditionsOOVVMix(VMO_OVOV, tMO, TFragOcc, TFragVir):
	CondMO = np.einsum('iakc,kjcb->ijab', VMO_OVOV, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFragOcc, TFragOcc, TFragVir, TFragVir)
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
	#VMO_OOVV = VMO[:nOcc, :nOcc, nOcc:, nOcc:]
	VMO_OVOV = VMO[:nOcc, nOcc:, :nOcc, nOcc:]
	
	# These give the FF block of the conditions
	CondOOVV = TwoConditionsOOVV(VMO_VVVV, tMO, TFragOcc, TFragVir)
	CondOOOO = TwoConditionsOOOO(VMO_OVOV, tMO, TFragOcc)
	CondVVVV = TwoConditionsVVVV(VMO_OVOV, tMO, TFragVir)
	CondOOVVMix = TwoConditionsOOVVMix(VMO_OVOV, tMO, TFragOcc, TFragVir)
	
	return [CondOOVV, CondOOOO, CondVVVV, CondOOVVMix]

def LossPacked(VEff, hEff, VMO, tMO, TFragOcc, TFragVir, FIndices, Conds, g = None):
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
	VMOEff, tEff, TFragEff = ERIEffectiveToFragMO(hEff, VEff, FIndices, g = g)
	nOccEff = tEff.shape[0]
	nVirEff = tEff.shape[2]
	TFragEffOcc = TFragEff[:nOccEff, :]
	TFragEffVir = TFragEff[nOccEff:, :]
	VMOEff_VVVV = VMOEff[nOccEff:, nOccEff:, nOccEff:, nOccEff:]
	VMOEff_OVOV = VMOEff[:nOccEff, nOccEff:, :nOccEff, nOccEff:]
	UnknOOVV = TwoConditionsOOVV(VMOEff_VVVV, tEff, TFragEffOcc, TFragEffVir)
	UnknOOOO = TwoConditionsOOOO(VMOEff_OVOV, tEff, TFragEffOcc)
	UnknVVVV = TwoConditionsVVVV(VMOEff_OVOV, tEff, TFragEffVir)
	UnknOOVVMix = TwoConditionsOOVVMix(VMOEff_OVOV, tEff, TFragEffOcc, TFragEffVir)
	
	Loss = [UnknOOVV - Conds[0], UnknOOOO - Conds[1], UnknVVVV - Conds[2], UnknOOVVMix - Conds[3]]
	return Loss
	
def Loss(VEffVec, hEff, VMO, tMO, TFragOcc, TFragVir, FIndices, BIndices, VUnmatched, Conds, g = None):
	VEff = VectorToVSymm(VEffVec, FIndices, BIndices)
	VEff[np.ix_(FIndices, FIndices, FIndices, FIndices)] = VUnmatched[0]
	VEff[np.ix_(BIndices, BIndices, BIndices, BIndices)] = VUnmatched[1]
	print(VEff)
	Losses = LossPacked(VEff, hEff, VMO, tMO, TFragOcc, TFragVir, FIndices, Conds, g = g)
	LossesVec = MakeLossVector(Losses)
	print(LossesVec)
	print(np.inner(LossesVec, LossesVec))
	return LossesVec


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

	#L = Loss(VEffVec, hEff, VMO, tMO, TFragOcc, TFragVir, FIndices, BIndices, [VFFFF, VBBBB])
	#print(L)
	VEff = VectorToVSymm(VEffVec, FIndices, BIndices)

	# Get Conditions which are fixed.
	Conds = GetConditions(VEff, hEff, VMO, tMO, TFragOcc, TFragVir, FIndices)

	# Do RHF to get a fixed g, if desired.
	if gFixed:
		VMOEff, g, TFragEff = FragmentRHF(hEff, VEff, FIndices)
	else:
		g = None

	VEffFinal = newton(Loss, VEffVec, args = [hEff, VMO, tMO, TFragOcc, TFragVir, FIndices, BIndices, [VFFFF, VBBBB], Conds, g])

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

	TTotal = np.dot(StoOrth, T)
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
	for i in SIndices:
		for j in SIndices:
			CoreContribution = 0.0
			for c in CoreIdx:
				CoreContribution += (2.0 * VSO[i, j, c, c] - VSO[i, c, c, j])
			hFrag[i, j] += CoreContribution

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

	#VMO_VVVV = np.zeros((Norb, Norb, Norb, Norb))
	#VMO_VVVV[Nocc:, Nocc:, Nocc:, Nocc:] = VMO[Nocc:, Nocc:, Nocc:, Nocc:]
	#VSO_VVVV = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, VMO_VVVV)

	#VMO_OOVV = np.zeros((Norb, Norb, Norb, Norb))
	#VMO_OOVV[:Nocc, :Nocc, Nocc:, Nocc:] = VMO[:Nocc, :Nocc, Nocc:, Nocc:]
	#VSO_OOVV = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, VMO_OOVV)

	#tMO_Occ = np.zeros((Norb, Norb, Norb, Norb))
	#tMO_Vir = np.zeros((Norb, Norb, Norb, Norb))

	MP2MLEmbedding(hFrag, VMO, t2, TFragOccMOtoSO, TFragVirMOtoSO, FIndices, VEff0 = VFrag, gFixed = True)
	
	#tZero = np.zeros((Norb, Norb, Norb, Norb))
	#testMFBath = MP2Bath(tZero, FIndex, BIndex, EIndex, PSch, PEnv, hSO, VSO)
	#mf0, mf1, mf2 = testMFBath.CalcH()
	#mf0.tofile("mf0")
	#mf1.tofile("mf1")
	#mf2.tofile("mf2")
	