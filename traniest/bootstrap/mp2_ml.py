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
def FragmentRHF(hEff, VEff, FIndices):
	mol = gto.M()
	nElec = hEff.shape[0] # Half occupied fragment
	mol.nelectron = nElec

	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: hEff
	mf.get_ovlp = lambda *args: np.eye(nElec)
	mf._eri = ao2mo.restore(8, VEff, nElec)

	mf.kernel()

	VMOEff = np.einsum('ijkl,ip,jq,kr,ls->pqrs', VEff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)
	TFrag = mf.mo_coeff.T[FIndices, :]
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
					tEff[i, j, a, b] = VEff[i, j, nOcc + a, nOcc + b] / (g[i] + g[j] - g[nOcc + a] - g[nOcc + b])
	return tEff
	
def ERIToVector(V):
	n = V.shape[0]
	#hVec = h.reshape(n * n)
	VVec = V.reshape(n * n * n * n)
	#ERIVec = np.vstack((hVec, VVec))
	return VVec

def VectorToERI(ERIVec):
	#n2 = int(n * n)
	#hVec = ERIVec[:n2]
	#VVec = ERIVec[n2:]
	#h = hVec.reshape(n, n)
	n = int(VVec.shape[0]**0.25)
	V = VVec.reshape(n, n, n, n)
	return V

def MakeLossVector(Losses):
	LossesVec = np.zeros(0)
	for Loss in Losses:
		Loss.shape[0] = n1
		Loss.shape[2] = n2
		Dim = n1 * n1 + n2 * n2
		LossVec = np.zeros(Dim)
		for i in Loss.shape[0]:
			for j in Loss.shape[1]:
				for k in Loss.shape[2]:
					for l in Loss.shape[3]:
						LossVec[i + j * Loss.shape[0] + k * Loss.shape[0] * Loss.shape[1] + l * Loss.shape[0] * Loss.shape[1] * Loss.shape[3]] = Loss[i, j, k, l]
		LossesVec = np.concatenate((LossesVec, LossVec))
	LossVec = np.zeros(Dim)
	for Loss in Losses:
		
	
def TwoConditionsOOVV(VMO_VVVV, tMO, TFrag):
	CondMO = np.einsum('abcd,ijcd->ijab', VMO_VVVV, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFrag, TFrag, TFrag, TFrag)
	return Cond

def TwoConditionsOOOO(VMO_OOVV, tMO, TFrag):
	CondMO = np.einsum('klcd,ijcd->ijkl', VMO_OOVV, tMO)
	Cond = np.einsum('ijkl,ip,jq,kr,ls->prqs', CondMO, TFrag, TFrag, TFrag, TFrag)
	return Cond

def TwoConditionsVVVV(VMO_OOVV, tMO, TFrag):
	CondMO = np.einsum('klcd,klab->abcd', VMO_OOVV, tMO)
	Cond = np.einsum('abcd,ap,bq,cr,ds->prqs', CondMO, TFrag, TFrag, TFrag, TFrag)
	return Cond

def TwoConditionsOOVVMix(VMO_OOVV, tMO, TFrag):
	CondMO = np.einsum('ikac,kjcb->ijab', VMO_OOVV, tMO)
	Cond = np.einsum('ijab,ip,jq,ar,bs->prqs', CondMO, TFrag, TFrag, TFrag, TFrag)
	return Cond

def OneConditionsOO(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('ikcd,jkcd->ij', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :][:, SIndices]
	return CondS

def OneConditionsVV(VSO_OOVV, tSO, SIndices):
	Cond = np.einsum('klad,klbd->ab', VSO_OOVV, tSO)
	CondS = Cond[SIndices, :][:, SIndices]
	return CondS

def ERIEffectiveToFragMO(hEff, VEff, FIndices):
	VMOEff, g, TFrag = FragmentRHF(hEff, VEff, FIndices)
	tEff = MaketEff(VMOEff, g)
	#nOcc = tEff.shape[0]
	#nVir = tEff.shape[2]
	#VMOEff_VVVV = VMOEff[nOcc:, nOcc:, nOcc:, nOcc:]
	#Unkn = TwoConditionsOOVV(VMOEff_VVVV, tEff, TFrag)
	return VMOEff, tEff, TFrag

def LossPacked(VEff, hEff, VMO, tMO, TFrag, FIndices):
	nOcc = tMO.shape[0]
	nVir = tMO.shape[2]
	VMO_VVVV = VMO[nOcc:, nOcc:, nOcc:, nOcc:]
	VMO_OOVV = VMO[:nOcc, :nOcc, nOcc:, nOcc:]
	
	# These give the FF block of the conditions
	CondOOVV = TwoConditionsOOVV(VMO_VVVV, tMO, TFrag)
	CondOOOO = TwoConditionsOOOO(VMO_OOVV, tMO, TFrag)
	CondVVVV = TwoConditionsVVVV(VMO_OOVV, tMO, TFrag)
	CondOOVVMix = TwoConditionsOOVVMix(VMO_OOVV, tMO, TFrag)
	
	# These give the unknowns
	VMOEff, tEff, TFragEff = ERIEffectiveToFragMO(hEff, VEff, FIndices)
	nOccEff = tEff.shape[0]
	nVirEff = tEff.shape[2]
	VMOEff_VVVV = VMOEff[nOccEff:, nOccEff:, nOccEff:, nOccEff:]
	VMOEff_OOVV = VMOEff[:nOccEff, :nOccEff, nOccEff:, nOccEff:]
	UnknOOVV = TwoConditionsOOVV(VMOEff_VVVV, tEff, TFragEff)
	UnknOOOO = TwoConditionsOOOO(VMOEff_OOVV, tEff, TFragEff)
	UnknVVVV = TwoConditionsVVVV(VMOEff_OOVV, tEff, TFragEff)
	UnknOOVVMix = TwoConditionsOOVVMix(VMOEff_OOVV, tEff, TFragEff)
	
	Loss = [UnknOOVV - CondOOVV, UnknOOOO - CondOOOO, UnknVVVV - CondVVVV, UnknOOVVMix - CondOOVVMix]
	return Loss
	
def Loss(VEffVec, hEff, VMO, tMO, TFrag, FIndices):
	VEff = VectorToERI(VEffVec)
	Losses = LossPacked(VEff, hEff, VMO, tMO, TFrag, FIndices)
	LossesVec = MakeLossVector(Losses)
	return LossUnpacked


def MP2MLEmbedding(VSO_VVVV, VSO_OOVV, tSO, SIndices, EIndices):
	N = int(len(SIndices))
	ERIVec = np.zeros(N**2 + N**4)
	L = Loss(ERIVec, VSO_VVVV, VSO_OOVV, tSO, SIndices)

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

	#VMO_VVVV = np.zeros((Norb, Norb, Norb, Norb))
	#VMO_VVVV[Nocc:, Nocc:, Nocc:, Nocc:] = VMO[Nocc:, Nocc:, Nocc:, Nocc:]
	#VSO_VVVV = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, VMO_VVVV)

	#VMO_OOVV = np.zeros((Norb, Norb, Norb, Norb))
	#VMO_OOVV[:Nocc, :Nocc, Nocc:, Nocc:] = VMO[:Nocc, :Nocc, Nocc:, Nocc:]
	#VSO_OOVV = np.einsum('ap,bq,cr,ds,abcd->pqrs', TTotal, TTotal, TTotal, TTotal, VMO_OOVV)


	print(np.dot(T.T, T))

	#tMO_Occ = np.zeros((Norb, Norb, Norb, Norb))
	#tMO_Vir = np.zeros((Norb, Norb, Norb, Norb))
			
	SIndex = list(range(PSch.shape[0]))
	FIndex = SIndex[:int(len(SIndex)/2)]
	BIndex = SIndex[int(len(SIndex)/2):]
	EIndex = list(range(PEnv.shape[0]))

	MP2MLEmbedding(VSO_VVVV, VSO_OOVV, tSO, SIndex, EIndex)
	
	#tZero = np.zeros((Norb, Norb, Norb, Norb))
	#testMFBath = MP2Bath(tZero, FIndex, BIndex, EIndex, PSch, PEnv, hSO, VSO)
	#mf0, mf1, mf2 = testMFBath.CalcH()
	#mf0.tofile("mf0")
	#mf1.tofile("mf1")
	#mf2.tofile("mf2")
	
