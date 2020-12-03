import numpy as np
from scipy.optimize import newton

def CheckSymmetry(V):
	Symmetric = True
	
# Assumes V is given as VFFFA in chemist notation
def OneExternal(V):
	nF = V.shape[0]
	B = V.copy()
	for i in range(nF):
		for j in range(nF):
			U, S, T = np.linalg.svd(V[i, j])
			print(S)
			B[i, j] = V[i, j] @ T.T
	return B

def TwoExternal(V):
	OrigDim = V.shape
	print(V[0,0])
	VExtended = V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
	print(VExtended[0].reshape(V.shape[2], V.shape[3]))
	U, S, T = np.linalg.svd(VExtended)
	VExtended = VExtended @ T.T
	return VExtended.reshape(OrigDim)

def ThreeExternal(V):
	OrigDim = V.shape
	VExtended = V.reshape(V.shape[0], V.shape[1] * V.shape[2] * V.shape[3])
	U, S, T = np.linalg.svd(VExtended)
	VExtended = VExtended @ T.T
	return VExtended.reshape(OrigDim)

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

	RDM2 = mp2.make_rdm2()

	VOneExt = VLO[np.ix_(FIndices, FIndices, FIndices, BEIndices)] #VLO[FIndices, :, :, :][:, FIndices, :, :][:, :, FIndices, :][:, :, :, :]
	print(VOneExt[0,0])
	print(VOneExt.shape)
	print(VOneExt)
	VV = OneExternal(VOneExt)
	print(VV[0, 0, 1, 1])
	print(VV[1, 1, 0, 0])
	print(VV[0, 1, 0, 1])
	print(VV[0, 1, 1, 0])
	print(VV[1, 0, 0, 1])
	VTwoExt = VLO[FIndices, :, :, :][:, FIndices, :, :][:, :, BEIndices, :][:, :, :, BEIndices]
	V2 = TwoExternal(VTwoExt)
	print(V2[:, :,  0, 0])
	print(V2[0, 0])
	print(V2[0, 1, 0, 1])
	print(V2[0, 1, 1, 0])
	VThreeExt = VLO[FIndices, :, :, :][:, :, :, :][:, :, :, :][:, :, :, :]
	V3 = ThreeExternal(VThreeExt)
	print(V3[0, 0, 1, 1])
	print(V3[1, 1, 0, 0])
	print(V3[0, 1, 0, 1])
	print(V3[0, 1, 1, 0])
