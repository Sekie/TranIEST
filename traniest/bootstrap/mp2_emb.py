import numpy as np
from scipy.optimize import newton
import math

def CheckSymmetry(V):
	Symmetric = True

def CompileFullV(VFFFF, VFFFB, VFFBB, VFBFB, VFBBB, VBBBB):
	nF = VFFFF.shape[0]
	V = np.zeros((2 * nF, 2 * nF, 2 * nF, 2 * nF))
	Fs = list(range(nF))
	Bs = list(range(nF, 2 * nF))
	
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

	
# Assumes V is given as VFFFA in chemist notation
def OneExternal(V, ReturnFull = False):
	nF = V.shape[0]
	B = V.copy()
	for i in range(nF):
		for j in range(nF):
			U, S, T = np.linalg.svd(V[i, j])
			B[i, j] = V[i, j] @ T.T
			#print(i, j, "\n", B[i, j])
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

def TwoExternal(V, VbbAA = None, ReturnFull = False):
	OrigDim = V.shape
	nF = OrigDim[0]
	VExtended = ReshapeTwo(V) #V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
	U, S, T = np.linalg.svd(VExtended)
	VExtended = VExtended @ T.T
	if ReturnFull:
		return ReshapeTwo(VExtended) #VExtended.reshape(OrigDim)
	VExtended = VExtended[:, :(S.shape[0])] #[:, :(nF * nF)]
	#print(ReshapeTwo(VExtended))
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
	return ReshapeTwo(VExtended) #VExtended.reshape(nF, nF, nF, nF)

#def ThreeExternal(V, ReturnFull = False):
#	OrigDim = V.shape
#	nF = OrigDim[0]
#	VExtended = V.reshape(V.shape[0], V.shape[1] * V.shape[2] * V.shape[3])
#	U, S, T = np.linalg.svd(VExtended)
#	VExtended = VExtended @ T.T
#	print(VExtended[:, :nF])
#	Idx = list(range(nF))
#	if ReturnFull:
#		return VExtended.reshape(OrigDim)
#	return VExtended.reshape(OrigDim)[np.ix_(Idx, Idx, Idx, Idx)]

def ThreeExternal(V):
	VFABB = TwoExternal(V)
	print(VFABB[0,0])
	VBBFA = np.swapaxes(VFABB, 0, 2)
	VBBFA = np.swapaxes(VBBFA, 1, 3)
	print(VBBFA[:, :, 0, 0])
	VBBFB = OneExternal(VBBFA)[0]
	print(VBBFB)
	VFBBB = np.swapaxes(VBBFB, 0, 2)
	VFBBB = np.swapaxes(VFBBB, 1, 3)
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

	VFFFA = VLO[np.ix_(FIndices, FIndices, FIndices, BEIndices)] #VLO[FIndices, :, :, :][:, FIndices, :, :][:, :, FIndices, :][:, :, :, :]
	VFFFB, TOneIdx = OneExternal(VFFFA)
	VFFAA = VLO[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]#[FIndices, :, :, :][:, FIndices, :, :][:, :, BEIndices, :][:, :, :, BEIndices]
	VbbAA = np.einsum('ijkl,ip,jq->pqkl', VLO, TBath, TBath)
	VbbAA = VbbAA[np.ix_(list(range(Nf)), list(range(Nf)), BEIndices, BEIndices)]
	VFFBB = TwoExternal(VFFAA, VbbAA = VbbAA)
	print(VFFBB)
	VFAFA = VLO[np.ix_(FIndices, BEIndices, FIndices, BEIndices)]
	VbAbA = np.einsum('ijkl,ip,kq->pjql', VLO, TBath, TBath)
	VbAbA = VbAbA[np.ix_(list(range(Nf)), BEIndices, list(range(Nf)), BEIndices)]
	VFAFA_Phys = np.swapaxes(VFAFA, 1, 2)
	VbAbA_Phys = np.swapaxes(VbAbA, 1, 2)
	VFBFB_Phys = TwoExternal(VFAFA_Phys, VbbAA = VbAbA_Phys)
	VFBFB = np.swapaxes(VFBFB_Phys, 1, 2)
	print(VFBFB)
	VFAAA = VLO[np.ix_(FIndices, BEIndices, BEIndices, BEIndices)]
	VFBBB = ThreeExternal(VFAAA)
	print(VFBBB)

	VFFFF = VLO[np.ix_(FIndices, FIndices, FIndices, FIndices)]
	VBBBB = VLO[np.ix_(BIndices, BIndices, BIndices, BIndices)]

	VTest = CompileFullV(VFFFF, VFFFB, VFFBB, VFBFB, VFBBB, VBBBB)
	print(VTest)	
