import numpy as np
from pyscf import gto, scf, ao2mo
from traniest.bootstrap import mp2_emb

def CustomMF(h, V, n, ENuc = 0.0, S = None):
	if S is None:
		S = np.eye(h.shape[0])
	mol = gto.M()
	mol.nelectron = n
	def energy_nuc():
		return ENuc
	mol.energy_nuc = energy_nuc

	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: h
	mf.get_ovlp = lambda *args: S
	mf._eri = ao2mo.restore(8, V, h.shape[0])

	return mf	

def MakeDFIntegrals(V):
	VThreeIndex = mp2_emb.TwoExternal(V)
	VThreeIndex = VThreeIndex.reshape(V.shape[0], V.shape[1], V.shape[2] * V.shape[3])
	V_DF = np.einsum('ijp,klp->ijkl', VThreeIndex, VThreeIndex)
	return V_DF

def CompareV(V, V1, V2):
	E1 = np.sqrt(((V - V1)**2).sum())
	E2 = np.sqrt(((V - V2)**2).sum())
	print("V Error:", E1, E2)
	return E1, E2

def CompareHF(h, V, n, mf, V1, V2, doMP2 = False, e_mp2 = None):
	mf1 = CustomMF(h, V1, n, ENuc = mf.energy_nuc())
	mf1.kernel()
	mf2 = CustomMF(h, V2, n, ENuc = mf.energy_nuc())
	mf2.kernel()

	E1 = (mf.e_tot - mf1.e_tot)
	E2 = (mf.e_tot - mf2.e_tot)

	print("HF Error:", E1, E2)
	if doMP2:
		from pyscf import mp
		mp1 = mp.MP2(mf1)
		mp1.kernel(with_t2=False)
		mp2 = mp.MP2(mf2)
		mp2.kernel(with_t2=False)
		print("MP2 Error:", e_mp2 - mp1.e_tot, e_mp2 - mp2.e_tot)
		return E1, E2, e_mp2 - mp1.e_tot, e_mp2 - mp2.e_tot
	return E1, E2

def RotateAuxBasis(L, C):
	CLC = np.zeros(L.shape)
	for i in range(CLC.shape[2]):
		CLC[:, :, i] = C.T @ L[:, :, i] @ C
	return CLC

def RotateTEIAuxBasis(L, C):
	CLC = RotateAuxBasis(L, C)
	VAux = np.zeros((L.shape[0], L.shape[1], L.shape[0], L.shape[1]))
	VAux = np.einsum('ijp,klp->ijkl', CLC, CLC)
	return VAux

def AuxBasisCD(V, Rank, tol = 1e-10):
	VExt = V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
	import tensorflow_probability as tfp
	L = np.asarray(tfp.math.pivoted_cholesky(VExt, Rank, diag_rtol = tol))
	L = L.reshape(V.shape[0], V.shape[1], L.shape[1])
	return np.asarray(L)

def AuxBasisSVD(V, thresh = 1e-4, UpTriReshape = False):
	if UpTriReshape:
		VExt = mp2_emb.ReshapeTwo(V)
	else:
		VExt = V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
	U, S, V = np.linalg.svd(VExt)
	x = np.argwhere(S > thresh)
	N = x.shape[0]
	M = np.zeros((N, N))
	np.fill_diagonal(M, S[x])
	return U[:, x[:, 0]], M, V[x[:, 0], :]

def AuxBasisDIAG(V, thresh = 1e-4, Rank = None, UpTriReshape = False):
	if UpTriReshape:
		VExt = mp2_emb.ReshapeTwo(V)
	else:
		VExt = V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
	# E, U = np.linalg.eigh(VExt)
	from scipy.sparse.linalg import eigsh # Lanczos
	E, L = eigsh(VExt, k = Rank)
	#if Rank is None:
	#	x = np.argwhere(E > thresh)
	#	x = list(x[:,0])
	#else:
	#	x = list(range(VExt.shape[1] - Rank, VExt.shape[1]))
	#N = len(x)
	#L = U[:, x]
	#for i, e in enumerate(E[x]):
	for i in range(len(E)):
		L[:, i] = L[:, i] * np.sqrt(max(E[i], 0.0))
	L = L.reshape(V.shape[0], V.shape[1], L.shape[1])
	return L

if __name__ == "__main__":
#def main():
	from functools import reduce
	from pyscf import mp, lo, ao2mo
	from frankenstein.tools.tensor_utils import get_symm_mat_pow

	mol = gto.Mole()
	mol.fromfile("/work/henry/Calculations/BE/Acene/sto-3g/geom/2.xyz")
	mol.basis = 'cc-pvdz'
	mol.build()

	mf = scf.RHF(mol)
	mf.kernel()

	mymp = mp.MP2(mf)
	mymp.kernel(with_t2=False)

	S = mol.intor_symmetric("int1e_ovlp")
	StoOrth = get_symm_mat_pow(S,  0.50)
	StoOrig = get_symm_mat_pow(S, -0.50)

	hAO = mf.get_hcore()
	I = np.eye(hAO.shape[0])
	VAO = ao2mo.kernel(mol, I)
	VAO = ao2mo.restore(1, VAO, hAO.shape[0])

	hMO = mf.mo_coeff.T @ hAO @ mf.mo_coeff
	VMO = ao2mo.kernel(mol, mf.mo_coeff)
	VMO = ao2mo.restore(1, VMO, hMO.shape[0])

	#hLO = reduce(np.dot, (StoOrig.T, mf.get_hcore(), StoOrig))
	#VLO = ao2mo.kernel(mol, StoOrig)
	#VLO = ao2mo.restore(1, VLO, hLO.shape[0])

	#return VAO, mf.mo_coeff

	Ranks = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
	TE_E1 = []
	TE_E2 = []
	HF_E1 = []
	HF_E2 = []
	MP_E1 = []
	MP_E2 = []

	for Rank in Ranks:
		L_CD = AuxBasisCD(VAO, Rank = Rank)
		L_DG = AuxBasisDIAG(VAO, Rank = Rank)

		VMO_CD = RotateTEIAuxBasis(L_CD, mf.mo_coeff)
		VMO_DG = RotateTEIAuxBasis(L_DG, mf.mo_coeff)

		V_E1, V_E2 = CompareV(VMO, VMO_CD, VMO_DG)
		E_E1, E_E2, F_E1, F_E2 = CompareHF(hMO, VMO, mol.nelectron, mf, VMO_CD, VMO_DG, doMP2 = True, e_mp2 = mymp.e_tot)
		TE_E1.append(V_E1)
		TE_E2.append(V_E2)
		HF_E1.append(E_E1)
		HF_E2.append(E_E2)
		MP_E1.append(F_E1)
		MP_E2.append(F_E2)

	print("HF Errors")
	print(HF_E1)
	print(HF_E2)
	print("MP2 Errors")
	print(MP_E1)
	print(MP_E2)
	N = len(Ranks)
	Results = np.zeros((N, 7))
	Results[:, 0] = Ranks
	Results[:, 1] = TE_E1
	Results[:, 2] = TE_E2
	Results[:, 3] = HF_E1
	Results[:, 4] = HF_E1
	Results[:, 5] = MP_E1
	Results[:, 6] = MP_E2

	np.savetxt("df_results.txt", Results, delimiter = '\t')
