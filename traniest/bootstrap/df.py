import numpy as np
from pyscf import gto, scf, ao2mo
from traniest.bootstrap import mp2_emb

def MakeDFIntegrals(V):
	VThreeIndex = mp2_emb.TwoExternal(V)
	VThreeIndex = VThreeIndex.reshape(V.shape[0], V.shape[1], V.shape[2] * V.shape[3])
	V_DF = np.einsum('ijp,klp->ijkl', VThreeIndex, VThreeIndex)
	return V_DF

def AuxBasisSVD(V, thresh = 1e-4):
	VExt = V.reshape(V.shape[0] * V.shape[1], V.shape[2] * V.shape[3])
	U, S, V = np.linalg.svd(VExt)
	x = np.argwhere(S > thresh)
	N = x.shape[0]
	M = np.zeros((N, N))
	np.fill_diagonal(M, S[x])
	return U[:, x[:, 0]], M, V[x[:, 0], :]

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

#if __name__ == "__main__":
def main():
	from functools import reduce
	from pyscf import mp, lo, ao2mo
	from frankenstein.tools.tensor_utils import get_symm_mat_pow

	mol = gto.Mole()
	mol.fromfile("/work/henry/Calculations/BE/Acene/sto-3g/geom/1.xyz")
	mol.basis = 'sto-3g'
	mol.build()

	mf = scf.RHF(mol)
	mf.kernel()

	S = mol.intor_symmetric("int1e_ovlp")
	StoOrth = get_symm_mat_pow(S,  0.50)
	StoOrig = get_symm_mat_pow(S, -0.50)

	hLO = reduce(np.dot, (StoOrig.T, mf.get_hcore(), StoOrig))
	VLO = ao2mo.kernel(mol, StoOrig)
	VLO = ao2mo.restore(1, VLO, hLO.shape[0])

	return VLO

	mf_LO = CustomMF(hLO, VLO, mol.nelectron, ENuc = mol.energy_nuc())
	mf_LO.kernel()

	VDF = MakeDFIntegrals(VLO)
	mf_DF = CustomMF(hLO, VDF, mol.nelectron, ENuc = mol.energy_nuc())
	mf_DF.kernel()
