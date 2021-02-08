import numpy as np
from scipy.optimize import newton
import math
from timer import Timer

def Rotate4DTensor(A, R):
	from functools import reduce
	Vmmaa = np.zeros((R.shape[1], R.shape[1], A.shape[2], A.shape[3]))
	V = np.zeros((R.shape[1], R.shape[1], R.shape[1], R.shape[1]))
	for i in range(A.shape[2]):
		for j in range(A.shape[3]):
			Y = reduce(np.dot, (R.T, A[:, :, i, j], R))
			Vmmaa[:, :, i, j] = Y
	for i in range(Vmmaa.shape[0]):
		for j in range(Vmmaa.shape[1]):
			Y = reduce(np.dot, (R.T, Vmmaa[i, j], R))
			V[i, j] = Y
	return V

def T2LocalizedBasis(VMO, FMO, TMOtoLO, NumOcc, NumVir):
	FLO = TMOtoLO.T @ FMO @ TMOtoLO
	OccList = list(range(NumOcc))
	VirList = list(range(NumOcc, NumOcc + NumVir))
	VMO_OVOV = np.zeros(VMO.shape)
	VMO_OVOV[np.ix_(OccList, VirList, OccList, VirList)] = VMO[np.ix_(OccList, VirList, OccList, VirList)]
	VLO_OVOV = Rotate4DTensor(VMO_OVOV, TMOtoLO)
	T2LO = np.zeros(VMO.shape)
	t2LO = np.zeros(VMO.shape)
	print(FLO)
	for i in range(T2LO.shape[0]):
		for j in range(T2LO.shape[1]):
			for a in range(T2LO.shape[2]):
				for b in range(T2LO.shape[3]):
					T2LO[i, j, a, b] = VLO_OVOV[i, a, j, b] / (FLO[a, a] + FLO[b, b] - FLO[i, i] - FLO[j, j])
	for i in range(T2LO.shape[0]):
		for j in range(T2LO.shape[1]):
			for a in range(T2LO.shape[2]):
				for b in range(T2LO.shape[3]):
					t2LO = 2 * T2LO[i, j, a, b] - T2LO[i, j, b, a]
	return T2LO, t2LO

def MakePairDensity(T, t):
	PairP = np.zeros(T.shape) 
	for i in range(PairP.shape[0]):
		for j in range(PairP.shape[1]):
			Pij = np.dot(t[i, j].T, T[i, j]) + np.dot(t[i, j], T[i, j].T)
			if i == j:
				Pij = Pij / 2
			PairP[i, j] = Pij
	return PairP
	
def PrintNumPNO(P, tol = 1e-10):
	for i in range(P.shape[0]):
		Occi = []
		for j in range(P.shape[1]):
			e, v = np.linalg.eigh(P[i, j])
			Occi.append(sum(n > tol for n in e))
		print('\t'.join(map(str, Occi)))
			

def UpdateOEIwithMu(h, hcore, mu, CenterIdx):
	hnew = h.copy()
	hcorenew = hcore.copy()
	for i in CenterIdx:
		hnew[i, i] += mu
		hcorenew[i, i] += mu
	return hnew, hcorenew

def ErrorChemicalPotential(mu, args, nElectron, CenterIdx):
	Err = 0.0
	hnew, hcorenew = UpdateOEIwithMu(args[0], args[1], mu, CenterIdx)
	argsnew = args.copy()
	argsnew[0] = hnew
	argsnew[1] = hcorenew
	argsnew.append(mu)
	E, P = FragmentRMP2(*argsnew)
	for i in CenterIdx:
		Err += P[i, i]
	return Err - nElectron

def dErrorChemicalPotential(mu, args, nElectron, CenterIdx, dmu = 1e-6):
	errP = ErrorChemicalPotential(mu + dmu, args, nElectron, CenterIdx)
	errM = ErrorChemicalPotential(mu - dmu, args, nElectron, CenterIdx)
	return (errP - errM) / (dmu + dmu)
	
def NewtonRaphson(f, x0, df, args, tol = 1e-10, eps = 1e-6):
	F = f(x0, *args)
	x = x0 #.copy()
	it = 1
	print("Starting Newton Raphson")
	while not abs(F) < tol: #all([abs(x) < tol for x in F]):
		print("NR step", it)
		#t1 = Timer("Derivative Calculation")
		#t1.start()
		J = df(x, *args)
		print(x, F)
		#t1.stop()
		#print("L =", F)
		#print("J\n", J)
		#try:
		#	JInv = np.linalg.inv(J)
		#except:
		#	e, V = np.linalg.eig(J)
		#	JMod = np.zeros(J.shape)
		#	np.fill_diagonal(JMod, e + eps)
		#	JMod = V @ JMod @ V.T
		#	JInv = np.linalg.inv(JMod)
		JInv = 1.0 / J
		dx = -1.0 * JInv * F #-1.0 * JInv @ F
		#print("dx =", dx)
		#x, F = BacktrackLineSearch(f, x, dx, args, y0 = F) #x - alp * JInv @ F
		x = x + dx
		#t2 = Timer("Calculate Loss")
		#t2.start()
		F = f(x, *args)
		#t2.stop()
		#print(Timer.timers)
		it += 1
		#print("x =", x)
		#print("L =", F)
		#print((F**2.0).sum())
		#input()
	return x

def BisectionMethod(f, x0, args, tol = 1e-10):
	xa = x0[0]
	xb = x0[1]
	Fa = f(xa, *args)
	Fb = f(xb, *args)
	xc = (xa + xb) / 2
	Fc = f(xc, *args)
	it = 1
	print("Starting Bisection Method")
	while not abs(Fc) < tol:
		print("Bisection step", it)
		xc = (xa + xb) / 2
		Fc = f(xc, *args)
		print(xc, Fc)
		if np.sign(Fc) == np.sign(Fa):
			xa = xc
			Fa = Fc
		else:
			xb = xc
			Fb = Fc
		it += 1
		if abs(xa - xb) < 1e-16:
			break
	return xc
	

def CalcFragEnergy(h, hcore, V, P, G, CenterIndices):
	n = h.shape[0]
	AllIndices = list(range(n))
	hhFrag = h[CenterIndices] + hcore[CenterIndices]
	PFrag = P[CenterIndices]
	E1 = (hhFrag * PFrag).sum() / 2
	VFrag = V[CenterIndices]
	GFrag = G[CenterIndices]
	E2 = (VFrag * GFrag).sum() / 2
	return E1 + E2

def CalcFragEnergy_(h, hcore, V, P, G, CenterIndices):
	n = h.shape[0]
	AllIndices = list(range(len(CenterIndices) * 2))
	hhFrag = h[CenterIndices] + hcore[CenterIndices]
	PFrag = P[CenterIndices]
	E1 = (hhFrag * PFrag)[:, AllIndices].sum() / 2
	VFrag = V[np.ix_(CenterIndices, AllIndices, AllIndices, AllIndices)]
	GFrag = G[np.ix_(CenterIndices, AllIndices, AllIndices, AllIndices)]
	E2 = (VFrag * GFrag).sum() / 2
	return E1 + E2


def FragmentRMP2(h, hcore, V, CenterIndices, mu = None, n = None, S = None):
	if S is None:
		S = np.eye(h.shape[0])
	mol = gto.M()
	if n is None:
		n = h.shape[0]
	mol.nelectron = n
	mol.max_memory = 32000

	x1 = Timer("do scf")
	x1.start()
	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: hcore
	mf.get_ovlp = lambda *args: S
	mf._eri = ao2mo.restore(8, V, n)
	mf.max_cycle = 1000

	mf.kernel()
	x1.stop()
	mo_coeff = mf.mo_coeff
	C = mo_coeff.T

	x2 = Timer("do mp2")
	x2.start()
	mp2 = mp.MP2(mf)
	mp2.verboseint = 0
	mp2.max_memory = 8000
	E, T2 = mp2.kernel(with_t2 = False)
	x2.stop()

	x3 = Timer("make rdm")
	x33 = Timer("xform rdm")
	x3.start()
	P = mp2.make_rdm1()
	x3.stop()
	x33.start()
	P = C.T @ P @ C
	x33.stop()
	x3.start()
	#print(P)
	G = mp2.make_rdm2()
	x3.stop()
	x33.start()
	G = Rotate4DTensor(G, C)
	#G = np.einsum('ijkl,ip,jq,kr,ls->pqrs', G, C, C, C, C)
	x33.stop()

	x4 = Timer("add mu")
	x4.start()
	#E1 = np.einsum('pq,qp->', hcore, P)
	#E2 = np.einsum('pqrs,pqrs->', G, V)
	#print(E1 + E2)
	if mu is not None:
		hnew, hcorenew = UpdateOEIwithMu(h, hcore, -mu, CenterIndices)
	else:
		hnew = h
		hcorenew = hcore
	x4.stop()

	x5 = Timer("calc frag energy")
	x5.start()
	FragE = CalcFragEnergy(hnew, hcorenew, V, P, G, CenterIndices)
	print("Frag E =", FragE)
	x5.stop()

	print(Timer.timers)
	return FragE, P

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


if __name__ == '__main__':
	from functools import reduce
	from pyscf import gto, scf, mp, lo, ao2mo
	from frankenstein.tools.tensor_utils import get_symm_mat_pow
	mol = gto.M()
	#mol.fromfile("0.xyz")
	#mol.atom = 'H 0 0 0; H 1 0 0; H 1 1 0; H 0 1 0'
	CC = 0.5 #1.39
	CH = 0.5 #1.09
	for i in range(20):
		angle = i / 20.0 * 2.0 * np.pi
		#mol.atom.append(('C', (CC * np.sin(angle), CC * np.cos(angle), 0)))
		mol.atom.append(('H', ((CH + CC) * np.sin(angle), (CH + CC) * np.cos(angle), 0)))
	mol.basis = 'sto-3g'
	# H - 5 basis functions, C - 14 basis functions
	mol.build()
	ne = mol.nelec
	nelec = ne[0] + ne[1]

	mf = scf.RHF(mol)
	mf.kernel()
	print("E_elec(HF) =", mf.e_tot - mf.energy_nuc())
	NBasisC = 14
	NBasisH = 5
	#FragStart = NBasisC * 2 + NBasisH * 5 # Leave out the first CH3CH2 chunk
	#FragEnd = FragStart + NBasisC * 1 + NBasisH * 2 # CH2 Fragment
	FragStart = 0
	Nf = 1 #NBasisC * 1 + NBasisH * 1
	FragEnd = FragStart + Nf

	nocc = int(np.sum(mf.mo_occ) / 2)

	S = mol.intor_symmetric("int1e_ovlp")
	N = S.shape[0]
	mo_coeff = mf.mo_coeff
	StoOrth = get_symm_mat_pow(S, 0.50)
	StoOrig = get_symm_mat_pow(S, -0.5)
	mo_occ = mo_coeff[:, :nocc]
	mo_occ = np.dot(StoOrth.T, mo_occ)
	mo_vir = mo_coeff[:, nocc:]
	#mo_vir = np.dot(StoOrth.T, mo_vir) # Maybe not the best idea to localize it
	P = np.dot(mo_occ, mo_occ.T)
	FIdx = list(range(FragStart, FragEnd))
	AIdx = list(range(N))
	BEIdx = list(set(AIdx) - set(FIdx))
	PFrag = P[np.ix_(FIdx, FIdx)]
	PEnvBath = P[np.ix_(BEIdx, BEIdx)]
	eEnv, vEnv = np.linalg.eigh(PEnvBath)
	thresh = 1.e-9
	BathOrbs = [i for i, v in enumerate(eEnv) if v > thresh and v < 1.0 - thresh]
	print(len(BathOrbs))
	EnvOrbs  = [i for i, v in enumerate(eEnv) if v < thresh or v > 1.0 - thresh]
	CoreOrbs = [i for i, v in enumerate(eEnv) if v > 1.0 - thresh]
	print(len(CoreOrbs))
	TBath = np.zeros((N, len(BathOrbs)))
	TBath[BEIdx, :] = vEnv[:,BathOrbs]
	TEnv  = np.zeros((N, len(EnvOrbs)))
	TEnv[BEIdx, :] = vEnv[:, EnvOrbs]
	TFrag = np.zeros((N, Nf))
	TFrag[FIdx, :] = np.eye(Nf)
	TSch = np.concatenate((TFrag, TBath), axis = 1)
	T = np.concatenate((TSch, TEnv), axis = 1)
	BathOrbs = [x + Nf for x in BathOrbs]
	SchOrbs = list(range(Nf)) + BathOrbs
	EnvOrbs = [x + Nf for x in EnvOrbs]
	PEnv = reduce(np.dot, (TEnv.T, P, TEnv))
	PSch = reduce(np.dot, (TSch.T, P, TSch))
	PEnv[PEnv < thresh] = 0.0
	PEnv[PEnv > 1.0 - thresh] = 1.0
	print(PSch)
	print(PEnv)
	FIndices = list(range(Nf)) # In the Schmidt space
	BIndices = list(range(Nf, Nf + len(BathOrbs))) # In the Schmidt space
	BEIndices = list(range(Nf, N))
	SIndices = FIndices + BIndices
	CoreIdx = [i + Nf for i in CoreOrbs]

	TTotal = np.dot(StoOrig, T) # AO to SO
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

	TFrag = TTotal[:, SIndices]
	#TFrag = np.concatenate((TFrag, mo_vir), axis = 1)

	hFrag = reduce(np.dot, (TFrag.T, mf.get_hcore(), TFrag))
	VFrag = ao2mo.kernel(mol, TFrag)
	VFrag = ao2mo.restore(1, VFrag, hFrag.shape[0])

	#VFrag = VSO[SIndices, :, :, :][:, SIndices, :, :][:, :, SIndices, :][:, :, :, SIndices]
	#hFrag = hSO[SIndices, :][:, SIndices]
	hNoCore = hFrag.copy()
	for i in SIndices:
		for j in SIndices:
			CoreContribution = 0.0
			for c in CoreIdx:
				CoreContribution += (2.0 * VSO[i, j, c, c] - VSO[i, c, c, j])
			hFrag[i, j] += CoreContribution

	mp2 = mp.MP2(mf)
	mp2.max_memory = 32000
	E, T2 = mp2.kernel()
	print("E_elec(MP2) =", mf.e_tot + E - mf.energy_nuc())
	#EMP2 = mf.e_tot + E - mf.energy_nuc()	

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

	EFragMP2 = FragmentRMP2(hNoCore, hFrag, VFrag, FIndices)
	#print(EFragMP2 * 6)
	NewtonRaphson(ErrorChemicalPotential, 0.0, dErrorChemicalPotential, [[hNoCore, hFrag, VFrag, FIndices], nelec / 20, FIndices])

	TMOtoLO = np.dot(TMOtoAO, StoOrig)

	#VMO_OVOV = np.zeros(VMO.shape)
	TMO = np.zeros(VMO.shape)
	tMO = np.zeros(VMO.shape)
	OccIdx = list(range(Nocc))
	VirIdx = list(range(Nocc, Norb))
	TMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = T2
	tMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = t2
	T2LO = Rotate4DTensor(TMO, TMOtoLO)
	t2LO = Rotate4DTensor(tMO, TMOtoLO)
	print(T2LO[0, 0])
	T2LO_FFAA = np.zeros((len(FIndices), len(FIndices), len(BEIndices), len(BEIndices)))	
	t2LO_FFAA = np.zeros((len(FIndices), len(FIndices), len(BEIndices), len(BEIndices)))	
	T2LO_FFAA = T2LO[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]
	t2LO_FFAA = t2LO[np.ix_(FIndices, FIndices, BEIndices, BEIndices)]
	PairP = MakePairDensity(T2LO_FFAA, t2LO_FFAA)
	ePNO, vPNO = np.linalg.eigh(PairP[0, 0])
	print(ePNO)
	PNOIdx = [i for i, v in enumerate(ePNO) if v > 2e-5]
	print(len(PNOIdx))
	PNOs = np.zeros((Norb, len(PNOIdx)))
	PNOs[BEIndices, :] = vPNO[:, PNOIdx]
	TAOtoPNO = np.dot(StoOrig, PNOs)
	TFrag = np.concatenate((TTotal[:, FIndices], TAOtoPNO), axis = 1)
	hFrag = reduce(np.dot, (TFrag.T, mf.get_hcore(), TFrag))
	VFrag = ao2mo.kernel(mol, TFrag)
	VFrag = ao2mo.restore(1, VFrag, hFrag.shape[0])

	#NewtonRaphson(ErrorChemicalPotential, 0.0, dErrorChemicalPotential, [[hFrag, hFrag, VFrag, FIndices], nelec / 20, FIndices])
	BisectionMethod(ErrorChemicalPotential, [-3., 3.], [[hFrag, hFrag, VFrag, FIndices], nelec / 20, FIndices])



	'''
	#VMO_OVOV = np.zeros(VMO.shape)
	TMO = np.zeros(VMO.shape)
	tMO = np.zeros(VMO.shape)
	OccIdx = list(range(Nocc))
	VirIdx = list(range(Nocc, Norb))
	TMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = T2
	tMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = t2

	T2MMVV = np.zeros((Norb, Norb, Nvir, Nvir))
	t2MMVV = np.zeros((Norb, Norb, Nvir, Nvir))
	T2MMVV[:Nocc, :, :, :][:, :Nocc, :, :] = T2
	t2MMVV[:Nocc, :, :, :][:, :Nocc, :, :] = t2
	T2LLVV = np.einsum('ijkl,ia,jb->abkl', T2MMVV, TMOtoLO, TMOtoLO)
	t2LLVV = np.einsum('ijkl,ia,jb->abkl', t2MMVV, TMOtoLO, TMOtoLO)

	#PairP = np.zeros((len(FIndices), len(FIndices), Nvir, Nvir))
	#for f, i in enumerate(FIndices):
	#	for g, j in enumerate(FIndices):
	#		Pij = np.dot(t2LLVV[i, j].T, T2LLVV[i, j]) + np.dot(t2LLVV[i, j], T2LLVV[i, j].T)
	#		if i == j:
	#			Pij = Pij / 2
	#		PairP[f, g] = Pij
	#		e, v = np.linalg.eigh(Pij)
	#PrintNumPNO(PairP, tol = 1e-5)
	#ePNO, vPNO = np.linalg.eigh(PairP[4, 4])
	#P88 = np.dot(t2LLVV[8, 8].T, T2LLVV[8, 8]) + np.dot(t2LLVV[8, 8], T2LLVV[8, 8].T)
	#P88 = P88 / 2
	#ePNO, vPNO = np.linalg.eigh(P88)
	PNOIdx = [i for i, v in enumerate(ePNO) if v < 1e-6]
	PNOs = vPNO[:, PNOIdx]
	TMOtoPNO = np.zeros((Norb, len(PNOIdx)))
	TMOtoPNO[VirIdx, :] = PNOs

	TAOtoPNO = np.dot(mo_coeff, TMOtoPNO)
	TFragAndPNO = np.concatenate((TFrag, TAOtoPNO), axis = 1)
	hFragAndPNO = reduce(np.dot, (TFragAndPNO.T, mf.get_hcore(), TFragAndPNO))
	VFragAndPNO = ao2mo.kernel(mol, TFragAndPNO)
	VFragAndPNO = ao2mo.restore(1, VFragAndPNO, hFragAndPNO.shape[0])

	hNoCore = hFragAndPNO.copy()
	for i in SIndices:
		for j in SIndices:
			CoreContribution = 0.0
			for c in CoreIdx:
				CoreContribution += (2.0 * VSO[i, j, c, c] - VSO[i, c, c, j])
			hFragAndPNO[i, j] += CoreContribution

	EFragMP2 = FragmentRMP2(hNoCore, hFragAndPNO, VFragAndPNO, FIndices)[0]
	print(EFragMP2 * 6)
	#NewtonRaphson(ErrorChemicalPotential, 0.0, dErrorChemicalPotential, [[hNoCore, hFragAndPNO, VFragAndPNO, FIndices], nelec / 6, FIndices])

	'''
