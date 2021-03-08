import numpy as np
from scipy.optimize import newton
import math
from timer import Timer

def GramSchmidt(A, Start = 0):
	B = A.copy()
	for j in range(Start, B.shape[1] - 1):
		CurrentColIdx = j + 1
		CurrentColVec = B[:, CurrentColIdx]
		Projections = CurrentColVec.T @ B[:, :CurrentColIdx]
		for i in range(j + 1):
			CurrentColVec -= Projections[i] * B[:, i]
		Norm = np.linalg.norm(CurrentColVec)
		if Norm < 1e-10:
			CurrentColVec = np.zeros(CurrentColVec.shape)
		else:
			CurrentColVec = CurrentColVec / Norm
		B[:, CurrentColIdx] = CurrentColVec
	return B

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

def HalfRotate4DTensor(A, R):
	from functools import reduce
	Vmmaa = np.zeros((R.shape[1], R.shape[1], A.shape[2], A.shape[3]))
	for i in range(A.shape[2]):
		for j in range(A.shape[3]):
			Y = reduce(np.dot, (R.T, A[:, :, i, j], R))
			Vmmaa[:, :, i, j] = Y
	return Vmmaa
	
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

def TracePairDensity(PairP, Idx = None):
	if Idx == None:
		Idx = list(range(PairP.shape[0]))
	P = np.zeros(PairP[0, 0].shape)
	for i in Idx:
		for j in Idx:
			P += PairP[i, j]
	return P
	
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
	while int(np.sign(Fa)) == int(np.sign(Fb)):
		xa -= 1.0
		xb += 1.0
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
	
def Scan(f, x0, args, tol = 1e-3):
	xa = x0[0]
	xb = x0[1]
	Fa = f(xa, *args)
	while xa < xb:
		print(xa, Fa)
		xa += tol
		Fa = f(xa, *args)

def GetGCore(VAO, PAO, VFrag, PFrag, TFrag):
	from pyscf.scf.hf import dot_eri_dm
	JAO, KAO = dot_eri_dm(VAO, PAO)
	GAO = 2 * JAO - KAO
	JFrag, KFrag = dot_eri_dm(VFrag, PFrag)
	GFrag = 2 * JFrag - KFrag
	GFull = TFrag.T @ GAO @ TFrag
	return GFull - GFrag
	
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

def CalcFragEnergy_(h, hcore, V, P, G):
	n = h.shape[0]
	CenterIndices = [11]
	AllIndices = list(range(n))
	hhFrag = h[CenterIndices] + hcore[CenterIndices]
	PFrag = P[CenterIndices]
	E1 = (hhFrag * PFrag).sum() / 2
	VFrag = V[CenterIndices]
	GFrag = G[CenterIndices]
	E2 = (VFrag * GFrag).sum() / 2
	return E1 + E2

def FragmentRMP2(h, hcore, V, CenterIndices, n = None, mu = None, S = None):
	if S is None:
		S = np.eye(h.shape[0])
	mol = gto.M()
	if n is None:
		n = h.shape[0]
	mol.nelectron = n
	mol.max_memory = 32000
	mol.verbose = 0

	#x1 = Timer("do scf")
	#x1.start()
	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: hcore
	mf.get_ovlp = lambda *args: S
	mf._eri = ao2mo.restore(8, V, h.shape[0])
	mf.max_cycle = 1000

	mf.kernel()
	#x1.stop()
	mo_coeff = mf.mo_coeff
	C = mo_coeff.T

	#x2 = Timer("do mp2")
	#x2.start()
	mp2 = mp.MP2(mf)
	mp2.max_memory = 32000
	E, T2 = mp2.kernel(with_t2 = False)
	print("E MP2 Raw =", E)
	#x2.stop()

	#x3 = Timer("make rdm")
	#x33 = Timer("xform rdm")
	#x3.start()
	P = mp2.make_rdm1()
	#x3.stop()
	#x33.start()
	P = C.T @ P @ C
	#x33.stop()
	#x3.start()
	#print(P)
	G = mp2.make_rdm2()
	#x3.stop()
	#x33.start()
	G = Rotate4DTensor(G, C)
	#G = np.einsum('ijkl,ip,jq,kr,ls->pqrs', G, C, C, C, C)
	#x33.stop()

	#x4 = Timer("add mu")
	#x4.start()
	#E1 = np.einsum('pq,qp->', hcore, P)
	#E2 = np.einsum('pqrs,pqrs->', G, V)
	#print(E1 + E2)
	if mu is not None:
		hnew, hcorenew = UpdateOEIwithMu(h, hcore, -mu, CenterIndices)
	else:
		hnew = h
		hcorenew = hcore
	#x4.stop()

	#x5 = Timer("calc frag energy")
	#x5.start()
	#FragE = CalcFragEnergy_(hnew, hcorenew, V, P, G)
	FragE = CalcFragEnergy(hnew, hcorenew, V, P, G, CenterIndices)
	print("Frag E =", FragE)
	#x5.stop()

	#print(Timer.timers)
	#print(np.diag(P))
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
	mol.fromfile("0.xyz")
	#mol.atom = 'H 0 0 0; H 1 0 0; H 1 1 0; H 0 1 0'
	#CC = 0.5 #1.39
	#CH = 0.5 #1.09
	#for i in range(20):
	#	angle = i / 20.0 * 2.0 * np.pi
	#	#mol.atom.append(('C', (CC * np.sin(angle), CC * np.cos(angle), 0)))
	#		mol.atom.append(('H', ((CH + CC) * np.sin(angle), (CH + CC) * np.cos(angle), 0)))
	mol.basis = 'sto-3g'
	# H - 5 basis functions, C - 14 basis functions
	mol.build()
	ne = mol.nelec
	nelec = ne[0] + ne[1]

	mf = scf.RHF(mol)
	mf.kernel()
	print("E_elec(HF) =", mf.e_tot - mf.energy_nuc())
	if mol.basis == 'cc-pvdz':
		NBasisC = 14
		NBasisH = 5
	if mol.basis == 'sto-3g':
		NBasisC = 5
		NBasisH = 1
	#FragStart = NBasisC * 2 + NBasisH * 5 # Leave out the first CH3CH2 chunk
	#FragEnd = FragStart + NBasisC * 1 + NBasisH * 2 # CH2 Fragment
	FragStart = 0
	Nf = NBasisC * 1 + NBasisH * 1
	NumFrag = 6
	FragEnd = FragStart + Nf
	print("Number of Fragments:", NumFrag)

	nocc = int(np.sum(mf.mo_occ) / 2)

	S = mol.intor_symmetric("int1e_ovlp")
	N = S.shape[0]
	mo_coeff = mf.mo_coeff
	StoOrth = get_symm_mat_pow(S, 0.50)
	StoOrig = get_symm_mat_pow(S, -0.5)
	np.save("StoOrth", StoOrth)
	np.save("StoOrig", StoOrig)
	mo_occ = mo_coeff[:, :nocc]
	PAO = np.dot(mo_occ, mo_occ.T)
	mo_occ = np.dot(StoOrth.T, mo_occ)
	mo_vir = mo_coeff[:, nocc:]
	#mo_vir = np.dot(StoOrth.T, mo_vir) # Maybe not the best idea to localize it
	P = np.dot(mo_occ, mo_occ.T)
	FIdx = list(range(FragStart, FragEnd))
	AIdx = list(range(N))
	BEIdx = list(set(AIdx) - set(FIdx))
	PEnvBath = P[np.ix_(BEIdx, BEIdx)]
	eEnv, vEnv = np.linalg.eigh(PEnvBath)
	thresh = 1.e-6
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
	PSO = reduce(np.dot, (T.T, P, T))
	FIndices = list(range(Nf)) # In the Schmidt space
	BIndices = list(range(Nf, Nf + len(BathOrbs))) # In the Schmidt space
	print("Num Frag and Bath", Nf, len(BathOrbs))
	BEIndices = list(range(Nf, N))
	SIndices = FIndices + BIndices
	CoreIdx = [i + Nf for i in CoreOrbs]

	PHFFrag = PSO[np.ix_(SIndices, SIndices)]
	NumOccFrag = int(round(np.trace(PHFFrag)))

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
	#for i in SIndices:
	#	for j in SIndices:
	#		CoreContribution = 0.0
	#		for c in CoreIdx:
	#			CoreContribution += (2.0 * VSO[i, j, c, c] - VSO[i, c, c, j])
	#		hFrag[i, j] += CoreContribution

	#STFrag = S @ TFrag
	#PFrag = STFrag.T @ PAO @ STFrag
	GCore = GetGCore(VAO, PAO, VFrag, PHFFrag, TFrag)
	np.save("GCore", GCore)
	hFrag = hNoCore + GCore
	mp2 = mp.MP2(mf)
	mp2.max_memory = 32000
	E, T2 = mp2.kernel()

	MP2RDM = mp2.make_rdm1()
	MP2RDMAO = TMOtoAO.T @ MP2RDM @ TMOtoAO

	print("E_elec(MP2) =", mf.e_tot + E - mf.energy_nuc())
	#EMP2 = mf.e_tot + E - mf.energy_nuc()	

	Nocc = T2.shape[0]
	Nvir = T2.shape[2]
	Norb = Nocc + Nvir
	print(T2.shape)
	time1 = Timer("Make t2")
	time1.start()
	T2X = np.swapaxes(T2, 2, 3)
	t2 = 2 * T2 - T2X
	time1.stop()

	EFragMP2, PFragMP2 = FragmentRMP2(hNoCore, hFrag, VFrag, FIndices, n = 2 * NumOccFrag)
	np.savetxt("be-density0.txt", PFragMP2, delimiter = '\t')
	#print(EFragMP2 * 6)
	mu = NewtonRaphson(ErrorChemicalPotential, 0.0, dErrorChemicalPotential, [[hNoCore, hFrag, VFrag, FIndices, 2 * NumOccFrag], nelec / NumFrag, FIndices])
	hnew, hcorenew = UpdateOEIwithMu(hNoCore, hFrag, mu, FIndices)
	EFrag, PFrag = FragmentRMP2(hnew, hcorenew, VFrag, FIndices, n = 2 * NumOccFrag, mu = mu)
	np.savetxt("be-density.txt", PFrag, delimiter = '\t')

	print("BE Energy:", NumFrag * EFrag)

	time2 = Timer("Make TMOtoLO")
	time2.start()
	TMOtoLO = np.dot(TMOtoAO, StoOrig)
	time2.stop()

	# PNO Bath theory
	'''
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

	#NewtonRaphson(ErrorChemicalPotential, 0.0, dErrorChemicalPotential, [[hFrag, hFrag, VFrag, FIndices], nelec / NumFrag, FIndices])
	BisectionMethod(ErrorChemicalPotential, [-3., 3.], [[hFrag, hFrag, VFrag, FIndices], nelec / NumFrag, FIndices])
	'''


	# PNO-BE Theory
	
	print("BEGIN PNO-BE")
	# pnobe_tol = 1e-3
	num_pno = 1
	
	time3 = Timer("def TMO")
	time3.start()
	#VMO_OVOV = np.zeros(VMO.shape)
	TMO = np.zeros(VMO.shape)
	tMO = np.zeros(VMO.shape)
	OccIdx = list(range(Nocc))
	VirIdx = list(range(Nocc, Norb))
	TMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = T2
	tMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = t2

	T2MMVV = np.zeros((Norb, Norb, Nvir, Nvir))
	t2MMVV = np.zeros((Norb, Norb, Nvir, Nvir))
	time3.stop()
	time33 = Timer("fill TMO")
	time33.start()
	T2MMVV[:Nocc, :, :, :][:, :Nocc, :, :] = T2
	t2MMVV[:Nocc, :, :, :][:, :Nocc, :, :] = t2
	time33.stop()
	time7 = Timer("rotate T")
	time7.start()
	T2LLVV = HalfRotate4DTensor(T2MMVV, TMOtoLO) #np.einsum('ijkl,ia,jb->abkl', T2MMVV, TMOtoLO, TMOtoLO)
	t2LLVV = HalfRotate4DTensor(t2MMVV, TMOtoLO) #np.einsum('ijkl,ia,jb->abkl', t2MMVV, TMOtoLO, TMOtoLO)
	time7.stop()

	time4 = Timer("Make Pair Density")
	time4.start()
	PairP = MakePairDensity(T2LLVV, t2LLVV) #np.zeros((len(FIndices), len(FIndices), Nvir, Nvir))
	time4.stop()
	print("Fragment-NonFragment block of Pair Density")
	print(np.sqrt((PairP[np.ix_(FIndices, BEIndices)]**2).sum()))
	time5 = Timer("Trace pair density")
	time5.start()
	PairPFrag = TracePairDensity(PairP, Idx = FIndices)
	time5.stop()
	ePNO, vPNO = np.linalg.eigh(PairPFrag)
	print(ePNO)
	np.savetxt("ePNO.txt", ePNO, delimiter = '\n')
	#PNOIdx = [i for i, v in enumerate(ePNO) if v > pnobe_tol]
	PNOIdx = list(range(len(ePNO) - num_pno, len(ePNO)))
	print("Number of included PNOs:", len(PNOIdx))
	PNOs = vPNO[:, PNOIdx]
	TMOtoPNO = np.zeros((Norb, len(PNOIdx)))
	TMOtoPNO[VirIdx, :] = PNOs

	time6 = Timer("rotate ints")
	time6.start()
	TAOtoPNO = np.dot(mo_coeff, TMOtoPNO)
	TFragAndPNO = np.concatenate((TFrag, TAOtoPNO), axis = 1)
	TLOtoFBPNO_NotOrth = StoOrth @ TFragAndPNO
	TLOtoFBPNO = GramSchmidt(TLOtoFBPNO_NotOrth, TFrag.shape[1] - 1)
	TFragAndPNO = StoOrig @ TLOtoFBPNO
	np.save("TFBPNO", TFragAndPNO)
	hFragAndPNO = reduce(np.dot, (TFragAndPNO.T, mf.get_hcore(), TFragAndPNO))
	VFragAndPNO = ao2mo.kernel(mol, TFragAndPNO)
	VFragAndPNO = ao2mo.restore(1, VFragAndPNO, hFragAndPNO.shape[0])
	time6.stop()

	MP2RDMFragAndPNO = TFragAndPNO.T @ MP2RDMAO @ TFragAndPNO
	np.savetxt("mp2_rdm1_fragpno.txt", MP2RDMFragAndPNO)

	hNoCore = hFragAndPNO.copy()
	#for i in SIndices:
	#	for j in SIndices:
	#		CoreContribution = 0.0
	#		for c in CoreIdx:
	#			CoreContribution += (2.0 * VSO[i, j, c, c] - VSO[i, c, c, j])
	#		hFragAndPNO[i, j] += CoreContribution

	STFragAndPNO = S @ TFragAndPNO
	PHFFragAndPNO = STFragAndPNO.T @ PAO @ STFragAndPNO
	GCore = GetGCore(VAO, PAO, VFragAndPNO, PHFFragAndPNO, TFragAndPNO)
	hFragAndPNO = hFragAndPNO + GCore
	np.save("GCorePNO", GCore)
	np.save("PHFFBPNO", PHFFragAndPNO)
	np.save("VFBPNO", VFragAndPNO)

	NumOccFrag = int(round(np.trace(PHFFragAndPNO)))
	print(np.diag(PHFFragAndPNO))
	print("PNO-BE Frag Occ:", NumOccFrag)
	
	EFragMP2, PFragMP2 = FragmentRMP2(hNoCore, hFragAndPNO, VFragAndPNO, FIndices, n = 2 * NumOccFrag)
	np.savetxt("bepno-density0.txt", PFragMP2, delimiter = '\t')
	#print(EFragMP2 * 6)
	print(Timer.timers)
	mu = BisectionMethod(ErrorChemicalPotential, [-2., 2.], [[hNoCore, hFragAndPNO, VFragAndPNO, FIndices, 2 * NumOccFrag], nelec / NumFrag, FIndices], tol = 1e-5)
	#mu = NewtonRaphson(ErrorChemicalPotential, 0.0, dErrorChemicalPotential, [[hNoCore, hFragAndPNO, VFragAndPNO, FIndices, 2 * NumOccFrag], nelec / NumFrag, FIndices])
	hnew, hcorenew = UpdateOEIwithMu(hNoCore, hFragAndPNO, mu, FIndices)
	EFrag, PFrag = FragmentRMP2(hnew, hcorenew, VFragAndPNO, FIndices, n = 2 * NumOccFrag, mu = mu)
	np.savetxt("bepno-density.txt", PFrag, delimiter = '\t')


	print("PNO-BE Energy:", NumFrag * EFrag)
	


	# PNO Disentanglement method
'''	
	print("BEGIN PNO DISENTANGLE")	
	PNO_RDM_lambda = 1.

	TMO = np.zeros(VMO.shape)
	tMO = np.zeros(VMO.shape)
	OccIdx = list(range(Nocc))
	VirIdx = list(range(Nocc, Norb))
	TMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = T2
	tMO[np.ix_(OccIdx, OccIdx, VirIdx, VirIdx)] = t2

	TLO = Rotate4DTensor(TMO, TMOtoLO)
	tLO = Rotate4DTensor(tMO, TMOtoLO)
	MP2RDM = TMOtoLO.T @ MP2RDM @ TMOtoLO
	np.save("MP2RDM", MP2RDM)

	PairP = MakePairDensity(TLO, tLO)
	PairPFrag = TracePairDensity(PairP, Idx = FIndices)

	mo_coeff = mf.mo_coeff
	StoOrth = get_symm_mat_pow(S, 0.50)
	StoOrig = get_symm_mat_pow(S, -0.5)
	mo_occ = mo_coeff[:, :nocc]
	PAO = np.dot(mo_occ, mo_occ.T)
	mo_occ = np.dot(StoOrth.T, mo_occ)
	mo_vir = mo_coeff[:, nocc:]
	#mo_vir = np.dot(StoOrth.T, mo_vir) # Maybe not the best idea to localize it
	P = np.dot(mo_occ, mo_occ.T)
	FIdx = list(range(FragStart, FragEnd))
	AIdx = list(range(N))
	BEIdx = list(set(AIdx) - set(FIdx))
	np.save("HFRDM", P)
	PEnvBath = (1.0 - PNO_RDM_lambda) * P[np.ix_(BEIdx, BEIdx)] + PNO_RDM_lambda * MP2RDM[np.ix_(BEIdx, BEIdx)] #PairPFrag[np.ix_(BEIdx, BEIdx)]
	eEnv, vEnv = np.linalg.eigh(PEnvBath)
	pnolambdaname = "mp2lambda_" + str(PNO_RDM_lambda)
	np.savetxt(pnolambdaname, eEnv, delimiter = '\n')
	thresh = 1.e-6
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
	PSO = reduce(np.dot, (T.T, P, T))
	FIndices = list(range(Nf)) # In the Schmidt space
	BIndices = list(range(Nf, Nf + len(BathOrbs))) # In the Schmidt space
	print("Num Frag and Bath", Nf, len(BathOrbs))
	BEIndices = list(range(Nf, N))
	SIndices = FIndices + BIndices
	CoreIdx = [i + Nf for i in CoreOrbs]

	PHFFrag = PSO[np.ix_(SIndices, SIndices)]
	NumOccFrag = int(round(np.trace(PHFFrag)))
	print("Num Electrons in Fragment:", NumOccFrag * 2)

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
	#for i in SIndices:
	#	for j in SIndices:
	#		CoreContribution = 0.0
	#		for c in CoreIdx:
	#			CoreContribution += (2.0 * VSO[i, j, c, c] - VSO[i, c, c, j])
	#		hFrag[i, j] += CoreContribution

	#STFrag = S @ TFrag
	#PFrag = STFrag.T @ PAO @ STFrag
	GCore = GetGCore(VAO, PAO, VFrag, PHFFrag, TFrag)
	hFrag = hNoCore + GCore
	mp2 = mp.MP2(mf)
	mp2.max_memory = 32000
	E, T2 = mp2.kernel()
	print("E_elec(MP2) =", mf.e_tot + E - mf.energy_nuc())
	#EMP2 = mf.e_tot + E - mf.energy_nuc()	

	Nocc = T2.shape[0]
	Nvir = T2.shape[2]
	Norb = Nocc + Nvir
	print(T2.shape)
	time1 = Timer("Make t2")
	time1.start()
	T2X = np.swapaxes(T2, 2, 3)
	t2 = 2 * T2 - T2X
	time1.stop()

	EFragMP2, PFragMP2 = FragmentRMP2(hNoCore, hFrag, VFrag, FIndices, n = 2 * NumOccFrag)
	np.save("PFrag", PFragMP2)
	#print(EFragMP2 * 6)
	mu = NewtonRaphson(ErrorChemicalPotential, 0.0, dErrorChemicalPotential, [[hNoCore, hFrag, VFrag, FIndices, 2 * NumOccFrag], nelec / NumFrag, FIndices])
	hnew, hcorenew = UpdateOEIwithMu(hNoCore, hFrag, mu, FIndices)
	EFrag, PFrag = FragmentRMP2(hnew, hcorenew, VFrag, FIndices, n = 2 * NumOccFrag, mu = mu)

	print("BE Energy:", NumFrag * EFrag)
'''	
