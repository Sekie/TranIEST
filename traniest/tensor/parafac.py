import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares
from tensorly.tenalg import khatri_rao
from traniest.tools.tensor_utils import get_symm_mat_pow

def VectorsToTensor(As, Ls = None):
	Dim = ()
	if Ls is None:
		Ls = np.ones((As[0].shape[1],))
	for A in As:
		iN = A.shape[0]
		Dim += (iN, )
	X = np.zeros(Dim)
	for r in range(As[0].shape[1]): # Rank of decomposition
		ais = []
		for A in As:
			ais.append(A[:, r])
		X += Ls[r] * np.prod(np.ix_(*ais))
	return X

def PARAFACError(X, As, Ls = None, ReturnVector = False):
	x = VectorsToTensor(As, Ls = Ls)
	if ReturnVector:
		return (X - x)
	return np.sqrt(((X - x)**2).sum())

def DIIS_Solve(Es):
	B = np.full((len(Es) + 1, len(Es) + 1), -1.0)
	for i in range(len(Es)):
		for j in range(len(Es)):
			B[i, j] = (Es[i] * Es[j]).sum()
	B[len(Es), len(Es)] = 0.0
	y = np.zeros((len(Es) + 1,))
	y[len(Es)] = -1.0
	return np.linalg.solve(B, y)[:len(Es)]

def DIIS(As, Es):
	cs = DIIS_Solve(Es)
	NewA = np.zeros(As[0].shape)
	for n in range(len(As)):
		NewA += cs[n] * As[n]
	return NewA 
	
def Matricize(X, m):
	I = X.shape
	IArr = np.array(I)
	
	Dim1 = I[m]
	Dim2 = int(IArr.prod() / Dim1)
	Xm = X.copy()
	for i in range(m, 0, -1):
		Xm = np.swapaxes(Xm, i - 1, i)
	return Xm.reshape(Dim1, Dim2)

def UnMatricize(X, m, Dim):
	Xu = X.reshape(Dim)
	for i in range(m):
		Xu = np.swapaxes(Xu, i, i + 1)
	return Xu

'''
Input
	X (np.array): I1 x .. x IN
	U (np.array): J x In
Output
	UX (np.array): I1 x .. x J x .. x IN
'''
def nModeMatProduct(X, U, n):
	Dim = list(X.shape)
	Dim[n] = U.shape[0]
	Dim = tuple(Dim)
	Xn = Matricize(X, n)
	Yn = U @ Xn
	return UnMatricize(Yn, n, Dim)


def InitHOSVD(X, R):
	As = []
	for i in range(X.ndim):
		Xi = Matricize(X, i)
		U, S, V = np.linalg.svd(Xi)
		U = U[:, :R]
		As.append(U)
	return As

def InitA(I, r):
	As = []
	for i in I:
		A = np.eye(i, r)
		As.append(A)
	return As

def PARAFAC_Orth(X, R, max_iter = 1000, DIISSpace = 0, verbose = 0):
	I = X.shape
	As = InitHOSVD(X, R)
	def OptL(weights, factors, tensor):
		return PARAFACError(tensor, factors, Ls = weights)
	Ls = np.ones((R,))
	if DIISSpace > 0:
		ErrorVecs = []
		PrevAs = []
		for i in range(len(I)):
			ErrorVecs.append([])
			PrevAs.append([])
	for i in range(max_iter):
		for n in range(len(I)):
			W = khatri_rao(As, skip_matrix = n)
			Xn = Matricize(X, n)
			Y = get_symm_mat_pow(W.T @ Xn.T @ Xn @ W, -0.50, check_symm = False) # W @ np.linalg.pinv(V)
			As[n] = Xn @ W @ Y
			if DIISSpace > 0:
				Ls = least_squares(OptL, Ls, args = [As, X]).x
				E = PARAFACError(X, As, Ls = Ls, ReturnVector = True)
				ErrorVecs[n].append(E)
				ErrorVecs[n] = ErrorVecs[n][-DIISSpace:]
				PrevAs[n].append(As[n])
				PrevAs[n] = PrevAs[n][-DIISSpace:]
				if i > DIISSpace:
					As[n] = DIIS(PrevAs[n], ErrorVecs[n])
					Ls = least_squares(OptL, Ls, args = [As, X]).x
					E = PARAFACError(X, As, Ls = Ls, ReturnVector = True)
					PrevAs[n][-1] = As[n]
					ErrorVecs[n][-1] = E
		Ls = least_squares(OptL, Ls, args = [As, X]).x
		Err = PARAFACError(X, As, Ls = Ls)
		if verbose > 0:
			print("PARAFAC Iteration", i, "complete with error", Err)
		if Err < 1e-6:
			break
	print("PARAFAC complete with error", Err)
	return As, Ls

def PARAFAC_OrthQR(X, R, max_iter = 1000, Normalize = False):
	I = X.shape
	As = InitHOSVD(X, R)
	for i in range(max_iter):
		Ls = np.ones((R,))
		# Pre orthogonalize all factor matrices
		for n in range(len(As)):
			As[n] = qr(As[n], mode = 'economic')[0]
		for n in range(len(I)):
			W = khatri_rao(As, skip_matrix = n)
			Xn = Matricize(X, n)
			As[n] = Xn @ W @ np.linalg.inv(W.T @ W)
		for A in As:
			for j in range(A.shape[1]):
				norm = np.linalg.norm(A[:, j])
				A[:, j] /= norm
				Ls[j] *= norm
		Err = PARAFACError(X, As, Ls = Ls)
		print("PARAFAC Iteration", i, "complete with error", Err)
		if Err < 1e-6:
			break
	return As, Ls


def PARAFAC(X, R, max_iter = 1000, Normalize = False):
	I = X.shape
	As = InitHOSVD(X, R)
	for i in range(max_iter):
		if Normalize:
			Ls = np.ones((R,))
		else:
			Ls = None
		for n in range(len(I)):
			#V = np.ones((R, R))
			#for m, A in enumerate(As):
			#	if m == n:
			#		continue
			#	else:
			#		V *= A.T @ A

			W = khatri_rao(As, skip_matrix = n)
			Xn = Matricize(X, n)
			As[n] = Xn @ W @ np.linalg.inv(W.T @ W)
			#As[n] = Xn @ W @ np.linalg.pinv(V)
		if Normalize:
			for A in As:
				for j in range(A.shape[1]):
					norm = np.linalg.norm(A[:, j])
					A[:, j] /= norm
					Ls[j] *= norm
		Err = PARAFACError(X, As, Ls = Ls)
		if Err < 1e-6:
			break
	print("PARAFAC complete with error", Err)
	return As, Ls

if __name__ == "__main__":
	X = np.zeros((3, 4, 2))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X[i, j, 0] = i + X.shape[0] * j + 1
			X[i, j, 1] = i + X.shape[0] * j + 13
	X = np.random.rand(10,10,10,10)

	#U = np.zeros((2, 3))
	#for i in range(U.shape[0]):
	#	for j in range(U.shape[1]):
	#		U[i, j] = i + U.shape[0] * j + 1

	#Y = nModeMatProduct(X, U, 0)
	#print(Y[:, :, 0])
	#print(Y[:, :, 1])	

	As, Ls = PARAFAC_Orth(X, 2, max_iter = 100, DIISSpace = 3, verbose = 1)
	As, Ls = PARAFAC_Orth(X, 2, max_iter = 100, DIISSpace = 0, verbose = 1)
	from tensorly.decomposition import parafac
	S, Us = parafac(X, 2, n_iter_max = 100, orthogonalise = True, normalize_factors = False)
	print(PARAFACError(X, Us, Ls = S))
