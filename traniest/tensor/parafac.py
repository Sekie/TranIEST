import numpy as np
from tensorly.tenalg import khatri_rao

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

def PARAFACError(X, As, Ls = None):
	x = VectorsToTensor(As, Ls = Ls)
	return np.sqrt(((X - x)**2).sum())

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

def PARAFAC(X, R, max_iter = 1000, Normalize = True):
	I = X.shape
	As = InitHOSVD(X, R)
	for i in range(max_iter):
		if Normalize:
			Ls = np.ones((R,))
		else:
			Ls = None
		for n in range(len(I)):
			V = np.ones((R, R))
			
			for m, A in enumerate(As):
				if m == n:
					continue
				else:
					V *= A.T @ A
			if Normalize:
				LMat = np.zeros((R, R))
				LMat = np.fill_diagonal(LMat, Ls)
				V = LMat @ V
			W = khatri_rao(As, skip_matrix = n)
			Xn = Matricize(X, n)
			As[n] = Xn @ W @ np.linalg.pinv(V)
			if Normalize:
				for j in range(As[n].shape[1]):
					norm = np.linalg.norm(As[n][:, j])
					As[n][:, j] /= norm
					Ls[j] *= norm
		Err = PARAFACError(X, As, Ls = Ls)
		print("PARAFAC Iteration", i, "complete with error", Err)
		if Err < 1e-6:
			break
	return As, Ls

if __name__ == "__main__":
	X = np.zeros((3, 4, 2))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X[i, j, 0] = i + X.shape[0] * j + 1
			X[i, j, 1] = i + X.shape[0] * j + 13

	#U = np.zeros((2, 3))
	#for i in range(U.shape[0]):
	#	for j in range(U.shape[1]):
	#		U[i, j] = i + U.shape[0] * j + 1

	#Y = nModeMatProduct(X, U, 0)
	#print(Y[:, :, 0])
	#print(Y[:, :, 1])	

	As, Ls = PARAFAC(X, 2, max_iter = 10, Normalize = False)
	from tensorly.decomposition import parafac
	S, Us = parafac(X, 2, n_iter_max = 10, normalize_factors = False)
	print(S)
	print(Us)
	print(PARAFACError(X, Us, Ls = S))	
