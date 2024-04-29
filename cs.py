import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import DictionaryLearning

'''
    This section defines omp algorithm , compress algorithm and reconstruct algorithm
'''

def compress(m,D,x):

    ''' 
        m means the length of compress signal
        D means the dictionary
        x means the oriiginal signal 
    '''
    n,_ = D.shape

    # Random Gaussian sensing matrix
    A = np.random.randn(m, n) / np.sqrt(m)

    y = np.dot(A,np.dot(D,x))

    return A,y 
    
def reconstruct(A,y,k = 15):

    '''
        A means random gaussian sensing matrix
        y means compressed signal
    '''
    # Reconstruction using Orthogonal Matching Pursuit

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    omp.fit(A, y)
    x_reconstructed = omp.coef_

    return x_reconstructed

def dl(X):

    # Create a Dictionary Learning model
    dl = DictionaryLearning(n_components=100, transform_algorithm='omp')

    # Fit the model to your data
    dl.fit(X)

    # Get the learned dictionary
    learned_dict = dl.components_

    return learned_dict

def k_svd_dl(Y, k,D ,max_iter=30 ):
    n_features , n_samples = Y.shape

    # K-SVD Training 
    for _ in range(max_iter):
        # Sparse coding (fixing D)
        X = sparse_coding(Y, D)

        # Dictionary update
        D = dictionary_update(Y, D, X)

    return D


def sparse_coding(Y, D):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=D.shape[0])
    X = np.zeros((D.shape[0], Y.shape[1]))

    for i in range(Y.shape[1]):
        omp.fit(D.T, Y[:,i].T)
        X[:, i] = omp.coef_

    return X


def dictionary_update(Y, D, X):
    for k in range(D.shape[0]):
        # Indices of data points that use atom k
        indices = np.where(X[k, :] != 0)[0]

        if len(indices) == 0:
            continue

        # Residual matrix
        Ek = Y[:, indices] - np.dot(D.T, X[:, indices])

        # Update atom k using the SVD of the residual matrix
        u, s, v = np.linalg.svd(Ek)
        D[k, :] = u[:, 0]

        # Update coefficients corresponding to atom k
        X[k, indices] = s[0] * v[0, :]

    return D






