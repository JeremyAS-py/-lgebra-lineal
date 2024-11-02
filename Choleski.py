import numpy as np

def Cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)  # Inicializa L como matriz compuesta por ceros

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))  # Suma parcial
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - sum_k)  # Elementos de la diagonal de L 
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]  # Elementos que no pertenecen a la diagonal
    return L

def regresion_cholesky(X, y):
    XtX = X.T @ X  # Matriz de correlación de X
    Xty = X.T @ y  # Producto de X transpuesta y y
    L = Cholesky(XtX)  # Descomposición de Cholesky de XtX
    z = np.linalg.solve(L, Xty)  # Resuelve Lz = Xty
    beta = np.linalg.solve(L.T, z)  # Resuelve L.T * beta = z
    return beta
