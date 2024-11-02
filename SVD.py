import numpy as np

def svd(X, tol=1e-10, max_iter=100):
    m, n = X.shape
    U = np.eye(m)  # Matriz ortogonal U
    V = np.eye(n)  # Matriz ortogonal V
    A = X.copy()  # Copia de X 

    for _ in range(max_iter):
        for i in range(n):
            for j in range(i + 1, n):
                # Ángulo de rotación
                a = A[:, i]
                b = A[:, j]
                theta = 0.5 * np.arctan2(2 * np.dot(a, b), np.dot(a, a) - np.dot(b, b))

                # Matriz de rotación
                c, s = np.cos(theta), np.sin(theta)
                R = np.eye(n)
                R[i, i] = c
                R[i, j] = s
                R[j, i] = -s
                R[j, j] = c

                A = A @ R  # Aplica rotación a A
                V = V @ R  # Actualiza V con la rotación

        if np.allclose(np.triu(A, 1), 0, atol=tol):  # Verifica si A es diagonal
            break

    Sigma = np.diag(np.sqrt(np.sum(A**2, axis=0)))  # Calcula valores singulares
    U = X @ V / Sigma  # Calcula U a partir de X y V

    return U, Sigma, V.T

def regresion_SVD(X, y):
    U, Sigma, VT = svd(X)  # Descomposición SVD de X
    Sigma_inv = np.linalg.inv(Sigma)  # Inversa de Sigma
    beta = VT.T @ Sigma_inv @ U.T @ y  # Calcula coeficientes de regresión
    return beta
