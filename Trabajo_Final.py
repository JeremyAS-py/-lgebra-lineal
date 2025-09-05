import numpy as np 
import matplotlib.pyplot as plt
import time

X = None
y = None

def cholesky_decomposition(A):
    """Realiza la descomposición de Cholesky de una matriz A"""
    n = A.shape[0]
    L = np.zeros((n, n))

    # Verificar que A es simétrica
    if not np.allclose(A, A.T):
        raise ValueError("La matriz no es simétrica.")

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:  # Elementos diagonales
                val = A[i][i] - sum_k
                if val <= 0:
                    raise ValueError("La matriz no es definida positiva.")
                L[i][j] = np.sqrt(val)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]

    return L

def solve_cholesky(X, y):
    """Resuelve la regresión lineal usando Cholesky."""
    XtX = X.T @ X
    Xty = X.T @ y
    L = cholesky_decomposition(XtX)

    # Resolver Lz = X^T y
    z = np.linalg.solve(L, Xty)

    # Resolver L.T * beta = z
    beta_hat = np.linalg.solve(L.T, z)

    return beta_hat

def verificar_datos_reales(matriz, vector):
    """
    Verifica que todos los elementos en la matriz y el vector sean números reales y finitos.

    Input:
    - matriz : Matriz X.
    - vector : Vector y.
    """
    if not np.isrealobj(matriz) or not np.isrealobj(vector):    # Condición: Verifica si matriz y vector contienen solo números reales
        return False, "Error: Los datos deben ser números reales. El programa aun no acepta números complejos."
    
    if not np.all(np.isfinite(matriz)) or not np.all(np.isfinite(vector)):  # Condición: Verifica que todos los valores sean finitos o no numéricos
        return False, "Error: Los datos contienen valores no numéricos o infinitos."

    return True, None

def householder_reflection(A, tol=1e-10):
    """
    Realiza la descomposición QR usando reflexiones de Householder.
    Input
    - A : Matriz a descomponer.
    - tol : Umbral de tolerancia para considerar valores como cero en R para estabilidad del algoritmo.

    Retorna:
    - Q : Matriz ortogonal.
    - R : Matriz triangular superior.
    """
    m, n = A.shape
    
    if m < n:       # Condición: Verifica que m >= n para que sea posible la descomposición QR
        return None, None, "Error: La matriz debe tener m >= n para realizar la descomposición QR."
    
    Q = np.eye(m)  
    R = A.copy()   

    for i in range(n):
        x = R[i:, i]
        norm_x = np.linalg.norm(x)
        
        if norm_x == 0:
            continue
        
        e1 = np.zeros_like(x)
        e1[0] = norm_x
        v = x - e1
        v = v / np.linalg.norm(v)

        H_i = np.eye(m)
        H_i[i:, i:] -= 2.0 * np.outer(v, v)
        
        R = H_i @ R
        Q = Q @ H_i

    R = np.triu(R, k=0) * (np.abs(R) > tol)

    if not np.allclose(Q.T @ Q, np.eye(m), atol=tol):   # Verifica que Q sea ortogonal
        return None, None, "Error: La matriz Q no es ortogonal."

    return Q, R, None

def linear_regression_householder(X, y, tol=1e-10):
    """
    Calcula los coeficientes beta en un modelo de regresión lineal mediante QR por Householder.

    Input:
    - X : Matriz de diseño.
    - y : Vector de observaciones.

    Retorna:
    - beta_hat : Estimación de los coeficientes beta.
    """
    
    # Verificar si la matriz X y el vector y no están vacíos
    if X.size == 0 or y.size == 0:
        return None, "Error: La matriz X y el vector y no deben estar vacíos."

    # Verificar que el número de filas sea mayor o igual al número de columnas
    m, n = X.shape
    if m < n:       # Condición: Verifica que m >= n para que sea posible la descomposición QR
        return None, "Error: La matriz debe tener m >= n para realizar la descomposición QR."

    # Verificar la independencia lineal de las columnas
    if np.linalg.matrix_rank(X) < n:        # Verifica que las columnas de X sean linealmente independientes
        return None, "Error: Las columnas de X deben ser linealmente independientes."

    # Verificar que los datos sean reales y no contengan valores no numéricos o infinitos
    datos_reales, error = verificar_datos_reales(X, y)
    if not datos_reales:
        return None, error

    # Realizar la descomposición QR mediante reflexiones de Householder
    Q, R, error = householder_reflection(X, tol)
    if error:
        return None, error

    print("Matriz Q:")
    print(Q)
    print("\nMatriz R:")
    print(R)

    # Resolver R * beta = Q^T * y
    Qt_y = Q.T @ y  

    # Verificar que R esté bien condicionada para estabilidad numérica
    if np.linalg.cond(R[:n, :]) > 1e10:          
        return None, "Error: La matriz R es mal condicionada o singular."

    try:
        beta_hat = np.linalg.solve(R[:n, :], Qt_y[:n])
        print("\nEstimación de beta_hat:", beta_hat)
    except np.linalg.LinAlgError:
        return None, "Error: No se pudo resolver el sistema lineal."

    return beta_hat, None
  
def gram_schmidt_linear_regression(X, y):
    # Validaciones de entrada
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X e y deben ser matrices y vectores de tipo numpy.ndarray.")
        
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X debe ser una matriz de 2 dimensiones y y un vector de 1 dimensión.")
        
    n, p = X.shape
    if n < p:
        raise ValueError("El número de muestras (n) debe ser mayor o igual al número de características (p).")
        
    if len(y) != n:
        raise ValueError("La longitud de y debe coincidir con el número de filas de X.")

    # Inicializar matrices Q y R
    Q = np.zeros((n, p))
    R = np.zeros((p, p))
    
    # Ortogonalización de Gram-Schmidt
    for j in range(p):
        v = X[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], X[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        
        # Verificar que no se genere un vector nulo, lo cual indicaría dependencia lineal
        if R[j, j] == 0:
            raise ValueError("Las columnas de X deben ser linealmente independientes.")
            
        Q[:, j] = v / R[j, j]
    beta_hat = np.linalg.solve(R, Q.T @ y)
    
    return beta_hat

def validar_matriz_diseno(X):
    n, p = X.shape

    # Verificar que X tenga más filas que columnas
    if n < p:
        raise ValueError("La matriz de diseño X debe tener al menos tantas filas como columnas (n >= p).")

    # Verificar que el rango de X sea completo
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("La matriz de diseño X no tiene rango completo (colinealidad detectada).")

def validar_vector_respuesta(X, y):
    """Valida que el vector de respuesta y tenga el mismo número de filas que X."""
    if X.shape[0] != y.size:
        raise ValueError("El vector de respuesta y debe tener el mismo número de filas que X.")

def svd(X, tol=1e-10, max_iter=100):
    """Realiza la descomposición SVD de una matriz X."""
    n, m = X.shape
    U = np.eye(n)  # Matriz ortogonal U
    V = np.eye(m)  # Matriz ortogonal V
    A = X.copy()   # Copia de X

    for _ in range(max_iter):
        for i in range(m):
            for j in range(i + 1, m):
                a, b = A[:, i], A[:, j]
                theta = 0.5 * np.arctan2(2 * np.dot(a, b), np.dot(a, a) - np.dot(b, b))

                # Matriz de rotación
                c, s = np.cos(theta), np.sin(theta)
                A[:, [i, j]] = A[:, [i, j]] @ np.array([[c, s], [-s, c]])
                V[:, [i, j]] = V[:, [i, j]] @ np.array([[c, s], [-s, c]])

        if np.allclose(np.triu(A, 1), 0, atol=tol):
            break

    Sigma = np.sqrt(np.sum(A**2, axis=0))
    U = X @ V / Sigma.clip(min=tol)  # Clip para evitar división por cero

    return U, np.diag(Sigma), V.T

def regresion_SVD(X, y):
    """Calcula los coeficientes de regresión de mínimos cuadrados usando la descomposición SVD."""
    validar_matriz_diseno(X)
    validar_vector_respuesta(X, y)

    U, Sigma, VT = svd(X)
    Sigma_pinv = np.linalg.pinv(Sigma)  
    beta = VT @ Sigma_pinv @ U.T @ y
    return beta

##############################################
##### Pruebas ( Test de tiempo entre los diferentes tiempos de los algoritmos)
##############################################

def ejecutar_pruebas(X, y):
    tiempos = {}

    # Medir tiempo para la descomposición de Cholesky
    inicio = time.perf_counter()
    try:
        beta_cholesky = solve_cholesky(X, y)
        tiempos["Cholesky"] = time.perf_counter() - inicio
        print("Coeficientes obtenidos (Cholesky):", beta_cholesky)
        print(f"Tiempo de ejecución (Cholesky): {tiempos['Cholesky']:.6f} segundos")
    except ValueError as e:
        tiempos["Cholesky"] = None
        print("Error en descomposición de Cholesky:", e)

    # Medir tiempo para la descomposición QR usando Householder
    inicio = time.perf_counter()
    try:
        beta_householder, error = linear_regression_householder(X, y)
        tiempos["Householder"] = time.perf_counter() - inicio
        if error:
            print("Error en descomposición QR Householder:", error)
        else:
            print("Coeficientes obtenidos (Householder):", beta_householder)
            print(f"Tiempo de ejecución (Householder): {tiempos['Householder']:.6f} segundos")
    except ValueError as e:
        tiempos["Householder"] = None
        print("Error en descomposición QR Householder:", e)

    # Medir tiempo para la ortogonalización de Gram-Schmidt
    inicio = time.perf_counter()
    try:
        beta_gram_schmidt = gram_schmidt_linear_regression(X, y)
        tiempos["Gram-Schmidt"] = time.perf_counter() - inicio
        print("Coeficientes obtenidos (Gram-Schmidt):", beta_gram_schmidt)
        print(f"Tiempo de ejecución (Gram-Schmidt): {tiempos['Gram-Schmidt']:.6f} segundos")
    except ValueError as e:
        tiempos["Gram-Schmidt"] = None
        print("Error en ortogonalización de Gram-Schmidt:", e)

    # Medir tiempo para la descomposición SVD
    inicio = time.perf_counter()
    try:
        beta_svd = regresion_SVD(X, y)
        tiempos["SVD"] = time.perf_counter() - inicio
        print("Coeficientes obtenidos (SVD):", beta_svd)
        print(f"Tiempo de ejecución (SVD): {tiempos['SVD']:.6f} segundos")
    except ValueError as e:
        tiempos["SVD"] = None
        print("Error en descomposición SVD:", e)

    # Mostrar gráfico comparativo de tiempos de ejecución
    mostrar_grafico_tiempos(tiempos)

def mostrar_grafico_tiempos(tiempos):
    """Genera un gráfico de barras para comparar los tiempos de ejecución de cada método."""
    metodos = [metodo for metodo, tiempo in tiempos.items() if tiempo is not None]
    tiempos_filtrados = [tiempo for tiempo in tiempos.values() if tiempo is not None]

    plt.figure(figsize=(10, 6))
    plt.bar(metodos, tiempos_filtrados, color='skyblue')
    plt.xlabel("Método de Descomposición")
    plt.ylabel("Tiempo de Ejecución (segundos)")
    plt.title("Comparación de Tiempos de Ejecución de Métodos de Regresión")
    plt.show()

###############################################

def menu_creacion_matriz():
    print("Opciones para la matriz y el vector:")
    print("1. Crear matriz y vector manualmente")
    print("2. Generar matriz y vector aleatoriamente")
   
    
    data_choice = int(input("Ingrese el número de la opción deseada: "))

    if data_choice == 1:
        rows = int(input("Ingrese el número de filas para la matriz: "))
        cols = int(input("Ingrese el número de columnas para la matriz: "))
        X = np.zeros((rows, cols))
        
        print("Ingrese los elementos de la matriz:")
        for i in range(rows):
            for j in range(cols):
                X[i, j] = float(input(f"Elemento [{i+1},{j+1}]: "))
        
        y = np.zeros(rows)
        print("Ingrese los elementos del vector:")
        for i in range(rows):
            y[i] = float(input(f"Elemento [{i+1}]: "))
    
            
    else:
        rows = int(input("Ingrese el número de filas para la matriz: "))
        cols = int(input("Ingrese el número de columnas para la matriz: "))
        X = np.random.randint(0, 10, (rows, cols))
        y = np.random.randint(0, 10, rows)
        
        print("\nMatriz generada aleatoriamente:")
        print(X)
        print("\nVector generado aleatoriamente:")
        print(y)

    return X, y

def menu_descomposicion():
    print("\nSeleccione una opción para la descomposición:")
    print("1. Cholesky")
    print("2. Householder")
    print("3. Gram-Schmidt")
    print("4. SVD")
    print("5. Realizar test de tiempo")
    print("6. Salir")
    choice = int(input("Ingrese el número de la opción deseada: "))
    return choice

def main():
    while True:
        X, y = menu_creacion_matriz()
        choice = menu_descomposicion()
        
        if choice == 1:
            try:
                beta = solve_cholesky(X, y)
                print("Coeficientes obtenidos (Cholesky):", beta)
            except ValueError as e:
                print("Error:", e)
        elif choice == 2:
            beta, error = linear_regression_householder(X, y)
            if error:
                print("Error:", error)
            else:
                print("Coeficientes obtenidos (Householder):", beta)
        elif choice == 3:
            try:
                beta = gram_schmidt_linear_regression(X, y)
                print("Coeficientes obtenidos (Gram-Schmidt):", beta)
            except ValueError as e:
                print("Error:", e)
        elif choice == 4:
            try:
                beta = regresion_SVD(X, y)
                print("Coeficientes obtenidos (SVD):", beta)
            except ValueError as e:
                print("Error:", e)
        elif choice == 5:
                ejecutar_pruebas(X,y)
        elif choice == 6:
                print("Fin")
                break
        else:
            print("Opción no válida.")

main()
