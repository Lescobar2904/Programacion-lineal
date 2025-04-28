import numpy as np
import operator
import matplotlib.pyplot as plt

operadores = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne
}


# funcion que resuelve un conjunto de ecuaciones lineales para 
# encontrar los puntos de intercepcion entre dos rectas
def puntos(a,b):
    return np.linalg.solve(a, b)

# funcion que verifica si los puntos son parte de la region factible o no
def verificacion(A, B, simbolos, punto):
    resultados = A @ punto  # Mutiplica la matriz por el vector de puntos (como resultado da un vector)
    for i, op_str in enumerate(simbolos):
        op_func = operadores[op_str]  # función según el símbolo
        if not op_func(resultados[i], B[i]):
            return False
    return True

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# A = Matriz de tamaño n x 2 que contiene los coeficientes de las variables
#     en las restricciones del sistema.
#     Por ejemplo, si una restricción es: a1*x + a2*y <= b1,
#     entonces la fila correspondiente en A es [a1, a2].


# B = Vector de tamaño n que contiene los términos independientes (lado derecho)
#     de las restricciones del sistema.
#     Por ejemplo, si una restricción es: a1*x + a2*y <= b1
#     entonces B incluye b1.


# C = vector de tamaño 2 que contiene los coeficientes que acompañan 
#     a las variables de la función objetivo que se desea maximizar o minimizar.
#     Por ejemplo, si la función es: z = c1*x1 + c2*x2, entonces:
#     C = [c1, c2]

def Pro_lineal(A,B,C,Sim,max_Min):
    Puntos = []
    valores = []
    for i in range(len(B)-1):
        for j in range(len(B)):
            if i != j:
                a = A[[i,j]]
                b = B[[i,j]]
                try:
                    pun = puntos(a, b)
                except np.linalg.LinAlgError:
                    continue 
                if verificacion(A,B,Sim,pun):
                    z = C[0]*pun[0] + C[1]*pun[1]
                    Puntos.append(pun)
                    valores.append(z)
    if max_Min == "max":
        idx = np.argmax(valores)
    elif max_Min == "min":
        idx = np.argmin(valores)
    return valores[idx], Puntos[idx]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def graficar(A, B, Sim, C, max_Min):
    x_vals = np.linspace(0, 100, 400)
    y_vals = np.linspace(0, 100, 400)
    
    plt.figure(figsize=(8, 8))

    for i in range(len(B)):
        if A[i, 1] != 0:
            y = (B[i] - A[i, 0] * x_vals) / A[i, 1]
            plt.plot(x_vals, y, label=f'restriccion {i+1}')
        else:
            plt.axvline(x=B[i] / A[i, 0], linestyle='--', label=f'restriccion {i+1}')

    if C[1] != 0:
        y_obj = (-C[0] * x_vals) / C[1]
        plt.plot(x_vals, y_obj, label="funcion objetivo", color='black', linestyle='--')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    
    opt_val, opt_punto = Pro_lineal(A, B, C, Sim, max_Min)
    plt.plot(opt_punto[0], opt_punto[1], 'ro', label=f'solucion optima {opt_punto}')
    plt.title(f"optimizacion: {max_Min.capitalize()}imizacion de z")
    plt.legend()
    plt.grid(True)
    plt.show()


A = np.array([
    [2, 1],
    [1, 3],
    [1, 0],
    [0, 1]
])

B = np.array([100, 80, 45, 100])
Sim = ["<=", "<=", "<=", "<="]
C = np.array([2, 3])

opt_val, opt_punto = Pro_lineal(A, B, C, Sim, "max")
print(f"valor optimo: {opt_val}")
print(f"punto optimo: {opt_punto}")

graficar(A, B, Sim, C, "max")

#Valor óptimo: 124.0
#Punto óptimo: [44. 12.]