import numpy as np
import operator
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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

def Pro_lineal(A, B, C, Sim, max_Min):
    Puntos = []
    valores = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            a = A[[i, j]]
            b = B[[i, j]]
            try:
                pun = puntos(a, b)
            except np.linalg.LinAlgError:
                continue
            if verificacion(A, B, Sim, pun):
                z = C[0]*pun[0] + C[1]*pun[1]
                Puntos.append(pun)
                valores.append(z)
    if max_Min == "max":
        idx = np.argmax(valores)
    elif max_Min == "min":
        idx = np.argmin(valores)
    return valores[idx], Puntos[idx], Puntos  # también devuelve todos los puntos válidos

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def graficar(A, B, Sim, C, max_Min):
    x_vals = np.linspace(0, 100, 400)

    plt.figure(figsize=(8, 8))

    for i in range(len(B)):
        if A[i, 1] != 0:
            y = (B[i] - A[i, 0] * x_vals) / A[i, 1]
            plt.plot(x_vals, y, label=f'Restricción {i+1}')
        else:
            plt.axvline(x=B[i] / A[i, 0], linestyle='--', label=f'Restricción {i+1}')

    #graficar funcion objetivo (solo referencia)
    if C[1] != 0:
        y_obj = (-C[0] * x_vals) / C[1]
        plt.plot(x_vals, y_obj, label="Función objetivo", color='black', linestyle='--')

    #obtener puntos factibles y optimo
    opt_val, opt_punto, puntos_validos = Pro_lineal(A, B, C, Sim, max_Min)

    # Graficar región factible (rellenada)
    if len(puntos_validos) >= 3:
        # ordenar los puntos alrededor del centro para graficar el polígono correctamente
        centro = np.mean(puntos_validos, axis=0)
        puntos_ordenados = sorted(puntos_validos, key=lambda p: np.arctan2(p[1] - centro[1], p[0] - centro[0]))
        polygon = Polygon(puntos_ordenados, color='lightblue', alpha=0.5, label="Región factible")
        plt.gca().add_patch(polygon)

    #graficar puntos factibles
    for p in puntos_validos:
        plt.plot(p[0], p[1], 'bo')  # semi-optimos en azul
        plt.text(p[0]+1, p[1]+1, f"({p[0]:.1f}, {p[1]:.1f})", fontsize=8)

    #graficar solucion optima
    plt.plot(opt_punto[0], opt_punto[1], 'ro', label=f'Óptimo: {opt_punto} \nz={opt_val:.1f}', markersize=8)

    #estetica general
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{max_Min.capitalize()}imización de Z = {C[0]}x + {C[1]}y")
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True)
    plt.legend()
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

# CAMBIO REALIZADO AQUÍ:
opt_val, opt_punto, _ = Pro_lineal(A, B, C, Sim, "max")

print(f"valor optimo: {opt_val}")
print(f"punto optimo: {opt_punto}")

graficar(A, B, Sim, C, "max")

#Valor óptimo: 124.0
#Punto óptimo: [44. 12.]
