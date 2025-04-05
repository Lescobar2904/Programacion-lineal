import numpy as np
import operator

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


#A = np.array([
#    [2, 1],
#    [1, 3],
#    [1, 0],
#    [0, 1],
#    [1, 0],
#    [0, 1]
#])
#
#B = np.array([100, 80, 45, 100, 0, 0])
#Sim = ["<=", "<=", "<=", "<=", ">=", ">="]
#C = np.array([2, 3])
#
#opt_val, opt_punto = Pro_lineal(A, B, C, Sim, "max")
#print("Valor óptimo:", opt_val)
#print("Punto óptimo:", opt_punto)
#
#Valor óptimo: 124.0
#Punto óptimo: [44. 12.]