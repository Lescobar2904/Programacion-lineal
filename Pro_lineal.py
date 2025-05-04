import numpy as np                   
import matplotlib.pyplot as plt     
import re                           #libreria para trabajar con expresiones regulares, en cadenas como "max z = 3x + 2y" y extraer los coeficientes de forma estructurada
from itertools import combinations  #para generar combinaciones de restricciones (pares)

def parse_input():
    A = []   #lista para coeficientes de x en cada restriccion
    B = []   #lista para coeficientes de y en cada restriccion
    C = []   #lista para terminos independientes de cada restriccion
    Sim = [] #lista para simbolos de desigualdad (<=, >=, =)

    #se pide al usuario cuantas restricciones ingresara
    n = int(input("¿cuantas restricciones vas a ingresar? "))
    print("ingresa cada restriccion (ejemplo: 2x + 3y <= 60):")

    for _ in range(n):
        restr = input("> ").replace(" ", "")  #se eliminan espacios
        #expresion regular para extraer coeficientes y simbolo
        m = re.match(r'([-+]?\d*)x([-+]?\d*)y(<=|>=|=)(-?\d+)', restr)
        if not m:
            raise ValueError("formato incorrecto de restriccion. Asegurate de seguir el formato como '2x+3y<=60'")
        
        #coeficiente de x (1 si no hay número, -1 si es solo '-')
        a = int(m.group(1) or '1') if m.group(1) != '-' else -1
        #coeficiente de y
        b = int(m.group(2) or '1') if m.group(2) != '-' else -1
        #simbolo de desigualdad
        s = m.group(3)
        #termino independiente (constante)
        c = int(m.group(4))

        A.append(a)
        B.append(b)
        Sim.append(s)
        C.append(c)

    #se agregan restricciones x >= 0 y y >= 0 para restringir al primer cuadrante
    A.extend([1, 0])
    B.extend([0, 1])
    C.extend([0, 0])
    Sim.extend([">=", ">="])

    #se solicita fc objetivo
    #---------------------------------------------------
    #(max|min)       # Grupo 1: coincide con "max" o "min"
    #z=              # Coincide literalmente con 'z='
    #([-+]?\d*)      # Grupo 2: un número opcional con signo (coeficiente de x)
    #x               # Coincide literalmente con 'x'
    #([-+]?\d*)      # Grupo 3: otro número opcional con signo (coeficiente de y)
    #y               # Coincide literalmente con 'y'
    #---------------------------------------------------
    funcion = input("ingresa la funcion objetivo (ejemplo: max z = 3x + 2y): ").replace(" ", "")
    m = re.match(r'(max|min)z=([-+]?\d*)x([-+]?\d*)y', funcion)
    if not m:
        raise ValueError("formato incorrecto. Ejemplo valido: max z = 3x + 5y")

    tipo = m.group(1)  #'max' o 'min'
    #---------------------------------
    #si esta vacio se asume que es 1, si no es solo un signo negativo, entonces es -1
    #---------------------------------
    cx = int(m.group(2) or '1') if m.group(2) != '-' else -1  #coeficiente de x
    cy = int(m.group(3) or '1') if m.group(3) != '-' else -1  #coeficiente de y

    return A, B, C, Sim, tipo, cx, cy

# -------------------- CALCULO DE INTERSECCIONES --------------------
def interseccion(a1, b1, c1, a2, b2, c2):
    #se resuelve el sistema de ecuaciones lineales:
    # a1*x + b1*y = c1
    # a2*x + b2*y = c2
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([c1, c2])
    
    #si el determinante es 0, las rectas son paralelas o coincidentes
    if np.linalg.det(A) == 0:
        return None
    # np.linalg.solve resuelve el sistema Ax = B
    return np.linalg.solve(A, B)

# -------------------- VALIDACION DE FACTIBILIDAD --------------------
def es_factible(x, y, A, B, C, Sim):
    #se recorre cada restriccion y se evalua si el punto (x, y) la cumple
    for a, b, c, s in zip(A, B, C, Sim):
        val = a*x + b*y
        if s == '<=' and val > c + 1e-5:
            return False
        elif s == '>=' and val < c - 1e-5:
            return False
        elif s == '=' and abs(val - c) > 1e-5:
            return False
    return True  #si todas las restricciones se cumplen

# -------------------- RESOLUCIÓN DEL PROBLEMA --------------------
def Pro_lineal(A, B, C, Sim, tipo, cx, cy):
    puntos = []
    #se prueban todas las combinaciones de pares de restricciones
    for i, j in combinations(range(len(A)), 2):
        #se calcula la interseccion entre las rectas i y j
        p = interseccion(A[i], B[i], C[i], A[j], B[j], C[j])
        #si hay interseccion y es factible, se guarda
        if p is not None and es_factible(p[0], p[1], A, B, C, Sim):
            puntos.append(p)

    if not puntos:
        raise ValueError("la region factible esta vacía. No hay solucion")

    valores = []
    for p in puntos:
        #se evalua la funcion objetivo z = cx*x + cy*y en cada punto
        valores.append(cx*p[0] + cy*p[1])

    #se obtiene el indice del valor optimo (maximo o minimo)
    idx = np.argmax(valores) if tipo == 'max' else np.argmin(valores)

    return valores[idx], puntos[idx], puntos

# -------------------- GRÁFICO --------------------
def graficar(A, B, C, puntos, mejor_punto, cx, cy):
    #rango de valores para x que usaremos para dibujar lineas
    x = np.linspace(0, max(p[0] for p in puntos)*1.2 + 10, 400)
    colores = plt.cm.get_cmap('tab10')  # mapa de colores para diferenciar restricciones

    for i, (a, b, c) in enumerate(zip(A, B, C)):
        color = colores(i % 10)  #color distinto para cada linea
        if b != 0:
            #se despeja y = (c - a*x) / b
            y = (c - a*x) / b
            plt.plot(x, y, label=f'{a}x + {b}y {Sim[i]} {c}', color=color)
            px, py = 2, (c - a*2) / b  #punto para poner la flecha
        else:
            #si b == 0, es una recta vertical x = c/a
            plt.axvline(x=c/a, label=f'{a}x {Sim[i]} {c}', color=color)
            px, py = c/a, 2  #posicion para flecha

        #se añade anotacion con flecha hacia la recta
        plt.annotate(f"{a}x + {b}y {Sim[i]} {c}", xy=(px, py), xytext=(px + 3, py + 3),
                     arrowprops=dict(arrowstyle="->", color=color), color=color, fontsize=9)

    puntos = np.array(puntos)
    #se dibuja la region factible como poligono relleno
    plt.fill(puntos[:,0], puntos[:,1], 'skyblue', alpha=0.4, label="Región factible")
    #se dibujam los puntos vertice
    plt.plot(puntos[:,0], puntos[:,1], 'ko')

    #se etiqueta cada vertice con sus coordenadas y valor de z
    for p in puntos:
        z = cx*p[0] + cy*p[1]
        plt.text(p[0]+0.3, p[1]+0.3, f"({p[0]:.1f},{p[1]:.1f})\nz={z:.1f}", fontsize=8)

    #se dibuja el punto optimo en rojo
    plt.plot(mejor_punto[0], mejor_punto[1], 'ro', label="Óptimo")

    #se dibuja la recta de nivel de la funcion objetivo
    z = cx*mejor_punto[0] + cy*mejor_punto[1]
    y_obj = (z - cx*x) / cy
    plt.plot(x, y_obj, 'r--', label="Función objetivo")

    #estetica del grafico
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Programación Lineal - Solución Gráfica')
    plt.grid(True)
    plt.axis('equal')  #mantener proporción 1:1
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------ EJECUCIÓN PRINCIPAL ------------------
if __name__ == '__main__':
    try:
        #lee las entradas del usuario
        A, B, C, Sim, tipo, cx, cy = parse_input()

        #calcula la solucion optima y los puntos de interseccion validos
        valor_optimo, punto_optimo, vertices = Pro_lineal(A, B, C, Sim, tipo, cx, cy)

        #muestra el resultado
        print("\n Solución óptima:")
        print(f"   Punto: x = {punto_optimo[0]:.2f}, y = {punto_optimo[1]:.2f}")
        print(f"   Valor óptimo de z: {valor_optimo:.2f}")

        #genera el grafico final
        graficar(A, B, C, vertices, punto_optimo, cx, cy)

    except Exception as e:
        print(f"\n Error: {e}")
