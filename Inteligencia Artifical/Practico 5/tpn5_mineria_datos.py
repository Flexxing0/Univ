"""
================================================================================
  TRABAJO PRÁCTICO N°5 — Aprendizaje Automático: Minería de Datos
  Materia: Minería de Datos
================================================================================

  PARTE 1: Árbol de Decisión — Algoritmo ID3 (desde cero)
  PARTE 2: Perceptrón de una capa (desde cero)

  Autor: TPN5 - Implementación completa
  Bibliotecas usadas: numpy, pandas, matplotlib, math, collections
================================================================================
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ==============================================================================
#  PARTE 1 — ALGORITMO ID3 (árbol de decisión)
# ==============================================================================
# ID3 (Iterative Dichotomiser 3) es un algoritmo que construye un árbol de
# decisión eligiendo, en cada nodo, el atributo que maximice la "ganancia de
# información", es decir, el que reduzca más la entropía del conjunto.
#
# CONCEPTO CENTRAL:
#   Entropía H(S) = -Σ p_i * log2(p_i)
#   Ganancia IG(S,A) = H(S) - Σ (|Sv|/|S|) * H(Sv)
# ==============================================================================

# ─── 1.1  Dataset: "Problema del Tiempo" ──────────────────────────────────────
# Cada fila = un ejemplo de entrenamiento (E1..E14)
# Columnas: Estado, Temperatura, Humedad, Viento, JuegoTenis (clase objetivo)

datos = [
    # EJ,  Estado,      Temperatura, Humedad, Viento,  Clase
    ("E1",  "soleado",  "caluroso",  "alta",   "calmo", "NO"),
    ("E2",  "soleado",  "caluroso",  "alta",   "medio", "NO"),
    ("E3",  "nublado",  "caluroso",  "alta",   "calmo", "SI"),
    ("E4",  "lluvioso", "templado",  "alta",   "calmo", "SI"),
    ("E5",  "lluvioso", "fresco",    "normal", "calmo", "SI"),
    ("E6",  "lluvioso", "fresco",    "normal", "medio", "NO"),
    ("E7",  "nublado",  "fresco",    "normal", "medio", "SI"),
    ("E8",  "soleado",  "templado",  "alta",   "calmo", "NO"),
    ("E9",  "soleado",  "fresco",    "normal", "calmo", "SI"),
    ("E10", "lluvioso", "templado",  "normal", "calmo", "SI"),
    ("E11", "soleado",  "templado",  "normal", "medio", "SI"),
    ("E12", "nublado",  "templado",  "alta",   "medio", "SI"),
    ("E13", "nublado",  "caluroso",  "normal", "calmo", "SI"),
    ("E14", "lluvioso", "templado",  "alta",   "medio", "NO"),
]

# Construimos un DataFrame de pandas para manipular los datos fácilmente
# pd.DataFrame: crea una tabla (como una planilla) con columnas nombradas
df = pd.DataFrame(datos, columns=["EJ", "Estado", "Temperatura", "Humedad", "Viento", "JuegoTenis"])
df = df.set_index("EJ")  # El identificador de ejemplo es el índice

# Atributos de entrada (todos menos la clase objetivo)
ATRIBUTOS = ["Estado", "Temperatura", "Humedad", "Viento"]
CLASE = "JuegoTenis"


# ─── 1.2  Función de Entropía ──────────────────────────────────────────────────
def entropia(subset: pd.DataFrame) -> float:
    """
    Calcula la entropía de un conjunto de ejemplos respecto a la clase objetivo.

    H(S) = -Σ p_i * log2(p_i)

    - subset: DataFrame con los ejemplos del nodo actual
    - Retorna: valor de entropía (float entre 0 y 1 para clasificación binaria)

    Entropía = 0  → todos los ejemplos son de la misma clase (puro)
    Entropía = 1  → exactamente mitad y mitad (máxima incertidumbre binaria)
    """
    total = len(subset)
    if total == 0:
        return 0.0

    # Counter cuenta cuántos hay de cada valor de clase: {"SI": 9, "NO": 5}
    conteos = Counter(subset[CLASE])
    h = 0.0
    for clase, cantidad in conteos.items():
        p = cantidad / total          # probabilidad de esa clase
        h -= p * math.log2(p)        # término de entropía
    return h


# ─── 1.3  Ganancia de Información ─────────────────────────────────────────────
def ganancia_informacion(subset: pd.DataFrame, atributo: str) -> float:
    """
    Calcula la ganancia de información al dividir 'subset' por 'atributo'.

    IG(S, A) = H(S) - Σ_{v ∈ V(A)} (|S_v| / |S|) * H(S_v)

    - Primero calcula la entropía del conjunto completo H(S)
    - Luego resta la entropía ponderada de cada subconjunto generado
      al separar por los valores posibles del atributo
    """
    total = len(subset)
    h_inicial = entropia(subset)

    # Suma ponderada de entropías de subconjuntos
    suma_ponderada = 0.0
    for valor in subset[atributo].unique():       # cada valor posible del atrib.
        sub = subset[subset[atributo] == valor]   # subconjunto con ese valor
        suma_ponderada += (len(sub) / total) * entropia(sub)

    return h_inicial - suma_ponderada


# ─── 1.4  Algoritmo ID3 (recursivo) ───────────────────────────────────────────
def id3(subset: pd.DataFrame, atributos_disponibles: list, profundidad: int = 0) -> dict:
    """
    Implementación recursiva del algoritmo ID3.

    Devuelve un diccionario que representa el árbol:
      - Si es hoja: {"hoja": True, "clase": "SI"/"NO", "ejemplos": [E1,...]}
      - Si es nodo: {"atributo": "Estado",
                     "hoja": False,
                     "hijos": {"soleado": <subárbol>, "nublado": <subárbol>, ...}}

    Casos base (cuando devolvemos una hoja):
      1. Todos los ejemplos son de la misma clase → hoja con esa clase
      2. No quedan atributos → hoja con la clase mayoritaria
      3. El subconjunto está vacío → hoja con la clase mayoritaria del padre
    """
    clases = subset[CLASE].values
    clase_mayoritaria = Counter(clases).most_common(1)[0][0]

    # ── Caso base 1: todos los ejemplos tienen la misma clase
    if len(set(clases)) == 1:
        return {
            "hoja": True, "clase": clases[0],
            "ejemplos": list(subset.index), "profundidad": profundidad
        }

    # ── Caso base 2: no quedan atributos para dividir
    if not atributos_disponibles:
        return {
            "hoja": True, "clase": clase_mayoritaria,
            "ejemplos": list(subset.index), "profundidad": profundidad
        }

    # ── Selección del mejor atributo: el de mayor ganancia de información
    ganancias = {a: ganancia_informacion(subset, a) for a in atributos_disponibles}
    mejor_atributo = max(ganancias, key=ganancias.get)

    # Construimos el nodo de decisión
    nodo = {
        "hoja": False,
        "atributo": mejor_atributo,
        "ganancia": ganancias[mejor_atributo],
        "entropia": entropia(subset),
        "ejemplos": list(subset.index),
        "profundidad": profundidad,
        "hijos": {}
    }

    # Atributos restantes (ID3 no reutiliza el mismo atributo en una rama)
    atribs_restantes = [a for a in atributos_disponibles if a != mejor_atributo]

    # Dividimos recursivamente para cada valor del mejor atributo
    for valor in df[mejor_atributo].unique():          # valores posibles
        sub = subset[subset[mejor_atributo] == valor]  # subconjunto
        if sub.empty:
            # Caso base 3: rama sin ejemplos → hoja con clase mayoritaria
            nodo["hijos"][valor] = {
                "hoja": True, "clase": clase_mayoritaria,
                "ejemplos": [], "profundidad": profundidad + 1
            }
        else:
            nodo["hijos"][valor] = id3(sub, atribs_restantes, profundidad + 1)

    return nodo


# ─── 1.5  Mostrar cálculos paso a paso ────────────────────────────────────────
def mostrar_calculos_id3(subset: pd.DataFrame, atributos: list, nivel: int = 0):
    """
    Imprime los cálculos de entropía y ganancia de información en cada nodo,
    tal como lo pide el enunciado: "realice los cálculos... manualmente".
    """
    indent = "  " * nivel
    total = len(subset)
    si = (subset[CLASE] == "SI").sum()
    no = (subset[CLASE] == "NO").sum()
    h = entropia(subset)

    print(f"\n{indent}{'─'*60}")
    print(f"{indent}Nodo (profundidad {nivel}) — Ejemplos: {list(subset.index)}")
    print(f"{indent}  Distribución: {si} SI, {no} NO  →  Total: {total}")
    print(f"{indent}  H(S) = -{si}/{total}·log2({si}/{total}) - {no}/{total}·log2({no}/{total})")
    print(f"{indent}  H(S) = {h:.6f} bits")

    if len(set(subset[CLASE])) == 1 or not atributos:
        clase = Counter(subset[CLASE]).most_common(1)[0][0]
        print(f"{indent}  → HOJA: clase = {clase}")
        return clase

    print(f"\n{indent}  Ganancias de información:")
    ganancias = {}
    for a in atributos:
        g = ganancia_informacion(subset, a)
        ganancias[a] = g
        # Detalle del cálculo
        print(f"\n{indent}    IG(S, {a}):")
        for v in subset[a].unique():
            sub_v = subset[subset[a] == v]
            h_v = entropia(sub_v)
            sv_si = (sub_v[CLASE] == "SI").sum()
            sv_no = (sub_v[CLASE] == "NO").sum()
            print(f"{indent}      {a}={v}: {len(sub_v)} ej. [{sv_si} SI, {sv_no} NO]  H={h_v:.4f}")
        print(f"{indent}    IG(S, {a}) = {h:.4f} - suma_pond = {g:.6f}")

    mejor = max(ganancias, key=ganancias.get)
    print(f"\n{indent}  ★ Mejor atributo: {mejor} (IG={ganancias[mejor]:.6f})")

    atribs_rest = [a for a in atributos if a != mejor]
    for v in df[mejor].unique():
        sub = subset[subset[mejor] == v]
        if not sub.empty:
            mostrar_calculos_id3(sub, atribs_rest, nivel + 1)


# ─── 1.6  Visualización del árbol ─────────────────────────────────────────────
def dibujar_arbol(nodo: dict, ax, x=0.5, y=1.0, dx=0.25, dy=0.18,
                  x_padre=None, y_padre=None, etiqueta_arista=""):
    """
    Dibuja el árbol de decisión aprendido por ID3 usando matplotlib.

    Parámetros:
      - nodo: diccionario devuelto por id3()
      - ax: eje de matplotlib donde dibujar
      - x, y: posición del nodo actual (normalizada 0..1)
      - dx: separación horizontal entre hijos (se reduce en cada nivel)
      - dy: separación vertical entre niveles
      - x_padre, y_padre: posición del nodo padre (para trazar la arista)
      - etiqueta_arista: texto que va sobre la arista (valor del atributo)
    """
    COLOR_NODO = "#4A90D9"    # azul para nodos internos
    COLOR_SI   = "#27AE60"    # verde para hojas SI
    COLOR_NO   = "#E74C3C"    # rojo para hojas NO

    # Dibujamos la arista desde el padre
    if x_padre is not None:
        ax.plot([x_padre, x], [y_padre, y], "k-", lw=1.2, zorder=1)
        mx, my = (x_padre + x) / 2, (y_padre + y) / 2
        ax.text(mx, my, etiqueta_arista, fontsize=7, ha="center", va="center",
                color="#555", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    if nodo["hoja"]:
        color = COLOR_SI if nodo["clase"] == "SI" else COLOR_NO
        ejs = "\n".join(nodo["ejemplos"]) if nodo["ejemplos"] else "—"
        texto = f"Clase:\n{nodo['clase']}\n({ejs})"
        ax.text(x, y, texto, ha="center", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="black", alpha=0.9),
                zorder=2, color="white", fontweight="bold")
    else:
        ejs_str = ",".join(nodo["ejemplos"])
        texto = f"{nodo['atributo']}\nH={nodo['entropia']:.3f}\n({ejs_str})"
        ax.text(x, y, texto, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.4", fc=COLOR_NODO, ec="black", alpha=0.9),
                zorder=2, color="white")

        hijos = nodo["hijos"]
        n = len(hijos)
        xs = np.linspace(x - dx * (n - 1) / 2, x + dx * (n - 1) / 2, n)
        for (valor, hijo), xh in zip(hijos.items(), xs):
            dibujar_arbol(hijo, ax, xh, y - dy, dx * 0.55, dy, x, y, valor)


# ==============================================================================
#  PARTE 2 — PERCEPTRÓN DE UNA CAPA
# ==============================================================================
# El perceptrón es la unidad básica de una red neuronal.
# Dado un vector de entradas X = (x1, x2), calcula:
#
#   net = w1*x1 + w2*x2 + bias
#   salida = 1  si net >= umbral (típicamente 0)
#            0  si net <  umbral
#
# APRENDIZAJE: Regla del Perceptrón (Rosenblatt):
#   Δw_i = η * (t - y) * x_i
#   w_i_nuevo = w_i + Δw_i
#
#   donde:
#     η (eta) = tasa de aprendizaje (usamos 1 por defecto)
#     t       = salida deseada (target)
#     y       = salida actual
#     x_i     = valor de la entrada i
# ==============================================================================

# ─── 2.1  Dataset del Perceptrón ──────────────────────────────────────────────
# Formato: ((x1, x2), target)
ejemplos_perceptron = [
    ((-1, -3), 0),
    (( 1,  2), 1),
    (( 2,  3), 0),
    (( 1,  0), 1),
]

# Pesos iniciales y bias según el enunciado
w1_init, w2_init = 0.2, 0.1
bias_init         = 0.3
ETA               = 1          # tasa de aprendizaje
UMBRAL            = 0          # umbral de activación (función escalón)
N_ITERACIONES     = 2          # número de épocas (pasadas completas) pedidas


# ─── 2.2  Función de activación escalón ───────────────────────────────────────
def escalon(net: float, umbral: float = 0) -> int:
    """
    Función de activación escalón (Heaviside):
      output = 1 si net >= umbral
               0 si net <  umbral
    """
    return 1 if net >= umbral else 0


# ─── 2.3  Entrenamiento del Perceptrón ────────────────────────────────────────
def entrenar_perceptron(ejemplos, w1, w2, bias, eta=1, n_iter=2, umbral=0):
    """
    Entrena un perceptrón de 2 entradas durante 'n_iter' épocas.

    Por cada época, recorre todos los ejemplos y aplica la regla de aprendizaje:
      - Calcula la salida actual: y = escalon(w1*x1 + w2*x2 + bias)
      - Calcula el error: error = target - y
      - Actualiza pesos: w_i += eta * error * x_i
                         bias += eta * error

    Devuelve:
      - historial: lista de dicts con el estado de cada paso
      - pesos_finales: (w1, w2, bias) al terminar
    """
    historial = []
    for iteracion in range(1, n_iter + 1):
        print(f"\n{'='*70}")
        print(f"  ITERACIÓN {iteracion}")
        print(f"{'='*70}")
        print(f"  Pesos iniciales de la iteración: w1={w1:.4f}, w2={w2:.4f}, bias={bias:.4f}")
        print(f"  {'Ej':<6} {'x1':>5} {'x2':>5} {'target':>8} {'net':>10} {'y':>5} {'error':>7} {'Δw1':>8} {'Δw2':>8} {'Δbias':>8} {'w1_new':>8} {'w2_new':>8} {'b_new':>8}")
        print(f"  {'-'*100}")

        for idx, ((x1, x2), t) in enumerate(ejemplos, start=1):
            # Paso 1: calcular suma ponderada (activación neta)
            net = w1 * x1 + w2 * x2 + bias

            # Paso 2: aplicar función de activación
            y = escalon(net, umbral)

            # Paso 3: calcular error
            error = t - y

            # Paso 4: actualizar pesos (regla del perceptrón)
            dw1   = eta * error * x1
            dw2   = eta * error * x2
            dbias = eta * error

            w1_new   = w1   + dw1
            w2_new   = w2   + dw2
            bias_new = bias + dbias

            print(f"  E{idx:<5} {x1:>5} {x2:>5} {t:>8} {net:>10.4f} {y:>5} {error:>7} "
                  f"{dw1:>8.4f} {dw2:>8.4f} {dbias:>8.4f} "
                  f"{w1_new:>8.4f} {w2_new:>8.4f} {bias_new:>8.4f}")

            historial.append({
                "iter": iteracion, "ej": idx,
                "x1": x1, "x2": x2, "target": t,
                "net": net, "y": y, "error": error,
                "w1": w1, "w2": w2, "bias": bias,
                "dw1": dw1, "dw2": dw2, "dbias": dbias,
                "w1_new": w1_new, "w2_new": w2_new, "bias_new": bias_new
            })

            # Actualizamos para el siguiente ejemplo
            w1, w2, bias = w1_new, w2_new, bias_new

    print(f"\n  Pesos FINALES: w1={w1:.4f}, w2={w2:.4f}, bias={bias:.4f}")
    return historial, (w1, w2, bias)


# ─── 2.4  Dibujar recta de separabilidad lineal ───────────────────────────────
def dibujar_perceptron(historial, pesos_finales, ejemplos):
    """
    Dibuja el plano 2D con los puntos de entrenamiento y la recta
    de separación lineal aprendida por el perceptrón.

    La recta de decisión es donde net = 0:
      w1*x1 + w2*x2 + bias = 0
      → x2 = -(w1*x1 + bias) / w2

    Cada subfigura muestra el estado después de una iteración completa.
    """
    w1_f, w2_f, bias_f = pesos_finales

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Perceptrón — Recta de Separabilidad Lineal", fontsize=14, fontweight="bold")

    # Colores por clase
    COLORES  = {0: "#E74C3C", 1: "#27AE60"}
    MARCAS   = {0: "o", 1: "^"}
    NOMBRES  = {0: "Clase 0", 1: "Clase 1"}

    for it_idx, (ax, n_iter_show) in enumerate(zip(axes, [1, 2])):
        # Obtenemos los pesos AL FINAL de la iteración n_iter_show
        pasos_iter = [h for h in historial if h["iter"] == n_iter_show]
        if pasos_iter:
            ultimo = pasos_iter[-1]
            w1_p, w2_p, bias_p = ultimo["w1_new"], ultimo["w2_new"], ultimo["bias_new"]
        else:
            w1_p, w2_p, bias_p = w1_f, w2_f, bias_f

        ax.set_title(f"Iteración {n_iter_show}\n"
                     f"w1={w1_p:.4f}, w2={w2_p:.4f}, bias={bias_p:.4f}",
                     fontsize=10)

        # Graficar puntos
        for (x1, x2), t in ejemplos:
            ax.scatter(x1, x2, c=COLORES[t], marker=MARCAS[t],
                       s=120, edgecolors="black", zorder=3)
            ax.annotate(f"({x1},{x2})", (x1, x2),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)

        # Recta de decisión: w1*x1 + w2*x2 + bias = 0  →  x2 = -(w1*x1 + bias) / w2
        x_vals = np.linspace(-4, 4, 200)
        if abs(w2_p) > 1e-9:
            x2_recta = -(w1_p * x_vals + bias_p) / w2_p
            ax.plot(x_vals, x2_recta, "b-", lw=2, label="Frontera de decisión")
        else:
            ax.axvline(x=-bias_p / w1_p if abs(w1_p) > 1e-9 else 0,
                       color="blue", lw=2, label="Frontera de decisión")

        # Verificación: clasificar cada punto con los pesos de esta iteración
        correctos = 0
        for (x1, x2), t in ejemplos:
            net_v = w1_p * x1 + w2_p * x2 + bias_p
            y_v   = escalon(net_v)
            if y_v == t:
                correctos += 1

        clasificacion = "✓ Todos correctos" if correctos == len(ejemplos) \
                        else f"✗ {correctos}/{len(ejemplos)} correctos"
        ax.set_xlabel("x₁", fontsize=11)
        ax.set_ylabel("x₂", fontsize=11)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-4, 4); ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F9F9F9")

        parches = [mpatches.Patch(color=COLORES[k], label=NOMBRES[k]) for k in [0, 1]]
        parches.append(mpatches.Patch(color="blue", label=f"Frontera | {clasificacion}"))
        ax.legend(handles=parches, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/perceptron.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Gráfica del perceptrón guardada.")


# ==============================================================================
#  EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("  TPN5 — MINERÍA DE DATOS: ID3 + PERCEPTRÓN")
    print("="*70)

    # ──────────────────────────────────────────────────────────────────────────
    #  PARTE 1: ID3
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  PARTE 1: ALGORITMO ID3 — Problema del Tiempo")
    print("─"*70)

    print("\nDataset de entrenamiento:")
    print(df.to_string())

    print(f"\nEntropía inicial del conjunto completo:")
    h_total = entropia(df)
    si_total = (df[CLASE] == "SI").sum()
    no_total = (df[CLASE] == "NO").sum()
    print(f"  {si_total} SI, {no_total} NO de {len(df)} ejemplos")
    print(f"  H(S) = -{si_total}/14·log2({si_total}/14) - {no_total}/14·log2({no_total}/14)")
    print(f"  H(S) = {h_total:.6f} bits")

    print("\n\n>>> CÁLCULOS PASO A PASO (nodo a nodo):")
    mostrar_calculos_id3(df, ATRIBUTOS)

    print("\n\n>>> ÁRBOL APRENDIDO POR ID3:")
    arbol = id3(df, ATRIBUTOS)

    # Verificación: clasificar todos los ejemplos con el árbol
    def clasificar(nodo, fila):
        if nodo["hoja"]:
            return nodo["clase"]
        valor = fila[nodo["atributo"]]
        return clasificar(nodo["hijos"][valor], fila)

    print("\nVerificación (clasificación de ejemplos de entrenamiento):")
    errores = 0
    for ej, fila in df.iterrows():
        pred = clasificar(arbol, fila)
        real = fila[CLASE]
        estado = "✓" if pred == real else "✗"
        if pred != real:
            errores += 1
        print(f"  {estado} {ej}: predicho={pred}, real={real}")
    print(f"\n  Exactitud: {(len(df)-errores)}/{len(df)} = {(len(df)-errores)/len(df)*100:.1f}%")

    # Graficar árbol
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    dibujar_arbol(arbol, ax, x=0.5, y=0.95, dx=0.22, dy=0.20)
    ax.set_title("Árbol de Decisión ID3 — Problema del Tiempo\n(Juego Tenis)", 
                 fontsize=13, fontweight="bold", pad=10)

    parches = [
        mpatches.Patch(color="#4A90D9", label="Nodo de decisión"),
        mpatches.Patch(color="#27AE60", label="Hoja: SI"),
        mpatches.Patch(color="#E74C3C", label="Hoja: NO"),
    ]
    ax.legend(handles=parches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/arbol_id3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✓ Árbol de decisión guardado como imagen.")

    # ──────────────────────────────────────────────────────────────────────────
    #  PARTE 2: PERCEPTRÓN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n\n" + "─"*70)
    print("  PARTE 2: PERCEPTRÓN — Aprendizaje con 2 iteraciones")
    print("─"*70)

    print(f"\nEjemplos: {ejemplos_perceptron}")
    print(f"Pesos iniciales: w1={w1_init}, w2={w2_init}, bias={bias_init}")
    print(f"Tasa de aprendizaje η={ETA}, Umbral={UMBRAL}")

    historial, pesos_finales = entrenar_perceptron(
        ejemplos_perceptron, w1_init, w2_init, bias_init, ETA, N_ITERACIONES, UMBRAL
    )

    # Verificación final
    w1_f, w2_f, b_f = pesos_finales
    print("\nVerificación con pesos finales:")
    correctos = 0
    for (x1, x2), t in ejemplos_perceptron:
        net = w1_f * x1 + w2_f * x2 + b_f
        y   = escalon(net)
        ok  = "✓" if y == t else "✗"
        if y == t:
            correctos += 1
        print(f"  {ok} ({x1},{x2}): net={net:.4f}, y={y}, target={t}")

    print(f"\n  Clasificación: {correctos}/{len(ejemplos_perceptron)} correctos")
    if correctos < len(ejemplos_perceptron):
        print("  → Los datos NO son linealmente separables con 2 iteraciones.")
        print("    Se necesitarían más épocas o un modelo no-lineal.")
    else:
        print("  → Todos los puntos clasificados correctamente.")

    dibujar_perceptron(historial, pesos_finales, ejemplos_perceptron)

    print("\n" + "="*70)
    print("  FIN DEL TPN5")
    print("="*70)


if __name__ == "__main__":
    main()
