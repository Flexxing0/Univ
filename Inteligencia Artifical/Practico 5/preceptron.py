import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron

x = np.array([[-1, -3], [1, 2], [2, 3], [1, 0]])
d = np.array([0, 1, 0, 1])
w1, w2, b = 0.2, 0.1, 0.3

red = Perceptron(max_iter=2, tol=None, shuffle=False)
red.fit(x,d,coef_init=[w1,w2],intercept_init=[b])
print(red.coef_[0][0],red.coef_[0][1],red.intercept_[0])

w1_final = red.coef_[0][0]
w2_final = red.coef_[0][1]
b_final = red.intercept_[0]

print("--- RESULTADOS TRAS 2 ITERACIONES ---")
print(f"Peso W1: {w1_final}")
print(f"Peso W2: {w2_final}")
print(f"Bias b:  {b_final}")

predicciones = red.predict(x)
print(f"\nSalidas reales esperadas: {d}")
print(f"Predicciones del modelo:  {predicciones}")

x1_linea = np.linspace(-3, 4, 100)
x2_linea = -(w1_final / w2_final) * x1_linea - (b_final / w2_final)


plt.figure(figsize=(8, 6))


colores = ['blue' if i == 1 else 'red' for i in d]
plt.scatter(x[:, 0], x[:, 1], color=colores, s=120, zorder=5, label="Datos de entrenamiento")

plt.plot(x1_linea, x2_linea, color='green', linestyle='-', linewidth=2.5, label="Frontera de decisión (Iteración 2)")
plt.axhline(0, color='black', linewidth=0.8, alpha=0.7)
plt.axvline(0, color='black', linewidth=0.8, alpha=0.7)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(-3, 4)
plt.ylim(-4, 4)
plt.xlabel('$X_1$ (Entrada 1)')
plt.ylabel('$X_2$ (Entrada 2)')
plt.title('Separabilidad Lineal del Perceptrón con Scikit-Learn')
plt.legend(loc='upper right')
plt.savefig("recta_separacion.png", dpi=150)
plt.close()