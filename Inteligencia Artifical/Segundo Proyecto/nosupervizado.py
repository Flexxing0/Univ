import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --- FUNCIONES DEL PROCESO ---

def cargar_y_describir_iris():
    """Punto 11"""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    
    print("PUNTO 11: DESCRIPCIÓN DE VARIABLES ")
    print(f"Cantidad de registros (filas): {X.shape[0]}")
    print(f"Cantidad de variables (columnas): {X.shape[1]}")
    print("Nombres de las variables:", list(X.columns))
    print("-" * 50)
    return X, y

def estandarizar_datos(X):
    """Punto 12"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print("PUNTO 12: DATOS ESTANDARIZADOS")
    print("Media de las variables transformadas (esperada ~0):\n", X_scaled_df.mean().round(2).values)
    print("Desviación estándar (esperada ~1):\n", X_scaled_df.std().round(2).values)
    print("-" * 50)
    return X_scaled

def aplicar_pca(X_scaled):
    """Punto 13"""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    varianza_explicada = pca.explained_variance_ratio_
    print(" PUNTO 13: ANÁLISIS DE PCA")
    print(f"Varianza explicada por la Componente 1: {varianza_explicada[0]*100:.2f}%")
    print(f"Varianza explicada por la Componente 2: {varianza_explicada[1]*100:.2f}%")
    print(f"Varianza acumulada total: {np.sum(varianza_explicada)*100:.2f}%")
    print("-" * 50)
    return X_pca

def analizar_kmeans_metodo_codo_y_silhouette(X_scaled):
    """Puntos 14, 15 y 16"""
    inercias = []
    siluetas = []
    valores_k = range(2, 8) 
    
    print("=== PUNTOS 14, 15 Y 16: ENTRENAMIENTO KMEANS ===")
    for k in valores_k:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(X_scaled)
        
        #Inercia (Punto 15)
        inercias.append(kmeans.inertia_)
        
        #Coeficiente de Silhouette (Punto 16)
        score_silueta = silhouette_score(X_scaled, kmeans.labels_)
        siluetas.append(score_silueta)
        
        print(f"Para K = {k} -> Inercia: {kmeans.inertia_:.2f} | Coeficiente Silhouette: {score_silueta:.4f}")
    print("-" * 50)
    
    # --- GRÁFICO 1: MÉTODO DEL CODO (INERCIA) ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(valores_k, inercias, marker='o', color='tab:blue', linewidth=2, label='Inercia')
    ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inercia (Suma de errores cuadrados)', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Evaluacion de Metricas Internas para KMeans', fontsize=14, fontweight='bold')
    
    # --- GRÁFICO 2: COEFIENTE DE SILHOUETTE (EJE DOBLE) ---
    ax2 = ax1.twinx()  # Comparten el mismo eje X
    ax2.plot(valores_k, siluetas, marker='s', color='tab:orange', linewidth=2, linestyle='--', label='Silhouette')
    ax2.set_ylabel('Coeficiente de Silhouette', color='tab:orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.tight_layout()
    plt.savefig('analisis_metricas_internas.png', dpi=300)
    plt.close()

def visualizar_clusters_y_clases_reales(X_pca, X_scaled, y_real):
    """Punto 17"""
    # Elegimos K=3 basándonos en la morfología biológica del dataset y los análisis métricos
    k_elegido = 3
    kmeans_final = KMeans(n_clusters=k_elegido, n_init='auto', random_state=42)
    clusters_predichos = kmeans_final.fit(X_scaled)
    labels_pred = clusters_predichos.labels_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico Izquierdo: Clusters identificados por KMeans (No Supervisado)
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pred, cmap='viridis', s=60, alpha=0.8)
    ax1.set_title(f'Agrupamiento No Supervisado (KMeans K={k_elegido})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Componente Principal 1')
    ax1.set_ylabel('Componente Principal 2')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Gráfico Derecho: Especies Reales del dataset Iris (Supervisado - Solo para comparar)
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real, cmap='plasma', s=60, alpha=0.8)
    ax2.set_title('Clases Reales (Especies del Dataset Iris)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Componente Principal 1')
    ax2.set_ylabel('Componente Principal 2')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Calcular e imprimir una métrica externa de control para la explicación teórica
    ari = adjusted_rand_score(y_real, labels_pred)
    print(f"PUNTO 17: ANÁLISIS DE COMPARACIÓN ")
    print(f"Métrica Externa - Adjusted Rand Index (ARI): {ari:.4f}")
    print("-" * 50)
    
    plt.tight_layout()
    plt.savefig('visualizacion_clusters_pca.png', dpi=300)
    plt.close()

# --- EJECUCIÓN DEL FLUJO NO SUPERVISADO ---
if __name__ == "__main__":
    X_iris, y_iris = cargar_y_describir_iris()
    X_estandarizado = estandarizar_datos(X_iris)
    X_plano_pca = aplicar_pca(X_estandarizado)
    analizar_kmeans_metodo_codo_y_silhouette(X_estandarizado)
    visualizar_clusters_y_clases_reales(X_plano_pca, X_estandarizado, y_iris)