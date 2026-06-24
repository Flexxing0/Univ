import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, v_measure_score # <-- Agregamos v_measure_score

def cargar_y_describir_iris():
    """Punto 11: Carga el dataset Iris y describe sus variables."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    
    print("PUNTO 11: DESCRIPCIÓN DE VARIABLES ")
    print(f"Cantidad de registros (filas): {X.shape[0]}")
    print(f"Cantidad de variables (columnas): {X.shape[1]}")
    print("Nombres de las variables:", list(X.columns))
    print("-" * 50)
    
    fig, ejes = plt.subplots(2, 2, figsize=(10, 8))
    columnas = list(X.columns)
    colores = ['#34495e', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, ax in enumerate(ejes.ravel()):
        ax.hist(X[columnas[i]], bins=15, color=colores[i], edgecolor='black', alpha=0.7)
        ax.set_title(columnas[i], fontsize=11, fontweight='bold')
        ax.set_ylabel('Frecuencia')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.suptitle('Punto 11 - Distribución y Rangos Originales de las Variables de Iris', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('punto11_distribucion_variables.png', dpi=300)
    plt.close()
    
    print("-> ¡Gráfico 'punto11_distribucion_variables.png' generado para la descripción inicial!")
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
    print("PUNTO 13: ANÁLISIS COMPONENTES PRINCIPALES (PCA)")
    print(f"Varianza explicada por componente: {pca.explained_variance_ratio_}")
    print(f"Varianza acumulada total (2 componentes): {sum(pca.explained_variance_ratio_):.4f}")
    print("-" * 50)
    return X_pca, pca

def analizar_kmeans_metodo_codo_y_silhouette(X_scaled):
    """Puntos 14, 15 y 16"""
    inercias = []
    siluetas = []
    valores_k = range(2, 8) 
    
    print("=== PUNTOS 14, 15 Y 16: ENTRENAMIENTO KMEANS ===")
    for k in valores_k:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(X_scaled)
        
        # Inercia (Punto 15)
        inercias.append(kmeans.inertia_)
        
        # Coeficiente de Silhouette (Punto 16)
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
    ax2 = ax1.twinx()
    ax2.plot(valores_k, siluetas, marker='s', color='tab:orange', linewidth=2, linestyle='--', label='Silhouette')
    ax2.set_ylabel('Coeficiente de Silhouette', color='tab:orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.tight_layout()
    plt.savefig('analisis_metricas_internas.png', dpi=300)
    plt.close()

def visualizar_clusters_y_clases_reales(X_pca, X_scaled, y_real, pca_transformer):
    """Punto 17 y 18: Visualiza clusters con centroides, esferas e imprime métricas externas de contraste."""
    k_elegido = 3
    kmeans_final = KMeans(n_clusters=k_elegido, n_init='auto', random_state=42)
    kmeans_final.fit(X_scaled)
    labels_pred = kmeans_final.labels_
    
    # 1. PASAMOS LOS CENTROIDES DE 4D A 2D USANDO EL MISMO PCA
    centroides_4d = kmeans_final.cluster_centers_
    centroides_2d = pca_transformer.transform(centroides_4d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    colores_clusters = ['#1f77b4', '#2ca02c', '#9467bd']
    
    # =========================================================================
    # GRÁFICO IZQUIERDO: KMEANS CON CENTROIDES Y CÍRCULOS (NO SUPERVISADO)
    # =========================================================================
    for i in range(k_elegido):
        puntos_cluster = X_pca[labels_pred == i]
        ax1.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], 
                    color=colores_clusters[i], s=50, alpha=0.7, label=f'Clúster {i}')
        
        cent_x, cent_y = centroides_2d[i, 0], centroides_2d[i, 1]
        
        # Centroide (X grande)
        ax1.scatter(cent_x, cent_y, color='black', marker='X', s=250, 
                    edgecolor='white', linewidth=1.5, zorder=10)
        
        # Círculo Envolvente
        distancias = np.sqrt(np.sum((puntos_cluster - [cent_x, cent_y]) ** 2, axis=1))
        radio_envolvente = np.max(distancias) if len(distancias) > 0 else 0.5
        
        circulo = plt.Circle((cent_x, cent_y), radio_envolvente, color=colores_clusters[i], 
                             fill=True, alpha=0.1, linestyle='--', linewidth=1.5, edgecolor=colores_clusters[i])
        ax1.add_patch(circulo)
        
    ax1.set_title(f'Agrupamiento No Supervisado (KMeans K={k_elegido})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Componente Principal 1')
    ax1.set_ylabel('Componente Principal 2')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right')
    
    # =========================================================================
    # GRÁFICO DERECHO: CLASES REALES (SOLO PARA COMPARATIVA VISUAL)
    # =========================================================================
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real, cmap='plasma', s=50, alpha=0.8)
    ax2.set_title('Clases Reales (Especies del Dataset Iris)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Componente Principal 1')
    ax2.set_ylabel('Componente Principal 2')
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    clases_nombres = ['Setosa', 'Versicolor', 'Virginica']
    handlers, _ = scatter2.legend_elements()
    ax2.legend(handlers, clases_nombres, loc='upper right')
    
    # =========================================================================
    # CÁLCULO E INYECCIÓN DE MÉTRICAS EXTERNAS (PUNTO 18)
    # =========================================================================
    ari = adjusted_rand_score(y_real, labels_pred)
    v_measure = v_measure_score(y_real, labels_pred)
    
    # Añadimos un cuadro de texto prolijo en el gráfico izquierdo (KMeans) abajo a la izquierda
    texto_metricas = f"Métricas Externas (Ground Truth):\n▶ ARI: {ari:.4f}\n▶ V-Measure: {v_measure:.4f}"
    propiedades_caja = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#bdc3c7', alpha=0.85)
    ax1.text(0.03, 0.04, texto_metricas, transform=ax1.transAxes, fontsize=10,
             fontweight='bold', color='#2c3e50', bbox=propiedades_caja, zorder=20)
    
    # Impresión estructurada en la consola
    print("\n" + "="*55)
    print("   PUNTO 17 Y 18: EVALUACIÓN MEDIANTE MÉTRICAS EXTERNAS   ")
    print("="*55)
    print(f" Adjusted Rand Index (ARI)  | Valor: {ari:.4f}")
    print(f" V-Measure Score (Homog/Comp) | Valor: {v_measure:.4f}")
    print("="*55 + "\n")
    
    plt.tight_layout()
    plt.savefig('visualizacion_clusters_pca.png', dpi=300)
    plt.close()
    print("-> ¡Gráfico 'visualizacion_clusters_pca.png' generado e inyectado con los valores de las métricas!")

# --- EJECUCIÓN DEL FLUJO NO SUPERVISADO ---
if __name__ == "__main__":
    X_iris, y_iris = cargar_y_describir_iris()
    X_estandarizado = estandarizar_datos(X_iris)
    X_plano_pca, objeto_pca = aplicar_pca(X_estandarizado)
    
    analizar_kmeans_metodo_codo_y_silhouette(X_estandarizado)
    visualizar_clusters_y_clases_reales(X_plano_pca, X_estandarizado, y_iris, objeto_pca)