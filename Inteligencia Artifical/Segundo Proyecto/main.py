import math
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.tree import DecisionTreeClassifier,plot_tree #arbol de decision y ploteo
from sklearn.model_selection import train_test_split #herramienta de corte de dataset
from sklearn.metrics import ConfusionMatrixDisplay #matriz de confusión
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, poisson
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#PARTEA
#1=benigno
#0=maligno
x,y = load_breast_cancer(return_X_y=True, as_frame=True)
#muestreo variables
def graficos_iniciales():
    filas, columnas = x.shape
    print(f"Cantidad de registros (pacientes): {filas}")
    print(f"Cantidad de variables (atribútos): {columnas}")
    clases = np.unique(y)
    print(f"Clases identificadas en el target: {clases} (0 = Maligno, 1 = Benigno)")
    #verifica desbalance
    conteo_clases = y.value_counts()
    porcentaje_clases = y.value_counts(normalize=True) * 100

    print("Distribución de clases (Absoluta):")
    print(conteo_clases)
    print("\nDistribución de clases (Porcentaje):")
    print(porcentaje_clases)

    plt.figure(figsize=(6, 4))
    conteo_clases.plot(kind='bar', color=['green', 'red'])
    plt.title('Distribución de Clases (0: Maligno, 1: Benigno)')
    plt.xlabel('Clase')
    plt.ylabel('Cantidad de Registros')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('distribucion_clases.png')
    plt.close
    plt.figure(figsize=(6,6))
    etiquetas = ['Benigno (1)', 'Maligno (0)']
    plt.pie(
        conteo_clases, 
        labels=etiquetas, 
        autopct='%1.1f%%',
        startangle=90,       
    )
    plt.title("Distribución de clases")
    plt.savefig("distribucion_clases2.png")
    plt.close()
    
def estadisticos_descriptivos():
    
    estadisticos = x.describe().T
    print(estadisticos.head(10)) 
    estadisticos.to_csv("estadisticos_descriptivos.csv")

def armar_df():
    df = x.copy()
    df['target'] = y
    print(df.isnull().sum().sum())  
    print(df.dtypes)
    return df

def split(df):
    columnas = df.columns.values.tolist()
    prediccion = columnas[:30]
    objetivo = columnas[30]
    x_train,x_test,y_train,y_test= train_test_split(df[prediccion],df[objetivo],test_size=0.30,stratify=df[objetivo],random_state=56)
    return x_train,x_test,y_train,y_test

def modelos():
    modelo_logistica = LogisticRegression(max_iter=10000, random_state=42)
    modelo_arbol     = DecisionTreeClassifier(random_state=42)
    modelo_bosque    = RandomForestClassifier(random_state=42)
    modelo_red       = MLPClassifier(max_iter=1000, random_state=42)
    modelos = {
        "Regresión Logística": modelo_logistica,
        "Árbol de Decisión": modelo_arbol,
        "Random Forest": modelo_bosque,
        "Red Neuronal (MLP)": modelo_red
    }
    return modelos
#con los datos de entrenamiento podemos sacar feature_importances(pto9), para saber variables determinantes
def entrenamiento(modelos,x_train,y_train):
    modelos_entrenados=copy.deepcopy(modelos)
    for nombre, modelo in modelos_entrenados.items():
        modelo.fit(x_train, y_train)
        modelos_entrenados[nombre]=modelo
        print(f"Modelo {nombre} entrenado exitosamente")
        
    return modelos_entrenados
#con esto hacemos pto6
def tests(modelos_entrenados,x_test,y_test):
    resultados = []
    
    for nombre, modelo in modelos_entrenados.items():
        # El método predict SOLO recibe x_test
        predicciones = modelo.predict(x_test)
        
        # Calculamos las métricas (usando pos_label=0 para Maligno)
        acc = accuracy_score(y_test, predicciones)
        prec = precision_score(y_test, predicciones, pos_label=0)
        rec = recall_score(y_test, predicciones, pos_label=0)
        f1 = f1_score(y_test, predicciones, pos_label=0)
        
        resultados.append({
            "Modelo": nombre,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
    # Devolvemos un DataFrame listo con la comparativa (Punto 6)
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    graficos_iniciales()
    estadisticos_descriptivos()
    df = armar_df()
    x_train,x_test,y_train,y_test= split(df)
    modelos_entrenados = entrenamiento(modelos(),x_train,y_train)
    resultados = tests(modelos_entrenados,x_test,y_test)
    print(resultados)
    