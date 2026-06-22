import math
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.tree import DecisionTreeClassifier,plot_tree, export_text #arbol de decision y ploteo
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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
        "Regresion Logistica": modelo_logistica,
        "Arbol de Decision": modelo_arbol,
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

#con esto hacemos pto6,7 y 9
def tests(modelos_entrenados,x_test,y_test):
    resultados = []
    mejor_f1 = 0
    mejor_nombre = ""
    mejor_preds = None
    for nombre, modelo in modelos_entrenados.items():
        
        predicciones = modelo.predict(x_test)
        
        # Calculamos las métricas (usando pos_label=0 para Maligno)
        acc = accuracy_score(y_test, predicciones)
        prec = precision_score(y_test, predicciones, pos_label=0)
        rec = recall_score(y_test, predicciones, pos_label=0)
        f1 = f1_score(y_test, predicciones, pos_label=0)
        
        #se crea la matriz para cada modelo
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            predicciones, 
            display_labels=['Maligno (0)', 'Benigno (1)'],
            cmap=plt.cm.Blues,
            ax=ax 
        )
        ax.set_title(f"Matriz de Confusion - {nombre}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        nombre_archivo = f"matriz_{nombre.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(nombre_archivo, dpi=300)
        plt.close()
        
        #se guardan resultados en diccionario
        resultados.append({
            "Modelo": nombre,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
        #se guarda mejor modelo
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_nombre = nombre
            mejor_preds = predicciones
        if nombre == "Random Forest":
            modelo_bosque = modelo
            
    #pto7 se imprime mejor modelo
    print(f"\nMejor modelo: {mejor_nombre} (F1={mejor_f1:.4f})")
    disp = ConfusionMatrixDisplay.from_predictions( y_test, mejor_preds, display_labels=["Maligno (0)", "Benigno (1)"], cmap="Blues")
    disp.ax_.set_title(f"Matriz de Confusion - {mejor_nombre}", fontsize=12, fontweight='bold')
    disp.ax_.set_xlabel("Predicción")
    disp.ax_.set_ylabel("Real")
    plt.savefig("punto7_matriz_confusion.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    #pto9
    variables_importantes(modelo_bosque)
    # dataframe con comparativa pto6
    return pd.DataFrame(resultados), resultados

def grafica_resultados(resultados):
    
    print("\n=== GENERANDO GRÁFICO COMPARATIVO GENERAL (PUNTO 6) ===")
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    posiciones = np.arange(len(metricas_nombres))
    num_modelos = len(resultados)
    ancho_barra = 0.8 / num_modelos 
    fig, ax = plt.subplots(figsize=(12, 7))
    colores = ['#34495e', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f']
    e=0
    # Iteramos sobre el diccionario para calcular métricas y dibujar las barras de cada uno
    for i in resultados:
        metricas = [
            i['Accuracy'],
            i['Precision'],
            i['Recall'],
            i['F1-Score']
        ]
        e+=1
        offset = (e - (num_modelos - 1) / 2) * ancho_barra
        barras = ax.bar(posiciones + offset, metricas, ancho_barra, label=i["Modelo"], color=colores[e % len(colores)])
        
        for barra in barras:
            alto = barra.get_height()
            ax.annotate(f'{alto:.2f}',
                        xy=(barra.get_x() + barra.get_width() / 2, alto),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Ajustes estéticos finales del gráfico
    ax.set_title('Comparativa General de Rendimiento - Modelos Originales', fontsize=14, fontweight='bold')
    ax.set_ylabel('Puntaje (Score)', fontsize=12)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(metricas_nombres, fontsize=11)
    ax.set_ylim(0, 1.15) # Espacio para las etiquetas y la leyenda
    ax.legend(loc='upper right', shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparativa_general_modelos.png', dpi=300)
    plt.close()
    
    print("¡Gráfico 'comparativa_general_modelos.png' generado con éxito!")

def optimizar_random_forest(x_train, y_train):
    print("\n=== OPTIMIZANDO HIPERPARÁMETROS CON GRIDSEARCHCV ===")
    
    # 1. Instanciamos un modelo base vacío
    rf_base = RandomForestClassifier(random_state=42)
    
    # 2. Definimos la "grilla" de parámetros que queremos probar
    # Probaremos combinaciones de cantidad de árboles, profundidad y criterio de división
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt','log2',None]
        }
    
    # 3. Configuramos el GridSearchCV
    # cv=5 significa que usará Validación Cruzada de 5 pliegues
    # scoring='f1' hará que busque la combinación que logre el mejor F1-Score
    grid_search = GridSearchCV(
        estimator=rf_base, 
        param_grid=param_grid, 
        cv=5, 
        scoring='f1', 
        n_jobs=-1 # Usa todos los núcleos de tu procesador para ir más rápido
    )
    
    # 4. Ejecutamos la búsqueda (esto va a probar las 18 combinaciones posibles 5 veces cada una)
    grid_search.fit(x_train, y_train)
    
    # 5. Mostramos los resultados en la terminal
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor F1-Score en entrenamiento: {grid_search.best_score_:.4f}")
    
    # Devuelve el mejor modelo ya entrenado con la configuración ganadora
    return grid_search.best_estimator_

def comparativa_bosques(bosque_original, bosque_optimizado, y_test):
    
    metricas_orig = [
        bosque_original['Accuracy'],
        bosque_original['Precision'],
        bosque_original['Recall'],
        bosque_original['F1-Score']
    ]
    
    metricas_tune = [
        accuracy_score(y_test, bosque_optimizado),
        precision_score(y_test, bosque_optimizado, pos_label=0),
        recall_score(y_test, bosque_optimizado, pos_label=0),
        f1_score(y_test, bosque_optimizado, pos_label=0)
    ]
    
    # 3. Configuración del gráfico
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    posiciones = np.arange(len(metricas_nombres))
    ancho_barra = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Usamos el nombre del modelo dinámicamente en las etiquetas
    barras_orig = ax.bar(posiciones - ancho_barra/2, metricas_orig, ancho_barra, label=f'{bosque_original["Modelo"]} Original', color='#34495e')
    barras_tune = ax.bar(posiciones + ancho_barra/2, metricas_tune, ancho_barra, label="Random Forest Optimizado", color='#e67e22')
    
    ax.set_title(f'Comparativa de Rendimiento: Random Forest Antes vs Despues', fontsize=14, fontweight='bold')
    ax.set_ylabel('Puntaje (Score)', fontsize=12)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(metricas_nombres, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Función interna para poner los números sobre las barras
    for barras in [barras_orig, barras_tune]:
        for barra in barras:
            alto = barra.get_height()
            ax.annotate(f'{alto:.3f}',
                        xy=(barra.get_x() + barra.get_width() / 2, alto),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    # Guardamos el archivo con el nombre del modelo (ej: "comparativa_regresión_logística.png")
    nombre_archivo = f"comparativa_optimizacion_{'Random Forest'.replace(' ', '_').lower()}.png"
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()
    
    print(f"¡Gráfico guardado como '{nombre_archivo}'!")

def variables_importantes(modelo_bosque):
    importancias = modelo_bosque.feature_importances_
    df_imp = pd.DataFrame({
        "Variable": x_test.columns,
        "Importancia": importancias
    }).sort_values("Importancia", ascending=False).head(10)

    print("\nPunto 9: Random Forest")
    print(df_imp.to_string(index=False))


    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["Variable"][::-1], df_imp["Importancia"][::-1], color="green")
    plt.title("Punto 9 – Importancia de Variables (Random Forest)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig("punto9_importancia_variables.png", dpi=150)
    plt.close()
    print("\n terminado")
    
def arbol_reducido(x_train, y_train, x_test, y_test):
    arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    arbol.fit(x_train, y_train)
    predicciones = arbol.predict(x_test)
    
    print(f"Árbol reducido")
    print(f"\nAccuracy : {accuracy_score(y_test, predicciones)}")
    print(f"\nF1 (Maligno): {f1_score(y_test, predicciones, pos_label=0)}")
    print("\nReglas:")
    print(export_text(arbol, feature_names=x_train.columns.tolist()))
    
    plt.figure(figsize=(20, 8))
    plot_tree(arbol, feature_names=x_train.columns.tolist(), class_names=["Maligno", "Benigno"], filled=True, rounded=True, fontsize=9)
    plt.title("Arbol Interpretable (max_depth=3)")
    plt.tight_layout()
    plt.savefig("punto10_arbol_reducido.png", bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    #PTO1Y2
    graficos_iniciales()
    #PTO3
    estadisticos_descriptivos()
    #PTO4
    df = armar_df()
    x_train,x_test,y_train,y_test= split(df)
    #PTO5
    modelos_entrenados = entrenamiento(modelos(),x_train,y_train)
    #PTO6,7Y9
    df_resultados,resultados = tests(modelos_entrenados,x_test,y_test)
    grafica_resultados(resultados)
    #PTO8
    mejor_bosque_tuneado = optimizar_random_forest(x_train, y_train)
    predicciones_tuneadas = mejor_bosque_tuneado.predict(x_test)
    for i in resultados:
        if i["Modelo"] == "Random Forest":
            comparativa_bosques(i, predicciones_tuneadas, y_test)
    print("\nReporte de Clasificación del Modelo Optimizado:")
    print(classification_report(y_test, predicciones_tuneadas))
    #PTO10
    arbol_reducido(x_train, y_train, x_test, y_test)
    