from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
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
from sklearn.neighbors import KNeighborsClassifier
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
    plt.close()
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

def estadisticos_descriptivos(df):
    print("\n" + "="*70)
    print("   ESTADÍSTICOS DESCRIPTIVOS DEL DATASET")
    print("="*70)

    desc = df.drop(columns=['target']).describe().T

    modas = df.drop(columns=['target']).mode().iloc[0]
    desc['moda'] = modas

    desc = desc.rename(columns={
        'mean': 'Media',
        'std': 'Desvío Std',
        'min': 'Mínimo',
        '50%': 'Mediana',
        'max': 'Máximo',
        'moda': 'Moda'
    })

    tabla_final = desc[['Media', 'Moda', 'Mediana', 'Desvío Std', 'Mínimo', 'Máximo']]

    print(tabla_final.round(3).to_string())
    print("="*70 + "\n")
    print("Nota: en variables continuas la moda suele ser poco representativa,")
    print("ya que rara vez hay valores exactamente repetidos.\n")

    tabla_final.to_csv('estadisticos_descriptivos.csv')
    print("-> 'estadisticos_descriptivos.csv' generado.")

    graficar_estadisticos(tabla_final)

    return tabla_final


def graficar_estadisticos(tabla):
    import matplotlib.pyplot as plt

    micro, media, macro = [], [], []
    for var in tabla.index:
        val_max = tabla.loc[var, 'Máximo']
        if val_max <= 1.0:
            micro.append(var)
        elif val_max <= 50.0:
            media.append(var)
        else:
            macro.append(var)

    grupos = [
        (micro, 'Escala Micro (0 a 1)', 'estadisticos_micro.png'),
        (media, 'Escala Media (hasta 50)', 'estadisticos_media.png'),
        (macro, 'Escala Macro (magnitudes grandes)', 'estadisticos_macro.png'),
    ]

    for variables, titulo_escala, nombre_archivo in grupos:
        if not variables:
            continue

        subset = tabla.loc[variables]
        fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.5 * len(variables))))

        # --- Gráfico 1: Media con barras de error (Desvío Std) ---
        axes[0].barh(subset.index, subset['Media'], xerr=subset['Desvío Std'],
                     color='#3498db', edgecolor='black', capsize=4, alpha=0.85)
        axes[0].set_title(f'Media ± Desvío Estándar — {titulo_escala}', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Valor')
        axes[0].grid(axis='x', linestyle='--', alpha=0.4)

        # --- Gráfico 2: tabla visual con Mínimo, Mediana, Máximo ---
        axes[1].axis('off')
        tabla_mostrar = subset[['Mínimo', 'Mediana', 'Máximo']].round(3)
        tabla_render = axes[1].table(
            cellText=tabla_mostrar.values,
            rowLabels=tabla_mostrar.index,
            colLabels=tabla_mostrar.columns,
            cellLoc='center',
            loc='center'
        )
        tabla_render.auto_set_font_size(False)
        tabla_render.set_fontsize(9)
        tabla_render.scale(1, 1.5)
        axes[1].set_title('Mínimo / Mediana / Máximo', fontsize=11, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"-> '{nombre_archivo}' generado.")
        
def boxplots(df):
    print("\n=== GENERANDO BOXPLOTS AUTOMÁTICOS BASADOS EN TOP 10 DE CORRELACIÓN ===")
    
    df_temp = copy.deepcopy(df)
    df_temp['Diagnóstico'] = df_temp['target'].map({0: 'Maligno (0)', 1: 'Benigno (1)'})
    
    matriz_corr = df_temp.drop(columns=['Diagnóstico']).corr()
    top_10_serie = matriz_corr['target'].abs().sort_values(ascending=False).iloc[1:11]
    top_10_atributos = top_10_serie.index.tolist()
    

    print("\n" + "="*60)
    print("   TOP 10 ATRIBUTOS CON MAYOR CORRELACIÓN A LAS CLASES")
    print("="*60)
    for puesto, (atributo, valor) in enumerate(top_10_serie.items(), 1):
        print(f"{puesto:02d} | {atributo:<25} | Correlación: {valor:.6f}")
    print("="*60 + "\n")
    
    
    atributos_micro = []  
    atributos_media = []  
    atributos_macro = []  
    
    for col in top_10_atributos:
        val_max = df_temp[col].abs().max()
        if val_max <= 1.0:
            atributos_micro.append(col)
        elif val_max <= 50.0:
            atributos_media.append(col)
        else:
            atributos_macro.append(col)
            

    
    if atributos_micro:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_temp.boxplot(column=atributos_micro, by='Diagnóstico', ax=ax)
        ax.set_title('Top Correlación - Características de Escala Micro (0 a 1)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor Decimal')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('boxplot_auto_micro.png', dpi=300)
        plt.close()
        print("-> 'boxplot_auto_micro.png' generado.")

    if atributos_media:
        fig, ax = plt.subplots(figsize=(8, 5))
        df_temp.boxplot(column=atributos_media, by='Diagnóstico', ax=ax)
        ax.set_title('Top Correlación - Características de Escala Media', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades de Medida')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('boxplot_auto_media.png', dpi=300)
        plt.close()
        print("-> 'boxplot_auto_media.png' generado.")

    if atributos_macro:

        num_macro = len(atributos_macro)
        fig, ejes = plt.subplots(1, num_macro, figsize=(6 * num_macro, 6))
    
        if num_macro == 1:
            ejes = [ejes]
            
        for idx, col in enumerate(atributos_macro):
            df_temp.boxplot(column=[col], by='Diagnóstico', ax=ejes[idx])
            ejes[idx].set_title(f'{col}')
            ejes[idx].set_ylabel('Escala Absoluta')
            
        plt.suptitle('Top Correlación - Características de Escala Macro (Magnitudes Grandes)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('boxplot_auto_macro.png', dpi=300)
        plt.close()
        print("-> 'boxplot_auto_macro.png' generado.")

def armar_df():
    print("normal")
    df = copy.deepcopy(x)
    df['target'] = copy.deepcopy(y)
    atributos = df.columns[:31]
    print(df.isnull().sum().sum()) 
    tabla_df = pd.DataFrame({
        'N°': range(1, 32),
        'Nombre del Atributo (X)': atributos
    }) 
    print("\n" + "="*50)
    print("      TABLA DE ATRIBUTOS DEL DATASET (30)      ")
    print("="*50)
    # index=False hace que no se dibuje el índice por defecto de Pandas
    print(tabla_df.to_string(index=False, justify='left'))
    print("="*50 + "\n")
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
    modelo_knn       = KNeighborsClassifier(n_neighbors=5)
    modelos = {
        "Regresion Logistica": modelo_logistica,
        "Arbol de Decision": modelo_arbol,
        "Random Forest": modelo_bosque,
        "Red Neuronal (MLP)": modelo_red,
        "KNN": modelo_knn
    }
    return modelos

#con los datos de entrenamiento podemos sacar feature_importances(pto9), para saber variables determinantes
def entrenamiento(modelos, x_train, y_train):
    modelo_entrenar=copy.deepcopy(modelos)
    for nombre, modelo in modelo_entrenar.items():
        modelo.fit(x_train, y_train)
        modelo_entrenar[nombre]=modelo
        print(f"{nombre} accuracy: {modelo.score(x_train, y_train)}")
    print("Modelos entrenados exitosamente")
    return modelo_entrenar

#con esto hacemos pto6,7 y 9
def tests(modelos_entrenados,x_test,y_test, opcion):
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
        ax.set_title(f"Matriz de Confusion - {nombre}-{opcion}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        nombre_archivo = f"matriz_{nombre.replace(' ', '_').lower()}-{opcion}.png"
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
    disp.ax_.set_title(f"Matriz de Confusion - {mejor_nombre}-{opcion}", fontsize=12, fontweight='bold')
    disp.ax_.set_xlabel("Predicción")
    disp.ax_.set_ylabel("Real")
    plt.savefig(f"punto7_matriz_confusion-{opcion}.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    #pto9
    
    # dataframe con comparativa pto6
    return pd.DataFrame(resultados), resultados

def grafica_resultados(resultados,opcion):
    
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
    ax.set_title(f'Comparativa General de Rendimiento - Modelos {opcion}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Puntaje (Score)', fontsize=12)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(metricas_nombres, fontsize=11)
    ax.set_ylim(0, 1.15) # Espacio para las etiquetas y la leyenda
    #ax.legend(loc='upper right', shadow=True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparativa_general_modelos_{opcion}.png', dpi=300)
    plt.close()
    
    print(f"¡Gráfico 'comparativa_general_modelos_{opcion}.png' creado")

def optimizar_random_forest(x_train, y_train):
    print("\n OPTIMIZANDO HIPERPARÁMETROS CON GRIDSEARCHCV ")
    
    rf_base = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 3, 5, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': ["sqrt", "log2"]
    }
    f1_maligno = make_scorer(f1_score,pos_label=0)
    
    grid_search = GridSearchCV(
        estimator=rf_base, 
        param_grid=param_grid, 
        cv=5, 
        scoring=f1_maligno, 
        n_jobs=-1 
    )
    
    grid_search.fit(x_train, y_train)
    
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor F1-Score en entrenamiento: {grid_search.best_score_:.4f}")

    resultados_grid = pd.DataFrame(grid_search.cv_results_)
    
    resultados_grid['descripcion_params'] = resultados_grid['params'].apply(
        lambda p: f"Crit: {p['criterion']} | Depth: {p['max_depth'] if p['max_depth'] is not None else 'Unlim'} | Trees: {p['n_estimators']} | Features: {p['max_features']}"
    )
    
    top_combinaciones = resultados_grid.sort_values(by='mean_test_score', ascending=False).head(5)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    scores = top_combinaciones['mean_test_score'].values[::-1]
    nombres_comb = top_combinaciones['descripcion_params'].values[::-1]
    
    colores_barras = ['#bdc3c7'] * 4 + ['#e67e22'] 
    barras = ax.barh(nombres_comb, scores, color=colores_barras, edgecolor='black', height=0.6)
    
    for barra in barras:
        ancho = barra.get_width()
        ax.annotate(f' F1: {ancho:.4f}',
                    xy=(ancho, barra.get_y() + barra.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
                    
    ax.set_title("Top 5 Combinaciones de Hiperparámetros en Grid Search", fontsize=12, fontweight='bold')
    ax.set_xlabel("F1-Score Promedio (Validación Cruzada - 5 Folds)", fontsize=10)
    ax.set_xlim(0, 1.15) 
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('punto8_gridsearch_mejores_parametros.png', dpi=300)
    plt.close()
    
    print("'punto8_gridsearch_mejores_parametros.png' creado")
    # =========================================================================
    
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
    
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    posiciones = np.arange(len(metricas_nombres))
    ancho_barra = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    barras_orig = ax.bar(posiciones - ancho_barra/2, metricas_orig, ancho_barra, label=f'{bosque_original["Modelo"]} Original', color='#34495e')
    barras_tune = ax.bar(posiciones + ancho_barra/2, metricas_tune, ancho_barra, label="Random Forest Optimizado", color='#e67e22')
    
    ax.set_title(f'Comparativa de Rendimiento: Random Forest Antes vs Despues', fontsize=14, fontweight='bold')
    ax.set_ylabel('Puntaje (Score)', fontsize=12)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(metricas_nombres, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for barras in [barras_orig, barras_tune]:
        for barra in barras:
            alto = barra.get_height()
            ax.annotate(f'{alto:.3f}',
                        xy=(barra.get_x() + barra.get_width() / 2, alto),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    nombre_archivo = f"comparativa_optimizacion_{'Random Forest'.replace(' ', '_').lower()}.png"
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()
    
    print(f"¡Gráfico guardado como '{nombre_archivo}'!")

def variables_importantes(modelo_bosque, opcion):
    importancias = modelo_bosque.feature_importances_
    df_imp = pd.DataFrame({
        "Variable": x.columns,
        "Importancia": importancias
    }).sort_values("Importancia", ascending=False).head(10)

    print("\nPunto 9: Random Forest")
    print(df_imp.to_string(index=False))


    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["Variable"][::-1], df_imp["Importancia"][::-1], color="green")
    plt.title(f"Punto 9 – Importancia de Variables (Random Forest) {opcion}")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(f"punto9_importancia_variables_{opcion}.png", dpi=150)
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
    return arbol
    
def graficar_importancia_atributos_arbol(modelo_arbol, nombre_columnas):
    print("\nGANANCIA DE INFORMACIÓN (FEATURE IMPORTANCE)")
    
    importancias = modelo_arbol.feature_importances_
    
    df_importancia = pd.DataFrame({
        'Atributo': nombre_columnas,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=True) 
    
    df_top = df_importancia.tail(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_top['Atributo'], df_top['Importancia'], color='#2980b9', edgecolor='black', alpha=0.8)
    
    ax.set_title("Top 10 Atributo de Mayor importancia (Reduccion impureza Gini)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Importancia Relativa ", fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('confirmacion_ganancia_informacion.png', dpi=300)
    plt.close()
    print(" ¡Gráfico 'confirmacion_ganancia_informacion.png'")
    
if __name__ == "__main__":
    # PTO 1 y 2
    graficos_iniciales()

    # CAMINO 1: EVALUACIÓN NORMAL

    df_normal = armar_df()
    estadisticos_descriptivos(df_normal)
    boxplots(df_normal) #
    x_train_norm, x_test_norm, y_train_norm, y_test_norm = split(df_normal)
    
    modelos_base_norm = modelos() 
    modelonormal_entrenados = entrenamiento(modelos_base_norm, x_train_norm, y_train_norm)
    
    df_resultadosnormal, resultadosnormal = tests(modelonormal_entrenados, x_test_norm, y_test_norm, "Normal")
    grafica_resultados(resultadosnormal, "Normal")
    

    # CAMINO 2: EVALUACIÓN ESTANDARIZADA ( KNN y Red Neuronal)
    #df_estandarizado = armardf_estandarizado()
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train_norm)
    x_train_std = pd.DataFrame(x_train_std, columns=x_train_norm.columns)
    x_test_std = scaler.transform(x_test_norm)
    x_test_std = pd.DataFrame(x_test_std, columns=x_test_norm.columns)
    y_train_std = copy.deepcopy(y_train_norm)
    y_test_std = copy.deepcopy(y_test_norm)
    
    modelos_base_std = modelos() 
    modeloestandarizado_entrenados = entrenamiento(modelos_base_std, x_train_std, y_train_std)
    
    df_resultadosestandarizado, resultadosestandarizado = tests(modeloestandarizado_entrenados, x_test_std, y_test_std, "Estandarizado")
    grafica_resultados(resultadosestandarizado, "Estandarizado")
    

    # PUNTOS (8, 9 y 10) 
    # PTO 8: Optimización
    mejor_bosque_tuneado = optimizar_random_forest(x_train_norm, y_train_norm)
    print(f"\nParámetros del modelo BASE (sin optimizar): {modelonormal_entrenados['Random Forest'].get_params()}")
    print(f"Parámetros del modelo OPTIMIZADO: {mejor_bosque_tuneado.get_params()}")

    preds_base = modelonormal_entrenados["Random Forest"].predict(x_test_norm)
    predicciones_tuneadas = mejor_bosque_tuneado.predict(x_test_norm)
    print(f"¿Predicciones idénticas?: {(preds_base == predicciones_tuneadas).all()}")
    print(f"Cantidad de predicciones distintas: {(preds_base != predicciones_tuneadas).sum()} de {len(preds_base)}")
    
    # PTO 9: Variables importantes
    variables_importantes(mejor_bosque_tuneado, "Optimizado")
    variables_importantes(modelonormal_entrenados["Random Forest"], "No Optimizado")
    # Comparativa de bosques (buscando el original estandarizado para contrastar con el optimizado)
    for i in resultadosnormal:
        if i["Modelo"] == "Random Forest":
            comparativa_bosques(i, predicciones_tuneadas, y_test_norm)
            
    print("\nReporte de Clasificación del Modelo Optimizado:")
    print(classification_report(y_test_norm, predicciones_tuneadas))
    
    # PTO 10: Árbol reducido (Se puede hacer con los datos norm o std, preferible norm para reglas legibles)
    arbol_reducido = arbol_reducido(x_train_norm, y_train_norm, x_test_norm, y_test_norm)
    graficar_importancia_atributos_arbol(arbol_reducido, x_train_norm.columns)
    