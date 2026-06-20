import math
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

df = pd.DataFrame(datos, columns=["EJ", "Estado", "Temperatura", "Humedad", "Viento", "JuegoTenis"])

print(df)