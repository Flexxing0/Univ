from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree #arbol de decision y ploteo
import copy

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

df.drop("EJ",axis=1,inplace=True)
df_transformado = copy.deepcopy(df)
columnas = df.columns.values.tolist()
prediccion = columnas[:4]
x = copy.deepcopy(df[prediccion])
objetivo = columnas[4]
y = copy.deepcopy(df[objetivo])
print(prediccion,objetivo)

print(df)
encoders = {}
for columna in df.columns:
    le = LabelEncoder()
    df_transformado[columna] = le.fit_transform(df[columna])
    encoders[columna] = le 

X = df_transformado[['Estado', 'Temperatura', 'Humedad', 'Viento']]
y = df_transformado['JuegoTenis']

arbol = DecisionTreeClassifier(criterion="entropy", min_samples_split=3, random_state=40)
arbol.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(
    arbol, 
    feature_names=list(X.columns), 
    class_names=list(encoders['JuegoTenis'].classes_), 
    filled=True, 
    rounded=True,
    fontsize=10
)

plt.savefig("arbol.png", dpi=150)
plt.close()