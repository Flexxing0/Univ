import numpy as py 
import pandas as pd
dato = pd.read_csv('../datos_ventas_suc1.csv', encoding='latin1')
datoframe = pd.DataFrame(dato)
print(datoframe)