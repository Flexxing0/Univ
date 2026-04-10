import numpy as py 
import pandas as pd
dato1 = pd.read_csv('../datos_ventas_suc1.csv', encoding='latin1')
dato2 = pd.read_csv('../datos_ventas_suc2.csv', encoding='latin1')
datoframe = pd.DataFrame(dato1)
datoframe2 = pd.DataFrame(dato2)
#Se imprimen ambos dataframe
# print(datoframe)
# print(datoframe2)
#Se combina ambos dataframe anteriores y imprime la tabla resultante
frames = [datoframe,datoframe2]
concatFrame = pd.concat(frames, ignore_index=True, sort=False)#, keys=['suc1','suc2'])
#print(concatFrame)
#Tratamiento de datos, cambio de formato de fecha
nuevoFormato = pd.to_datetime(concatFrame['Fecha'])
##print(nuevoFormato)
#productoMasVendido = concatFrame.groupby("Producto").agg(Cantidad=pd.NamedAgg(column="Cantidad",aggfunc="sum")).sort_values("Cantidad",ascending=False)
meses = concatFrame['Fecha'].dt.month
agrupadoFechas = concatFrame.groupby('Fecha')
print(agrupadoFechas.get_group(meses))

