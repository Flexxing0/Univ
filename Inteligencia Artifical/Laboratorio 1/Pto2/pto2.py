import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 

dato1 = pd.read_csv('../datos_ventas_suc1.csv', encoding='latin1')
dato2 = pd.read_csv('../datos_ventas_suc2.csv', encoding='latin1')
datoframe = pd.DataFrame(dato1)
datoframe2 = pd.DataFrame(dato2)
#Se combina ambos dataframe anteriores y imprime la tabla resultante
frames = [datoframe,datoframe2]
concatFrame = pd.concat(frames, ignore_index=True, sort=False)#, keys=['suc1','suc2'])
ventasProducto = concatFrame.groupby('Producto')['Cantidad'].sum()
fig, ax = plt.subplots()
# ax.pie(ventasProducto, labels=ventasProducto.index, autopct='%1.1f%%', startangle=90)
# ax.set_title('Ventas por Producto')
frameFormatoNuevo = concatFrame.copy()
frameFormatoNuevo['Fecha'] = pd.to_datetime(concatFrame['Fecha'])
meses = frameFormatoNuevo['Fecha'].dt.month
nombres_meses={1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}
agrupadoFechas = frameFormatoNuevo.groupby(meses).agg(Ventas=pd.NamedAgg(column="Total Venta",aggfunc="sum"))
agrupadoFechas = agrupadoFechas.rename(index=nombres_meses)
ax.plot(agrupadoFechas,marker ='o')
ax.set_title('Ventas por mes')
ax.set_xlabel('Ventas')
ax.set_ylabel('Fecha')
ax.grid(True)
plt.show()