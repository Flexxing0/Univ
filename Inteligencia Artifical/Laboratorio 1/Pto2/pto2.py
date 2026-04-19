import matplotlib.pyplot as plt 
import pandas as pd

concatFrame = pd.read_csv('../Pto1/datos_ventas_combinados.csv')
ventasProducto = concatFrame.groupby('Producto')['Cantidad'].sum()
fig, ax = plt.subplots()
ax.pie(ventasProducto, labels=ventasProducto.index, autopct='%1.1f%%', startangle=90)
ax.set_title('Ventas por Producto')
frameFormatoNuevo = concatFrame.copy()
frameFormatoNuevo['Fecha'] = pd.to_datetime(concatFrame['Fecha'])
meses = frameFormatoNuevo['Fecha'].dt.month
nombres_meses={1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}
agrupadoFechas = frameFormatoNuevo.groupby(meses).agg(Ventas=pd.NamedAgg(column="Total Venta",aggfunc="sum"))
agrupadoFechas = agrupadoFechas.rename(index=nombres_meses)
fig2, ax2 = plt.subplots()
ax2.plot(agrupadoFechas,marker ='o')
ax2.set_title('Ventas por mes')
ax2.grid(True)
plt.show()