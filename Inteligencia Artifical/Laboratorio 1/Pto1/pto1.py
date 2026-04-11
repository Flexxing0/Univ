import numpy as py 
import pandas as pd
dato1 = pd.read_csv('../datos_ventas_suc1.csv', encoding='latin1')
dato2 = pd.read_csv('../datos_ventas_suc2.csv', encoding='latin1')
datoframe = pd.DataFrame(dato1)
datoframe2 = pd.DataFrame(dato2)
#Se imprimen ambos dataframe
print(datoframe)
print(datoframe2)
#Se combina ambos dataframe anteriores y imprime la tabla resultante
frames = [datoframe,datoframe2]
concatFrame = pd.concat(frames, ignore_index=True, sort=False)#, keys=['suc1','suc2'])
print(concatFrame)
#Tratamiento de datos, cambio de formato de fecha
frameFormatoNuevo = concatFrame.copy()
frameFormatoNuevo['Fecha'] = pd.to_datetime(concatFrame['Fecha'])
print(frameFormatoNuevo)
#Para producto mas vendido
productos = concatFrame.groupby("Producto")
productoMasVendido = productos.agg(Cantidad=pd.NamedAgg(column="Cantidad",aggfunc="sum")).sort_values("Cantidad",ascending=False)
print(productoMasVendido)
#Para ventas por mes
meses = frameFormatoNuevo['Fecha'].dt.month
nombres_meses={1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}
agrupadoFechas = frameFormatoNuevo.groupby(meses).agg(Ventas=pd.NamedAgg(column="Total Venta",aggfunc="sum")).sort_values("Ventas",ascending=False)
agrupadoFechas = agrupadoFechas.rename(index=nombres_meses)
print(agrupadoFechas)
#Por categoria de producto
catProductos = productos.agg(Ventas=pd.NamedAgg(column="Total Venta",aggfunc="sum")).sort_values("Ventas",ascending=False)
print(catProductos)
#Se guarda el dataframe resultante
concatFrame.to_csv('datos_ventas_combinados.csv',index=False)


