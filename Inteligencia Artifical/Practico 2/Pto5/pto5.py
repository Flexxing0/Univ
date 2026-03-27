lista = ['lunes','martes','miercoles','jueves','viernes']
print(f"\nDias de Lunes a Viernes: {lista}")
lista.append('sabado')
lista.append('domingo')
print(f"\nSe agregaron Sabado y Domingo{lista}")
dia = input('Ingrese un dia de semana: ')
if dia.lower() in lista:
    print(f"\n{dia} si es un dia")
else:
    print(f"\n{dia} no es un dia")