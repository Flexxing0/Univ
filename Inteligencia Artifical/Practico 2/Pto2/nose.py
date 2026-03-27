edad = int(input("ingrese edad: "))
if edad>21:
    print("ya eres mayor de edad")
else:
    print("aun eres menor de edad")
print("No puedes salir al boliche")
#El codigo del a no es el mismo que el b porque print esta fuera del bloque condicional else if, no importa el numero que se ingrese en b, "no puedes salir al boliche" siempre se va a ejecutar dado que es la ultima linea de codigo
