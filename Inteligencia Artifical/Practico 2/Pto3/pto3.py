import random
def crealista():
    return [random.randint(0,10) for i in range(20)]

def convierteConjunto(lista):
    return set(lista)

lista1 = crealista()
print(f"Lista original: {lista1}")
lista2 = convierteConjunto(lista1)
print(f"Lista conjunto: {lista2}")

print(lista1-lista2)
