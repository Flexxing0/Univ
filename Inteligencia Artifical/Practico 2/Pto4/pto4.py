import random
num = random.randint(-20,20)
print(num)
print("Es negativo" if num<0 else "Es positivo" if num>0 else "Es Nulo")