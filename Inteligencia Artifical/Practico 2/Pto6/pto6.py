listaInicial = ['Base de datos','POO','Laboratorio','Estadistica','Algebra','AyP','AyP2','Analisis Matematico','EPYA']
listaAprobados = []
for i in listaInicial.copy():
    print(f"\nQue nota sacaste en {i}?")
    nota = int(input("Nota: "))
    if nota > 4:
        listaAprobados.append(i)
        listaInicial.remove(i)
        
print(f"Tienes que volver a rendir: {listaInicial}\n")
print(f"Aprobaste: {listaAprobados}")

 