import random
class piso:
    __cuadros = [0, 1] #estado inicial
    def ensuciar(self):
        for i in range(len(self.__cuadros)):
            if self.__cuadros[i] == 0:
                self.__cuadros[i] = 1 if random.randint(0,10) > 5 else 0

    def set0(self,i):
        self.__cuadros[i] = 0
        
    def estaSucio(self,i: int):
        return self.__cuadros[i]
    
    def muestraPiso(self):
        print(f"Piso: {self.__cuadros}")
    