from piso import piso
class aspiradora:
    __localizacion = 0
    def limpia(self, piso: piso):
        for i in range(0,20):
            if piso.estaSucio(self.__localizacion) == 1:
                piso.set0(self.__localizacion)
            else:
                self.__localizacion += self.__localizacion ^ 1
            piso.muestraPiso()
            self.muestraAspiradora()
    
    def muestraAspiradora(self):
        print("Aspiradora:" + (" ^" if self.__localizacion == 1 else "^ "))