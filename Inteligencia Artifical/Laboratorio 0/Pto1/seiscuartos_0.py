import entornos_o
import copy
from random import choice

estado = {
    'robot': ['PisoInferior','E'],
    'PisoSuperior': {'A':'sucio','B':'limpio','C':'sucio'},
    'PisoInferior': {'D':'sucio','E':'limpio','F':'sucio'}
}

estados = {
    'A': ['derecha'],
    'B': ['bajar','izquierda','derecha'],
    'C': ['izquierda'],
    'D': ['subir','derecha'],
    'E': ['derecha','izquierda'],
    'F': ['subir','izquierda']
}

acciones = {
    "derecha": {'A':'B', 'B':'C', 'D':'E', 'E':'F'},
    "izquierda": {'B':'A', 'C':'B', 'E':'D', 'F':'E'},
    "subir": {'D':'A', 'F':'C'},
    "bajar": {'B':'E'},
    "nada": None,
    "limpiar": None,
}

penalizaciones = {
    'subir':3,
    'bajar':3,
    'derecha':2,
    'izquierda':2,
    'limpiar':1,
    'nada':0
}

#Comenzamos con el robot en PisoSuperior, cuadro A, tambien funciona cambiando de piso y cuadro
class SeisCuartos(entornos_o.Entorno):
    def __init__(self,x0=None):
        if x0 is None:
            x0 = [estado['robot'],estado['PisoSuperior'],estado['PisoInferior']]
            self.x = copy.deepcopy(x0)
            self.desempeño = 0
        
    def accion_legal(self,accion):
        print(f"Parado en {self.x[0][1]}")
        match self.x[0][1]:
            case 'A':
                return accion in ("derecha","limpiar","nada")
            case 'B':
                return accion in ("izquierda","bajar","derecha","limpiar","nada")
            case 'C':
                return accion in ("izquierda","limpiar","nada")
            case 'D':
                return accion in ("derecha","subir","limpiar","nada")
            case 'E':
                return accion in ("izquierda","derecha","limpiar","nada")
            case 'F':
                return accion in ("izquierda","subir","limpiar","nada")

    def transicion(self,accion):
        if not self.accion_legal(accion):
            print(f"Accion ilegal, quiere moverse a: {accion}, se cambia a nada")
            self.desempeño -= penalizaciones[accion]  
            accion = 'nada' 
            #raise ValueError("La accion no es legal para este estado")
        
        robot,piso1,piso2 = self.x
        d = robot[1]
        self.desempeño -= penalizaciones[accion]
        if accion == "limpiar":
            for i in range (1,3):
                if self.x[0][1] in self.x[i]:
                    self.x[i][self.x[0][1]] = 'limpio'
        elif accion != "nada":
            
            if accion == "subir":
                self.x[0][0] = 'PisoSuperior'
            elif accion == "bajar":
                self.x[0][0] = 'PisoInferior'
            self.x[0][1] = acciones[accion][d]
        print(f"Nueva transicion: {self.x[0][1]}")
           
    def percepcion(self):
        for i in range(1,3): 
            print(f"Estado {self.x[0][1]} en {self.x[i]}")
            if self.x[0][1] in self.x[i]:
                situacion = self.x[i][self.x[0][1]]
        print(f"Percepcion actual: {self.x[0],situacion}")
        return self.x[0], situacion
    
class AgenteAleatorio(entornos_o.Agente):
    
    def __init__(self,acciones):
        self.acciones = acciones
    def programa(self,percepcion):
        return choice(self.acciones)
    
class AgenteReactivoSeisCuartos(entornos_o.Agente):
    
    def programa(self,percepcion):
        robot, situacion = percepcion
        
        print(f"Situacion: {situacion}")
        return (
            'limpiar' if situacion == 'sucio' else choice(estados[robot[1]])
        )
        
class AgenteReactivoModeloSeisCuartos(entornos_o.Agente):
    
    def __init__(self): 
        self.modelo = {
    'robot': ['PisoSuperior','A'],
    'PisoSuperior': {'A':'sucio','B':'sucio','C':'sucio'},
    'PisoInferior': {'D':'sucio','E':'sucio','F':'sucio'}
}    
    def programa(self,percepcion):
        robot, situacion = percepcion
        llaves = list(self.modelo.keys())
        llaves.remove('robot')
        suciedadPiso = {}
        pisoActual, cuadroActual = robot
        #actualiza estado interno
        self.modelo['robot'] = [pisoActual, cuadroActual]
        self.modelo[pisoActual][cuadroActual] = situacion
        #busca los pisos que estan sucios
        for i in range(len(llaves)):
            suciedadPiso[llaves[i]] = any(v=='sucio' for v in self.modelo[llaves[i]].values())   
        #decide el sigMovimiento
        sigMovimiento = None
        if self.modelo[pisoActual][cuadroActual] == 'sucio':
            sigMovimiento = 'limpiar'
        else:
            llaves.remove(pisoActual)
            valores = list(self.modelo[pisoActual].values())
            indice = list(self.modelo[pisoActual].keys()).index(cuadroActual)
            izquierda = valores[:indice]
            derecha = valores[indice+1:]
            if any(v == 'sucio' for v in izquierda):
                print("Mi izquierda esta sucia")
                sigMovimiento = 'izquierda'
            elif any((v == 'sucio' for v in derecha)):
                print("Mi derecha esta sucia")
                sigMovimiento = 'derecha'
            elif suciedadPiso[llaves[0]]:
                print("Mi piso esta limpio, me fijo el de abajo/arriba")
                if 'subir' in estados[cuadroActual]:
                    sigMovimiento = 'subir'
                    print("Puedo subir, subo")
                elif 'bajar' in estados[cuadroActual]:
                    sigMovimiento = 'bajar'
                    print("Puedo bajar, bajo")
                elif 'derecha' in estados[cuadroActual]:
                    sigMovimiento = 'derecha'
                    print("Piso arriba/abajo 2 sucio, intentare algo")
                elif 'izquierda' in estados[cuadroActual]:
                    sigMovimiento = 'izquierda'
                    print("Piso 1 arriba/abajo sucio, intentare algo")
            else:
                sigMovimiento = 'nada'
        return sigMovimiento
            
        
def test():
    #print("Prueba del entorno con un agente aleatorio")
    #entornos_o.simulador(SeisCuartos(),AgenteAleatorio(list(acciones)),100)

    #print("Prueba del entorno con un agente reactivo")
    #entornos_o.simulador(SeisCuartos(), AgenteReactivoSeisCuartos(), 20)

    print("Prueba del entorno con un agente reactivo con modelo")
    entornos_o.simulador(SeisCuartos(), AgenteReactivoModeloSeisCuartos(), 100)
    
if __name__ == "__main__":
    test()