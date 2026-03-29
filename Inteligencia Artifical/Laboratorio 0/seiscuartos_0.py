import entornos_o
import copy
from random import choice

estado = {
    'robot': ['PisoSuperior','A'],
    'PisoSuperior': {'A':'sucio','B':'sucio','C':'sucio'},
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

movimientos = {
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
    'nada':2
}

#Comenzamos con el robot en Piso 1, cuadro A
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
        #subir solo en los extremos del piso inferior
        #bajar solo en cuarto del centro del piso superior
    def transicion(self,accion):
        if not self.accion_legal(accion):
            print(f"Accion ilegal, quiere moverse a: {accion}, se cambia a nada")
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
            self.x[0][1] = movimientos[accion][d]
        print(f"Nueva transicion: {self.x[0][1]}")
           
    def percepcion(self):
        for i in range(1,3): 
            print(f"Estado {self.x[0][1]} en {self.x[i]}")
            if self.x[0][1] in self.x[i]:
                situacion = self.x[i][self.x[0][1]]
        print(f"Percepcion actual: {self.x[0],situacion}")
        return self.x[0], situacion
        #medio raro pero busca en piso superior e inferior la situacion de A
        #puedo usar [0][0] para primera tupla y case para ver el piso, y luego hacer [0].find [0][2] por ejemplo, o usar directamente diccionario para cambiar valores. 
    
class AgenteAleatorio(entornos_o.Agente):
    
    def __init__(self,acciones):
        self.acciones = acciones
    def programa(self,percepcion):
        return choice(self.acciones)
    
class AgenteReactivoSeisCuartos(entornos_o.Agente):
    
    def programa(self,percepcion):
        robot, situacion = percepcion
        sigMovimiento = choice(estados[robot[1]])
        
        print(f"Situacion: {situacion}")
        return (
            'limpiar' if situacion == 'sucio' else sigMovimiento
        )
        
class AgenteReactivoModeloSeisCuartos(entornos_o.Agente):
    
    def __init__(self):
        
        self.modelo = {
            'robot': ['PisoSuperior','A'],
            'PisoSuperior': {'A':'sucio','B':'sucio','C':'sucio'},
            'PisoInferior': {'D':'sucio','E':'limpio','F':'sucio'}}
        
    def programa(self,percepcion):
        robot, situacion = percepcion
        keys = ['PisoSuperior','PisoInferior']
        self.modelo['robot'] = robot
        for i in range(1,3):
            if self.modelo[0][1] in self.modelo[keys[i]]:
                self.modelo[keys[i]][1] = situacion
        for cuadro, estCuadro in self.modelo
        
def test():
    entornos_o.simulador(SeisCuartos(),AgenteAleatorio(list(movimientos)),20)

    print("Prueba del entorno con un agente reactivo")
    entornos_o.simulador(SeisCuartos(), AgenteReactivoSeisCuartos(), 20)

    print("Prueba del entorno con un agente reactivo con modelo")
    entornos_o.simulador(SeisCuartos(), AgenteReactivoModeloSeisCuartos(), 20)
    
if __name__ == "__main__":
    test()