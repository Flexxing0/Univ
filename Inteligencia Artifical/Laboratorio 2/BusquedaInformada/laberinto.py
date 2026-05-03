from collections import deque
import heapq
import math

maze = [
    ['L','L','L','L','L','L','L','L','L','L','L','L','L'],
    ['S','L','L','L','L','L','L','#','L','L','L','L','L'],
    ['L','L','L','L','L','L','L','L','L','L','L','L','L'],
    ['L','#','#','#','#','#','L','#','L','#','#','#','L'],
    ['L','L','L','L','L','L','L','L','L','L','L','L','L'],
    ['L','L','L','L','L','#','L','#','L','#','L','L','L'],
    ['L','L','L','L','L','L','L','L','L','L','L','L','L'],
    ['#','#','L','#','L','#','L','#','L','#','L','#','#'],
    ['L','L','L','L','L','L','L','L','L','L','L','L','L'],
    ['L','L','L','#','L','L','L','#','L','L','L','#','G'],
    ['L','L','L','L','L','L','L','L','L','L','L','L','L'],
]
FILAS = len(maze)
COLUMNAS = len(maze[0])
class Laberinto:
    def __get_inicio_fin(maze):
        start = goal = None
        for r in range(FILAS):
            for c in range(COLUMNAS):
                if maze[r][c] == 'S': start = (r, c)
                if maze[r][c] == 'G': goal = (r, c)
        return start, goal

    def __get_vecinos(maze, r, c):
        """Devuelve celdas adyacentes válidas (sin obstáculos, dentro del grid)."""
        directions = [(-1,0),(1,0),(0,-1),(0,1)]  # arriba, abajo, izq, der
        result = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < FILAS and 0 <= nc < COLUMNAS and maze[nr][nc] != '#':
                result.append((nr, nc))
        return result
    def print_path(maze, path, name):
        """
        Imprime el laberinto con el camino marcado con '*'.
        El inicio y fin conservan sus letras (S, G).
        Permite visualizar visualmente el recorrido de cada algoritmo.
        """
        grid = [row[:] for row in maze]  # copia para no modificar el original
        for (r, c) in path:
            if grid[r][c] not in ('S', 'G'):
                grid[r][c] = '*'
        print(f"\n=== {name} ===")
        for row in grid:
            print(' '.join(row))

    def print_path(maze, path, name):
        """
        Imprime el laberinto con el camino marcado con '*'.
        El inicio y fin conservan sus letras (S, G).
        Permite visualizar visualmente el recorrido de cada algoritmo.
        """
        grid = [row[:] for row in maze]  # copia para no modificar el original
        for (r, c) in path:
            if grid[r][c] not in ('S', 'G'):
                grid[r][c] = '*'
        print(f"\n=== {name} ===")
        for row in grid:
            print(' '.join(row))
    # ── Heurísticas ──────────────────────────────────────────────────────────────

    def h_manhattan(node, goal):
        """
        Distancia Manhattan: suma de diferencias absolutas en filas y columnas.
        |fila_actual - fila_goal| + |col_actual - col_goal|

        Es admisible en una grilla con movimientos en 4 direcciones porque
        nunca sobreestima — el camino real siempre requiere al menos esa
        cantidad de pasos. No cuenta obstáculos, así que puede ser optimista.
        """
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def h_euclidiana(node, goal):
        """
        Distancia Euclidiana: distancia en línea recta al objetivo.
        sqrt((fila_actual - fila_goal)^2 + (col_actual - col_goal)^2)

        También es admisible: la línea recta siempre es menor o igual
        al camino real. Además, h_manhattan >= h_euclidiana siempre,
        por lo que Manhattan DOMINA a Euclidiana (es mejor heurística).
        """
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    # ── Greedy Best-First Search ──────────────────────────────────────────────────

    def greedy(maze, h):
        """
        Búsqueda Primero el Mejor Voraz (Greedy Best-First Search).

        Expande siempre el nodo con menor h(n) — la estimación al objetivo.
        NO considera el costo g(n) del camino ya recorrido.
        Esto lo hace rápido pero NO garantiza el camino óptimo.

        heap: (h(n), nodo, camino)
        heapq ordena por el primer elemento → siempre expande el de menor h.
        """
        start, goal = Laberinto.__get_inicio_fin(maze)
        heap = [(h(start, goal), start, [start])]
        visited = set()
        expanded = 0

        while heap:
            _, (r, c), path = heapq.heappop(heap)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            expanded += 1
            if (r, c) == goal:
                return path, expanded
            for nb in Laberinto.__get_vecinos(maze, r, c):
                if nb not in visited:
                    heapq.heappush(heap, (h(nb, goal), nb, path + [nb]))
        return None, expanded

    # ── A* ────────────────────────────────────────────────────────────────────────

    def a_star(maze, h):
        """
        Búsqueda A* (A-estrella).

        Expande el nodo con menor f(n) = g(n) + h(n), donde:
        g(n) = costo real acumulado desde el inicio hasta n
        h(n) = estimación heurística de n al objetivo
        f(n) = estimación del costo total del camino a través de n

        A diferencia de Greedy, sí considera el costo ya pagado (g).
        Esto garantiza el camino óptimo si la heurística es ADMISIBLE
        (nunca sobreestima el costo real).

        heap: (f(n), g(n), nodo, camino)
        Se incluye g en la tupla porque si dos nodos tienen igual f,
        heapq desempata por el segundo elemento (g).
        """
        start, goal = Laberinto.__get_inicio_fin(maze)
        heap = [(h(start, goal), 0, start, [start])]
        visited = {}
        expanded = 0

        while heap:
            f, g, (r, c), path = heapq.heappop(heap)
            if (r, c) in visited:
                continue
            visited[(r, c)] = g
            expanded += 1
            if (r, c) == goal:
                return path, g, expanded
            for nb in Laberinto.__get_vecinos(maze, r, c):
                if nb not in visited:
                    g_nuevo = g + 1          # cada paso cuesta 1
                    f_nuevo = g_nuevo + h(nb, goal)
                    heapq.heappush(heap, (f_nuevo, g_nuevo, nb, path + [nb]))
        return None, float('inf'), expanded

    # ── comparar_informada ────────────────────────────────────────────────────────

    def comparar_informada(self, maze):
        g_m, g_m_exp         = Laberinto.greedy(maze, Laberinto.h_manhattan)
        g_e, g_e_exp         = Laberinto.greedy(maze, Laberinto.h_euclidiana)
        a_m, a_m_c, a_m_exp  = Laberinto.a_star(maze, Laberinto.h_manhattan)
        a_e, a_e_c, a_e_exp  = Laberinto.a_star(maze, Laberinto.h_euclidiana)

        Laberinto.print_path(maze, g_m, "Greedy Manhattan")
        Laberinto.print_path(maze, g_e, "Greedy Euclidiana")
        Laberinto.print_path(maze, a_m, "A* Manhattan")
        Laberinto.print_path(maze, a_e, "A* Euclidiana")

        print("\n=== Nodos expandidos y longitud de camino ===")
        print(f"  Greedy Manhattan : {g_m_exp:3} nodos | longitud: {len(g_m)}")
        print(f"  Greedy Euclidiana: {g_e_exp:3} nodos | longitud: {len(g_e)}")
        print(f"  A* Manhattan     : {a_m_exp:3} nodos | longitud: {len(a_m)} | costo: {a_m_c}")
        print(f"  A* Euclidiana    : {a_e_exp:3} nodos | longitud: {len(a_e)} | costo: {a_e_c}")

        print("\n=== Complejidad ===")
        print("  Greedy — Tiempo: O(b^m) | Espacio: O(b^m)")
        print("           No óptimo, no completo (puede ciclarse sin visited).")
        print("  A*     — Tiempo: O(b^d) | Espacio: O(b^d)")
        print("           Óptimo y completo si h es admisible.")