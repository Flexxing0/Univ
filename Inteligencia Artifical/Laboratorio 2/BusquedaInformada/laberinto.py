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
        
        directions = [(-1,0),(1,0),(0,-1),(0,1)]  # arriba, abajo, izq, der
        result = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < FILAS and 0 <= nc < COLUMNAS and maze[nr][nc] != '#':
                result.append((nr, nc))
        return result
    def print_path(maze, path, name):

        grid = [row[:] for row in maze]  # copia para no modificar el original
        for (r, c) in path:
            if grid[r][c] not in ('S', 'G'):
                grid[r][c] = '*'
        print(f"\n=== {name} ===")
        for row in grid:
            print(' '.join(row))

    def print_path(maze, path, name):

        grid = [row[:] for row in maze]  # copia para no modificar el original
        for (r, c) in path:
            if grid[r][c] not in ('S', 'G'):
                grid[r][c] = '*'
        print(f"\n=== {name} ===")
        for row in grid:
            print(' '.join(row))
    # ── Heurísticas ──────────────────────────────────────────────────────────────

    def h_manhattan(node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def h_euclidiana(node, goal):
    
        return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    # ── Greedy Best-First Search ──────────────────────────────────────────────────

    def greedy(maze, h):

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

        print("\n=== PUNTO 4: Nodos expandidos ===")
        print(f"  Greedy Manhattan : {g_m_exp:3} nodos | longitud: {len(g_m)}")
        print(f"  Greedy Euclidiana: {g_e_exp:3} nodos | longitud: {len(g_e)}")
        print(f"  A* Manhattan     : {a_m_exp:3} nodos | longitud: {len(a_m)} | costo: {a_m_c}")
        print(f"  A* Euclidiana    : {a_e_exp:3} nodos | longitud: {len(a_e)} | costo: {a_e_c}")

        print("\n=== PUNTO 5: Complejidad ===")
        print("  Greedy — Tiempo: O(b^m) | Espacio: O(b^m)")
        print("           No óptimo, no completo (puede ciclarse sin visited).")
        print("  A*     — Tiempo: O(b^d) | Espacio: O(b^d)")
        print("           Óptimo y completo si h es admisible.")