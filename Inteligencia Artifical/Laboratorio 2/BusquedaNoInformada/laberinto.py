from collections import deque
import heapq

estados = {
    'S': 'Inicial',
    '#': 'Obscatulo',
    'L': 'Libre',
    'G': 'Final'
}
# El laberinto del lab
maze = [
    ['L', 'L', 'L', 'L', 'L','L','L','L','L'],
    ['S', '#', 'L', 'L', 'L','L','L','#','G'],
    ['L', 'L', 'L', 'L', 'L','L','L','L','L'],
    ['L', '#', 'L', '#', '#','#','L','#','L'],
    ['L', 'L', 'L', 'L', 'L','L','L','L','L'],
    ['L', '#', 'L', '#', 'L','L','L','#','L'],
    ['L', 'L', 'L', 'L', 'L','L','L','L','L'],
    ['L', 'L', 'L', '#', 'L','#','#','#','L'],
    ['L', 'L', 'L', 'L', 'L','L','L','L','L'],
    ['#', '#', 'L', '#', 'L','L','L','L','L'],
    ['L', 'L', 'L', 'L', 'L','L','L','L','L'],
    ['L', 'L', 'L', 'L', 'L','#','#','#','#'],
    ['L', 'L', 'L', 'L', 'L','L','L','L','L'],
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

    # --- BFS ---
    def bfs(maze):
        start, goal = Laberinto.__get_inicio_fin(maze)
        # deque: append y popleft son O(1), ideal para colas FIFO
        queue = deque()
        queue.append((start, [start]))   # (posición actual, camino hasta acá)
        visited = set([start])
        expanded = 0

        while queue:
            (r, c), path = queue.popleft()
            expanded += 1
            if (r, c) == goal:
                return path, expanded
            for neighbor in Laberinto.__get_vecinos(maze, r, c):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, expanded

    # --- DFS ---
    def dfs(maze):
        start, goal = Laberinto.__get_inicio_fin(maze)
        stack = [(start, [start])]
        visited = set()          # ← vacío, no marcamos start todavía
        expanded = 0

        while stack:
            (r, c), path = stack.pop()
            if (r, c) in visited:    # ← chequeamos al SACAR
                continue
            visited.add((r, c))      # ← marcamos al SACAR
            expanded += 1
            if (r, c) == goal:
                return path, expanded
            for neighbor in Laberinto.__get_vecinos(maze, r, c):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None, expanded

    # --- UCS ---
    def ucs(maze, cost_map=None):
        """
        cost_map: dict {(r,c): costo} para celdas con costo variable.
        Si es None, todas las celdas tienen costo 1 (equivale a BFS en resultado).
        """
        start, goal = Laberinto.__get_inicio_fin(maze)
        # heapq: min-heap, pop() devuelve el elemento de menor costo → O(log n)
        # Cada elemento: (costo_acumulado, posición, camino)
        heap = [(0, start, [start])]
        visited = {}   # nodo → menor costo con que fue visitado
        expanded = 0

        while heap:
            cost, (r, c), path = heapq.heappop(heap)
            if (r, c) in visited:
                continue   # ya fue procesado con menor costo
            visited[(r, c)] = cost
            expanded += 1
            if (r, c) == goal:
                return path, cost, expanded
            for neighbor in Laberinto.__get_vecinos(maze, r, c):
                if neighbor not in visited:
                    step_cost = cost_map[neighbor] if cost_map else 1
                    heapq.heappush(heap, (cost + step_cost, neighbor, path + [neighbor]))
        return None, float('inf'), expanded

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

    def comparar(self, maze):
        bfs_path, bfs_exp            = Laberinto.bfs(maze)
        dfs_path, dfs_exp            = Laberinto.dfs(maze)
        ucs_path, ucs_cost, ucs_exp  = Laberinto.ucs(maze)

        Laberinto.print_path(maze, bfs_path, "BFS")
        Laberinto.print_path(maze, dfs_path, "DFS")
        Laberinto.print_path(maze, ucs_path, "UCS")

        # ── PUNTO 4: Comparación de nodos expandidos ──────────────────────────
        print("\n=== PUNTO 4: Nodos expandidos ===")
        print(f"  BFS: {bfs_exp} nodos  | longitud camino: {len(bfs_path)}")
        print(f"  DFS: {dfs_exp} nodos  | longitud camino: {len(dfs_path)}")
        print(f"  UCS: {ucs_exp} nodos  | longitud camino: {len(ucs_path)} | costo total: {ucs_cost}")

        # ── PUNTO 5: Complejidad temporal y espacial ───────────────────────────
        # b = factor de ramificación = máximo de vecinos posibles = 4 (grilla)
        # d = profundidad de la solución = longitud del camino encontrado
        # m = profundidad máxima del árbol = total de celdas libres
        celdas_libres = sum(1 for fila in maze for c in fila if c != '#')
        b = 4
        d_bfs = len(bfs_path)
        d_dfs = len(dfs_path)
        d_ucs = len(ucs_path)

        print("\n=== PUNTO 5: Complejidad ===")
        print(f"  Celdas libres (m): {celdas_libres} | Factor ramificación (b): {b}")
        print(f"")
        print(f"  BFS  — Tiempo: O(b^d) = O({b}^{d_bfs})  | Espacio: O(b^d) = O({b}^{d_bfs})")
        print(f"         Guarda TODOS los nodos del nivel actual en memoria.")
        print(f"")
        print(f"  DFS  — Tiempo: O(b^m) = O({b}^{celdas_libres}) | Espacio: O(b*m) = O({b}*{celdas_libres}={b*celdas_libres})")
        print(f"         Solo guarda el camino actual + ramas pendientes.")
        print(f"")
        print(f"  UCS  — Tiempo: O(b^d) = O({b}^{d_ucs})  | Espacio: O(b^d) = O({b}^{d_ucs})")
        print(f"         Similar a BFS pero con cola de prioridad (heap).")
        print(f"         Cada operación del heap cuesta O(log n) extra.")
