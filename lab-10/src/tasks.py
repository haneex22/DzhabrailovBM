from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


from graph_representation import GraphAdjacencyList
from graph_traversal import bfs, reconstruct_path


@dataclass(frozen=True)
class Maze:
    """Лабиринт в виде прямоугольной сетки.

    grid[r][c] == 0 -> свободно
    grid[r][c] == 1 -> стена
    """

    grid: List[List[int]]

    def rows(self) -> int:
        return len(self.grid)

    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0


def shortest_path_in_maze(
    maze: Maze,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Кратчайший путь в лабиринте (BFS по клеткам).

    Сложность:
        O(R*C) по вершинам сетки и рёбрам между соседними клетками.
    """
    r_count = maze.rows()
    c_count = maze.cols()
    if r_count == 0 or c_count == 0:
        return []

    def inside(r: int, c: int) -> bool:
        return 0 <= r < r_count and 0 <= c < c_count

    sr, sc = start
    gr, gc = goal
    if not inside(sr, sc) or not inside(gr, gc):
        return []
    if maze.grid[sr][sc] == 1 or maze.grid[gr][gc] == 1:
        return []

    # Нумеруем клетки в вершины графа.
    def vid(r: int, c: int) -> int:
        return r * c_count + c

    def cell(v: int) -> Tuple[int, int]:
        return divmod(v, c_count)

    n = r_count * c_count
    g = GraphAdjacencyList(n, directed=False)

    # Строим граф смежности по 4 направлениям.
    for r in range(r_count):
        for c in range(c_count):
            if maze.grid[r][c] == 1:
                continue
            u = vid(r, c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if inside(nr, nc) and maze.grid[nr][nc] == 0:
                    v = vid(nr, nc)
                    g.add_edge(u, v, 1.0)

    res = bfs(g, vid(sr, sc))
    path_vertices = reconstruct_path(res.parent, vid(sr, sc), vid(gr, gc))
    return [cell(v) for v in path_vertices]


def is_network_connected(num_vertices: int, edges: List[Tuple[int,
                                                              int]]) -> bool:
    """Задача: определить связность сети (неориентированный граф).

    Сложность:
        Построение O(V + E), BFS O(V + E).
    """
    if num_vertices <= 0:
        return False

    g = GraphAdjacencyList(num_vertices, directed=False)
    for u, v in edges:
        g.add_edge(u, v, 1.0)

    res = bfs(g, 0)
    return all(d is not None for d in res.distances)


def find_dependencies_order(
    num_vertices: int,
    directed_edges: List[Tuple[int, int]],
) -> List[int]:
    """Задача: порядок выполнения задач по зависимостям (топосорт).

    Вершины 0..num_vertices-1, ребро u->v означает: u должен быть раньше v.

    Сложность:
        O(V + E).
    """
    from shortest_path import topo_sort_kahn  # локальный импорт

    g = GraphAdjacencyList(num_vertices, directed=True)
    for u, v in directed_edges:
        g.add_edge(u, v, 1.0)
    return topo_sort_kahn(g)


def format_edges(edges: List[Tuple[int, int]]) -> str:
    """Печатает список рёбер в удобном виде."""
    if not edges:
        return "(нет рёбер)"
    return ", ".join(f"{u}—{v}" for u, v in edges)


def format_toposort(order: List[int]) -> str:
    """Печатает порядок задач с русскими пояснениями."""
    return " → ".join(str(v) for v in order) if order else "(порядок пуст)"
