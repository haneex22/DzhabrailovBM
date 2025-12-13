from __future__ import annotations

from graph_representation import GraphAdjacencyList, GraphAdjacencyMatrix
from graph_traversal import (
    bfs,
    connected_components,
    dfs_iterative,
    dfs_recursive,
    reconstruct_path,
)
from shortest_path import dijkstra, topo_sort_kahn
from tasks import (
    Maze,
    find_dependencies_order,
    format_edges,
    format_toposort,
    is_network_connected,
    shortest_path_in_maze,
)


PC_INFO = """\
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def demo_representations() -> None:
    print_section("1) Представления графа: матрица и список смежности")

    g_list = GraphAdjacencyList(5, directed=False)
    g_mat = GraphAdjacencyMatrix(5, directed=False)

    edges = [(0, 1), (0, 2), (1, 3), (3, 4)]
    for u, v in edges:
        g_list.add_edge(u, v, 1.0)
        g_mat.add_edge(u, v, 1.0)

    print("Граф (неориентированный). Рёбра:", format_edges(edges))
    print()
    print("Оценка памяти (условно):")
    print(f"  Список смежности: V + E(ориент.) = {g_list.memory_units()}")
    print(f"  Матрица смежности: V^2 = {g_mat.memory_cells()}")
    print()
    print("Проверка ребра 1—3:")
    print(f"  В списке смежности: {g_list.has_edge(1, 3)}")
    print(f"  В матрице смежности: {g_mat.has_edge(1, 3)}")


def demo_traversals() -> None:
    print_section("2) Обходы графа: BFS, DFS, компоненты связности")

    g = GraphAdjacencyList(6, directed=False)
    edges = [(0, 1), (1, 2), (0, 3), (4, 5)]
    for u, v in edges:
        g.add_edge(u, v, 1.0)

    print("Граф (неориентированный). Рёбра:", format_edges(edges))
    print()

    bfs_res = bfs(g, 0)
    print("BFS (поиск в ширину) от вершины 0:")
    print("  Расстояния (по числу рёбер):")
    for v, d in enumerate(bfs_res.distances):
        print(f"    до вершины {v}: {d}")

    path_0_2 = reconstruct_path(bfs_res.parent, 0, 2)
    print("\n  Восстановленный кратчайший путь 0 → 2:", path_0_2)

    print("\nDFS (поиск в глубину):")
    print("  Рекурсивный порядок обхода от 0:", dfs_recursive(g, 0))
    print("  Итеративный порядок обхода от 0:", dfs_iterative(g, 0))

    comps = connected_components(g)
    print("\nКомпоненты связности:")
    for i, comp in enumerate(comps, start=1):
        print(f"  Компонента {i}: {comp}")


def demo_dijkstra_and_toposort() -> None:
    print_section("3) Алгоритмы: Дейкстра (кратчайший путь по вес) и топосорт")

    g = GraphAdjacencyList(5, directed=True)
    edges = [
        (0, 1, 2.0),
        (0, 2, 4.0),
        (1, 2, 1.0),
        (1, 3, 7.0),
        (2, 4, 3.0),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    print("Взвешенный ориентированный граф. Рёбра вида u → v (вес):")
    for u, v, w in edges:
        print(f"  {u} → {v} (вес = {w})")

    res = dijkstra(g, 0)
    print("\nАлгоритм Дейкстры от вершины 0 (минимальная стоимость пути):")
    for v, d in enumerate(res.dist):
        print(f"  до вершины {v}: {d}")

    print("\nПример DAG для топологической сортировки:")
    dag = GraphAdjacencyList(6, directed=True)
    dag_edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
    for u, v in dag_edges:
        dag.add_edge(u, v, 1.0)

    print("Рёбра зависимостей u → v (u должен быть раньше v):")
    for u, v in dag_edges:
        print(f"  {u} → {v}")

    order = topo_sort_kahn(dag)
    print("\nТопологический порядок (один из возможных):")
    print(" ", format_toposort(order))


def demo_tasks() -> None:
    print_section("Практические задачи: лабиринт/связность сети/зависимости")

    # --- Задача 1: Лабиринт ---
    print("\n№ 1: найти кратчайший путь в лабиринте (BFS по клеткам)")
    maze = Maze(
        grid=[
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
        ],
    )
    start = (0, 0)
    goal = (3, 3)

    path = shortest_path_in_maze(maze, start, goal)

    print(f"Старт: {start}, Финиш: {goal}")

    if path:
        print("\nНайденный путь (координаты):")
        print("  " + " → ".join(str(p) for p in path))
        print("\nЛабиринт с отмеченным путём (*):")

        print(f"\nДлина пути (число шагов): {len(path) - 1}")
    else:
        print("\nПуть не найден (старт и финиш могут быть разделены стенами).")

    # --- Задача 2: Связность сети ---
    print("\n" + "-" * 80)
    print("№ 2: проверить, связна ли сеть (компьютеры и кабели)")

    n = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    print(f"Число компьютеров: {n}")
    print("Кабели (соединения):", format_edges(edges))

    connected = is_network_connected(n, edges)
    if connected:
        print("Результат: сеть СВЯЗНА (каждый компьютер достижим из другого).")
    else:
        print("Результат: сеть НЕ связна (есть изолированные части).")

    # --- Задача 3: Зависимости (топосорт) ---
    print("\n" + "-" * 80)
    print("№ 3: порядок выполнения задач при наличии зависимостей (топосорт)")

    tasks_count = 4
    deps = [(0, 1), (1, 2), (0, 3)]
    print(f"Число задач: {tasks_count} (задачи 0..{tasks_count - 1})")
    print("Зависимости u → v (u должно быть выполнено раньше v):")
    for u, v in deps:
        print(f"  {u} → {v}")

    order = find_dependencies_order(tasks_count, deps)
    print("\nКорректный порядок выполнения:")
    print(" ", format_toposort(order))
    print("\nПроверка смысла:")
    print("  - задача 0 стоит раньше задач 1 и 3;")
    print("  - задача 1 стоит раньше задачи 2.")


def main() -> None:
    print(PC_INFO)
    demo_representations()
    demo_traversals()
    demo_dijkstra_and_toposort()
    demo_tasks()


if __name__ == "__main__":
    main()
