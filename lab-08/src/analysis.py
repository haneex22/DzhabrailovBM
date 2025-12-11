from __future__ import annotations

import random
import string
import timeit
from dataclasses import dataclass
from typing import Callable, List


from greedy_algorithms import (
    Item,
    fractional_knapsack,
)


PC_INFO = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


@dataclass(frozen=True)
class KnapsackInstance:
    """Описание тестового примера для задачи о рюкзаке."""

    capacity: int
    weights: List[int]
    values: List[int]


def knapsack_01_dp(instance: KnapsackInstance) -> int:
    """Точный алгоритм для 0-1 задачи рюкзаке (динамическое программирование).

    Сложность:
        Пусть n — число предметов, W — вместимость.
        Временная:  O(n * W).
        Пространственная: O(W).
    """
    n = len(instance.weights)
    capacity = instance.capacity

    dp = [0] * (capacity + 1)

    for i in range(n):
        weight_i = instance.weights[i]
        value_i = instance.values[i]
        for w in range(capacity, weight_i - 1, -1):
            dp[w] = max(dp[w], dp[w - weight_i] + value_i)

    return dp[capacity]


def generate_random_knapsack_instance(
    n_items: int,
    max_weight: int = 15,
    max_value: int = 30,
) -> KnapsackInstance:
    """Генерирует случайный пример для задачи о рюкзаке.

    Используется для сравнения жадного и точного подходов.
    """
    weights = [random.randint(1, max_weight) for _ in range(n_items)]
    values = [random.randint(1, max_value) for _ in range(n_items)]
    capacity = random.randint(max_weight, max_weight * n_items // 2)
    return KnapsackInstance(capacity=capacity, weights=weights, values=values)


def measure_time(func: Callable[[], None], repeats: int = 5) -> float:
    """Измеряет среднее время выполнения функции в миллисекундах.

    Сложность:
        Вызов функции `func` происходит `repeats` раз.
        Если T — время одного вызова, то общая сложность O(repeats * T).
    """
    timer = timeit.Timer(func)
    total = timer.timeit(number=repeats)
    return (total / repeats) * 1000.0


def experiment_knapsack_comparison() -> None:
    """Сравнивает жадный дробный рюкзак с точным 0-1 для маленьких примеров.

    Печатает таблицу и строит два графика:
        1) отношение greedy_value / optimal_0-1 в зависимости от n;
        2) времена работы жадного алгоритма и DP в зависимости от n.

    Сложность:
        Пусть k = число разных значений n в n_values.
        Пусть n_max — максимальное количество предметов.
        Тогда:
            генерация примеров  O(k * n_max),
            жадный алгоритм     O(k * n_max log n_max),
            DP                  O(k * n_max * W) (для маленьких W),
        построение графиков    O(k).
    """
    print("=" * 80)
    print("Сравнение жадного алгоритма для дробного рюкзака и точного 0-1 DP")
    print("=" * 80)

    # Разные количества предметов в тестах.
    n_values = [5, 7, 9, 11]

    # Для последующей отрисовки графиков.
    ns: List[int] = []
    greedy_values: List[float] = []
    optimal_values: List[int] = []
    greedy_times_ms: List[float] = []
    dp_times_ms: List[float] = []

    print(
        " n  |  greedy_value | optimal_0-1 | time_greedy (ms) | time_dp (ms) "
    )
    print("-" * 68)

    for n in n_values:
        instance = generate_random_knapsack_instance(n)

        items = [
            Item(weight=float(w), value=float(v))
            for w, v in zip(instance.weights, instance.values, strict=True)
        ]

        # Обёртки без аргументов для функции measure_time.
        def greedy_call() -> None:
            fractional_knapsack(float(instance.capacity), items)

        def dp_call() -> None:
            knapsack_01_dp(instance)

        greedy_time_ms = measure_time(greedy_call)
        dp_time_ms = measure_time(dp_call)

        greedy_value, _ = fractional_knapsack(float(instance.capacity), items)
        optimal_value = knapsack_01_dp(instance)

        ns.append(n)
        greedy_values.append(greedy_value)
        optimal_values.append(optimal_value)
        greedy_times_ms.append(greedy_time_ms)
        dp_times_ms.append(dp_time_ms)

        print(
            f"{n:3d} | "
            f"{greedy_value:13.2f} | "
            f"{optimal_value:11d} | "
            f"{greedy_time_ms:15.4f} | "
            f"{dp_time_ms:11.4f}"
        )

    # -------------------- График качества решения -------------------- #
    # Относительное качество: отношение жадного значения к оптимальному.
    ratios: List[float] = []
    for g, opt in zip(greedy_values, optimal_values, strict=True):
        if opt > 0:
            ratios.append(g / opt)
        else:
            ratios.append(1.0)

    import matplotlib.pyplot as plt  # локальный импорт допустим

    plt.figure(figsize=(8, 5))
    plt.plot(ns, ratios, marker="o")
    plt.xlabel("Число предметов n")
    plt.ylabel("greedy_value / optimal_0-1")
    plt.title(
        "Качество жадного алгоритма для дробного рюкзака\n"
        "по сравнению с точным 0-1 DP"
    )
    plt.ylim(0.0, 1.2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("knapsack_greedy_vs_dp_quality.png", dpi=200)
    plt.show()

    # -------------------- График времени работы ---------------------- #
    plt.figure(figsize=(8, 5))
    plt.plot(ns, greedy_times_ms, marker="o", label="Жадный дробный рюкзак")
    plt.plot(ns, dp_times_ms, marker="s", label="Точный 0-1 DP")
    plt.xlabel("Число предметов n")
    plt.ylabel("Время работы, мс")
    plt.title("Время работы: жадный дробный рюкзак vs 0-1 DP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("knapsack_greedy_vs_dp_time.png", dpi=200)
    plt.show()

    print(
        "\nКомментарий: график качества показывает, что жадный алгоритм "
        "даёт верхнюю оценку для дробного рюкзака, но для дискретного 0-1 "
        "решение может отличаться от оптимального (отношение < 1). "
        "График времени демонстрирует, что DP растёт значительно быстрее, "
        "чем жадный алгоритм, по мере увеличения числа предметов и "
        "вместимости рюкзака."
    )


def generate_random_text(length: int,
                         alphabet: str = string.ascii_lowercase) -> str:
    """Генерирует случайную строку заданной длины из алфавита. O(n)"""
    return "".join(random.choice(alphabet) for _ in range(length))


def run_all_experiments() -> None:
    """Точка входа для запуска всех экспериментов из одной функции."""
    print(PC_INFO)
    experiment_knapsack_comparison()
    print()
    print()


if __name__ == "__main__":
    run_all_experiments()
