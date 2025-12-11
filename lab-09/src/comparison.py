from __future__ import annotations

import random
import timeit
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

from dynamic_programming import (
    KnapsackInstance,
    coin_change_min_coins,
    fib_bottom_up,
    fib_memoized,
    fib_recursive_naive,
    knapsack_01_dp,
    lis_length_dp,
    levenshtein_distance,
)


from greedy_algorithms import Item, fractional_knapsack  # из темы 08


PC_INFO = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


def measure_time_ms(func: Callable[[], None], repeats: int = 1) -> float:
    """Возвращает среднее время выполнения функции в миллисекундах.

    Временная сложность:
        O(repeats * T(func)), где T(func) — время работы func.
    """
    total = 0.0
    for _ in range(repeats):
        start = timeit.default_timer()
        func()
        end = timeit.default_timer()
        total += (end - start) * 1000.0
    return total / repeats


def measure_time_and_memory(
    func: Callable[[], None],
    repeats: int = 1,
) -> Tuple[float, int]:
    """Измеряет среднее время и максимальное потребление памяти (tracemalloc).

    Возвращает (avg_time_ms, peak_bytes).

    Временная сложность:
        O(repeats * T(func)).
    Пространственная сложность:
        Дополнительная память tracemalloc зависит от поведения func.
    """
    times: List[float] = []
    peak_bytes = 0

    for _ in range(repeats):
        tracemalloc.start()
        start = timeit.default_timer()
        func()
        end = timeit.default_timer()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append((end - start) * 1000.0)
        peak_bytes = max(peak_bytes, peak)

    avg_time = sum(times) / len(times)
    return avg_time, peak_bytes


@dataclass(frozen=True)
class RandomKnapsackParams:
    """Параметры генерации случайных экземпляров рюкзака 0-1."""
    num_items: int
    max_weight: int
    max_value: int
    capacity_factor: float = 0.5


def generate_random_knapsack_instance(
    params: RandomKnapsackParams,
) -> KnapsackInstance:
    """Генерирует случайный экземпляр задачи 0-1 рюкзака.

    Вес и ценность каждого предмета — случайные целые числа.
    Вместимость рюкзака пропорциональна суммарному весу.

    Временная сложность:
        O(n), где n = params.num_items.
    """
    values: List[int] = []
    weights: List[int] = []

    total_weight = 0
    for _ in range(params.num_items):
        weight = random.randint(1, params.max_weight)
        value = random.randint(1, params.max_value)
        weights.append(weight)
        values.append(value)
        total_weight += weight

    capacity = int(total_weight * params.capacity_factor)
    return KnapsackInstance(values=values, weights=weights, capacity=capacity)


def experiment_fibonacci_time_and_memory() -> None:
    """Сравнение трёх подходов к вычислению чисел Фибоначчи.

    Сравниваем:
        - fib_recursive_naive
        - fib_memoized
        - fib_bottom_up

    Строим графики:
        fib_time_comparison.png
        fib_memory_comparison.png
    """
    print("=" * 80)
    print("Сравнение подходов ДП для чисел Фибоначчи")
    print("=" * 80)

    n_values = [5, 10, 20, 25, 30]

    naive_times: List[float] = []
    memo_times: List[float] = []
    bottom_times: List[float] = []

    naive_memory: List[int] = []
    memo_memory: List[int] = []
    bottom_memory: List[int] = []

    for n in n_values:
        print(f"\n--- n = {n} ---")

        if n <= 30:
            def naive_call() -> None:
                fib_recursive_naive(n)

            t_naive, m_naive = measure_time_and_memory(naive_call)
        else:
            t_naive, m_naive = float("nan"), 0

        def memo_call() -> None:
            fib_memoized(n)

        def bottom_call() -> None:
            fib_bottom_up(n)

        t_memo, m_memo = measure_time_and_memory(memo_call)
        t_bottom, m_bottom = measure_time_and_memory(bottom_call)

        naive_times.append(t_naive)
        memo_times.append(t_memo)
        bottom_times.append(t_bottom)

        naive_memory.append(m_naive)
        memo_memory.append(m_memo)
        bottom_memory.append(m_bottom)

        print(f"Наивная рекурсия : {t_naive:8.4f} ms, peak = {m_naive} bytes")
        print(f"Мемоизация       : {t_memo:8.4f} ms, peak = {m_memo} bytes")
        print(f"Bottom-up ДП     : {t_bottom:8.4f} ms, peak ={m_bottom} bytes")

    plt.figure(figsize=(8, 5))
    plt.plot(n_values, naive_times, marker="o", label="Наивная рекурсия")
    plt.plot(n_values, memo_times, marker="s", label="Мемоизация (top-down)")
    plt.plot(n_values, bottom_times, marker="^", label="Bottom-up ДП")
    plt.xlabel("n")
    plt.ylabel("Время, ms")
    plt.title("Числа Фибоначчи: время для разных подходов ДП")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fib_time_comparison.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(n_values, naive_memory, marker="o", label="Наивная рекурсия")
    plt.plot(n_values, memo_memory, marker="s", label="Мемоизация (top-down)")
    plt.plot(n_values, bottom_memory, marker="^", label="Bottom-up ДП")
    plt.xlabel("n")
    plt.ylabel("Пиковая память, bytes")
    plt.title("Числа Фибоначчи: память для разных подходов ДП")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fib_memory_comparison.png", dpi=200)
    plt.show()


def build_levenshtein_table(a: str, b: str) -> List[List[int]]:
    """Строит DP-таблицу для расстояния Левенштейна между строками a и b.

    dp[i][j] — минимальное число операций (вставка, удаление, замена),
    чтобы превратить a[:i] в b[:j].

    Временная сложность:
        O(len(a) * len(b)).
    Пространственная сложность:
        O(len(a) * len(b)).
    """
    n = len(a)
    m = len(b)

    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]

    # Базовые случаи: превращаем пустую строку в префикс другой строки
    for i in range(n + 1):
        dp[i][0] = i  # i удалений
    for j in range(m + 1):
        dp[0][j] = j  # j вставок

    # Заполняем таблицу
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # удаление
                dp[i][j - 1] + 1,      # вставка
                dp[i - 1][j - 1] + cost,  # замена (или ничего, если совпали)
            )

    return dp


def print_levenshtein_table(a: str, b: str, dp: List[List[int]]) -> None:
    """Печатает DP-таблицу Левенштейна с подписями строк и столбцов.

    Формат (пример для a='kitten', b='sitting'):

          #   s   i   t   t   i   n   g
      #   0   1   2   3   4   5   6   7
      k   1   1   2   3   4   5   6   7
      i   2   2   1   2   3   4   5   6
      ...

    Сложность печати: O(len(a) * len(b)).
    """
    n = len(a)
    m = len(b)

    # Заголовок
    header = ["    "]  # уголок
    header.extend(f"{ch:>3}" for ch in "#" + b)
    print("".join(header))

    for i in range(n + 1):
        if i == 0:
            row_label = "#"
        else:
            row_label = a[i - 1]
        row_cells = [f"{row_label:>3} "]
        for j in range(m + 1):
            row_cells.append(f"{dp[i][j]:>3}")
        print("".join(row_cells))


def experiment_levenshtein_visualization() -> None:
    """Визуализация принципов Левенштейна через DP-таблицу.

    Для пары слов строится таблица dp[i][j] и:
        - выводится сама таблица с подписями;
        - поясняется, что значат базовые случаи и переходы.

    Используем несколько классических примеров:
        - 'kitten' -> 'sitting'
        - 'intention' -> 'execution'
    """
    print("=" * 80)
    print("Визуализация расстояния Левенштейна (редакционное расстояние)")
    print("=" * 80)

    examples = [
        ("kitten", "sitting"),
        ("intention", "execution"),
    ]

    for a, b in examples:
        print(f"\nСлова: {a!r} -> {b!r}")
        dist = levenshtein_distance(a, b)
        print(f"Расстояние Левенштейна: {dist}")

        dp = build_levenshtein_table(a, b)
        print("\nDP-таблица (dp[i][j] — минимум операций для a[:i] -> b[:j]):")
        print_levenshtein_table(a, b, dp)


def experiment_knapsack_greedy_vs_dp() -> None:
    """Сравнение жадного дробного рюкзака и точного 0-1 ДП.

    Используем:
        - fractional_knapsack (из темы 08)
        - knapsack_01_dp (тема 09)

    Строим графики:
        - относительного качества greedy_value / optimal_value
        - времени работы для обоих подходов.
    """
    print("=" * 80)
    print("Сравнение дробного рюкзака (жадный) и 0-1 рюкзака (ДП)")
    print("=" * 80)

    n_values = [5, 10, 15, 20]

    ratios: List[float] = []
    greedy_times: List[float] = []
    dp_times: List[float] = []

    print(" n  | greedy_value | optimal_0-1 | time_greedy (ms) | time_dp (ms)")
    print("-" * 68)

    for n in n_values:
        params = RandomKnapsackParams(
            num_items=n,
            max_weight=10,
            max_value=20,
            capacity_factor=0.5,
        )
        instance = generate_random_knapsack_instance(params)

        items: List[Item] = [
            Item(weight=float(w), value=float(v))
            for w, v in zip(instance.weights, instance.values, strict=True)
        ]

        def greedy_call() -> None:
            fractional_knapsack(float(instance.capacity), items)

        def dp_call() -> None:
            knapsack_01_dp(instance)

        t_greedy = measure_time_ms(greedy_call, repeats=5)
        t_dp = measure_time_ms(dp_call, repeats=5)

        greedy_value, _ = fractional_knapsack(float(instance.capacity), items)
        optimal_value = float(knapsack_01_dp(instance))

        ratios.append(greedy_value / optimal_value if
                      optimal_value > 0 else 1.0)
        greedy_times.append(t_greedy)
        dp_times.append(t_dp)

        print(
            f"{n:3d} | "
            f"{greedy_value:12.2f} | "
            f"{int(optimal_value):11d} | "
            f"{t_greedy:15.4f} | "
            f"{t_dp:11.4f}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(n_values, ratios, marker="o")
    plt.xlabel("Число предметов n")
    plt.ylabel("greedy_value / optimal_0-1")
    plt.title(
        "Качество жадного дробного рюкзака\n"
        "по сравнению с точным 0-1 ДП"
    )
    plt.ylim(0.0, 1.2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("knapsack_greedy_vs_dp_quality.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(n_values, greedy_times, marker="o", label="Жадный дробный")
    plt.plot(n_values, dp_times, marker="s", label="0-1 рюкзак (ДП)")
    plt.xlabel("Число предметов n")
    plt.ylabel("Время, ms")
    plt.title("Время работы: дробный жадный рюкзак vs 0-1 ДП")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("knapsack_greedy_vs_dp_time.png", dpi=200)
    plt.show()


def experiment_knapsack_scalability_dp() -> None:
    """Исследование масштабируемости 0-1 рюкзака (ДП) при росте n.

    Строим график:
        dp_knapsack_scalability.png
    """
    print("=" * 80)
    print("Масштабируемость ДП для задачи 0-1 рюкзака")
    print("=" * 80)

    n_values = [10, 20, 30, 40, 50]
    times: List[float] = []

    for n in n_values:
        params = RandomKnapsackParams(
            num_items=n,
            max_weight=20,
            max_value=30,
            capacity_factor=0.5,
        )
        instance = generate_random_knapsack_instance(params)

        def dp_call() -> None:
            knapsack_01_dp(instance)

        t = measure_time_ms(dp_call, repeats=3)
        times.append(t)
        print(f"n = {n:3d} => time_dp ≈ {t:8.4f} ms")

    plt.figure(figsize=(8, 5))
    plt.plot(n_values, times, marker="o")
    plt.xlabel("Число предметов n")
    plt.ylabel("Время, ms")
    plt.title("Масштабируемость ДП для 0-1 рюкзака (по n)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dp_knapsack_scalability.png", dpi=200)
    plt.show()


def demo_additional_dp_tasks() -> None:
    """Короткая демонстрация задач: размен монет и LIS."""
    print("=" * 80)
    print("Размен монет (минимальное число монет)")
    print("=" * 80)
    amount = 27
    coins = [1, 5, 10]
    min_coins = coin_change_min_coins(amount, coins)
    print(f"Сумма: {amount}, монеты: {coins}, минимум монет: {min_coins}")

    print("\n" + "=" * 80)
    print("Наибольшая возрастающая подпоследовательность (LIS)")
    print("=" * 80)
    seq = [3, 1, 5, 2, 6, 4, 9]
    lis_len = lis_length_dp(seq)
    print(f"Последовательность: {seq}")
    print(f"Длина LIS (DP O(n^2)): {lis_len}")


def main() -> None:
    """Главная точка входа для запуска всех экспериментов темы 09."""
    print(PC_INFO)
    experiment_fibonacci_time_and_memory()
    experiment_knapsack_greedy_vs_dp()
    experiment_knapsack_scalability_dp()
    demo_additional_dp_tasks()
    experiment_levenshtein_visualization()


if __name__ == "__main__":
    main()
