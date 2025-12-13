from __future__ import annotations

import random
import timeit
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt

from kmp_search import kmp_search_all
from string_matching import naive_search_all, rabin_karp_search_all
from string_matching import z_search_all

PC_INFO = """\
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


def measure_ms(func: Callable[[], None], repeats: int = 5) -> float:
    """Среднее время выполнения функции в миллисекундах O(repeats * T(func))"""
    timer = timeit.Timer(func)
    total = timer.timeit(number=repeats)
    return (total / repeats) * 1000.0


def random_string(n: int, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> str:
    """Генерирует случайную строку длины n. O(n)."""
    return "".join(random.choice(alphabet) for _ in range(n))


def periodic_string(n: int, base: str = "ab") -> str:
    """Генерирует периодическую строку длины n. O(n)."""
    return (base * (n // len(base) + 1))[:n]


def worst_case_for_naive(n: int, m: int) -> Tuple[str, str]:
    """Почти худший случай для наивного поиска: 'aaaa...a' и 'aaa...ab'.

    Временная сложность генерации: O(n + m).
    """
    text = "a" * n
    if m == 0:
        return text, ""
    pattern = "a" * (m - 1) + "b"
    return text, pattern


def run_experiment_case(
    title: str,
    texts: List[str],
    patterns: List[str],
) -> Dict[str, List[float]]:
    """Замеряет время для 4 алгоритмов на заданных данных.

    Возвращает словарь:
        algo_name -> список времен на каждой паре (text, pattern).
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    times: Dict[str, List[float]] = {
        "naive": [],
        "z": [],
        "rabin_karp": [],
        "kmp": [],
    }

    for text, pattern in zip(texts, patterns, strict=True):
        # чтобы условия были одинаковыми
        def f_naive() -> None:
            naive_search_all(text, pattern)

        def f_z() -> None:
            z_search_all(text, pattern)

        def f_rk() -> None:
            rabin_karp_search_all(text, pattern)

        def f_kmp() -> None:
            kmp_search_all(text, pattern)

        times["naive"].append(measure_ms(f_naive))
        times["z"].append(measure_ms(f_z))
        times["rabin_karp"].append(measure_ms(f_rk))
        times["kmp"].append(measure_ms(f_kmp))

        print(
            f"n={len(text):6d}, m={len(pattern):4d} | "
            f"naive={times['naive'][-1]:9.4f} ms | "
            f"z={times['z'][-1]:9.4f} ms | "
            f"rk={times['rabin_karp'][-1]:9.4f} ms | "
            f"kmp={times['kmp'][-1]:9.4f} ms"
        )

    return times


def plot_times(x: List[int], times: Dict[str, List[float]], filename: str,
               title: str) -> None:
    """Строит график времени выполнения алгоритмов."""
    plt.figure(figsize=(9, 5))
    plt.plot(x, times["naive"], marker="o", label="Наивный")
    plt.plot(x, times["z"], marker="s", label="Z-поиск")
    plt.plot(x, times["rabin_karp"], marker="^", label="Рабин-Карп")
    plt.plot(x, times["kmp"], marker="D", label="KMP")
    plt.xlabel("Длина текста n")
    plt.ylabel("Время, мс")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def main() -> None:
    print(PC_INFO)

    # 1) Случайные строки
    n_values = [1_000, 2_000, 4_000, 8_000, 16_000]
    m = 25
    texts = [random_string(n) for n in n_values]
    patterns = [random_string(m) for _ in n_values]
    times_random = run_experiment_case(
        "Случайные строки: поиск паттерна в тексте",
        texts,
        patterns,
    )
    plot_times(
        n_values,
        times_random,
        "string_matching_random.png",
        "Случайные строки: время поиска vs длина текста",
    )

    # 2) Периодические строки (много повторений)
    texts = [periodic_string(n, base="abababab") for n in n_values]
    patterns = ["ababababababababababababa"[:m] for _ in n_values]
    times_periodic = run_experiment_case(
        "Периодические строки: много повторов",
        texts,
        patterns,
    )
    plot_times(
        n_values,
        times_periodic,
        "string_matching_periodic.png",
        "Периодические строки: время поиска vs длина текста",
    )

    # 3) Почти худший случай для наивного
    texts = []
    patterns = []
    for n in n_values:
        text, pattern = worst_case_for_naive(n, m)
        texts.append(text)
        patterns.append(pattern)
    times_worst = run_experiment_case(
        "Худший случай для наивного поиска",
        texts,
        patterns,
    )
    plot_times(
        n_values,
        times_worst,
        "string_matching_worst_naive.png",
        "Худший случай для наивного: сравнение алгоритмов",
    )

    print("\nИтог:")
    print("  - KMP и Z-поиск показывают линейный рост и стабильное поведение.")
    print("  - Наивный алгоритм резко замедляется на повторяющихся данных.")
    print("  - Рабин-Карп обычно быстрый, но теоретически имеет худший случай")


if __name__ == "__main__":
    main()
