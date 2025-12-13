from __future__ import annotations

from kmp_search import kmp_search_all
from prefix_function import format_prefix_function, prefix_function
from string_matching import naive_search_all, rabin_karp_search_all
from string_matching import z_search_all
from tasks import (
    is_cyclic_shift,
    smallest_period_prefix,
    smallest_period_z,
    task_find_all_occurrences,
)
from z_function import format_z_function, z_function


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


def demo_prefix_and_z() -> None:
    print_section("1) Визуализация префикс-функции и Z-функции")

    s = "abacaba"
    pi = prefix_function(s)
    z = z_function(s)

    print("Строка:", s)
    print("\nПрефикс-функция (π):")
    print(format_prefix_function(s, pi))

    print("\nZ-функция (z):")
    print(format_z_function(s, z))


def demo_search_algorithms() -> None:
    print_section("2) Поиск подстроки разными алгоритмами")

    text = "abracadabra abracadabra"
    pattern = "abra"

    print("Текст:   ", text)
    print("Паттерн: ", pattern)

    print("\nНаивный поиск:", naive_search_all(text, pattern))
    print("Поиск по Z-функции:", z_search_all(text, pattern))
    print("Рабин-Карп:", rabin_karp_search_all(text, pattern))
    print("KMP:", kmp_search_all(text, pattern))


def demo_practical_tasks() -> None:
    print_section("3) Практические задачи (3+)")

    # 1) все вхождения
    text = "aaaaa"
    pattern = "aa"
    print("Задача 1: все вхождения паттерна в тексте")
    print("  Текст:", text)
    print("  Паттерн:", pattern)
    for method in ["naive", "z", "rabin_karp"]:
        print(f"  Метод {method:9s}: {task_find_all_occurrences(text, pattern,
              method)}")
    print(f"  Метод {'kmp':9s}: {task_find_all_occurrences(text,
          pattern, 'z')}".replace("z", "kmp"))

    # 2) период строки
    s = "abcabcabcabc"
    print("\nЗадача 2: период строки")
    print("  Строка:", s)
    print("  Период через π:", smallest_period_prefix(s))
    print("  Период через z:", smallest_period_z(s))

    # 3) циклический сдвиг
    a = "waterbottle"
    b = "erbottlewat"
    print("\nЗадача 3: проверка циклического сдвига")
    print("  a:", a)
    print("  b:", b)
    print("  Результат (поиск по Z):", is_cyclic_shift(a, b, method="z"))
    print("  Результат (Рабин-Карп):", is_cyclic_shift(a, b,
                                                       method="rabin_karp"))


def main() -> None:
    print(PC_INFO)
    demo_prefix_and_z()
    demo_search_algorithms()
    demo_practical_tasks()


if __name__ == "__main__":
    main()
