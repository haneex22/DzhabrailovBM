"""Модуль с оптимизированными рекурсивными алгоритмами с мемоизацией."""

import time
from typing import Dict, Optional


def fibonacci_memo(n: int, memo: Optional[Dict[int, int]] = None) -> int:
    """
    Вычисляет n-е число Фибоначчи с мемоизацией.

    Args:
        n: Порядковый номер числа Фибоначчи
        memo: Словарь для хранения вычисленных значений

    Returns:
        n-е число Фибоначчи
    """
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n == 0:
        return 0
    if n == 1:
        return 1

    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


def compare_fibonacci(n: int = 35) -> None:
    """
    Сравнивает производительность наивной и мемоизированной версий.

    Args:
        n: Номер числа Фибоначчи для тестирования
    """
    from recursion import fibonacci

    # Наивная версия
    start_time = time.time()
    naive_result = fibonacci(n)
    naive_time = time.time() - start_time

    # Мемоизированная версия
    start_time = time.time()
    memo_result = fibonacci_memo(n)
    memo_time = time.time() - start_time

    print(f'Результат для n={n}:')
    print(f'Наивная версия: {naive_result}, время: {naive_time:.6f} сек')
    print(f'Мемоизированная версия: {memo_result}, время: {memo_time:.6f} сек')
    print(f'Ускорение: {naive_time/memo_time:.2f} раз')


if __name__ == '__main__':
    compare_fibonacci(35)
