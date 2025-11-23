"""Модуль для экспериментального исследования рекурсивных алгоритмов."""

import time
import os
import sys
import shutil
import matplotlib.pyplot as plt
from recursion import fibonacci
from memoization import fibonacci_memo


def measure_fibonacci_performance(max_n: int = 35) -> None:
    """
    Замеряет время выполнения наивного и мемоизированного вычисления Фибоначчи.

    Args:
        max_n: Максимальное значение n для тестирования
    """
    naive_times = []
    memo_times = []
    n_values = list(range(1, max_n + 1))

    for n in n_values:
        # Наивная версия
        start_time = time.time()
        fibonacci(n)
        naive_times.append(time.time() - start_time)

        # Мемоизированная версия
        start_time = time.time()
        fibonacci_memo(n)
        memo_times.append(time.time() - start_time)

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, naive_times, label='Наивная рекурсия', marker='o')
    plt.plot(n_values, memo_times, label='С мемоизацией', marker='s')
    plt.xlabel('n')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение времени вычисления чисел Фибоначчи')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Логарифмическая шкала для наглядности
    plt.savefig('fibonacci_performance.png')
    plt.show()

    print('График сохранен как fibonacci_performance.png')


def measure_max_recursion_depth() -> None:
    """
    Измеряет максимальную глубину рекурсии для обхода файловой системы.
    """
    print(f'Текущее ограничение глубины рекурсии: {sys.getrecursionlimit()}')

    # Создание глубоко вложенной структуры каталогов для тестирования
    test_dir = 'deep_test'
    os.makedirs(test_dir, exist_ok=True)

    current_path = test_dir
    depth = 0

    try:
        while depth < 1000:  # Безопасный предел
            new_dir = os.path.join(current_path, f'level_{depth}')
            os.makedirs(new_dir, exist_ok=True)
            current_path = new_dir
            depth += 1
    except (OSError, RecursionError) as e:
        print(f'Достигнута максимальная глубина: {depth}')
        print(f'Ошибка: {e}')

    # Очистка тестовых данных
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    measure_fibonacci_performance(35)
    measure_max_recursion_depth()
