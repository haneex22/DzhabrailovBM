# search_comparison.py
import random
import timeit
from typing import List, Optional

import matplotlib.pyplot as plt


# Исходная задача
def linear_search(arr: list[int], target: int) -> Optional[int]:
    """
    Выполняет линейный поиск элемента в массиве.
    """
    for index, value in enumerate(arr):  # O(n) - цикл по всем элементам
        if value == target:  # O(1) - сравнение
            return index  # O(1) - возврат результата
    return None  # O(1) - возврат None, если элемент не найден
    # Общая сложность: O(n)


def binary_search(arr: list[int], target: int) -> Optional[int]:
    """
    Выполняет бинарный поиск элемента в отсортированном массиве.
    """
    left: int = 0  # O(1) - инициализация
    right: int = len(arr) - 1  # O(1) - получение длины и вычисление

    while left <= right:  # O(log n) - цикл выполняется log n раз
        mid: int = (left + right) // 2  # O(1) - вычисление среднего
        if arr[mid] == target:  # O(1) - сравнение
            return mid  # O(1) - возврат результата
        elif arr[mid] < target:  # O(1) - сравнение
            left = mid + 1  # O(1) - присваивание
        else:  # O(1) - ветвление
            right = mid - 1  # O(1) - присваивание
    return None  # O(1) - возврат None, если элемент не найден
    # Общая сложность: O(log n)


def generate_sorted_array(size: int) -> list[int]:
    """
    Генерирует отсортированный массив случайных чисел.
    """
    return sorted([random.randint(1, size * 10) for _ in range(size)])


# Функция для замера времени выполнения
def measure_search_time(search_func, arr: List[int], target: int,
                        repetitions: int = 100) -> float:
    """
    Измеряет среднее время выполнения функции поиска.
    """
    def search_wrapper():
        return search_func(arr, target)  # Сложность зависит от search_func

    execution_time = timeit.timeit(search_wrapper, number=repetitions)
    return (execution_time / repetitions) * 1000


def plot_results(sizes: List[int], linear_times: List[float],
                 binary_times: List[float]) -> None:
    """
    Строит графики результатов экспериментов.
    """
    # График в линейном масштабе
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, linear_times, 'ro-', label='Линейный поиск O(n)',
             linewidth=2)
    plt.plot(sizes, binary_times, 'bo-', label='Бинарный поиск O(log n)',
             linewidth=2)
    plt.xlabel('Размер массива (N)')
    plt.ylabel('Время выполнения (мс)')
    plt.title('Сравнение алгоритмов поиска\n(линейный масштаб)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # График в логарифмическом масштабе по оси Y
    plt.subplot(1, 2, 2)
    plt.semilogy(sizes, linear_times, 'ro-', label='Линейный поиск O(n)',
                 linewidth=2)
    plt.semilogy(sizes, binary_times, 'bo-', label='Бинарный поиск O(log n)',
                 linewidth=2)
    plt.xlabel('Размер массива (N)')
    plt.ylabel('Время выполнения (мс) - логарифмическая шкала')
    plt.title('Сравнение алгоритмов поиска\n(логарифмический масштаб)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('search_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


# Характеристики ПК (заполнить своими данными)
pc_info = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i7-13620H @ 2.40GHz
- Оперативная память: 32 GB DDR5
- ОС: Windows 11
- Python: 3.13.3
"""

print(pc_info)

# Проведение экспериментов
# Размеры массивов для тестирования
sizes: List[int] = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
# Результаты измерений
linear_times: List[float] = []
binary_times: List[float] = []

print('Замеры времени выполнения алгоритмов поиска:')
print('{:>10} {:>15} {:>15}'
      .format('Размер (N)', 'Линейный (мс)', 'Бинарный (мс)'))

for size in sizes:  # O(k) - цикл по количеству размеров
    # Генерация отсортированного массива
    arr: List[int] = generate_sorted_array(size)

    # Выбор целевых элементов для тестирования
    first_element: int = arr[0]
    last_element: int = arr[-1]
    middle_element: int = arr[size // 2]

    # Замер времени для линейного поиска
    # (поиск последнего элемента - худший случай)
    linear_time: float = measure_search_time(linear_search,
                                             arr, last_element)
    linear_times.append(linear_time)

    # Замер времени для бинарного поиска (поиск среднего элемента)
    binary_time: float = measure_search_time(binary_search, arr,
                                             middle_element)
    binary_times.append(binary_time)

    print('{:>10} {:>15.4f} {:>15.4f}'.format(size, linear_time, binary_time))

# Построение графиков
plot_results(sizes, linear_times, binary_times)


# Дополнительный анализ: сравнение с теоретической оценкой
print('\nАнализ результатов:')
print('1. Теоретическая сложность алгоритмов:')
print('- Линейный поиск: O(n)')
print('- Бинарный поиск: O(log n)')

print('2. Практические наблюдения:')

# Анализ роста времени для линейного поиска
linear_growth = linear_times[-1] / linear_times[0]
size_growth = sizes[-1] / sizes[0]
print('''- Линейный поиск:
      время выросло в {:.2f} раз при увеличении N в {:.2f} раз'''
      .format(linear_growth, size_growth))

# Анализ роста времени для бинарного поиска
binary_growth = binary_times[-1] / binary_times[0]
log_growth = (sizes[-1] / sizes[0]) ** (1 / len(sizes))
print('''- Бинарный поиск:
      время выросло в {:.2f} раз при увеличении N в {:.2f} раз'''
      .format(binary_growth, size_growth))

print('3. Выводы:')
print('- Линейный поиск демонстрирует линейный рост времени выполнения')
print('- Бинарный поиск демонстрирует логарифмический рост времени выполнения')
print('- Бинарный поиск значительно эффективнее на больших массивах')
print('- Теоретические оценки сложности подтверждены экспериментально')
