"""
Модуль для тестирования производительности алгоритмов сортировки.
"""

import timeit
from typing import List, Dict, Tuple, Callable
import sorts
import generate_data


def test_correctness() -> None:
    """Тестирование корректности всех алгоритмов сортировки."""
    algorithms: Dict[str, Callable[[List[int]], List[int]]] = {
        'Bubble Sort': sorts.bubble_sort,
        'Selection Sort': sorts.selection_sort,
        'Insertion Sort': sorts.insertion_sort,
        'Merge Sort': sorts.merge_sort,
        'Quick Sort': sorts.quick_sort
    }

    test_cases: Dict[str, List[int]] = {
        'Small array': [64, 34, 25, 12, 22, 11, 90],
        'Already sorted': [1, 2, 3, 4, 5, 6, 7],
        'Reverse sorted': [7, 6, 5, 4, 3, 2, 1],
        'With duplicates': [3, 1, 4, 1, 5, 9, 2, 6, 5]
    }

    print("=== ТЕСТИРОВАНИЕ КОРРЕКТНОСТИ ===")

    for test_name, test_data in test_cases.items():
        print(f'\n--- {test_name}: {test_data} ---')
        for algo_name, algo_func in algorithms.items():
            test_copy = test_data.copy()
            try:
                result = algo_func(test_copy)
                status = '✓ OK' if sorts.is_sorted(result) else '✗ FAIL'
                print(f'{algo_name:15}: {status}')
            except Exception as e:
                print(f'{algo_name:15}: ✗ ERROR -> {e}')


def measure_algorithm_performance(algo_func: Callable[[List[int]], List[int]],
                                  data: List[int]) -> float:
    """Измерение времени выполнения одного алгоритма на одном наборе данных."""
    def sort_wrapper() -> List[int]:
        arr_copy = data.copy()
        return algo_func(arr_copy)

    # Выполняем 3 раза и берем минимальное время
    timer = timeit.Timer(sort_wrapper)
    times = timer.repeat(repeat=3, number=1)
    return min(times)


def run_performance_tests() -> Dict[Tuple[str, str, int], float]:
    """Запуск всех тестов производительности."""
    algorithms: Dict[str, Callable[[List[int]], List[int]]] = {
        'bubble': sorts.bubble_sort,
        'selection': sorts.selection_sort,
        'insertion': sorts.insertion_sort,
        'merge': sorts.merge_sort,
        'quick': sorts.quick_sort
    }

    test_data = generate_data.generate_all_test_data()
    results: Dict[Tuple[str, str, int], float] = {}

    print("\n=== ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ===")

    for size in test_data:
        print(f'\n--- Размер массива: {size} ---')
        for data_type in test_data[size]:
            print(f'\n  {data_type} данные:')

            for algo_name, algo_func in algorithms.items():
                time_taken = measure_algorithm_performance(
                    algo_func,
                    test_data[size][data_type]
                )

                results[(algo_name, data_type, size)] = time_taken
                print(f'    {algo_name:10}: {time_taken:.6f} сек')

    return results


def print_summary_table(results: Dict[Tuple[str, str, int], float]) -> None:
    """Вывод сводной таблицы результатов."""
    print("\n" + "="*60)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*60)

    sizes = sorted(set(size for _, _, size in results.keys()))
    data_types = sorted(set(data_type for _, data_type, _ in results.keys()))
    algorithms = sorted(set(algo for algo, _, _ in results.keys()))

    # Заголовок таблицы
    header_part1 = f"{'Алгоритм':<12} {'Тип данных':<12}"
    header_part2 = "".join(f"{size:>8} " for size in sizes)
    header = header_part1 + header_part2
    print(header)
    print("-" * len(header))

    # Данные таблицы
    for algo in algorithms:
        for data_type in data_types:
            row = f"{algo:<12} {data_type:<12}"
            for size in sizes:
                key = (algo, data_type, size)
                if key in results:
                    row += f"{results[key]:>8.4f} "
                else:
                    row += " " * 9
            print(row)
        print("-" * len(header))


if __name__ == '__main__':
    # Запуск всех тестов
    test_correctness()
    results: Dict[Tuple[str, str, int], float] = run_performance_tests()
    print_summary_table(results)
