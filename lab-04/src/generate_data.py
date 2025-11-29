"""
Модуль для генерации тестовых данных.
"""

import random
from typing import Dict, List


def generate_random_array(size: int) -> List[int]:
    """Генерация массива случайных чисел."""
    return [random.randint(1, 1000) for _ in range(size)]


def generate_sorted_array(size: int) -> List[int]:
    """Генерация отсортированного массива."""
    return list(range(1, size + 1))


def generate_reversed_array(size: int) -> List[int]:
    """Генерация массива, отсортированного в обратном порядке."""
    return list(range(size, 0, -1))


def generate_almost_sorted_array(size: int) -> List[int]:
    """Генерация почти отсортированного массива."""
    arr = list(range(1, size + 1))
    # Перемешиваем 5% элементов
    num_swaps = max(1, size // 20)
    for _ in range(num_swaps):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def generate_all_test_data() -> dict:
    """Генерация всех типов тестовых данных для разных размеров."""
    sizes = [100, 500, 1000, 2000, 5000]
    data_types = {
        'random': generate_random_array,
        'sorted': generate_sorted_array,
        'reversed': generate_reversed_array,
        'almost_sorted': generate_almost_sorted_array
    }

    test_data: Dict[int, Dict[str, List[int]]] = {}
    for size in sizes:
        test_data[size] = {}
        for data_type, generator in data_types.items():
            test_data[size][data_type] = generator(size)

    return test_data


if __name__ == '__main__':
    # Пример генерации данных
    test_data = generate_all_test_data()
    print("Сгенерированы тестовые данные:")
    for size in test_data:
        print(f"Размер {size}: {len(test_data[size]['random'])} элементов")
