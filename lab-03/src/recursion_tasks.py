"""Модуль с практическими задачами на рекурсию."""

import os
from typing import List, Optional


def binary_search_recursive(arr: List[int], target: int,
                            left: int = 0, right:
                                Optional[int] = None) -> Optional[int]:
    """
    Рекурсивная реализация бинарного поиска.

    Args:
        arr: Отсортированный массив для поиска
        target: Искомый элемент
        left: Левая граница поиска
        right: Правая граница поиска

    Returns:
        Индекс элемента или None если не найден
    """
    if right is None:
        right = len(arr) - 1

    if left > right:
        return None

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def file_system_walk(path: str, level: int = 0) -> None:
    """
    Рекурсивный обход файловой системы с выводом дерева каталогов.

    Args:
        path: Начальный путь для обхода
        level: Текущий уровень вложенности
    """
    try:
        items = os.listdir(path)
    except PermissionError:
        print('  ' * level + f'[Доступ запрещен: {os.path.basename(path)}]')
        return

    for item in sorted(items):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            print('  ' * level + f'📁 {item}/')
            file_system_walk(item_path, level + 1)
        else:
            print('  ' * level + f'📄 {item}')


def hanoi_towers(n: int, source: str = 'A',
                 auxiliary: str = 'B', target: str = 'C') -> None:
    """
    Решает задачу Ханойских башен для n дисков.

    Args:
        n: Количество дисков
        source: Исходный стержень
        auxiliary: Вспомогательный стержень
        target: Целевой стержень
    """
    if n == 1:
        print(f'Переместить диск 1 с {source} на {target}')
        return

    hanoi_towers(n - 1, source, target, auxiliary)
    print(f'Переместить диск {n} с {source} на {target}')
    hanoi_towers(n - 1, auxiliary, source, target)


if __name__ == '__main__':
    # Тестирование бинарного поиска
    sorted_array = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search_recursive(sorted_array, target)
    print(f'Бинарный поиск {target} в {sorted_array}: индекс {result}')

    # Тестирование Ханойских башен
    print('\nХанойские башни для 3 дисков:')
    hanoi_towers(3)

    # Тестирование обхода файловой системы
    print('\nОбход текущей директории:')
    file_system_walk('.')
