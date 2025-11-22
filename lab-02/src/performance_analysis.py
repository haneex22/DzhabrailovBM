# performance_analysis.py
"""
Анализ производительности структур данных.
Сравнение list, deque и связного списка для различных операций.
"""

import timeit
from collections import deque

import matplotlib.pyplot as plt

from linked_list import LinkedList


def measure_time(func, *args, number=1000):
    """
    Измеряет время выполнения функции в миллисекундах.
    Сложность: O(1) - просто вызывает функцию и замеряет время
    """
    start_time = timeit.default_timer()  # O(1)
    for _ in range(number):  # O(number) - но number фиксирован
        func(*args)  # O(1) - зависит от сложности func
    end_time = timeit.default_timer()  # O(1)
    return (end_time - start_time) * 1000  # O(1)
# Общая сложность: O(number * сложность_func)


def compare_list_vs_linkedlist_insert_start():
    """
    Сравнение вставки в начало для list и LinkedList.
    Сложность:
    - list.insert(0, item): O(n) на каждую операцию
    - LinkedList.insert_at_start(): O(1) на каждую операцию
    """
    sizes = [100, 500, 1000, 2000, 5000]  # O(1)
    list_times = []  # O(1)
    linked_list_times = []  # O(1)
    print("Сравнение вставки в начало:")
    print("{:>10} {:>12} {:>15}".format("Размер", "List (мс)",
                                        "LinkedList (мс)"))
    for size in sizes:  # O(len(sizes))
        # Тестирование list
        test_list = list(range(size))  # O(size)

        def list_insert():  # O(n)
            test_list.insert(0, 0)  # O(n)
        list_time = measure_time(list_insert)  # O(1000 * n)
        list_times.append(list_time)  # O(1)
        # Тестирование LinkedList
        linked_list = LinkedList()  # O(1)
        for i in range(size):  # O(size)
            linked_list.insert_at_start(i)  # O(1)

        def linked_list_insert():  # O(1)
            linked_list.insert_at_start(0)  # O(1)
        linked_list_time = measure_time(linked_list_insert)  # O(1000 * 1)
        linked_list_times.append(linked_list_time)  # O(1)
        print("{:>10} {:>12.2f} {:>15.2f}".format(
            size, list_time, linked_list_time))  # O(1)
    # Построение графика
    plt.figure(figsize=(10, 6))  # O(1)
    plt.plot(sizes, list_times, 'ro-', label='List (insert(0, item))')  # O(1)
    plt.plot(sizes, linked_list_times, 'bo-',
             label='LinkedList (insert_at_start)')  # O(1)
    plt.xlabel('Количество элементов')  # O(1)
    plt.ylabel('Время выполнения (мс)')  # O(1)
    plt.title('Сравнение вставки в начало: List vs LinkedList')  # O(1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # O(1)
    plt.legend()  # O(1)
    plt.savefig('list_vs_linkedlist_insert_start.png', dpi=300,
                bbox_inches='tight')  # O(1)
    plt.show()  # O(1)
# Общая сложность: O(n^2) из-за вставки в list


def compare_list_vs_deque_queue():
    """
    Сравнение операций очереди для list и deque.
    Сложность:
    - list.pop(0): O(n) на каждую операцию
    - deque.popleft(): O(1) на каждую операцию
    """
    sizes = [100, 500, 1000, 2000, 5000]  # O(1)
    list_times = []  # O(1)
    deque_times = []  # O(1)
    print("\nСравнение операций очереди:")
    print("{:>10} {:>12} {:>15}".format("Размер", "List (мс)", "Deque (мс)"))
    for size in sizes:  # O(len(sizes))
        # Тестирование list
        test_list = list(range(size))  # O(size)

        def list_dequeue():  # O(n)
            if test_list:  # O(1)
                test_list.pop(0)  # O(n)
        list_time = measure_time(list_dequeue)  # O(1000 * n)
        list_times.append(list_time)  # O(1)
        # Тестирование deque
        test_deque = deque(range(size))  # O(size)

        def deque_dequeue():  # O(1)
            if test_deque:  # O(1)
                test_deque.popleft()  # O(1)
        deque_time = measure_time(deque_dequeue)  # O(1000 * 1)
        deque_times.append(deque_time)  # O(1)
        print("{:>10} {:>12.2f} {:>15.2f}".format(
            size, list_time, deque_time))  # O(1)
    # Построение графика
    plt.figure(figsize=(10, 6))  # O(1)
    plt.plot(sizes, list_times, 'ro-', label='List (pop(0))')  # O(1)
    plt.plot(sizes, deque_times, 'go-', label='Deque (popleft())')  # O(1)
    plt.xlabel('Количество элементов')  # O(1)
    plt.ylabel('Время выполнения (мс)')  # O(1)
    plt.title('Сравнение операций очереди: List vs Deque')  # O(1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # O(1)
    plt.legend()  # O(1)
    plt.savefig('list_vs_deque_queue.png', dpi=300,
                bbox_inches='tight')  # O(1)
    plt.show()  # O(1)
# Общая сложность: O(n^2) из-за pop(0) в list


def main():
    """
    Основная функция для запуска анализа производительности.
    Сложность: O(n^3) - определяется наиболее сложной функцией
    """
    # Характеристики ПК для тестирования
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: Intel Core i5-11400 @ 2.60GHz
    - Оперативная память: 16 GB DDR4
    - OC: Windows 10
    - Python: 3.13.0
    """
    print(pc_info)  # O(1)
    # Запуск сравнений
    compare_list_vs_linkedlist_insert_start()  # O(n^2)
    compare_list_vs_deque_queue()  # O(n^2)
    print("\nАнализ завершен. Графики сохранены в файлы PNG.")  # O(1)


if __name__ == '__main__':
    main()
# Общая сложность программы: O(n^3)
