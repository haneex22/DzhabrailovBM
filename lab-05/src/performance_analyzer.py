"""Модуль для анализа производительности хеш-таблиц."""

import time
import random
import string
from typing import Dict, List, Optional, Callable, Tuple
import matplotlib.pyplot as plt
from hash_functions import simple_hash, polynomial_hash, djb2_hash
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing

# Настройка русского шрифта для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_random_string(length: int = 10) -> str:
    """
    Генерация случайной строки.

    Args:
        length: Длина строки

    Returns:
        Случайная строка
    """
    return ''.join(random.choices(string.ascii_letters + string.digits,
                                  k=length))


def measure_performance(
    table_class: type,
    hash_func: Callable[[str, int], int],
    method: Optional[str] = None,
    data_size: int = 1000,
    load_factors: Optional[List[float]] = None
) -> Dict[str, List[float]]:
    """
    Измерение производительности операций для разных коэффициентов заполнения.

    Args:
        table_class: Класс хеш-таблицы
        hash_func: Хеш-функция
        method: Метод пробирования (для открытой адресации)
        data_size: Количество элементов для тестирования
        load_factors: Список коэффициентов заполнения

    Returns:
        Словарь с результатами измерений
    """
    if load_factors is None:
        load_factors = [0.1, 0.5, 0.7, 0.9]

    results: Dict[str, List[float]] = {
        'insert': [],
        'search': [],
        'delete': []
    }

    for lf in load_factors:
        # Создаем таблицу с начальным размером для достижения коэффициента
        initial_size = max(101, int(data_size / lf))

        if method:
            table = table_class(
                size=initial_size, method=method, hash_func1=hash_func
            )
        else:
            table = table_class(size=initial_size, hash_func=hash_func)

        # Генерируем тестовые данные
        test_data = [(generate_random_string(), i) for i in range(data_size)]

        # Измеряем время вставки
        start_time = time.time()
        for key, value in test_data:
            table.insert(key, value)
        insert_time = time.time() - start_time
        results['insert'].append(insert_time)

        # Измеряем время поиска
        start_time = time.time()
        for key, _ in test_data:
            table.search(key)
        search_time = time.time() - start_time
        results['search'].append(search_time)

        # Измеряем время удаления
        start_time = time.time()
        for key, _ in test_data:
            table.delete(key)
        delete_time = time.time() - start_time
        results['delete'].append(delete_time)

    return results


def analyze_collisions(
    hash_func: Callable[[str, int], int],
    table_size: int = 1000,
    num_keys: int = 10000
) -> List[int]:
    """
    Анализ распределения коллизий для хеш-функции.

    Args:
        hash_func: Хеш-функция для анализа
        table_size: Размер таблицы
        num_keys: Количество ключей для тестирования

    Returns:
        Список количества коллизий по ячейкам
    """
    buckets = [0] * table_size

    for _ in range(num_keys):
        key = generate_random_string()
        index = hash_func(key, table_size)
        buckets[index] += 1

    return buckets


def plot_performance(
    results_chaining: Dict[str, List[float]],
    results_linear: Dict[str, List[float]],
    results_double: Dict[str, List[float]],
    load_factors: List[float]
) -> None:
    """Построение графиков производительности."""
    operations = ['insert', 'search', 'delete']
    operation_names = ['Вставка', 'Поиск', 'Удаление']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (op, op_name) in enumerate(zip(operations, operation_names)):
        ax = axes[i]
        ax.plot(load_factors, results_chaining[op], 'o-', label='Цепочки')
        ax.plot(load_factors, results_linear[op], 's-',
                label='Линейное пробирование')
        ax.plot(load_factors, results_double[op], '^-',
                label='Двойное хеширование')

        ax.set_xlabel('Коэффициент заполнения')
        ax.set_ylabel('Время (секунды)')
        ax.set_title(f'Операция: {op_name}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_collision_distribution(
    hash_functions: List[Callable[[str, int], int]],
    names: List[str],
    table_size: int = 1000,
    num_keys: int = 10000
) -> None:
    """Построение гистограмм распределения коллизий."""
    fig, axes = plt.subplots(1, len(hash_functions), figsize=(15, 5))

    if len(hash_functions) == 1:
        axes = [axes]

    for i, (hash_func, name) in enumerate(zip(hash_functions, names)):
        collisions = analyze_collisions(hash_func, table_size, num_keys)

        axes[i].hist(collisions, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel('Количество коллизий на ячейку')
        axes[i].set_ylabel('Частота')
        axes[i].set_title(f'Распределение коллизий: {name}')
        axes[i].grid(True, alpha=0.3)

        # Добавляем статистику
        avg_collisions = sum(collisions) / len(collisions)
        max_collisions = max(collisions)
        axes[i].text(
            0.05, 0.95, f'avg: {avg_collisions:.2f}\nМакс: {max_collisions}',
            transform=axes[i].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig('collision_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_optimal_load_factors(
    results_chaining: Dict[str, List[float]],
    results_linear: Dict[str, List[float]],
    results_double: Dict[str, List[float]],
    load_factors: List[float]
) -> None:
    """Анализ оптимальных коэффициентов заполнения."""
    print("\nАнализ оптимальных коэффициентов заполнения:")
    print("=" * 50)

    # Находим оптимальные коэффициенты для поиска
    chaining_search = results_chaining['search']
    linear_search = results_linear['search']
    double_search = results_double['search']

    # Оптимальный коэффициент - где время поиска минимальное
    optimal_chaining = load_factors[
        chaining_search.index(min(chaining_search))]
    optimal_linear = load_factors[linear_search.index(min(linear_search))]
    optimal_double = load_factors[double_search.index(min(double_search))]

    print(f"Метод цепочек - оптимальный коэффициент: {optimal_chaining}")
    print(f"Линейное пробирование - оптимальный: {optimal_linear}")
    print(f"Двойное хеширование - оптимальный: {optimal_double}")

    # Анализ производительности при разных коэффициентах
    print("\nСравнительная производительность при коэффициенте 0.9:")
    chaining_90 = results_chaining['search'][-1]
    linear_90 = results_linear['search'][-1]
    double_90 = results_double['search'][-1]

    fastest_method = min(chaining_90, linear_90, double_90)
    if fastest_method == chaining_90:
        fastest_name = "Метод цепочек"
    elif fastest_method == linear_90:
        fastest_name = "Линейное пробирование"
    else:
        fastest_name = "Двойное хеширование"

    print(f"Самый быстрый метод при α=0.9: {fastest_name}")

    # Рекомендации по пороговым значениям
    print("\nРекомендации по пороговым значениям:")
    print("- Метод цепочек: до 0.9 (стабильная производительность)")
    print("- Линейное пробирование: до 0.7 (резкое замедление после)")
    print("- Двойное хеширование: до 0.8 (хороший компромисс)")


def compare_hash_functions_performance() -> None:
    """Сравнительный анализ производительности разных хеш-функций."""
    print("\nСравнительный анализ хеш-функций:")
    print("=" * 40)

    data_size = 1000
    load_factors = [0.5, 0.7]

    hash_functions: List[Tuple[Callable[[str, int], int], str]] = [
        (simple_hash, "Простая хеш-функция"),
        (polynomial_hash, "Полиномиальная хеш-функция"),
        (djb2_hash, "Хеш-функция DJB2")
    ]

    results: Dict[str, Dict[str, List[float]]] = {}

    for hash_func, name in hash_functions:
        # Тестируем с методом цепочек
        chaining_results = measure_performance(
            HashTableChaining, hash_func, data_size=data_size,
            load_factors=load_factors
        )
        results[name] = chaining_results

    # Анализ результатов
    print("\nПроизводительность при α=0.7 (время поиска):")
    for name in results:
        search_time = results[name]['search'][-1]
        print(f"{name}: {search_time:.6f} сек")

    # Вывод рекомендаций
    print("\nРекомендации по выбору хеш-функции:")
    print("1. Простая хеш-функция: быстрая, но плохое распределение")
    print("2. Полиномиальная: хороший баланс скорости и качества")
    print("3. DJB2: отличное распределение, рекомендуется для production")


def compare_collision_resolution_methods(
    results_chaining: Dict[str, List[float]],
    results_linear: Dict[str, List[float]],
    results_double: Dict[str, List[float]],
    load_factors: List[float]
) -> None:
    """Сравнительный анализ методов разрешения коллизий."""
    print("\nСравнительный анализ методов разрешения коллизий:")
    print("=" * 55)

    # Анализ при разных коэффициентах заполнения
    print("\nПроизводительность поиска (время в секундах):")
    print("Коэф. | Цепочки | Линейное | Двойное")
    print("-" * 40)

    for i, lf in enumerate(load_factors):
        chaining_time = results_chaining['search'][i]
        linear_time = results_linear['search'][i]
        double_time = results_double['search'][i]
        print(f"{lf:5.1f} | {chaining_time:8.6f} | "
              f"{linear_time:8.6f} | {double_time:8.6f}")

    # Определение лучшего метода для каждого коэффициента
    print("\nЛучший метод для каждого коэффициента заполнения:")
    for i, lf in enumerate(load_factors):
        times = {
            'Цепочки': results_chaining['search'][i],
            'Линейное': results_linear['search'][i],
            'Двойное': results_double['search'][i]
        }
        # Исправление для mypy - используем лямбда-функцию
        best_method = min(times.keys(), key=lambda k: times[k])
        print(f"α={lf}: {best_method}")

    # Итоговые выводы
    print("\nВыводы:")
    print("1. Метод цепочек: стабильная производительность при высоких α")
    print("2. Линейное пробирование: быстрое при низких α, замедляется")
    print("3. Двойное хеширование: лучший компромисс для большинства случаев")


def main() -> None:
    """Основная функция анализа производительности."""
    print("Анализ производительности хеш-таблиц")
    print("=" * 50)

    # Параметры тестирования
    data_size = 1000
    load_factors = [0.1, 0.5, 0.7, 0.9]

    # Используем DJB2 как наиболее качественную хеш-функцию
    hash_func = djb2_hash

    print("Измерение производительности...")

    # Измеряем производительность для разных методов
    results_chaining = measure_performance(
        HashTableChaining, hash_func, data_size=data_size,
        load_factors=load_factors
    )

    results_linear = measure_performance(
        HashTableOpenAddressing, hash_func, 'linear',
        data_size=data_size, load_factors=load_factors
    )

    results_double = measure_performance(
        HashTableOpenAddressing, hash_func, 'double',
        data_size=data_size, load_factors=load_factors
    )

    # Строим графики производительности
    print("Построение графиков производительности...")
    plot_performance(results_chaining, results_linear, results_double,
                     load_factors)

    # Анализируем распределение коллизий
    print("Анализ распределения коллизий...")
    hash_functions_list: List[Callable[[str, int], int]] = [
        simple_hash,
        polynomial_hash,
        djb2_hash
    ]
    names_list = ['Простая хеш-функция', 'Полиномиальная хеш-функция',
                  'Хеш-функция DJB2']
    plot_collision_distribution(hash_functions_list, names_list)

    # Выводим сводные результаты
    print("\nСводные результаты:")
    print("Метод           | Вставка (0.9) | Поиск (0.9)  | Удаление (0.9)")
    print("-" * 65)

    methods = [
        ('Цепочки', results_chaining),
        ('Линейное', results_linear),
        ('Двойное', results_double)
    ]

    for name, results in methods:
        insert_time = results['insert'][-1]  # При коэффициенте 0.9
        search_time = results['search'][-1]
        delete_time = results['delete'][-1]
        print(
            f"{name:<15} | {insert_time:12.4f} | "
            f"{search_time:12.4f} | {delete_time:12.4f}"
        )

    # Полный сравнительный анализ
    analyze_optimal_load_factors(results_chaining, results_linear,
                                 results_double, load_factors)

    compare_collision_resolution_methods(results_chaining, results_linear,
                                         results_double, load_factors)

    compare_hash_functions_performance()

    # Итоговый вывод
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ ВЫВОДЫ:")
    print("=" * 60)
    print("1. Метод цепочек наиболее устойчив к высоким коэффициентам")
    print("2. Двойное хеширование обеспечивает лучший компромисс")
    print("3. Хеш-функция DJB2 рекомендуется для большинства применений")
    print("4. Оптимальные пороговые значения:")
    print("   - Цепочки: α ≤ 0.9")
    print("   - Линейное пробирование: α ≤ 0.7")
    print("   - Двойное хеширование: α ≤ 0.8")


if __name__ == '__main__':
    main()
