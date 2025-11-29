"""
Модуль для визуализации результатов тестирования.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
import performance_test


def plot_performance_by_size(results: Dict[Tuple[str, str, int],
                                           float]) -> None:
    """Построение графиков зависимости времени от размера массива."""
    algorithms: List[str] = ['bubble', 'selection', 'insertion', 'merge',
                             'quick']
    algo_names: Dict[str, str] = {
        'bubble': 'Bubble Sort',
        'selection': 'Selection Sort',
        'insertion': 'Insertion Sort',
        'merge': 'Merge Sort',
        'quick': 'Quick Sort'
    }

    # Собираем данные для случайных данных
    sizes: List[int] = sorted(set(size for _, _, size in results.keys()))

    plt.figure(figsize=(12, 8))

    for algo in algorithms:
        times: List[float] = []
        for size in sizes:
            key = (algo, 'random', size)
            if key in results:
                times.append(results[key])
            else:
                times.append(0.0)

        if any(times):  # Если есть данные для этого алгоритма
            plt.plot(sizes, times, marker='o', label=algo_names[algo],
                     linewidth=2, markersize=6)

    plt.xlabel('Размер массива', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения от размера массива\n',
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации
    plt.xscale('log')
    plt.savefig('performance_by_size.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_by_data_type(results: Dict[Tuple[str, str, int],
                                                float]) -> None:
    """Построение графиков зависимости времени от типа данных."""
    algorithms: List[str] = ['bubble', 'selection', 'insertion', 'merge',
                             'quick']
    data_types: List[str] = ['random', 'sorted', 'reversed', 'almost_sorted']
    data_type_names: Dict[str, str] = {
        'random': 'Случайные',
        'sorted': 'Отсорт.',
        'reversed': 'Обратные',
        'almost_sorted': 'Почти отсорт.'
    }

    # Берем данные для размера 1000
    target_size: int = 1000

    plt.figure(figsize=(14, 8))

    # Подготовка данных
    plot_data: Dict[str, List[float]] = {}
    for algo in algorithms:
        plot_data[algo] = []
        for data_type in data_types:
            key = (algo, data_type, target_size)
            if key in results:
                plot_data[algo].append(results[key])
            else:
                plot_data[algo].append(0.0)

    # Построение группированной столбчатой диаграммы
    x = np.arange(len(data_types))
    width = 0.15
    multiplier = 0

    algo_display_names: Dict[str, str] = {
        'bubble': 'Пузырьком',
        'selection': 'Выбором',
        'insertion': 'Вставками',
        'merge': 'Слиянием',
        'quick': 'Быстрая'
    }

    for algo, times in plot_data.items():
        offset = width * multiplier
        plt.bar(x + offset, times, width, label=algo_display_names[algo])
        multiplier += 1

    plt.xlabel('Тип данных', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title(f'Сравнение на разных типах данных\n(размер: {target_size})',
              fontsize=14)
    plt.xticks(x + width * 2, [data_type_names[dt] for dt in data_types])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('performance_by_type.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_algorithm_comparison(results: Dict[Tuple[str, str, int],
                                            float]) -> None:
    """Сравнительный анализ алгоритмов на разных типах данных."""
    algorithms: List[str] = ['bubble', 'selection', 'insertion', 'merge',
                             'quick']
    data_types: List[str] = ['random', 'sorted', 'reversed', 'almost_sorted']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    sizes: List[int] = sorted(set(size for _, _, size in results.keys()))

    for i, data_type in enumerate(data_types):
        ax = axes[i]

        for algo in algorithms:
            times: List[float] = []
            for size in sizes:
                key = (algo, data_type, size)
                if key in results:
                    times.append(results[key])
                else:
                    times.append(0.0)

            if any(times):
                ax.plot(sizes, times, marker='o', label=algo, linewidth=2)

        ax.set_xlabel('Размер массива')
        ax.set_ylabel('Время (сек)')
        ax.set_title(f'Тип данных: {data_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Запускаем тесты и строим графики
    print("Запуск тестов производительности...")
    results = performance_test.run_performance_tests()

    print("\nПостроение графиков...")
    plot_performance_by_size(results)
    plot_performance_by_data_type(results)
    plot_algorithm_comparison(results)

    print("Все графики сохранены в файлы PNG")
