from __future__ import annotations

from typing import Generic, Iterable, List, Optional, TypeVar, Protocol
from typing import runtime_checkable


@runtime_checkable
class SupportsComparison(Protocol):
    """Протокол: тип, поддерживающий операции < и >."""

    def __lt__(self, other: "SupportsComparison", /) -> bool:
        ...

    def __gt__(self, other: "SupportsComparison", /) -> bool:
        ...


T = TypeVar("T", bound=SupportsComparison)


class BinaryHeap(Generic[T]):
    """Массивная реализация бинарной кучи (min-heap или max-heap).

    Элементы хранятся в списке self._data, индексация:
    - родитель: (i - 1) // 2
    - левый потомок: 2 * i + 1
    - правый потомок: 2 * i + 2

    При is_min=True — min-heap, при is_min=False — max-heap.

    Основные операции:
    - insert(value): O(log n) в худшем случае;
    - extract(): O(log n);
    - build_heap(iterable): O(n).

    Хранение в виде массива позволяет эффективно использовать кучу
    как для сортировки (Heapsort), так и для приоритетной очереди.
    """

    def __init__(
        self,
        is_min: bool = True,
        data: Optional[Iterable[T]] = None,
    ) -> None:
        """Создаёт пустую кучу или строит кучу из iterable.

        Если data не None, используется алгоритм build_heap с O(n).

        Сложность:
        - без data: O(1);
        - с data: O(n), где n — число элементов.
        """
        self.is_min = is_min
        self._data: List[T] = []

        if data is not None:
            self.build_heap(data)
        # Общая сложность конструктора: O(1) или O(n).

    def __len__(self) -> int:
        """Возвращает количество элементов в куче. O(1)."""
        return len(self._data)

    def is_empty(self) -> bool:
        """Проверяет, пуста ли куча. O(1)."""
        return len(self._data) == 0

    @staticmethod
    def _parent(index: int) -> int:
        """Индекс родителя. O(1)."""
        return (index - 1) // 2

    @staticmethod
    def _left_child(index: int) -> int:
        """Индекс левого потомка. O(1)."""
        return 2 * index + 1

    @staticmethod
    def _right_child(index: int) -> int:
        """Индекс правого потомка. O(1)."""
        return 2 * index + 2

    def _compare(self, a: T, b: T) -> bool:
        """Сравнение с учётом типа кучи.

        Возвращает True, если a должно быть "выше" b в куче.

        Для min-heap: a < b.
        Для max-heap: a > b.

        Сложность: O(1).
        """
        if self.is_min:
            return a < b

        return a > b

    def _sift_up(self, index: int) -> None:
        """Всплытие элемента вверх до восстановления свойства кучи.

        Алгоритм:
        1. Пока есть родитель и текущий элемент "лучше" родителя
           (меньше для min-heap или больше для max-heap),
           меняем их местами.
        2. Обновляем индекс и продолжаем.

        Сложность: O(log n) в худшем случае (по высоте кучи).
        """
        current = index

        while current > 0:
            parent = self._parent(current)
            if not self._compare(self._data[current],
                                 self._data[parent]):
                break

            # Меняем местами ребёнка и родителя.
            self._data[current], self._data[parent] = (
                self._data[parent],
                self._data[current],
            )
            current = parent

        # Общая сложность: O(h) ~ O(log n).

    def _sift_down(self, index: int) -> None:
        """Погружение элемента вниз до восстановления свойства кучи.

        Алгоритм:
        1. Сравниваем элемент с его детьми.
        2. Выбираем "лучшего" ребёнка (min или max).
        3. Если ребёнок "лучше" текущего, меняем их местами и
           продолжаем погружение.

        Сложность: O(log n) в худшем случае.
        """
        size = len(self._data)
        current = index

        while True:
            left = self._left_child(current)
            right = self._right_child(current)
            best = current

            if (
                left < size
                and self._compare(self._data[left], self._data[best])
            ):
                best = left

            if (
                right < size
                and self._compare(self._data[right], self._data[best])
            ):
                best = right

            if best == current:
                break

            self._data[current], self._data[best] = (
                self._data[best],
                self._data[current],
            )
            current = best

        # Общая сложность: O(h) ~ O(log n).

    def insert(self, value: T) -> None:
        """Вставляет элемент в кучу.

        Элемент добавляется в конец массива и всплывает вверх.

        Сложность:
        - амортизированная: O(log n);
        - худший случай: O(log n).
        """
        self._data.append(value)
        self._sift_up(len(self._data) - 1)
        # Общая сложность: O(log n).

    def peek(self) -> T:
        """Возвращает корень кучи без удаления.

        Сложность: O(1).
        """
        if self.is_empty():
            raise IndexError("Heap is empty.")

        return self._data[0]

    def extract(self) -> T:
        """Извлекает корень кучи и возвращает его.

        Алгоритм:
        1. Запоминаем корень.
        2. Перемещаем последний элемент на его место.
        3. Удаляем последний элемент.
        4. Погружаем новый корень вниз (sift_down).

        Сложность: O(log n) в худшем случае.
        """
        if self.is_empty():
            raise IndexError("Cannot extract from an empty heap.")

        root_value = self._data[0]
        last_value = self._data.pop()

        if not self.is_empty():
            self._data[0] = last_value
            self._sift_down(0)

        return root_value
        # Общая сложность: O(log n).

    def build_heap(self, data: Iterable[T]) -> None:
        """Строит кучу из итерируемой последовательности элементов.

        Используется эффективный алгоритм "heapify" (bottom-up):
        - сначала копируем данные;
        - затем вызываем _sift_down для всех внутренних узлов
          от последнего к корню.

        Сложность: O(n), где n — число элементов.
        """
        self._data = list(data)
        size = len(self._data)

        # Последний внутренний узел имеет индекс (size // 2) - 1.
        for index in range(size // 2 - 1, -1, -1):
            self._sift_down(index)
        # Общая сложность: O(n).

    def to_tree_string(self) -> str:
        """Возвращает текстовое представление кучи уровнями.

        Пример (min-heap):

        Уровень 0: 1
        Уровень 1: 3 5
        Уровень 2: 4 8 7 10

        Сложность: O(n).
        """
        if self.is_empty():
            return "<empty heap>"

        result_lines: List[str] = []
        level = 0
        index = 0
        size = len(self._data)

        while index < size:
            level_count = 2 ** level
            level_indices = range(index, min(index + level_count, size))

            values_str = " ".join(
                str(self._data[i]) for i in level_indices
            )

            result_lines.append(f"Уровень {level}: {values_str}")

            index += level_count
            level += 1

        return "\n".join(result_lines)
        # Общая сложность: O(n).


class MinHeap(BinaryHeap[T]):
    """Удобный класс для min-heap (корень — минимум)."""

    def __init__(self, data: Optional[Iterable[T]] = None) -> None:
        """Создаёт min-кучу.

        Сложность: O(1) без data, O(n) с data.
        """
        super().__init__(is_min=True, data=data)


class MaxHeap(BinaryHeap[T]):
    """Удобный класс для max-heap (корень — максимум)."""

    def __init__(self, data: Optional[Iterable[T]] = None) -> None:
        """Создаёт max-кучу.

        Сложность: O(1) без data, O(n) с data.
        """
        super().__init__(is_min=False, data=data)
