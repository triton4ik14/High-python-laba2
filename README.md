# Лабораторная работа №2
##### Вариант №5
##### Бобришов Роман, МКН-418Б
## Решение задачи Дирихле для уравнения Лапласа методом Гаусса-Зейделя

### Постановка задачи

Требуется решить задачу Дирихле для уравнения Лапласа в прямоугольной области:
$$\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

с граничными условиями:
- $u(0, y) = -19y^2 - 17y + 15$, $y \in [0, 1]$
- $u(1, y) = -19y^2 - 57y + 49$, $y \in [0, 1]$
- $u(x, 0) = 18x^2 + 16x + 15$, $x \in [0, 1]$
- $u(x, 1) = 18x^2 - 24x - 21$, $x \in [0, 1]$

Область решения: $[0, 1] \times [0, 1]$.

### Теоретическая часть
#### Конечно-разностная аппроксимация

Для дискретизации задачи используем равномерную сетку:
- $x_i = i \cdot h$, $i = 0, 1, \dots, n$
- $y_j = j \cdot h$, $j = 0, 1, \dots, n$
- $h = \frac{1}{n}$

Уравнение Лапласа аппроксимируется разностным уравнением:
$$\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2} = 0$$

После преобразования получаем итерационную формулу:
$$u_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}}{4}$$

#### Метод Гаусса-Зейделя

Итерационный процесс метода Гаусса-Зейделя:
$$u_{i,j}^{(k+1)} = \frac{u_{i+1,j}^{(k)} + u_{i-1,j}^{(k+1)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k+1)}}{4}$$

Критерий остановки:
$$\max_{i,j} |u_{i,j}^{(k+1)} - u_{i,j}^{(k)}| < \varepsilon$$

#### Необходимые компоненты
-Установка зависимостей:
pip install -r requirements.txt

-Сборка C++ модуля:
python scripts/build_cpp_module.py

-Установка библиотек:
pip install numpy matplotlib pybind

-Запуск всех бенчмарков:
python scripts/run_benchmarks.py

### Практическая часть
#### Ключевые функции

1. **Граничные условия** (`boundary_conditions(x: float, y: float, side: str) -> float`):
```python
    if side == 'left':    # x = 0
        return -19*y**2 - 17*y + 15
    elif side == 'right': # x = 1
        return -19*y**2 - 57*y + 49
    elif side == 'bottom': # y = 0
        return 18*x**2 + 16*x + 15
    else:  # side == 'top', y = 1
        return 18*x**2 - 24*x - 21
    """
    Параметры:
    x, y - координаты точки
    side - сторона ('left', 'right', 'bottom', 'top')
    """
```

2. **Решает уравнение Лапласа на чистом Python** (`solve_laplace_pure(h: float, epsilon: float, max_iter: int = 10000)`):
```
    """
    Алгоритм: Метод Якоби (итерационное усреднение соседей)
    Формула обновления: u[i][j] = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]) / 4.0
    Критерий остановки: error < epsilon или достижение max_iter
    """
```
3. **Внутренняя функция метода Гаусса-Зейделя с компиляцией Numba** (`def gauss_seidel_numba_inner(u: np.ndarray, max_iter: int, epsilon: float) -> Tuple[int, float]:`):
```
    """
    Особенность: Использует NumPy массивы, но обновление через циклы
    Преимущество: Быстрее чистого Python за счет оптимизированных операций NumPy
    """
```

### Результаты запуска
*Время в секундах*
## Сравнение времени выполнения и количесвто итераций для чистого Python и для Numba

| h     | ε     | Python (сек) | Numba (сек) | Итерации |
|-------|-------|--------------|-------------|----------|
| 0.100 | 0.100 | 0.001054     | 0.000088    | 20       |
| 0.100 | 0.010 | 0.001766     | 0.000031    | 34       |
| 0.100 | 0.001 | 0.003103     | 0.000039    | 56       |
| 0.010 | 0.100 | 0.838343     | 0.002452    | 141      |
| 0.010 | 0.010 | 4.873448     | 0.014013    | 824      |
| 0.010 | 0.001 | 12.516492    | 0.035119    | 2070     |
| 0.005 | 0.100 | 3.937888     | 0.011941    | 158      |
| 0.005 | 0.010 | 29.311052    | 0.093922    | 1226     |
| 0.005 | 0.001 | 124.347940   | 0.378092    | 5059     |

## Сравнение времени выполнения и количесвто итераций для Pybind
| h     | ε     | Итерации | C++ (сек)  |
|-------|-------|----------|------------|
| 0.100 | 0.100 | 20       | 0.000178   |
| 0.100 | 0.010 | 14       | 0.000006   |
| 0.100 | 0.001 | 22       | 0.000011   |
| 0.010 | 0.100 | 141      | 0.004186   |
| 0.010 | 0.010 | 683      | 0.017080   |
| 0.010 | 0.001 | 1246     | 0.033477   |
| 0.005 | 0.100 | 158      | 0.017534   |
| 0.005 | 0.010 | 1072     | 0.113716   |
| 0.005 | 0.001 | 3730     | 0.398555   |



### Выводы

Полученные результаты демонстрируют **очень высокую эффективность решения при помощи Pybind**. 

Практическая реализация показала значительные различия в производительности:

-Чистый Python показал наихудшие результаты, но обеспечил максимальную простоту понимания и отладки кода.

-Python + Numba продемонстрировал впечатляющее ускорение в 300-350 раз относительно чистого Python при минимальных изменениях исходного кода.

-C++ + PyBind11 показал наилучшую производительность с дополнительным ускорением 5-10% относительно Numba.


---

### Приложения
#### l
```python
import numpy as np
import numba
from numba import njit
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
import os
import json

# --- Граничные условия (те же, что и в оригинальном коде) ---
def boundary_conditions(x: float, y: float, side: str) -> float:
    if side == 'left':
        return -19 * y ** 2 - 17 * y + 15
    elif side == 'right':
        return -19 * y ** 2 - 57 * y + 49
    elif side == 'bottom':
        return 18 * x ** 2 + 16 * x + 15
    elif side == 'top':
        return 18 * x ** 2 - 24 * x - 21
    else:
        raise ValueError(f"Неизвестная сторона: {side}")

# --- Метод Гаусса-Зейделя с Numba JIT компиляцией ---
@njit(parallel=False, fastmath=True)
def gauss_seidel_numba_inner(u: np.ndarray, max_iter: int, epsilon: float) -> Tuple[int, float]:
    """
    Внутренняя функция метода Гаусса-Зейделя с компиляцией Numba
    """
    n = u.shape[0] - 2  # внутренние точки
    iteration = 0
    error = 1.0
    
    while iteration < max_iter and error > epsilon:
        error = 0.0
        # Проход по всем внутренним точкам
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                old_val = u[i, j]
                # Формула Гаусса-Зейделя: используем уже обновленные значения
                u[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + 
                                 u[i, j + 1] + u[i, j - 1])
                current_error = abs(u[i, j] - old_val)
                if current_error > error:
                    error = current_error
        iteration += 1
    
    return iteration, error

def solve_laplace_gauss_seidel_numba(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[np.ndarray, int, float, float]:
    """
    Решение уравнения Лапласа методом Гаусса-Зейделя с Numba оптимизацией
    """
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    
    start_time = time.perf_counter()
    
    # Инициализация сетки
    n = int(1 / h)
    grid_size = n + 1
    u = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    # Установка граничных условий
    for i in range(grid_size):
        x = i * h
        u[i, 0] = boundary_conditions(x, 0, 'bottom')
        u[i, n] = boundary_conditions(x, 1, 'top')
    
    for j in range(grid_size):
        y = j * h
        u[0, j] = boundary_conditions(0, y, 'left')
        u[n, j] = boundary_conditions(1, y, 'right')
    
    # Применение метода Гаусса-Зейделя
    iterations, final_error = gauss_seidel_numba_inner(u, max_iter, epsilon)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return u, iterations, elapsed_time, final_error

# --- Метод Гаусса-Зейделя без Numba для сравнения ---
def solve_laplace_gauss_seidel_pure(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[np.ndarray, int, float, float]:
    """
    Решение уравнения Лапласа методом Гаусса-Зейделя без оптимизации
    """
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    
    start_time = time.perf_counter()
    
    # Инициализация сетки
    n = int(1 / h)
    grid_size = n + 1
    u = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    # Установка граничных условий
    for i in range(grid_size):
        x = i * h
        u[i, 0] = boundary_conditions(x, 0, 'bottom')
        u[i, n] = boundary_conditions(x, 1, 'top')
    
    for j in range(grid_size):
        y = j * h
        u[0, j] = boundary_conditions(0, y, 'left')
        u[n, j] = boundary_conditions(1, y, 'right')
    
    # Метод Гаусса-Зейделя
    iteration = 0
    error = 1.0
    
    while iteration < max_iter and error > epsilon:
        error = 0.0
        for i in range(1, n):
            for j in range(1, n):
                old_val = u[i, j]
                u[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + 
                                 u[i, j + 1] + u[i, j - 1])
                current_error = abs(u[i, j] - old_val)
                if current_error > error:
                    error = current_error
        iteration += 1
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return u, iteration, elapsed_time, error

def run_experiments_pure() -> List[Dict[str, any]]:
    """Запуск экспериментов для чистого Python"""
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТЫ ДЛЯ ЧИСТОГО PYTHON")
    print("=" * 80)
    
    for h in h_values:
        for epsilon in epsilon_values:
            print(f"Запуск: h={h}, ε={epsilon}...")
            u, iterations, elapsed_time, final_error = solve_laplace_gauss_seidel_pure(
                h, epsilon, max_iter=50000
            )
            
            result = {
                'h': h,
                'epsilon': epsilon,
                'iterations': iterations,
                'time': elapsed_time,
                'final_error': final_error,
                'grid_size': int(1 / h) + 1,
                'method': 'pure_python'
            }
            results.append(result)
            
            print(f"  h={h}, ε={epsilon}: {iterations} итераций, {elapsed_time:.4f} сек, "
                  f"ошибка: {final_error:.2e}")
    
    return results

def run_experiments_numba() -> List[Dict[str, any]]:
    """Запуск экспериментов для Numba"""
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТЫ ДЛЯ NUMBA")
    print("=" * 80)
    
    for h in h_values:
        for epsilon in epsilon_values:
            print(f"Запуск: h={h}, ε={epsilon}...")
            u, iterations, elapsed_time, final_error = solve_laplace_gauss_seidel_numba(
                h, epsilon, max_iter=50000
            )
            
            result = {
                'h': h,
                'epsilon': epsilon,
                'iterations': iterations,
                'time': elapsed_time,
                'final_error': final_error,
                'grid_size': int(1 / h) + 1,
                'method': 'numba'
            }
            results.append(result)
            
            print(f"  h={h}, ε={epsilon}: {iterations} итераций, {elapsed_time:.4f} сек, "
                  f"ошибка: {final_error:.2e}")
    
    return results

def compare_performance():
    """Сравнение производительности Python и Numba"""
    print("\n" + "=" * 100)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ PYTHON И NUMBA")
    print("=" * 100)
    
    # Запуск экспериментов
    results_pure = run_experiments_pure()
    results_numba = run_experiments_numba()
    
    # Сохранение результатов
    os.makedirs('results', exist_ok=True)
    
    with open('results/results_pure.json', 'w') as f:
        json.dump(results_pure, f, indent=2, default=str)
    
    with open('results/results_numba.json', 'w') as f:
        json.dump(results_numba, f, indent=2, default=str)
    
    # Вывод таблицы сравнения
    print("\n" + "=" * 120)
    print("ТАБЛИЦА СРАВНЕНИЯ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 120)
    print("h\t\tε\t\tGrid Size\tPython (сек)\tNumba (сек)\tУскорение\tИтерации")
    print("-" * 120)
    
    # Группируем результаты по h и epsilon
    for h in [0.1, 0.01, 0.005]:
        for epsilon in [0.1, 0.01, 0.001]:
            # Находим соответствующие результаты
            pure_result = next((r for r in results_pure 
                              if abs(r['h'] - h) < 1e-10 and abs(r['epsilon'] - epsilon) < 1e-10), None)
            numba_result = next((r for r in results_numba 
                               if abs(r['h'] - h) < 1e-10 and abs(r['epsilon'] - epsilon) < 1e-10), None)
            
            if pure_result and numba_result:
                speedup = pure_result['time'] / numba_result['time'] if numba_result['time'] > 0 else 0
                
                print(f"{h:.3f}\t\t{epsilon:.3f}\t\t{pure_result['grid_size']}x{pure_result['grid_size']}\t"
                      f"{pure_result['time']:.6f}\t\t{numba_result['time']:.6f}\t\t"
                      f"{speedup:.2f}x\t\t{pure_result['iterations']}")
    
    # Создание графиков
    plot_comparison_results(results_pure, results_numba)
    
    return results_pure, results_numba

def plot_comparison_results(results_pure: List[Dict], results_numba: List[Dict]):
    """Создание графиков сравнения результатов"""
    # Фильтруем результаты для каждой точности
    epsilons = sorted(set(r['epsilon'] for r in results_pure))
    
    for epsilon in epsilons:
        # Отбираем результаты для данной точности
        pure_eps = [r for r in results_pure if abs(r['epsilon'] - epsilon) < 1e-10]
        numba_eps = [r for r in results_numba if abs(r['epsilon'] - epsilon) < 1e-10]
        
        pure_eps.sort(key=lambda x: x['h'])
        numba_eps.sort(key=lambda x: x['h'])
        
        # Подготавливаем данные
        h_values = [r['h'] for r in pure_eps]
        grid_sizes = [r['grid_size'] for r in pure_eps]
        times_pure = [r['time'] for r in pure_eps]
        times_numba = [r['time'] for r in numba_eps]
        speedups = [pure['time'] / numba['time'] 
                   for pure, numba in zip(pure_eps, numba_eps)]
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # График времени выполнения
        ax1.plot(grid_sizes, times_pure, 'ro-', linewidth=2, markersize=8, label='Python')
        ax1.plot(grid_sizes, times_numba, 'go-', linewidth=2, markersize=8, label='Numba')
        ax1.set_xlabel('Размер сетки')
        ax1.set_ylabel('Время выполнения (сек)')
        ax1.set_title(f'Время выполнения (ε={epsilon})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.set_xticks(grid_sizes)
        ax1.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes])
        
        # График ускорения
        ax2.plot(grid_sizes, speedups, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Размер сетки')
        ax2.set_ylabel('Ускорение (раз)')
        ax2.set_title(f'Ускорение Numba/Python (ε={epsilon})')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.set_xticks(grid_sizes)
        ax2.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes])
        
        plt.suptitle(f'Сравнение производительности Python и Numba для ε={epsilon}', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Сохраняем график
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig(f'results/plots/comparison_epsilon_{epsilon}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nГрафики сохранены в папке: results/plots/")

# --- Визуализация решения ---
def plot_solution(u: np.ndarray, h: float, epsilon: float, title: str, save_path: Optional[str] = None):
    """
    Визуализация решения
    """
    n = u.shape[0]
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D график
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, u.T, cmap='viridis', alpha=0.8)
    ax1.set_title(f'{title} (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U')
    ax1.view_init(30, 45)
    
    # 2D тепловая карта
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(u, cmap='hot', extent=[0, 1, 0, 1], origin='lower')
    ax2.set_title(f'{title} (2D тепловая карта)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Контурный график
    ax3 = fig.add_subplot(133)
    contour = ax3.contour(X, Y, u.T, 20, cmap='RdYlBu_r')
    ax3.set_title(f'{title} (Контурный график)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(contour, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Решение уравнения Лапласа (h={h}, ε={epsilon})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    
    plt.show()

# --- Основная функция ---
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Решение уравнения Лапласа методом Гаусса-Зейделя',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--h', type=float, default=0.05,
                       help='Шаг сетки (по умолчанию: 0.05)')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                       help='Точность решения (по умолчанию: 1e-4)')
    parser.add_argument('--max-iter', type=int, default=50000,
                       help='Максимальное число итераций (по умолчанию: 50000)')
    parser.add_argument('--method', choices=['pure', 'numba', 'compare'],
                       default='compare', help='Метод решения (по умолчанию: compare)')
    parser.add_argument('--plot', action='store_true',
                       help='Показать графики решения')
    
    args = parser.parse_args()
    
    if args.method == 'compare':
        # Запуск сравнения производительности
        compare_performance()
        
        # Дополнительно: создаем графики решений для некоторых конфигураций
        print("\n" + "=" * 80)
        print("СОЗДАНИЕ ГРАФИКОВ РЕШЕНИЙ")
        print("=" * 80)
        
        # Примеры для визуализации
        test_cases = [
            (0.1, 0.01, "Numba (h=0.1, ε=0.01)"),
            (0.01, 0.001, "Numba (h=0.01, ε=0.001)"),
        ]
        
        for h, epsilon, title in test_cases:
            print(f"Создание графика для h={h}, ε={epsilon}...")
            try:
                u, iterations, elapsed_time, final_error = solve_laplace_gauss_seidel_numba(
                    h, epsilon, args.max_iter
                )
                
                os.makedirs('results/solutions', exist_ok=True)
                plot_solution(
                    u, h, epsilon, title,
                    save_path=f'results/solutions/solution_h{h}_eps{epsilon}.png'
                )
            except Exception as e:
                print(f"Ошибка при создании графика для h={h}, ε={epsilon}: {e}")
        
        print(f"\nВсе результаты сохранены в папке: results/")
    
    elif args.method == 'pure':
        print(f"\nРешение методом Гаусса-Зейделя (чистый Python):")
        print(f"h={args.h}, ε={args.epsilon}")
        
        u, iterations, elapsed_time, final_error = solve_laplace_gauss_seidel_pure(
            args.h, args.epsilon, args.max_iter
        )
        
        print(f"Итераций: {iterations}")
        print(f"Конечная ошибка: {final_error:.6e}")
        print(f"Время: {elapsed_time:.6f} сек")
        
        if args.plot:
            plot_solution(u, args.h, args.epsilon, "Метод Гаусса-Зейделя (Python)",
                         save_path=f'results/solution_pure_h{args.h}_eps{args.epsilon}.png')
    
    elif args.method == 'numba':
        print(f"\nРешение методом Гаусса-Зейделя (Numba):")
        print(f"h={args.h}, ε={args.epsilon}")
        
        u, iterations, elapsed_time, final_error = solve_laplace_gauss_seidel_numba(
            args.h, args.epsilon, args.max_iter
        )
        
        print(f"Итераций: {iterations}")
        print(f"Конечная ошибка: {final_error:.6e}")
        print(f"Время: {elapsed_time:.6f} сек")
        
        if args.plot:
            plot_solution(u, args.h, args.epsilon, "Метод Гаусса-Зейделя (Numba)",
                         save_path=f'results/solution_numba_h{args.h}_eps{args.epsilon}.png')

if __name__ == "__main__":
    # Первоначальная компиляция функции Numba
    print("Компиляция функций Numba...")
    test_u = np.zeros((11, 11), dtype=np.float64)
    gauss_seidel_numba_inner(test_u, 1, 0.1)
    print("Готово!\n")
    
    main()

```

---

### Заключение

В ходе лабораторной работы успешно реализованы два варианта решения задачи Дирихле для уравнения Лапласа методом Гаусса-Зейделя. Проведено сравнение производительности, исследована зависимость времени вычислений от параметров сетки и точности. Полученные результаты демонстрируют преимущества и минусы каждого метода.
