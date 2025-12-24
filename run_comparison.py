import numpy as np
import time
import matplotlib.pyplot as plt
import os
import json
from laplace_gauss_seidel import (
    solve_laplace_gauss_seidel_pure,
    solve_laplace_gauss_seidel_numba
)

def run_benchmark():
    """
    Бенчмарк для разных размеров сетки
    """
    h_values = [0.1, 0.05, 0.02, 0.01]
    epsilon = 1e-4
    max_iter = 20000
    
    print("\n" + "=" * 100)
    print("БЕНЧМАРК МЕТОДОВ ГАУССА-ЗЕЙДЕЛЯ")
    print("=" * 100)
    
    all_results = {}
    
    for h in h_values:
        print(f"\nРазмер сетки: {int(1/h)+1}x{int(1/h)+1} (h={h})")
        print("-" * 80)
        
        results = {}
        grid_size = int(1/h) + 1
        
        # Python
        start = time.perf_counter()
        _, iter_py, _, _ = solve_laplace_gauss_seidel_pure(h, epsilon, max_iter)
        elapsed_py = time.perf_counter() - start
        results['python'] = elapsed_py
        
        # Numba
        start = time.perf_counter()
        _, iter_numba, _, _ = solve_laplace_gauss_seidel_numba(h, epsilon, max_iter)
        elapsed_numba = time.perf_counter() - start
        results['numba'] = elapsed_numba
        
        print(f"  Python: {elapsed_py:.4f} сек, {iter_py} итераций")
        print(f"  Numba:  {elapsed_numba:.4f} сек, {iter_numba} итераций")
        print(f"  Ускорение: {elapsed_py/elapsed_numba:.2f}x")
        
        all_results[f"grid_{grid_size}"] = results
    
    # Визуализация бенчмарка
    plot_benchmark_results(all_results, h_values)
    
    return all_results

def plot_benchmark_results(all_results, h_values):
    """
    Визуализация результатов бенчмарка
    """
    grid_sizes = [int(1/h) + 1 for h in h_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График времени выполнения
    for method in ['python', 'numba']:
        times = []
        for h, grid_size in zip(h_values, grid_sizes):
            key = f"grid_{grid_size}"
            if method in all_results.get(key, {}):
                times.append(all_results[key][method])
        
        if times:
            if method == 'python':
                ax1.plot(grid_sizes, times, 'ro-', linewidth=2, markersize=8, label='Python')
            elif method == 'numba':
                ax1.plot(grid_sizes, times, 'go-', linewidth=2, markersize=8, label='Numba')
    
    ax1.set_xlabel('Размер сетки')
    ax1.set_ylabel('Время выполнения (сек)')
    ax1.set_title('Время выполнения для разных размеров сетки')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xticks(grid_sizes)
    ax1.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes])
    
    # График ускорения
    speedups_numba = []
    for h, grid_size in zip(h_values, grid_sizes):
        key = f"grid_{grid_size}"
        if 'python' in all_results[key] and 'numba' in all_results[key]:
            speedups_numba.append(all_results[key]['python'] / all_results[key]['numba'])
    
    if speedups_numba:
        ax2.plot(grid_sizes[:len(speedups_numba)], speedups_numba, 'bo-', 
                linewidth=2, markersize=8)
        ax2.set_xlabel('Размер сетки')
        ax2.set_ylabel('Ускорение (раз)')
        ax2.set_title('Ускорение Numba относительно Python')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(grid_sizes[:len(speedups_numba)])
        ax2.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes[:len(speedups_numba)]])
    
    plt.suptitle('Бенчмарк методов Гаусса-Зейделя', fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs('benchmark_results', exist_ok=True)
    plt.savefig('benchmark_results/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    
    # Сохранение данных
    with open('benchmark_results/benchmark_data.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Сравнение методов Гаусса-Зейделя для уравнения Лапласа'
    )
    
    parser.add_argument('--h', type=float, default=0.05,
                       help='Шаг сетки (по умолчанию: 0.05)')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                       help='Точность (по умолчанию: 1e-4)')
    parser.add_argument('--max-iter', type=int, default=20000,
                       help='Максимальное число итераций (по умолчанию: 20000)')
    parser.add_argument('--mode', choices=['single', 'benchmark'],
                       default='benchmark', help='Режим работы')
    
    args = parser.parse_args()
    
    if args.mode == 'benchmark':
        print("Запуск бенчмарка для разных размеров сетки...")
        results = run_benchmark()
        print(f"\nРезультаты бенчмарка сохранены в директории: benchmark_results/")
    else:
        # Одиночный запуск
        print(f"\nРешение для h={args.h}, ε={args.epsilon}")
        print("-" * 50)
        
        # Python
        start = time.perf_counter()
        u_py, iter_py, time_py, error_py = solve_laplace_gauss_seidel_pure(args.h, args.epsilon, args.max_iter)
        elapsed_py = time.perf_counter() - start
        
        # Numba
        start = time.perf_counter()
        u_numba, iter_numba, time_numba, error_numba = solve_laplace_gauss_seidel_numba(args.h, args.epsilon, args.max_iter)
        elapsed_numba = time.perf_counter() - start
        
        print(f"Python: {elapsed_py:.4f} сек, {iter_py} итераций, ошибка: {error_py:.2e}")
        print(f"Numba:  {elapsed_numba:.4f} сек, {iter_numba} итераций, ошибка: {error_numba:.2e}")
        print(f"Ускорение: {elapsed_py/elapsed_numba:.2f}x")

if __name__ == "__main__":
    main()