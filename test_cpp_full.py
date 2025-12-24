import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os

try:
    import laplace_cpp
    print("✓ C++ модуль успешно загружен!")
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"✗ Ошибка загрузки C++ модуля: {e}")
    print("Сначала нужно собрать модуль:")
    print("python setup.py build_ext --inplace")
    CPP_AVAILABLE = False

def run_cpp_experiments():
    """Запуск всех экспериментов на C++"""
    if not CPP_AVAILABLE:
        print("C++ модуль не доступен!")
        return None
    
    print("\n" + "=" * 80)
    print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ НА C++")
    print("=" * 80)
    print("h = [0.1, 0.01, 0.005]")
    print("epsilon = [0.1, 0.01, 0.001]")
    print("-" * 80)
    
    # Запускаем все эксперименты
    start_time = time.time()
    results_dict = laplace_cpp.run_all_experiments_cpp()
    total_time = time.time() - start_time
    
    # Конвертируем в Python dict для удобства
    results = {}
    for key in results_dict:
        exp = results_dict[key]
        h = exp["h"]
        epsilon = exp["epsilon"]
        
        if h not in results:
            results[h] = {}
        
        results[h][epsilon] = {
            'iterations': exp["iterations"],
            'time': exp["time"],
            'final_error': exp["final_error"],
            'grid_size': exp["grid_size"]
        }
    
    # Выводим таблицу
    print("\nРЕЗУЛЬТАТЫ C++:")
    print("-" * 100)
    print("h\t\tepsilon\t\tGrid Size\tИтерации\tВремя (сек)\tОшибка")
    print("-" * 100)
    
    for h in [0.1, 0.01, 0.005]:
        for epsilon in [0.1, 0.01, 0.001]:
            if h in results and epsilon in results[h]:
                r = results[h][epsilon]
                print(f"{h:.3f}\t\t{epsilon:.3f}\t\t{r['grid_size']}x{r['grid_size']}\t"
                      f"{r['iterations']}\t\t{r['time']:.6f}\t\t{r['final_error']:.2e}")
    
    print("-" * 100)
    print(f"Общее время выполнения всех экспериментов: {total_time:.2f} сек")
    
    # Сохраняем результаты
    os.makedirs('cpp_results', exist_ok=True)
    
    # Сохраняем в JSON
    with open('cpp_results/cpp_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Сохраняем в текстовый файл с правильной кодировкой
    with open('cpp_results/cpp_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ C++ ЭКСПЕРИМЕНТОВ\n")
        f.write("=" * 80 + "\n")
        f.write(f"Общее время: {total_time:.2f} сек\n\n")
        
        for h in sorted(results.keys()):
            for epsilon in sorted(results[h].keys()):
                r = results[h][epsilon]
                f.write(f"h={h}, epsilon={epsilon}:\n")
                f.write(f"  Сетка: {r['grid_size']}x{r['grid_size']}\n")
                f.write(f"  Итераций: {r['iterations']}\n")
                f.write(f"  Время: {r['time']:.6f} сек\n")
                f.write(f"  Ошибка: {r['final_error']:.2e}\n")
                f.write("-" * 40 + "\n")
    
    print(f"\nРезультаты сохранены в папке: cpp_results/")
    
    return results

def compare_with_python():
    """Сравнение C++ с Python реализацией"""
    if not CPP_AVAILABLE:
        return None
    
    # Импортируем Python функцию
    try:
        from laplace_gauss_seidel import solve_laplace_gauss_seidel_pure
        
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ C++ И PYTHON")
        print("=" * 80)
        
        # Тестовые параметры
        test_cases = [
            (0.1, 0.01),
            (0.01, 0.001),
            (0.005, 0.001)
        ]
        
        comparison_results = []
        
        for h, epsilon in test_cases:
            print(f"\nh={h}, epsilon={epsilon}")
            print("-" * 60)
            
            # Python
            print("Python:")
            start = time.perf_counter()
            u_py, iter_py, time_py, error_py = solve_laplace_gauss_seidel_pure(h, epsilon, 50000)
            elapsed_py = time.perf_counter() - start
            print(f"  Итераций: {iter_py}")
            print(f"  Время: {elapsed_py:.4f} сек")
            print(f"  Ошибка: {error_py:.2e}")
            
            # C++
            print("\nC++:")
            start = time.perf_counter()
            u_cpp, iter_cpp, error_cpp, elapsed_cpp = laplace_cpp.solve_laplace_gauss_seidel_cpp(h, epsilon, 50000)
            elapsed_cpp = time.perf_counter() - start
            print(f"  Итераций: {iter_cpp}")
            print(f"  Время: {elapsed_cpp:.4f} сек")
            print(f"  Ошибка: {error_cpp:.2e}")
            
            # Сравнение
            speedup = elapsed_py / elapsed_cpp
            print(f"\nУскорение C++: {speedup:.2f}x")
            
            # Проверка точности
            if u_py.shape == u_cpp.shape:
                diff = np.max(np.abs(u_py - u_cpp))
                print(f"Максимальная разница решений: {diff:.2e}")
            
            comparison_results.append({
                'h': h,
                'epsilon': epsilon,
                'grid_size': int(1/h) + 1,
                'python_time': elapsed_py,
                'cpp_time': elapsed_cpp,
                'speedup': speedup,
                'python_iterations': iter_py,
                'cpp_iterations': iter_cpp
            })
        
        # Сохраняем сравнение
        os.makedirs('comparison_results', exist_ok=True)
        with open('comparison_results/cpp_vs_python.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Визуализация
        plot_comparison(comparison_results)
        
        return comparison_results
        
    except ImportError as e:
        print(f"Не удалось импортировать Python модуль: {e}")
        return None

def plot_comparison(results):
    """Визуализация сравнения"""
    if not results:
        return
    
    # Подготавливаем данные
    grid_sizes = [r['grid_size'] for r in results]
    python_times = [r['python_time'] for r in results]
    cpp_times = [r['cpp_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График времени выполнения
    x = np.arange(len(grid_sizes))
    width = 0.35
    
    ax1.bar(x - width/2, python_times, width, label='Python', color='red', alpha=0.7)
    ax1.bar(x + width/2, cpp_times, width, label='C++', color='blue', alpha=0.7)
    ax1.set_xlabel('Размер сетки')
    ax1.set_ylabel('Время выполнения (сек)')
    ax1.set_title('Время выполнения')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for i, (py_time, cpp_time) in enumerate(zip(python_times, cpp_times)):
        ax1.text(i - width/2, py_time, f'{py_time:.2f}', 
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, cpp_time, f'{cpp_time:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # График ускорения
    ax2.bar(x, speedups, color='green', alpha=0.7)
    ax2.set_xlabel('Размер сетки')
    ax2.set_ylabel('Ускорение (раз)')
    ax2.set_title('Ускорение C++ относительно Python')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # Добавляем значения
    for i, speedup in enumerate(speedups):
        ax2.text(i, speedup, f'{speedup:.2f}x', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Сравнение производительности C++ и Python', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('comparison_results/cpp_vs_python_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_solutions():
    """Визуализация решений для некоторых случаев"""
    if not CPP_AVAILABLE:
        return
    
    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ РЕШЕНИЙ")
    print("=" * 80)
    
    # Выбираем несколько случаев для визуализации
    visualization_cases = [
        (0.1, 0.01, "Маленькая сетка (11x11)"),
        (0.01, 0.001, "Средняя сетка (101x101)"),
    ]
    
    os.makedirs('visualization', exist_ok=True)
    
    for h, epsilon, title in visualization_cases:
        try:
            print(f"\nСоздание визуализации: {title}")
            print(f"h={h}, epsilon={epsilon}")
            
            # Получаем решение
            u, iterations, error, elapsed_time = laplace_cpp.solve_laplace_gauss_seidel_cpp(h, epsilon)
            
            print(f"  Итераций: {iterations}")
            print(f"  Ошибка: {error:.2e}")
            print(f"  Время: {elapsed_time:.4f} сек")
            
            # Создаем график
            plt.figure(figsize=(12, 5))
            
            # 2D тепловая карта
            plt.subplot(1, 2, 1)
            im = plt.imshow(u, cmap='hot', extent=[0, 1, 0, 1], origin='lower')
            plt.colorbar(im, label='Значение U', fraction=0.046, pad=0.04)
            plt.title(f'Тепловая карта\n{title}')
            plt.xlabel('X')
            plt.ylabel('Y')
            
            # 3D поверхность (для маленьких сеток)
            if u.shape[0] <= 101:  # Ограничиваем для больших сеток
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(1, 2, 2, projection='3d')
                
                n = u.shape[0]
                x = np.linspace(0, 1, n)
                y = np.linspace(0, 1, n)
                X, Y = np.meshgrid(x, y)
                
                surf = ax.plot_surface(X, Y, u.T, cmap='viridis', alpha=0.8)
                ax.set_title('3D поверхность')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('U')
                ax.view_init(30, 45)
                plt.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
            
            plt.suptitle(f'Решение уравнения Лапласа (C++)\nh={h}, epsilon={epsilon}, {iterations} итераций', 
                        fontsize=12, y=1.05)
            plt.tight_layout()
            
            # Сохраняем
            filename = f'visualization/solution_h{h}_eps{epsilon}.png'
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"  График сохранен: {filename}")
            
        except Exception as e:
            print(f"  Ошибка при создании визуализации: {e}")

def main():
    """Основная функция"""
    print("ПРОВЕРКА РАБОТЫ C++ МОДУЛЯ С PyBind11")
    print("=" * 50)
    
    if not CPP_AVAILABLE:
        return
    
    # 1. Запускаем все эксперименты
    cpp_results = run_cpp_experiments()
    
    # 2. Сравниваем с Python (если доступно)
    comparison = compare_with_python()
    
    # 3. Создаем визуализации
    visualize_solutions()
    
    print("\n" + "=" * 80)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("=" * 80)
    
    if cpp_results:
        print("\nКраткая сводка по C++ экспериментам:")
        print("-" * 60)
        
        # Считаем среднее ускорение
        if comparison:
            avg_speedup = np.mean([r['speedup'] for r in comparison])
            print(f"Среднее ускорение C++ относительно Python: {avg_speedup:.2f}x")
        
        # Самый быстрый и самый медленный случай
        times = []
        for h in cpp_results:
            for epsilon in cpp_results[h]:
                times.append((h, epsilon, cpp_results[h][epsilon]['time']))
        
        if times:
            fastest = min(times, key=lambda x: x[2])
            slowest = max(times, key=lambda x: x[2])
            
            print(f"\nСамый быстрый случай:")
            print(f"  h={fastest[0]}, epsilon={fastest[1]}: {fastest[2]:.4f} сек")
            
            print(f"\nСамый медленный случай:")
            print(f"  h={slowest[0]}, epsilon={slowest[1]}: {slowest[2]:.4f} сек")
        
        print(f"\nВсе результаты сохранены в папках:")
        print("  - cpp_results/ - результаты C++ экспериментов")
        print("  - comparison_results/ - сравнение с Python")
        print("  - visualization/ - графики решений")

if __name__ == "__main__":
    main()