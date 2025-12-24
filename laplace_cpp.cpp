#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <tuple>

namespace py = pybind11;

// Функция для расчета граничных условий
double boundary_conditions_cpp(double x, double y, const std::string& side) {
    if (side == "left") {
        return -19.0 * y * y - 17.0 * y + 15.0;
    } else if (side == "right") {
        return -19.0 * y * y - 57.0 * y + 49.0;
    } else if (side == "bottom") {
        return 18.0 * x * x + 16.0 * x + 15.0;
    } else if (side == "top") {
        return 18.0 * x * x - 24.0 * x - 21.0;
    } else {
        throw std::invalid_argument("Неизвестная сторона: " + side);
    }
}

// Метод Гаусса-Зейделя на C++ с возвратом времени
py::tuple solve_laplace_gauss_seidel_cpp(
    double h, 
    double epsilon, 
    int max_iter = 50000
) {
    // Проверка параметров
    if (h <= 0.0) {
        throw std::invalid_argument("Шаг сетки должен быть положительным");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("Точность должна быть положительной");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Размер сетки
    int n = static_cast<int>(1.0 / h);
    int grid_size = n + 1;
    
    // Создание массива numpy
    auto u = py::array_t<double>({grid_size, grid_size});
    auto u_buf = u.request();
    double* u_ptr = static_cast<double*>(u_buf.ptr);
    
    // Инициализация граничных условий
    for (int i = 0; i < grid_size; ++i) {
        double x = i * h;
        u_ptr[i * grid_size + 0] = boundary_conditions_cpp(x, 0.0, "bottom");
        u_ptr[i * grid_size + n] = boundary_conditions_cpp(x, 1.0, "top");
    }
    
    for (int j = 0; j < grid_size; ++j) {
        double y = j * h;
        u_ptr[0 * grid_size + j] = boundary_conditions_cpp(0.0, y, "left");
        u_ptr[n * grid_size + j] = boundary_conditions_cpp(1.0, y, "right");
    }
    
    // Метод Гаусса-Зейделя
    int iteration = 0;
    double error = 1.0;
    
    while (iteration < max_iter && error > epsilon) {
        error = 0.0;
        
        // Обновление внутренних точек
        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < n; ++j) {
                int idx = i * grid_size + j;
                double old_val = u_ptr[idx];
                
                // Формула Гаусса-Зейделя
                u_ptr[idx] = 0.25 * (
                    u_ptr[(i + 1) * grid_size + j] +  // ниже
                    u_ptr[(i - 1) * grid_size + j] +  // выше
                    u_ptr[i * grid_size + (j + 1)] +  // справа
                    u_ptr[i * grid_size + (j - 1)]    // слева
                );
                
                double current_error = std::abs(u_ptr[idx] - old_val);
                if (current_error > error) {
                    error = current_error;
                }
            }
        }
        
        ++iteration;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    double elapsed_time = duration.count();
    
    // Возвращаем кортеж: (решение, итерации, ошибка, время)
    return py::make_tuple(u, iteration, error, elapsed_time);
}

// Функция для запуска всех экспериментов
py::dict run_all_experiments_cpp() {
    std::vector<double> h_values = {0.1, 0.01, 0.005};
    std::vector<double> epsilon_values = {0.1, 0.01, 0.001};
    int max_iter = 50000;
    
    py::dict results_dict;
    
    for (double h : h_values) {
        for (double epsilon : epsilon_values) {
            // Создаем ключ для словаря
            std::string key = "h_" + std::to_string(h) + "_eps_" + std::to_string(epsilon);
            
            // Запускаем решение
            auto result_tuple = solve_laplace_gauss_seidel_cpp(h, epsilon, max_iter);
            
            // Извлекаем результаты из кортежа
            py::tuple result = result_tuple.cast<py::tuple>();
            py::array_t<double> u = result[0].cast<py::array_t<double>>();
            int iterations = result[1].cast<int>();
            double final_error = result[2].cast<double>();
            double elapsed_time = result[3].cast<double>();
            
            // Создаем словарь для этого эксперимента
            py::dict exp_dict;
            exp_dict["h"] = h;
            exp_dict["epsilon"] = epsilon;
            exp_dict["iterations"] = iterations;
            exp_dict["time"] = elapsed_time;
            exp_dict["final_error"] = final_error;
            exp_dict["grid_size"] = static_cast<int>(1.0 / h) + 1;
            
            // Добавляем в общий словарь
            results_dict[py::str(key)] = exp_dict;
            
            std::cout << "C++: h=" << h << ", epsilon=" << epsilon 
          << ": " << iterations << " итераций, " 
          << elapsed_time << " сек, ошибка=" << final_error << std::endl;
        }
    }
    
    return results_dict;
}

// Функция для сравнения с Python (если Python модуль доступен)
py::dict benchmark_cpp_vs_python(py::function python_solver) {
    std::vector<double> h_values = {0.1, 0.01, 0.005};
    std::vector<double> epsilon_values = {0.1, 0.01, 0.001};
    int max_iter = 50000;
    
    py::dict comparison_dict;
    
    for (double h : h_values) {
        for (double epsilon : epsilon_values) {
            // Создаем ключ
            std::string key = "h_" + std::to_string(h) + "_eps_" + std::to_string(epsilon);
            py::dict result_dict;
            
            // C++ версия
            auto start_cpp = std::chrono::high_resolution_clock::now();
            auto cpp_result = solve_laplace_gauss_seidel_cpp(h, epsilon, max_iter);
            auto end_cpp = std::chrono::high_resolution_clock::now();
            auto cpp_duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpp - start_cpp);
            
            py::tuple cpp_tuple = cpp_result.cast<py::tuple>();
            int cpp_iterations = cpp_tuple[1].cast<int>();
            double cpp_error = cpp_tuple[2].cast<double>();
            double cpp_time = cpp_duration.count();
            
            // Python версия (если функция передана)
            double python_time = 0.0;
            int python_iterations = 0;
            double python_error = 0.0;
            
            if (!python_solver.is_none()) {
                try {
                    auto start_python = std::chrono::high_resolution_clock::now();
                    py::tuple python_result = python_solver(h, epsilon, max_iter).cast<py::tuple>();
                    auto end_python = std::chrono::high_resolution_clock::now();
                    auto python_duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_python - start_python);
                    
                    python_time = python_duration.count();
                    python_iterations = python_result[1].cast<int>();
                    python_error = python_result[2].cast<double>();
                } catch (const std::exception& e) {
                    std::cerr << "Ошибка при вызове Python функции: " << e.what() << std::endl;
                }
            }
            
            // Сохраняем результаты
            result_dict["h"] = h;
            result_dict["epsilon"] = epsilon;
            result_dict["grid_size"] = static_cast<int>(1.0 / h) + 1;
            result_dict["cpp_time"] = cpp_time;
            result_dict["cpp_iterations"] = cpp_iterations;
            result_dict["cpp_error"] = cpp_error;
            result_dict["python_time"] = python_time;
            result_dict["python_iterations"] = python_iterations;
            result_dict["python_error"] = python_error;
            
            if (python_time > 0) {
                double speedup = python_time / cpp_time;
                result_dict["speedup"] = speedup;
            }
            
            comparison_dict[py::str(key)] = result_dict;
        }
    }
    
    return comparison_dict;
}

// Модуль PyBind11
PYBIND11_MODULE(laplace_cpp, m) {
    m.doc() = "Решение уравнения Лапласа методом Гаусса-Зейделя на C++";
    
    m.def("solve_laplace_gauss_seidel_cpp", &solve_laplace_gauss_seidel_cpp,
          "Решение уравнения Лапласа методом Гаусса-Зейделя",
          py::arg("h"), py::arg("epsilon"), py::arg("max_iter") = 50000);
    
    m.def("run_all_experiments_cpp", &run_all_experiments_cpp,
          "Запуск всех экспериментов для h=[0.1,0.01,0.005], ε=[0.1,0.01,0.001]");
    
    m.def("benchmark_cpp_vs_python", &benchmark_cpp_vs_python,
          "Сравнение производительности C++ и Python",
          py::arg("python_solver") = py::none());
    
    m.def("boundary_conditions_cpp", &boundary_conditions_cpp,
          "Граничные условия",
          py::arg("x"), py::arg("y"), py::arg("side"));
}