from setuptools import setup, Extension
import pybind11
import sys

# Флаги компиляции
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/arch:AVX2", "/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-march=native", "-ffast-math", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "laplace_cpp",
        ["laplace_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="laplace_solver_cpp",
    version="1.0",
    author="Your Name",
    description="Решение уравнения Лапласа на C++ с PyBind11",
    ext_modules=ext_modules,
    python_requires=">=3.7",
)s