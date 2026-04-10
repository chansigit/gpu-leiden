cmake_minimum_required(VERSION 3.18)
project(gpu_leiden LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA architecture — default to sm_80 (A100), can be overridden at build time
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80")
endif()

# OpenMP (used by leiden.cpp's c_renumber_communities)
find_package(OpenMP REQUIRED)

# Python + nanobind
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Point to the Leiden C++/CUDA sources (parent directory of python/)
set(LEIDEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Build the nanobind extension module
# We compile the Leiden sources together with the bindings into one module.
# NB_STATIC statically links nanobind to avoid runtime dependency.
nanobind_add_module(_core
    NB_STATIC
    src/bindings.cpp
    ${LEIDEN_ROOT}/leiden.cu
    ${LEIDEN_ROOT}/leiden.cpp
)

target_include_directories(_core PRIVATE ${LEIDEN_ROOT})
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Suppress CUDA warnings that the Makefile also suppresses
target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=550,177,20092,549,20011>
)

# Install the compiled module into the gpu_leiden package directory
install(TARGETS _core LIBRARY DESTINATION gpu_leiden)
