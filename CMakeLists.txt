cmake_minimum_required(VERSION 3.15...3.27)
project(mcpower_native LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find pybind11 — try CONFIG first, then query the Python module as fallback
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fallback: ask the pybind11 Python module where its cmake files live
    # (scikit-build-core may not always add site-packages to CMAKE_PREFIX_PATH on Windows)
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _pybind11_result
        )
        if(_pybind11_result EQUAL 0 AND _pybind11_cmake_dir)
            find_package(pybind11 CONFIG REQUIRED PATHS "${_pybind11_cmake_dir}" NO_DEFAULT_PATH)
        endif()
    endif()
endif()
if(NOT pybind11_FOUND)
    if(DEFINED ENV{CIBUILDWHEEL})
        message(FATAL_ERROR "pybind11 not found in CI — cannot build wheel without native extension")
    else()
        message(WARNING "pybind11 not found — building Python-only package (no C++ backend)")
        return()
    endif()
endif()

# Try to find Eigen3, make it optional
find_package(Eigen3 3.3 CONFIG)
if(NOT Eigen3_FOUND)
    # Try pkg-config as fallback
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 eigen3)
        if(EIGEN3_FOUND)
            set(Eigen3_FOUND TRUE)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIRS}"
            )
        endif()
    endif()
endif()

# If Eigen still not found, fetch it
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen)
    message(STATUS "Eigen3 not found, fetching from source")
endif()

# Fetch Boost.Math (header-only, for statistical distributions)
# Only need headers — skip building Boost.Math's own targets
include(FetchContent)

# Allow FetchContent_Populate for header-only deps (avoid add_subdirectory)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

FetchContent_Declare(
    boost_math
    GIT_REPOSITORY https://github.com/boostorg/math.git
    GIT_TAG boost-1.87.0
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(boost_math)
if(NOT boost_math_POPULATED)
    FetchContent_Populate(boost_math)
endif()

# Fetch LBFGSPP (header-only, for L-BFGS-B optimization)
FetchContent_Declare(
    lbfgspp
    GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
    GIT_TAG v0.3.0
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(lbfgspp)
if(NOT lbfgspp_POPULATED)
    FetchContent_Populate(lbfgspp)
endif()

# Source files
set(MCPOWER_SOURCES
    cpp/src/ols.cpp
    cpp/src/data_generation.cpp
    cpp/src/lme_solver.cpp
    cpp/src/optimizers.cpp
    cpp/src/distributions.cpp
    cpp/src/bindings.cpp
)

# Create pybind11 module
pybind11_add_module(mcpower_native ${MCPOWER_SOURCES})

# Link Eigen
target_link_libraries(mcpower_native PRIVATE Eigen3::Eigen)

# Include directories
target_include_directories(mcpower_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
    ${boost_math_SOURCE_DIR}/include
    ${lbfgspp_SOURCE_DIR}/include
)

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mcpower_native PRIVATE
        -O3
        -fno-math-errno
        -fno-trapping-math
        -DNDEBUG
    )
    # Only use -march=native for local builds, not CI/cross-compilation
    if(NOT DEFINED ENV{CIBUILDWHEEL} AND NOT CMAKE_CROSSCOMPILING)
        target_compile_options(mcpower_native PRIVATE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(mcpower_native PRIVATE
        /O2
        /fp:precise
        /DNDEBUG
    )
    target_compile_definitions(mcpower_native PRIVATE _USE_MATH_DEFINES)
endif()

# OpenMP support (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mcpower_native PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(mcpower_native PRIVATE MCPOWER_USE_OPENMP)
endif()

# Install the module
install(TARGETS mcpower_native DESTINATION mcpower/backends)
