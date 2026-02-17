cmake_minimum_required(VERSION 3.15...3.27)
project(mcpower_native LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find pybind11 — optional so sdist installs without a compiler still succeed
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(WARNING "pybind11 not found — building Python-only package (no C++ backend)")
    return()
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

# Source files
set(MCPOWER_SOURCES
    cpp/src/ols.cpp
    cpp/src/data_generation.cpp
    cpp/src/lme_solver.cpp
    cpp/src/bindings.cpp
)

# Create pybind11 module
pybind11_add_module(mcpower_native ${MCPOWER_SOURCES})

# Link Eigen
target_link_libraries(mcpower_native PRIVATE Eigen3::Eigen)

# Include directories
target_include_directories(mcpower_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
)

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mcpower_native PRIVATE
        -O3
        -ffast-math
        -DNDEBUG
    )
    # Only use -march=native for local builds, not CI/cross-compilation
    if(NOT DEFINED ENV{CIBUILDWHEEL} AND NOT CMAKE_CROSSCOMPILING)
        target_compile_options(mcpower_native PRIVATE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(mcpower_native PRIVATE
        /O2
        /fp:fast
        /DNDEBUG
    )
endif()

# OpenMP support (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mcpower_native PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(mcpower_native PRIVATE MCPOWER_USE_OPENMP)
endif()

# Install the module
install(TARGETS mcpower_native DESTINATION mcpower/backends)
