cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")
project(veystrix_decode LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Bit-exactness requires the SAME MLX build as the Python engine (0.32.2).
# Point MLX_ROOT at the venv's installed MLX package.
set(MLX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python3.12/site-packages/mlx" CACHE PATH "MLX install root")
find_package(MLX CONFIG REQUIRED PATHS "${MLX_ROOT}/share/cmake/MLX" NO_DEFAULT_PATH)
message(STATUS "MLX version: ${MLX_VERSION}  include: ${MLX_INCLUDE_DIRS}")

add_executable(mlx_probe tools/mlx_probe.cpp)
target_include_directories(mlx_probe PRIVATE ${MLX_INCLUDE_DIRS})
target_link_libraries(mlx_probe PRIVATE mlx)
set_target_properties(mlx_probe PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib"
  INSTALL_RPATH "${MLX_ROOT}/lib")
