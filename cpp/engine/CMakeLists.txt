cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")
project(veystrix_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Bit-exactness requires the SAME MLX build as the Python engine (0.32.2).
set(MLX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python3.12/site-packages/mlx" CACHE PATH "MLX install root")
find_package(MLX CONFIG REQUIRED PATHS "${MLX_ROOT}/share/cmake/MLX" NO_DEFAULT_PATH)
message(STATUS "MLX version: ${MLX_VERSION}  include: ${MLX_INCLUDE_DIRS}")

add_library(veystrix_engine STATIC src/engine.cpp src/expert_store.cpp)
target_include_directories(veystrix_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${MLX_INCLUDE_DIRS})
target_link_libraries(veystrix_engine PUBLIC mlx)
set_target_properties(veystrix_engine PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Smoke test: constructs arrays, runs router + MoE on synthetic weights.
add_executable(test_engine_smoke tests/test_engine_smoke.cpp)
target_link_libraries(test_engine_smoke PRIVATE veystrix_engine)
set_target_properties(test_engine_smoke PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Bit-exact parity test against Python-generated vectors.
add_executable(test_engine_parity tests/test_engine_parity.cpp)
target_link_libraries(test_engine_parity PRIVATE veystrix_engine)
set_target_properties(test_engine_parity PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# End-to-end native layer load (no Python feeding weights).
add_executable(test_native_layer tests/test_native_layer.cpp)
target_link_libraries(test_native_layer PRIVATE veystrix_engine)
set_target_properties(test_native_layer PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

enable_testing()
add_test(NAME engine_smoke COMMAND test_engine_smoke)
add_test(NAME engine_parity COMMAND test_engine_parity
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors")
# Real-checkpoint parity (requires gen_real_vectors.py to have been run).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/real/meta.json")
  add_test(NAME engine_parity_real COMMAND test_engine_parity
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/real" --real)
  # End-to-end: C++ loads the real layer from the checkpoint natively.
  add_test(NAME engine_native_layer COMMAND test_native_layer
    "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP/model-00001-of-00022.safetensors"
    "language_model.model.layers.0.mlp.switch_mlp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/real")
endif()
