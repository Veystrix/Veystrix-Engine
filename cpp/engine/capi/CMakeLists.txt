# Standalone build of the C ABI dylib for offline ctypes tests.
#
# The Phase 4 C++ side is deliberately self-contained: it does not include MLX
# headers or link the GPU runtime (see docs/engineering/capi_design.md).  This
# makes the ABI surface, exception safety and ctypes round-trip buildable and
# testable on a CPU-only machine with no checkpoint and no .venv.
#
#   cmake -S cpp/engine/capi -B /tmp/vx-capi-build -DCMAKE_BUILD_TYPE=Release
#   cmake --build /tmp/vx-capi-build --parallel
#
# The production build adds the guarded `veystrix_c` target from
# cpp/engine/CMakeLists.txt (opt in with -DVEYSTRIX_BUILD_CAPI=ON), which links
# the real veystrix_engine static library.
cmake_minimum_required(VERSION 3.16)
project(veystrix_capi LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(veystrix_c SHARED veystrix_c.cpp)
target_include_directories(veystrix_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(veystrix_c PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
