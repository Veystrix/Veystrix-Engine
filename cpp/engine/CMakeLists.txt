cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")
project(veystrix_engine LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Bit-exactness requires the SAME MLX build as the Python engine (0.32.2).
set(MLX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python3.12/site-packages/mlx" CACHE PATH "MLX install root")
find_package(MLX CONFIG REQUIRED PATHS "${MLX_ROOT}/share/cmake/MLX" NO_DEFAULT_PATH)
message(STATUS "MLX version: ${MLX_VERSION}  include: ${MLX_INCLUDE_DIRS}")

# Native batched pread (compiled as C) + C++ pager, reused from cpp/decode.
# Copy native_read.c into this build so CMake tracks it as a source of this target
# (an absolute path outside the tree is silently dropped by the Makefile generator).
set(NATIVE_READ_C ${CMAKE_CURRENT_BINARY_DIR}/native_read.c)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../models/flashnext/native_read.c
               ${NATIVE_READ_C} COPYONLY)
set(PAGER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../decode)
add_library(veystrix_pager STATIC ${NATIVE_READ_C} ${PAGER_DIR}/src/io_pager.cpp)
target_include_directories(veystrix_pager PUBLIC ${PAGER_DIR}/src)
set_source_files_properties(${NATIVE_READ_C} PROPERTIES LANGUAGE C)

add_library(veystrix_engine STATIC src/engine.cpp src/expert_store.cpp src/gdn_kernel.cpp src/gdn_conv.cpp src/qsa.cpp src/qsa_full.cpp src/sampling.cpp src/model_adapter.cpp src/gdn_layer.cpp src/decoder.cpp src/decoder_full.cpp src/moe_stream.cpp)
target_include_directories(veystrix_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${MLX_INCLUDE_DIRS})
target_link_libraries(veystrix_engine PUBLIC mlx veystrix_pager)
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

# Nemotron relu2 MoE parity.
add_executable(test_relu2_parity tests/test_relu2_parity.cpp)
target_link_libraries(test_relu2_parity PRIVATE veystrix_engine)
set_target_properties(test_relu2_parity PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# GDN delta-rule STEP KERNEL bit-exact parity (production decode path).
add_executable(test_gdn_delta tests/test_gdn_delta.cpp)
target_link_libraries(test_gdn_delta PRIVATE veystrix_engine)
set_target_properties(test_gdn_delta PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Mixed-layer decoder loop (GDN + full attn + shared expert) bit-exact parity.
add_executable(test_decoder_full tests/test_decoder_full.cpp)
target_include_directories(test_decoder_full PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_decoder_full PRIVATE veystrix_engine)
set_target_properties(test_decoder_full PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Streaming MoE with real-checkpoint I/O (expert_store + io_pager).
add_executable(test_moe_stream tests/test_moe_stream.cpp)
target_include_directories(test_moe_stream PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${PAGER_DIR}/src)
target_link_libraries(test_moe_stream PRIVATE veystrix_engine)
set_target_properties(test_moe_stream PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Full multi-layer decoder loop bit-exact parity (real mlx_vlm modules).
add_executable(test_decoder tests/test_decoder.cpp)
target_include_directories(test_decoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_decoder PRIVATE veystrix_engine)
set_target_properties(test_decoder PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Full QSA indexer (RoPE + block scoring + KV cache) bit-exact parity.
add_executable(test_qsa_full tests/test_qsa_full.cpp)
target_include_directories(test_qsa_full PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_qsa_full PRIVATE veystrix_engine)
set_target_properties(test_qsa_full PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Full GDN linear-attention layer bit-exact parity.
add_executable(test_gdn_layer tests/test_gdn_layer.cpp)
target_include_directories(test_gdn_layer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_gdn_layer PRIVATE veystrix_engine)
set_target_properties(test_gdn_layer PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Final head (gated residual no-combine -> lm_head) bit-exact parity.
add_executable(test_final_head tests/test_final_head.cpp)
target_include_directories(test_final_head PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_final_head PRIVATE veystrix_engine)
set_target_properties(test_final_head PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Model-agnostic adapter config verification (Qwen + Nemotron + future).
add_executable(test_model_adapter tests/test_model_adapter.cpp)
target_include_directories(test_model_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_model_adapter PRIVATE veystrix_engine)
set_target_properties(test_model_adapter PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# QSA indexer (pre-RoPE) bit-exact parity.
add_executable(test_qsa tests/test_qsa.cpp)
target_include_directories(test_qsa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_qsa PRIVATE veystrix_engine)
set_target_properties(test_qsa PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Extended (non-greedy) sampling bit-exact parity, incl. seeded categorical draw.
add_executable(test_sampling tests/test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE veystrix_engine)
set_target_properties(test_sampling PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# GDN causal-conv1d decode chain bit-exact parity.
add_executable(test_conv tests/test_conv.cpp)
target_link_libraries(test_conv PRIVATE veystrix_engine)
set_target_properties(test_conv PROPERTIES
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
add_test(NAME engine_relu2 COMMAND test_relu2_parity
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/case3_nemotron_relu2")
# GDN delta-rule kernel vs the exact Python gated_delta_kernel (bit-for-bit).
add_test(NAME gdn_delta COMMAND test_gdn_delta
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/conv/meta.json")
  add_test(NAME gdn_conv COMMAND test_conv
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/conv")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_qsa/cases.json")
  add_test(NAME qsa COMMAND test_qsa
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_qsa")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/sampling_vectors/cases.txt")
  add_test(NAME sampling COMMAND test_sampling
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/sampling_vectors")
endif()
add_test(NAME model_adapter COMMAND test_model_adapter
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/model_adapter_flat")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_final_head/expected.npy")
  add_test(NAME final_head COMMAND test_final_head
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_final_head")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_gdn_layer/meta.json")
  add_test(NAME gdn_layer COMMAND test_gdn_layer
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_gdn_layer")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_qsa_full/cases.json")
  add_test(NAME qsa_full COMMAND test_qsa_full
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_qsa_full")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder/meta.json")
  add_test(NAME decoder COMMAND test_decoder
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_moe_stream/meta.json")
  add_test(NAME moe_stream COMMAND test_moe_stream
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_moe_stream"
           "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder_full/meta.json")
  add_test(NAME decoder_full COMMAND test_decoder_full
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder_full")
endif()
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
