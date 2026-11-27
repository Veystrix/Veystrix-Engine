cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "")
project(veystrix_engine LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
get_filename_component(VEYSTRIX_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_compile_options("-ffile-prefix-map=${VEYSTRIX_REPO_ROOT}=.")

# Bit-exactness requires the SAME MLX build as the Python engine (0.32.2).
set(MLX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python3.12/site-packages/mlx" CACHE PATH "MLX install root")
find_package(MLX CONFIG REQUIRED PATHS "${MLX_ROOT}/share/cmake/MLX" NO_DEFAULT_PATH)
message(STATUS "MLX version: ${MLX_VERSION}  include: ${MLX_INCLUDE_DIRS}")

# Native batched pread (compiled as C) + C++ pager, reused from cpp/decode.
# Copy native_read.c into this build so CMake tracks it as a source of this target
# (an absolute path outside the tree is silently dropped by the Makefile generator).
set(NATIVE_READ_C ${CMAKE_CURRENT_BINARY_DIR}/native_read.c)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../decode/src/native_read.c
               ${NATIVE_READ_C} COPYONLY)
set(PAGER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../decode)
add_library(veystrix_pager STATIC ${NATIVE_READ_C} ${PAGER_DIR}/src/io_pager.cpp ${PAGER_DIR}/src/io_pager_async.cpp)
target_include_directories(veystrix_pager PUBLIC ${PAGER_DIR}/src)
set_source_files_properties(${NATIVE_READ_C} PROPERTIES LANGUAGE C)

add_library(veystrix_engine STATIC src/engine.cpp src/mamba_prefill.cpp src/nemo_snapshot.cpp src/sampler_json.cpp src/expert_store.cpp src/gdn_kernel.cpp src/mamba_kernel.cpp src/gdn_conv.cpp src/mamba_conv.cpp src/mamba_layer.cpp src/qsa.cpp src/qsa_full.cpp src/sampling.cpp src/prefix_cache.cpp src/model_adapter.cpp src/gdn_layer.cpp src/decoder.cpp src/decoder_full.cpp src/parity_gaps.cpp src/rope_fused.cpp src/ple.cpp src/prefill.cpp src/nemo_attention.cpp src/kv_quant.cpp src/nemo_pool.cpp src/shared_expert_pool.cpp src/session_state.cpp src/admission.cpp src/ms_scheduler.cpp src/layer_cursor.cpp src/memory_budget.cpp src/gpu_keepwarm.cpp src/row_fetch.cpp src/gpu_gate.cpp)
target_include_directories(veystrix_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${MLX_INCLUDE_DIRS})
target_link_libraries(veystrix_engine PUBLIC mlx veystrix_pager)
set_target_properties(veystrix_engine PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Phase 4 / A2: narrow C ABI shared library. Opt-in so existing builds are
# unchanged. NOTE: the C++ side is currently a documented stub (default
# vx_engine_open -> VX_ERR_NOT_WIRED) plus a CPU-only fake behind
# VEYSTRIX_C_CPP_FAKE; it does not yet drive veystrix_engine. Linking it here
# reserves the production seam without claiming a GPU wiring that does not
# exist. The standalone CPU-only build lives in capi/CMakeLists.txt.
option(VEYSTRIX_BUILD_CAPI "Build the narrow C ABI shared library" OFF)
if(VEYSTRIX_BUILD_CAPI)
  # Phase G: each live runtime TU includes its tool's generator with main()
  # renamed and owns one MLX driver thread. Both TUs are compiled only into the
  # C ABI dylib so other tools keep their original binary size.
  add_library(veystrix_c SHARED
    capi/veystrix_c.cpp
    src/nemotron_engine_runtime.cpp src/nemotron_runtime.cpp
    src/nemo_snapshot.cpp src/prefix_cache.cpp)
  target_compile_definitions(veystrix_c PRIVATE VEYSTRIX_CAPI_LIVE=1)
  target_include_directories(veystrix_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/capi
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${PAGER_DIR}/src
    ${MLX_INCLUDE_DIRS})
  target_link_libraries(veystrix_c PRIVATE veystrix_engine)
  target_compile_options(veystrix_c PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
  set_target_properties(veystrix_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")
  if(APPLE)
    target_link_options(veystrix_c PRIVATE
      "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/capi/veystrix_c.exports")
  endif()
endif()

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

# Mamba/SSM fused STEP KERNEL bit-exact parity (production decode path).
add_executable(test_mamba_kernel tests/test_mamba_kernel.cpp)
target_include_directories(test_mamba_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_mamba_kernel PRIVATE veystrix_engine)
set_target_properties(test_mamba_kernel PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Full Mamba-2 layer bit-exact parity (production decode path).
add_executable(test_mamba_layer tests/test_mamba_layer.cpp)
target_include_directories(test_mamba_layer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_mamba_layer PRIVATE veystrix_engine)
set_target_properties(test_mamba_layer PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Fused-Metal RoPE (0 ULP vs production) + GDN prefill S>1.
add_executable(test_rope_fused tests/test_rope_fused.cpp)
target_include_directories(test_rope_fused PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_rope_fused PRIVATE veystrix_engine)
set_target_properties(test_rope_fused PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

add_executable(test_prefill tests/test_prefill.cpp)
target_include_directories(test_prefill PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_prefill PRIVATE veystrix_engine)
set_target_properties(test_prefill PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Remaining parity gaps (quantized shared expert + indexer-None fallback).
add_executable(test_parity_gaps tests/test_parity_gaps.cpp)
target_include_directories(test_parity_gaps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_parity_gaps PRIVATE veystrix_engine)
set_target_properties(test_parity_gaps PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Decisive: standalone C++ end-to-end token generation (no Python in loop).
add_executable(native_generate tools/veystrix_native_generate.cpp)
target_include_directories(native_generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(native_generate PRIVATE veystrix_engine)
set_target_properties(native_generate PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Real-checkpoint Nemotron 3.5 Lightning C++ greedy decode tool. Gated by a
# Python reference under real_run_guard.py (no ctest: real checkpoint required).
add_executable(nemotron_generate tools/nemotron_generate.cpp)
target_include_directories(nemotron_generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nemotron_generate PRIVATE veystrix_engine)
set_target_properties(nemotron_generate PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Phase F (both-model parity): include-with-renamed-main TU that provides the
# N=1 per-layer cursor gate and the N-session MsScheduler driver for Nemotron.
# Includes nemotron_generate.cpp, so it needs the tools + pager include dirs.
add_executable(nemotron_runtime src/nemotron_runtime.cpp)
target_include_directories(nemotron_runtime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/tools
  ${PAGER_DIR}/src)
target_link_libraries(nemotron_runtime PRIVATE veystrix_engine)
set_target_properties(nemotron_runtime PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Mixed-layer decoder loop (GDN + full attn + shared expert) bit-exact parity.
add_executable(test_decoder_full tests/test_decoder_full.cpp)
target_include_directories(test_decoder_full PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_decoder_full PRIVATE veystrix_engine)
set_target_properties(test_decoder_full PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
  set(Python3_EXECUTABLE python3)
endif()

# Production PLE layer (layer index 1) bit-exact parity.
add_executable(test_ple tests/test_ple.cpp)
target_include_directories(test_ple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_ple PRIVATE veystrix_engine)
set_target_properties(test_ple PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Fused-Metal MRoPE bit-exact parity (Python MRoPERotaryEmbedding fused path).
add_executable(test_rope_fused_mrope tests/test_rope_fused_mrope.cpp)
target_include_directories(test_rope_fused_mrope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_rope_fused_mrope PRIVATE veystrix_engine)
set_target_properties(test_rope_fused_mrope PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Batched vs sequential prefill primitives bit-exact parity.
add_executable(test_prefill_modes tests/test_prefill_modes.cpp)
target_include_directories(test_prefill_modes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_prefill_modes PRIVATE veystrix_engine)
set_target_properties(test_prefill_modes PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Packed linear projection bit-exact parity (Python nn.QuantizedLinear).
add_executable(test_linear_quantized tests/test_linear_quantized.cpp)
target_include_directories(test_linear_quantized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_linear_quantized PRIVATE veystrix_engine)
set_target_properties(test_linear_quantized PROPERTIES
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

# Nemotron attention decode primitive (bf16 KV path) bit-exact parity.
add_executable(test_nemo_attention tests/test_nemo_attention.cpp)
target_include_directories(test_nemo_attention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_nemo_attention PRIVATE veystrix_engine)
set_target_properties(test_nemo_attention PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Nemotron INT8 affine block KV attention (production default) bit-exact parity.
add_executable(test_nemo_int8 tests/test_nemo_int8.cpp)
target_include_directories(test_nemo_int8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_nemo_int8 PRIVATE veystrix_engine)
set_target_properties(test_nemo_int8 PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Final head (gated residual no-combine -> lm_head) bit-exact parity.
add_executable(test_final_head tests/test_final_head.cpp)
target_include_directories(test_final_head PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_final_head PRIVATE veystrix_engine)
set_target_properties(test_final_head PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

# Cooperative single-thread decode scheduler: interleaved == solo (bit-exact).
add_executable(test_cooperative_decode tests/test_cooperative_decode.cpp)
target_include_directories(test_cooperative_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME cooperative_decode COMMAND test_cooperative_decode)
# Phase 1 SharedExpertPool: CPU-only synthetic fixtures (no checkpoint, no MLX
# headers). Optional golden dir is produced by gen_shared_expert_pool_vectors.py.
add_executable(test_shared_expert_pool tests/test_shared_expert_pool.cpp)
target_include_directories(test_shared_expert_pool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_shared_expert_pool PRIVATE veystrix_engine)
set_target_properties(test_shared_expert_pool PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")
add_test(NAME shared_expert_pool COMMAND test_shared_expert_pool
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_shared_expert_pool")

# Multi-session Phase 1 runtime: CPU-only fixtures, no MLX / checkpoint / SSD.
# Compiled from the three new .cpp files directly so the test needs no engine
# link (and no real model). Guarded so a partial tree still configures.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ms_scheduler.cpp")
  add_executable(test_ms_scheduler tests/test_ms_scheduler.cpp
    src/session_state.cpp src/admission.cpp src/ms_scheduler.cpp)
  target_include_directories(test_ms_scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME ms_scheduler COMMAND test_ms_scheduler)
endif()

# KV tier quantization primitives (affine INT4/INT8 + tiered4 split) parity.
add_executable(test_kv_quant tests/test_kv_quant.cpp)
target_include_directories(test_kv_quant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_kv_quant PRIVATE veystrix_engine)
set_target_properties(test_kv_quant PROPERTIES
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
add_executable(test_prefix_cache tests/test_prefix_cache.cpp)
target_link_libraries(test_prefix_cache PRIVATE veystrix_engine)
set_target_properties(test_prefix_cache PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

add_executable(test_sampling tests/test_sampling.cpp)
# Strict sampler_json parser: no MLX needed, but built with the engine so it is
# part of the same ctest run as the numeric sampler gates.
add_executable(test_sampler_json tests/test_sampler_json.cpp src/sampler_json.cpp)
target_include_directories(test_sampler_json PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
# Phase 0 opt-in profiler: CPU-only standalone unit test (no MLX link needed).
add_executable(test_cpp_profile tests/test_cpp_profile.cpp)
target_include_directories(test_cpp_profile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME cpp_profile COMMAND test_cpp_profile)

# Dynamic-memory contract (memory_budget.*): env/profile-driven sizes, resident
# fail-closed rule, zero-read warm pools, MLX cache management. Offline: no
# checkpoint, no guarded run. Links the engine for NemoPool + MLX.
add_executable(test_memory_budget tests/test_memory_budget.cpp)
target_include_directories(test_memory_budget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${MLX_INCLUDE_DIRS})
target_link_libraries(test_memory_budget PRIVATE veystrix_engine)
set_target_properties(test_memory_budget PROPERTIES
  BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")
add_test(NAME memory_budget COMMAND test_memory_budget)
add_executable(test_dense_manifest tests/test_dense_manifest.cpp)
target_include_directories(test_dense_manifest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME dense_manifest COMMAND test_dense_manifest)
add_test(NAME ple COMMAND test_ple ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_ple)
add_test(NAME rope_fused_mrope COMMAND test_rope_fused_mrope
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_rope_mrope)
add_test(NAME prefill_modes COMMAND test_prefill_modes
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_prefill_modes)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_quant_linear/meta.json")
  add_test(NAME linear_quantized COMMAND test_linear_quantized
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_quant_linear")
endif()
add_test(NAME engine_smoke COMMAND test_engine_smoke)
add_test(NAME engine_parity COMMAND test_engine_parity
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors")
add_executable(test_nemo_snapshot tests/test_nemo_snapshot.cpp)
target_link_libraries(test_nemo_snapshot PRIVATE veystrix_engine mlx)
add_test(NAME nemo_snapshot COMMAND test_nemo_snapshot
         "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/engine/build/nemo-snapshot-test")
add_test(NAME engine_relu2 COMMAND test_relu2_parity
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/case3_nemotron_relu2")
# P5b: sorted relu2 MoE for S>1 chunks (real mlx_lm SwitchMLP sorted branch).
add_executable(test_relu2_sorted tests/test_relu2_sorted.cpp)
target_link_libraries(test_relu2_sorted PRIVATE veystrix_engine mlx)
add_test(NAME relu2_sorted COMMAND test_relu2_sorted
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_relu2_sorted")
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
add_test(NAME sampler_json COMMAND test_sampler_json)
# G-NG-KEYSEQ: per-session KeySequence == keyless default draws (no hardware).
add_executable(test_prng_keyseq tests/test_prng_keyseq.cpp)
target_link_libraries(test_prng_keyseq PRIVATE mlx)
add_test(NAME prng_keyseq COMMAND test_prng_keyseq)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/sampling_vectors/cases.txt")
  add_test(NAME sampling COMMAND test_sampling
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/sampling_vectors")
endif()
add_test(NAME prefix_cache COMMAND test_prefix_cache)
add_test(NAME model_adapter COMMAND test_model_adapter
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/model_adapter_flat")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_final_head/expected.npy")
  add_test(NAME final_head COMMAND test_final_head
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_final_head")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_kv_quant/meta.json")
  add_test(NAME kv_quant COMMAND test_kv_quant
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_kv_quant")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_nemo_attention/meta.json")
  add_test(NAME nemo_attention COMMAND test_nemo_attention
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_nemo_attention")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_nemo_int8/meta.json")
  add_test(NAME nemo_int8 COMMAND test_nemo_int8
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_nemo_int8")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_gdn_layer/meta.json")
  add_test(NAME gdn_layer COMMAND test_gdn_layer
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_gdn_layer")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_mamba_kernel/cases.json")
  add_test(NAME mamba_kernel COMMAND test_mamba_kernel
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_mamba_kernel")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_mamba_layer/meta.json")
  add_test(NAME mamba_layer COMMAND test_mamba_layer
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_mamba_layer")
endif()
# P5b: S>1 Mamba prefill vs real mlx_lm primitives (state carry across chunks).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_mamba_prefill/cases.json")
  add_executable(test_mamba_prefill tests/test_mamba_prefill.cpp)
  target_link_libraries(test_mamba_prefill PRIVATE veystrix_engine mlx)
  add_test(NAME mamba_prefill COMMAND test_mamba_prefill
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_mamba_prefill")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_qsa_full/cases.json")
  add_test(NAME qsa_full COMMAND test_qsa_full
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_qsa_full")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder/meta.json")
  add_test(NAME decoder COMMAND test_decoder
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder_full/meta.json")
  add_test(NAME decoder_full COMMAND test_decoder_full
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_decoder_full")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_parity_gaps/meta.json")
  add_test(NAME parity_gaps COMMAND test_parity_gaps
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_parity_gaps")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_rope_fused/cases.json")
  add_test(NAME rope_fused COMMAND test_rope_fused
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_rope_fused")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_prefill/meta.json")
  add_test(NAME prefill COMMAND test_prefill
           "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_prefill")
endif()
# End-to-end C++ generation vs Python (token ids + logit digests); runner script.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_native_generate/config.txt")
  add_test(NAME native_generate
    COMMAND ${CMAKE_COMMAND} -E env
      VENV_PYTHON=${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python
      ${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_native_generate.py
      --out ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors_native_generate
      --binary $<TARGET_FILE:native_generate> --steps 3)
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

# ---------------------------------------------------------------------------
# Phase 2 — resumable per-layer cursor + async I/O (offline/synthetic only).
# Both tests are CPU-only: no MLX headers, no checkpoint, no GPU, no benchmark.
# Guarded so a partial tree still configures.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_layer_cursor.cpp")
  add_executable(test_layer_cursor tests/test_layer_cursor.cpp src/layer_cursor.cpp)
  target_include_directories(test_layer_cursor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME layer_cursor COMMAND test_layer_cursor)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_io_pager_async.cpp")
  add_executable(test_io_pager_async tests/test_io_pager_async.cpp
    ${PAGER_DIR}/src/io_pager_async.cpp)
  target_include_directories(test_io_pager_async PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${PAGER_DIR}/src)
  add_test(NAME io_pager_async COMMAND test_io_pager_async)
endif()
# Phase F steps 1-4: CPU-only contract for the split-layer path — real
# LayerCursor + real io_pager_async on a temp file, NO MLX / checkpoint / GPU.
# Links the pager sources directly (guarded like the layer_cursor test).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_phase_f_contract.cpp")
  add_executable(test_phase_f_contract tests/test_phase_f_contract.cpp
    src/layer_cursor.cpp ${PAGER_DIR}/src/io_pager.cpp
    ${PAGER_DIR}/src/io_pager_async.cpp ${NATIVE_READ_C})
  set_source_files_properties(${NATIVE_READ_C} PROPERTIES LANGUAGE C)
  target_include_directories(test_phase_f_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${PAGER_DIR}/src)
  add_test(NAME phase_f_contract COMMAND test_phase_f_contract)
endif()

# ---------------------------------------------------------------------------
# Universal engine: generic recipe-driven runtime (docs/engineering/UNIVERSAL_CPP_ENGINE.md).
# Kept in its own library and its own block at the end of this file so merges
# with the hand-written runtimes stay clean.
add_library(veystrix_generic STATIC
  src/generic/json_lite.cpp src/generic/st_index.cpp src/generic/kv_store.cpp
  src/generic/kv_snapshot_store.cpp
  src/generic/expert_stream.cpp src/generic/blocks.cpp src/generic/blocks_qwen4.cpp
  src/generic/blocks_inkling.cpp src/generic/inkling_dspark.cpp
  src/generic/generic_model.cpp src/generic/generic_engine_runtime.cpp)
target_include_directories(veystrix_generic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${MLX_INCLUDE_DIRS})
# gpu_gate.cpp uses MLX's Metal command encoder (metal-cpp headers ship with MLX).
set_source_files_properties(src/gpu_gate.cpp PROPERTIES INCLUDE_DIRECTORIES "${MLX_ROOT}/include/metal_cpp")
target_link_libraries(veystrix_generic PUBLIC veystrix_engine)
if(APPLE)
  target_link_libraries(veystrix_generic PUBLIC compression)
endif()
set_target_properties(veystrix_generic PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET veystrix_c)
  target_link_libraries(veystrix_c PRIVATE veystrix_generic)
endif()
add_executable(test_gpu_gate tests/test_gpu_gate.cpp)
target_link_libraries(test_gpu_gate PRIVATE veystrix_generic)
set_target_properties(test_gpu_gate PROPERTIES BUILD_RPATH "${MLX_ROOT}/lib")
add_test(NAME gpu_gate COMMAND test_gpu_gate)

add_executable(test_kv_snapshot_store tests/test_kv_snapshot_store.cpp)
target_link_libraries(test_kv_snapshot_store PRIVATE veystrix_generic)
set_target_properties(test_kv_snapshot_store PROPERTIES BUILD_RPATH "${MLX_ROOT}/lib")
add_test(NAME kv_snapshot_store COMMAND test_kv_snapshot_store)
