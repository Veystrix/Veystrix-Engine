#!/bin/bash
# Build the Nemotron Phase F multi-session binaries (solo + runtime) against the
# SAME engine sources / MLX build. NO git, NO hardware run.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # worktree root
MLX=/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx
OBJ=/tmp/nemo_ms_obj
mkdir -p "$OBJ"

# native batched pread helper (C)
xcrun clang -O2 -c models/flashnext/native_read.c -o "$OBJ/native_read.o"

# Engine library: all src/*.cpp EXCEPT the two include-with-renamed-main TUs
# (qwen_runtime.cpp, nemotron_runtime.cpp) which each define main(). Include
# BOTH pager TUs because the engine links io_pager_async.
for f in cpp/engine/src/*.cpp; do
  b=$(basename "$f")
  if [ "$b" != "qwen_runtime.cpp" ] && [ "$b" != "nemotron_runtime.cpp" ]; then
    xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src \
      -Icpp/decode/src -c "$f" -o "$OBJ/${b%.cpp}.o"
  fi
done
xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src \
  -Icpp/decode/src -c cpp/decode/src/io_pager.cpp -o "$OBJ/io_pager.o"
xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src \
  -Icpp/decode/src -c cpp/decode/src/io_pager_async.cpp -o "$OBJ/io_pager_async.o"
xcrun ar rcs "$OBJ/libvey.a" "$OBJ"/*.o

# Solo (unmodified) binary: original Nemotron run_step path.
xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src \
  -Icpp/engine/tools -Icpp/decode/src cpp/engine/tools/nemotron_generate.cpp \
  "$OBJ/libvey.a" -L"$MLX/lib" -lmlx -Wl,-rpath,"$MLX/lib" \
  -o /tmp/nemotron_generate

# Runtime binary: N=1 per-layer cursor gate + N-session MsScheduler driver.
# `multi` selects the driver; otherwise it is the N=1 gate CLI.
xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src \
  -Icpp/engine/tools -Icpp/decode/src cpp/engine/src/nemotron_runtime.cpp \
  "$OBJ/libvey.a" -L"$MLX/lib" -lmlx -Wl,-rpath,"$MLX/lib" \
  -o /tmp/nemotron_runtime

echo "built: /tmp/nemotron_generate /tmp/nemotron_runtime"
