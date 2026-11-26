#pragma once
#include <cstdint>
#include <vector>
namespace veystrix::engine {
// Raw bytes + shape + dtype of one cache array, stored little-endian and
// reinterpreted verbatim on restore (no numeric conversion).
struct TensorBlob {
  std::vector<int> shape;
  int dtype = 0;               // mx::Dtype::Val int
  std::vector<uint8_t> bytes;  // raw host bytes
};
}  // namespace veystrix::engine
