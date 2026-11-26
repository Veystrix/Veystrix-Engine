// Read hints for expert shard descriptors.
//
// VEYSTRIX_CPP_RDAHEAD=0 clears the kernel's read-ahead (F_RDAHEAD) on every
// expert shard descriptor, as the Python engine does (FLASHNEXT_RDAHEAD=0):
// expert rows are random-access, so read-ahead only adds SSD bytes and evicts
// useful page-cache pages. Same bytes land in every destination (exact).
#pragma once

#include <cstdlib>
#include <cstring>

#include <fcntl.h>

namespace veystrix::engine {

inline bool rdahead_off() {
  static const bool off = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_RDAHEAD");
    return e && std::strcmp(e, "0") == 0;
  }();
  return off;
}

inline void apply_shard_read_hints(int fd) {
#ifdef F_RDAHEAD
  if (fd >= 0 && rdahead_off()) ::fcntl(fd, F_RDAHEAD, 0);
#else
  (void)fd;
#endif
}

}  // namespace veystrix::engine
