#include "layer_cursor.hpp"

namespace veystrix::engine {

const char* phase_name(LayerPhase p) noexcept {
  switch (p) {
    case LayerPhase::PreRoute:   return "PreRoute";
    case LayerPhase::AwaitReads: return "AwaitReads";
    case LayerPhase::PostMoE:    return "PostMoE";
  }
  return "Unknown";
}

}  // namespace veystrix::engine
