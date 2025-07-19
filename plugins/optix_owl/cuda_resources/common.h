#pragma once

namespace megamol {
namespace optix_owl {
inline size_t lChild(size_t P) {
    return 2 * P + 1;
}
inline size_t rChild(size_t P) {
    return 2 * P + 2;
}
} // namespace optix_owl
} // namespace megamol
