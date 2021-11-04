#include "Profiling_Service.hpp"

namespace megamol {
namespace frontend {

bool Profiling_Service::init(void* configPtr) {
    m_providedResourceReferences = {
        {"PerformanceManager", m_pman}
    }

    return true;
}

} // namespace frontend
} // namespace megamol
