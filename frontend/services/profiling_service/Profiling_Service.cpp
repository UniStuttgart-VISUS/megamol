#include "Profiling_Service.hpp"

namespace megamol {
namespace frontend {

    bool Profiling_Service::init(void* configPtr) {
        _providedResourceReferences =
        { {"PerformanceManager", _perf_man} };

        return true;
    }

    void Profiling_Service::updateProvidedResources() {
        _perf_man.startFrame();
    }

    void Profiling_Service::resetProvidedResources() {
        core::CallProfiling::CollectGPUPerformance();
        _perf_man.endFrame();
        // TODO append performance log file
    }
} // namespace frontend
} // namespace megamol
