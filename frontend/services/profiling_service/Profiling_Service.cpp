#include "Profiling_Service.hpp"

namespace megamol {
namespace frontend {

    bool Profiling_Service::init(void* configPtr) {
        _providedResourceReferences =
        { {"PerformanceManager", _perf_man} };

#ifdef PROFILING
        auto conf = static_cast<Config*>(configPtr);
        if (conf != nullptr && !conf->log_file.empty()) {
            _perf_man.subscribe_to_updates([&](const frontend_resources::PerformanceManager::frame_info& fi) {
                // TODO append to perf log.
            });
        }
#endif
        return true;
    }

    void Profiling_Service::updateProvidedResources() {
        _perf_man.startFrame();
    }

    void Profiling_Service::resetProvidedResources() {
        core::CallProfiling::CollectGPUPerformance();
        _perf_man.endFrame();
    }
} // namespace frontend
} // namespace megamol
