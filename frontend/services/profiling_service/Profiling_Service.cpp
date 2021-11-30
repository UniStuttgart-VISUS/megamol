#include "Profiling_Service.hpp"

namespace megamol {
namespace frontend {

bool Profiling_Service::init(void* configPtr) {
    _providedResourceReferences = {{"PerformanceManager", _perf_man}};

#ifdef PROFILING
    const auto conf = static_cast<Config*>(configPtr);
    if (conf != nullptr && !conf->log_file.empty()) {
        log_file = std::ofstream(conf->log_file, std::ofstream::trunc);
        // header
        log_file << "frame;parent;name;frame_index;api;type;time (ms)" << std::endl;
        _perf_man.subscribe_to_updates([&](const frontend_resources::PerformanceManager::frame_info& fi) {
            auto frame = fi.frame;
            for (auto& e : fi.entries) {
                auto name = _perf_man.lookup_name(e.handle);
                auto parent = _perf_man.lookup_parent(e.handle);
                std::string type_string, time_string, api_string;
                switch (e.type) {
                case frontend_resources::PerformanceManager::entry_type::START:
                    type_string = "start";
                    break;
                case frontend_resources::PerformanceManager::entry_type::END:
                    type_string = "end";
                    break;
                case frontend_resources::PerformanceManager::entry_type::DURATION:
                    type_string = "duration";
                    break;
                }
                switch (e.api) {
                case frontend_resources::PerformanceManager::query_api::CPU:
                    api_string = "CPU";
                    break;
                case frontend_resources::PerformanceManager::query_api::OPENGL:
                    api_string = "OpenGL";
                    break;
                }
                const auto dur = std::chrono::duration<double, std::milli>(e.timestamp.time_since_epoch());
                time_string = std::to_string(dur.count());

                log_file << frame << ";" << parent << ";" << name << ";" << e.frame_index << ";" << api_string << ";"
                         << type_string << ";" << time_string << std::endl;
            }
        });
    }
#endif
    return true;
}

void Profiling_Service::close() {
#ifdef PROFILING
    if (log_file.is_open()) {
        log_file.close();
    }
#endif
}

void Profiling_Service::updateProvidedResources() {
    _perf_man.startFrame();
}

void Profiling_Service::resetProvidedResources() {
    _perf_man.endFrame();
}
} // namespace frontend
} // namespace megamol
