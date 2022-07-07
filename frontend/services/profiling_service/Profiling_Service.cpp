#include "Profiling_Service.hpp"

#include <chrono>
#include <sstream>

#include "LuaCallbacksCollection.h"

namespace megamol {
namespace frontend {

bool Profiling_Service::init(void* configPtr) {
    _providedResourceReferences = {{"PerformanceManager", _perf_man}};

#ifdef PROFILING
    const auto conf = static_cast<Config*>(configPtr);
    if (conf != nullptr && !conf->log_file.empty()) {
        log_file = std::ofstream(conf->log_file, std::ofstream::trunc);
        // header
        log_file << "frame;parent;name;comment;frame_index;api;type;time (ms)" << std::endl;
        _perf_man.subscribe_to_updates([&](const frontend_resources::PerformanceManager::frame_info& fi) {
            auto frame = fi.frame;
            for (auto& e : fi.entries) {
                auto conf = _perf_man.lookup_config(e.handle);
                auto name = conf.name;
                auto parent = _perf_man.lookup_parent(e.handle);
                auto comment = conf.comment;
                std::string time_string;
                const auto dur = std::chrono::duration<double, std::milli>(e.timestamp.time_since_epoch());
                time_string = std::to_string(dur.count());

                log_file << frame << ";" << parent << ";" << name << ";" << comment << ";" << e.frame_index << ";"
                         << megamol::frontend_resources::PerformanceManager::query_api_string(e.api) << ";"
                         << megamol::frontend_resources::PerformanceManager::entry_type_string(e.type) << ";"
                         << time_string << std::endl;
            }
        });
    }
#endif
    _requestedResourcesNames = {"RegisterLuaCallbacks"};

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

void Profiling_Service::fill_lua_callbacks() {
    frontend_resources::LuaCallbacksCollection callbacks;

    callbacks.add<frontend_resources::LuaCallbacksCollection::StringResult>("mmGetTimeStamp",
        "(void)\n\tReturns a timestamp in ISO format.",
        {[&]() -> frontend_resources::LuaCallbacksCollection::StringResult {
            auto const tp = std::chrono::system_clock::now();

            auto const t = std::chrono::system_clock::to_time_t(tp);
            auto const lt = std::localtime(&t);
            auto const fs = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count() % 1000;
            std::stringstream str;
            str << std::to_string(1900 + lt->tm_year) << "-" << std::to_string(1 + lt->tm_mon) << "-"
                << std::to_string(lt->tm_mday) << "T" << std::to_string(lt->tm_hour) << ":"
                << std::to_string(lt->tm_min) << ":" << std::to_string(lt->tm_sec) << "." << std::to_string(fs);
            return frontend_resources::LuaCallbacksCollection::StringResult(str.str());
        }});

    auto& register_callbacks =
        _requestedResourcesReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

} // namespace frontend
} // namespace megamol
