#include "Profiling_Service.hpp"

#include "ModuleGraphSubscription.h"

namespace megamol {
namespace frontend {

bool Profiling_Service::init(void* configPtr) {
    _providedResourceReferences = {
        {"PerformanceManager", _perf_man},
    };

    _requestedResourcesNames = {
        frontend_resources::MegaMolGraph_SubscriptionRegistry_Req_Name,
    };

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

    return true;
}

void Profiling_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    _requestedResourcesReferences = resources;

    auto& megamolgraph_subscription = const_cast<frontend_resources::MegaMolGraph_SubscriptionRegistry&>(
        resources[0].getResource<frontend_resources::MegaMolGraph_SubscriptionRegistry>());

#ifdef PROFILING
    frontend_resources::ModuleGraphSubscription profiling_manager_subscription("Profiling Manager");

    profiling_manager_subscription.AddCall = [&](core::CallInstance_t const& call_inst) {
        auto the_call = call_inst.callPtr.get();
        //printf("adding timers for @ %p = %s \n", reinterpret_cast<void*>(the_call), the_call->GetDescriptiveText().c_str());
        the_call->cpu_queries = _perf_man.add_timers(the_call, frontend_resources::PerformanceManager::query_api::CPU);
        if (the_call->GetCapabilities().OpenGLRequired()) {
            the_call->gl_queries =
                _perf_man.add_timers(the_call, frontend_resources::PerformanceManager::query_api::OPENGL);
        }
        the_call->perf_man = &_perf_man;
        return true;
    };

    profiling_manager_subscription.DeleteCall = [&](core::CallInstance_t const& call_inst) {
        auto the_call = call_inst.callPtr.get();
        _perf_man.remove_timers(the_call->cpu_queries);
        if (the_call->GetCapabilities().OpenGLRequired()) {
            _perf_man.remove_timers(the_call->gl_queries);
        }
        return true;
    };

    megamolgraph_subscription.subscribe(profiling_manager_subscription);
#endif
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
