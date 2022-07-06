#include "Profiling_Service.hpp"

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

    this->_requestedResourcesNames = {"RegisterLuaCallbacks"};

    distro_ = std::uniform_int_distribution<int64_t>(1);
    rng_ = std::mt19937_64(42);

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
    start_timer_queries();
}

void Profiling_Service::resetProvidedResources() {
    _perf_man.endFrame();
    notify_timer_queries();
}

void Profiling_Service::fill_lua_callbacks() {
    frontend_resources::LuaCallbacksCollection callbacks;

    callbacks.add<frontend_resources::LuaCallbacksCollection::LongResult>("mmCreateTimeQuery",
        "(void)\n\tCreates a time query to time a specified number of frames.\n\tReturn UID of the query.",
        {[&]() -> frontend_resources::LuaCallbacksCollection::LongResult {
            //auto uid = distro_(rng_);
            auto uid = ++timer_id_;
            auto fit = timer_map_.find(uid);
            if (fit != timer_map_.end()) {
                return frontend_resources::LuaCallbacksCollection::LongResult(0);
            }
            timer_map_[uid] = std::tuple<int, int64_t, int64_t>(-1, -1, -1);
            return frontend_resources::LuaCallbacksCollection::LongResult(uid);
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::BoolResult, int64_t, int>("mmStartTimeQuery",
        "(long UID, int num_frames)\n\tStart the query specified by the UID. After num_frames a timestamp is "
        "recorded.\n\tReturns true if successful.",
        {[&](int64_t uid, int num_frames) -> frontend_resources::LuaCallbacksCollection::BoolResult {
            auto fit = timer_map_.find(uid);
            if (fit != timer_map_.end()) {
                std::get<0>(fit->second) = num_frames;
                /*std::get<1>(fit->second) = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                std::get<2>(fit->second) = -1;*/
                return frontend_resources::LuaCallbacksCollection::BoolResult(true);
            }
            return frontend_resources::LuaCallbacksCollection::BoolResult(false);
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::BoolResult, int64_t>("mmPokeTimeQuery",
        "(long UID)\n\tPokes the query specified by UID if num_frames have passed.",
        {[&](int64_t uid) -> frontend_resources::LuaCallbacksCollection::BoolResult {
            auto fit = timer_map_.find(uid);
            if (fit != timer_map_.end() && std::get<0>(fit->second) == 0) {
                /*auto val = std::get<2>(fit->second);
                timer_map_.erase(fit->first);*/
                return frontend_resources::LuaCallbacksCollection::BoolResult(true);
            }
            return frontend_resources::LuaCallbacksCollection::BoolResult(false);
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::LongResult, int64_t>("mmEndTimeQuery",
        "(long UID)\n\tDestroys the query.\n\tReturns timestamp at end of query.",
        {[&](int64_t uid) -> frontend_resources::LuaCallbacksCollection::LongResult {
            auto fit = timer_map_.find(uid);
            if (fit != timer_map_.end() && std::get<0>(fit->second) == 0) {
                auto val = std::get<2>(fit->second);
                timer_map_.erase(fit->first);
                return frontend_resources::LuaCallbacksCollection::LongResult(val);
            }
            return frontend_resources::LuaCallbacksCollection::LongResult(-1);
        }});

    auto& register_callbacks =
        _requestedResourcesReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

void Profiling_Service::start_timer_queries() {
    for (auto& query : timer_map_) {
        if (std::get<0>(query.second) != 0 && std::get<1>(query.second) == -1) {
            std::get<1>(query.second) = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        }
    }
}

void Profiling_Service::notify_timer_queries() {
    for (auto& query : timer_map_) {
        if (std::get<0>(query.second) == 0) {
            std::get<2>(query.second) = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        } else if (std::get<0>(query.second) > 0) {
            --std::get<0>(query.second);
        }
    }
}
} // namespace frontend
} // namespace megamol
