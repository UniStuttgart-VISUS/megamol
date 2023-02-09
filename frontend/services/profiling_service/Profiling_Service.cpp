#include "Profiling_Service.hpp"

#include "mmcore/MegaMolGraph.h"
#include "mmcore/utility/SampleCameraScenes.h"
#include "mmcore/view/AbstractViewInterface.h"
#include "mmcore/view/CameraSerializer.h"

#include "FrameStatistics.h"
#include "LuaCallbacksCollection.h"
#include "ModuleGraphSubscription.h"

namespace megamol {
namespace frontend {

bool Profiling_Service::init(void* configPtr) {
#ifdef MEGAMOL_USE_PROFILING
    _providedResourceReferences = {{frontend_resources::PerformanceManager_Req_Name, _perf_man},
        {frontend_resources::Performance_Logging_Status_Req_Name, profiling_logging}};

    const auto conf = static_cast<Config*>(configPtr);
    profiling_logging.active = conf->autostart_profiling;
    include_graph_events = conf->include_graph_events;

    if (conf != nullptr && !conf->log_file.empty()) {
        if (!log_file.is_open()) {
            log_file = std::ofstream(conf->log_file, std::ofstream::trunc);
        }
        // header
        log_file << "frame;parent;name;comment;frame_index;api;type;time (ms)" << std::endl;
        _perf_man.subscribe_to_updates([&](const frontend_resources::PerformanceManager::frame_info& fi) {
            if (!profiling_logging.active) {
                return;
            }
            auto frame = fi.frame;
            if (frame > 0) {
                auto& _frame_stats =
                    _requestedResourcesReferences[4].getResource<frontend_resources::FrameStatistics>();
                log_file << (frame - 1) << ";MegaMol;FrameTime;;0;CPU;Duration;"
                         << _frame_stats.last_rendered_frame_time_milliseconds << std::endl;
            }
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

    _requestedResourcesNames = {"RegisterLuaCallbacks", frontend_resources::MegaMolGraph_Req_Name, "RenderNextFrame",
        frontend_resources::MegaMolGraph_SubscriptionRegistry_Req_Name, frontend_resources::FrameStatistics_Req_Name};

    return true;
}

void Profiling_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    _requestedResourcesReferences = resources;

    auto& megamolgraph_subscription = const_cast<frontend_resources::MegaMolGraph_SubscriptionRegistry&>(
        resources[3].getResource<frontend_resources::MegaMolGraph_SubscriptionRegistry>());

#ifdef MEGAMOL_USE_PROFILING
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

        if (include_graph_events) {
            const auto frames_rendered = static_cast<int64_t>(_requestedResourcesReferences[4]
                                                                  .getResource<frontend_resources::FrameStatistics>()
                                                                  .rendered_frames_count);
            log_file << frames_rendered - 1 << ";" << call_inst.callPtr->ClassName()
                     << ";" << call_inst.callPtr->GetDescriptiveText() << ";AddCall;0;;GraphEvent;"
                     << std::endl;
        }

        return true;
    };

    profiling_manager_subscription.DeleteCall = [&](core::CallInstance_t const& call_inst) {
        auto the_call = call_inst.callPtr.get();
        _perf_man.remove_timers(the_call->cpu_queries);
        if (the_call->GetCapabilities().OpenGLRequired()) {
            _perf_man.remove_timers(the_call->gl_queries);
        }

        if (include_graph_events) {
            const auto frames_rendered = static_cast<int64_t>(_requestedResourcesReferences[4]
                                                                  .getResource<frontend_resources::FrameStatistics>()
                                                                  .rendered_frames_count);
            log_file << frames_rendered - 1 << ";" << call_inst.callPtr->GetDescriptiveText() << ";;DeleteCall;0;;GraphEvent;"
                     << std::endl;
        }

        return true;
    };

    profiling_manager_subscription.AddModule = [&](core::ModuleInstance_t const& mod_inst) {
        if (include_graph_events) {
            const auto frames_rendered = static_cast<int64_t>(_requestedResourcesReferences[4]
                                                                  .getResource<frontend_resources::FrameStatistics>()
                                                                  .rendered_frames_count);
            log_file << frames_rendered - 1 << ";" << mod_inst.modulePtr->ClassName()
                     << ";" << mod_inst.modulePtr->FullName() << ";AddModule;0;;GraphEvent;" << std::endl;
        }
        return true;
    };

    profiling_manager_subscription.DeleteModule = [&](core::ModuleInstance_t const& mod_inst) {
        if (include_graph_events) {
            const auto frames_rendered = static_cast<int64_t>(_requestedResourcesReferences[4]
                                                                  .getResource<frontend_resources::FrameStatistics>()
                                                                  .rendered_frames_count);
            log_file << frames_rendered - 1 << ";;" << mod_inst.modulePtr->FullName() << ";DeleteModule;0;;GraphEvent;"
                     << std::endl;
        }
        return true;
    };

    profiling_manager_subscription.RenameModule = [&](std::string const& old_name, std::string const& new_name,
                                                      core::ModuleInstance_t const& mod_inst) {
        if (include_graph_events) {
            const auto frames_rendered = static_cast<int64_t>(_requestedResourcesReferences[4]
                                                                  .getResource<frontend_resources::FrameStatistics>()
                                                                  .rendered_frames_count);
            log_file << frames_rendered - 1 << ";" << old_name << "->" << new_name << ";;RenameModule;0;;GraphEvent;"
                     << std::endl;
        }
        return true;
    };

    profiling_manager_subscription.ParameterChanged = [&](core::param::ParamSlot* const& param,
                                                          std::string const& new_value) {
        if (include_graph_events) {
            const auto frames_rendered = static_cast<int64_t>(_requestedResourcesReferences[4]
                                                                  .getResource<frontend_resources::FrameStatistics>()
                                                                  .rendered_frames_count);
            log_file << frames_rendered - 1 << ";" << param->Parent()->FullName() << ";" << param->Name()
                     << ";'ParamValueChanged=" << new_value << "';0;;GraphEvent;" << std::endl;
        }
        return true;
    };

    megamolgraph_subscription.subscribe(profiling_manager_subscription);
#endif

    fill_lua_callbacks();
}

void Profiling_Service::close() {
#ifdef MEGAMOL_USE_PROFILING
    if (log_file.is_open()) {
        log_file.close();
    }
#endif
}

void Profiling_Service::updateProvidedResources() {
    _perf_man.startFrame(
        _requestedResourcesReferences[4].getResource<frontend_resources::FrameStatistics>().rendered_frames_count);
}

void Profiling_Service::resetProvidedResources() {
    _perf_man.endFrame();
}

void Profiling_Service::fill_lua_callbacks() {
    frontend_resources::LuaCallbacksCollection callbacks;

    auto& graph = const_cast<core::MegaMolGraph&>(_requestedResourcesReferences[1].getResource<core::MegaMolGraph>());
    auto& render_next_frame = _requestedResourcesReferences[2].getResource<std::function<bool()>>();

    callbacks.add<frontend_resources::LuaCallbacksCollection::VoidResult, bool>(
        "mmSetProfilingLogging", "(bool on)", {[&](bool on) -> frontend_resources::LuaCallbacksCollection::VoidResult {
            this->profiling_logging.active = on;
            return frontend_resources::LuaCallbacksCollection::VoidResult{};
        }});

    callbacks.add<frontend_resources::LuaCallbacksCollection::StringResult, std::string, std::string, int>(
        "mmGenerateCameraScenes", "(string entrypoint, string camera_path_pattern, uint num_samples)",
        {[&graph](std::string entrypoint, std::string camera_path_pattern,
             int num_samples) -> frontend_resources::LuaCallbacksCollection::StringResult {
            auto entry = graph.FindModule(entrypoint);
            if (!entry)
                return frontend_resources::LuaCallbacksCollection::Error{"could not find entrypoint"};
            auto view = std::dynamic_pointer_cast<core::view::AbstractViewInterface>(entry);
            if (!view)
                return frontend_resources::LuaCallbacksCollection::Error{"requested entrypoint is not a view"};
            auto cam_func = megamol::core::utility::GetCamScenesFunctional(camera_path_pattern);
            if (!cam_func)
                return frontend_resources::LuaCallbacksCollection::Error{"could not request camera path pattern"};
            auto camera_samples = megamol::core::utility::SampleCameraScenes(view, cam_func, num_samples);
            if (camera_samples.empty())
                return frontend_resources::LuaCallbacksCollection::Error{"could not sample camera"};
            return frontend_resources::LuaCallbacksCollection::StringResult{camera_samples};
        }});


    callbacks.add<frontend_resources::LuaCallbacksCollection::StringResult, std::string, std::string, int, bool>(
        "mmProfile", "(string entrypoint, string cameras, unsigned int num_frames, bool pretty)",
        {[&graph, &render_next_frame](std::string entrypoint, std::string cameras, int num_frames,
             bool pretty) -> frontend_resources::LuaCallbacksCollection::StringResult {
            auto entry = graph.FindModule(entrypoint);
            if (!entry)
                return frontend_resources::LuaCallbacksCollection::Error{"could not find entrypoint"};
            auto view = std::dynamic_pointer_cast<core::view::AbstractViewInterface>(entry);
            if (!view)
                return frontend_resources::LuaCallbacksCollection::Error{"requested entrypoint is not a view"};

            auto serializer = core::view::CameraSerializer();
            std::vector<core::view::Camera> cams;
            serializer.deserialize(cams, cameras);

            auto const old_cam = view->GetCamera();

            uint64_t tot_num_frames = num_frames * cams.size();

            auto const tp_start = std::chrono::system_clock::now();
            for (auto const& cam : cams) {
                view->SetCamera(cam);
                for (unsigned int f_idx = 0; f_idx < num_frames; ++f_idx) {
                    render_next_frame();
                }
            }
            auto const tp_end = std::chrono::system_clock::now();

            view->SetCamera(old_cam);

            auto const duration = tp_end - tp_start;

            auto const time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

            auto const time_per_frame = static_cast<float>(time_in_ms) / static_cast<float>(tot_num_frames);

            std::stringstream sstr;
            if (pretty) {
                sstr << "Total Number of Frames: " << tot_num_frames << "; Elapsed Time (ms): "
                     << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
                     << "; Time per Frame (ms): " << time_per_frame;
            } else {
                sstr << tot_num_frames << ", "
                     << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << ", "
                     << time_per_frame;
            }

            return frontend_resources::LuaCallbacksCollection::StringResult{sstr.str()};
        }});


    auto& register_callbacks =
        _requestedResourcesReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

} // namespace frontend
} // namespace megamol
