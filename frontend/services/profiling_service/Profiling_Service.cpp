/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "Profiling_Service.hpp"

#include "FrameStatistics.h"
#include "LuaApiResource.h"
#include "ModuleGraphSubscription.h"
#include "mmcore/LuaAPI.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/utility/SampleCameraScenes.h"
#include "mmcore/view/AbstractViewInterface.h"
#include "mmcore/view/CameraSerializer.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::frontend {

bool Profiling_Service::init(void* configPtr) {
#ifdef MEGAMOL_USE_NVPERF
    nv::perf::InitializeNvPerf();
#endif
#ifdef MEGAMOL_USE_PROFILING
    _providedResourceReferences = {{frontend_resources::PerformanceManager_Req_Name, _perf_man},
        {frontend_resources::Performance_Logging_Status_Req_Name, profiling_logging}};

    const auto conf = static_cast<Config*>(configPtr);
    profiling_logging.active = conf->autostart_profiling;
    include_graph_events = conf->include_graph_events;

    const auto unit_name = "ns";
    using timer_ratio = std::nano;
    //const auto unit_name = "us";
    //using timer_ratio = std::micro;
    //const auto unit_name = "ms";
    //using timer_ratio = std::milli;

    if (conf != nullptr && !conf->log_file.empty()) {
        if (!log_file.is_open()) {
            log_file = std::ofstream(conf->log_file, std::ofstream::trunc);
            flush_frequency = conf->flush_frequency;
        }
        // header
        log_buffer << "frame;type;parent;name;comment;global_index;frame_index;api;start (" << unit_name << ");end ("
                   << unit_name << ");duration (" << unit_name << ")" << std::endl;
        _perf_man.subscribe_to_updates([&](const frontend_resources::PerformanceManager::frame_info& fi) {
            if (!profiling_logging.active) {
                return;
            }
            auto frame = fi.frame;
            if (frame > 0) {
                auto& _frame_stats =
                    _requestedResourcesReferences[4].getResource<frontend_resources::FrameStatistics>();
                log_buffer << (frame - 1) << ";MegaMol;MegaMol;FrameTime;;0;0;CPU;;;"
                           << _frame_stats.last_rendered_frame_time_milliseconds << std::endl;
            }
            for (auto& e : fi.entries) {
                auto conf = _perf_man.lookup_config(e.handle);
                auto name = conf.name;
                auto parent = _perf_man.lookup_parent(e.handle);
                auto comment = conf.comment;

                const auto the_start = std::chrono::duration<double, timer_ratio>(e.start.time_since_epoch()).count();
                const auto the_end = std::chrono::duration<double, timer_ratio>(e.end.time_since_epoch()).count();
                const auto the_duration =
                    std::chrono::duration<double, timer_ratio>(e.duration.time_since_epoch()).count();

                log_buffer << frame << ";" << frontend_resources::PerformanceManager::parent_type_string(e.parent_type)
                           << ";" << parent << ";" << name << ";" << comment << ";" << e.global_index << ";"
                           << e.frame_index << ";"
                           << megamol::frontend_resources::PerformanceManager::query_api_string(e.api) << ";"
                           << std::to_string(the_start) << ";" << std::to_string(the_end) << ";"
                           << std::to_string(the_duration) << std::endl;
            }
            if (frame % flush_frequency == flush_frequency - 1) {
                log_file << log_buffer.rdbuf();
                log_buffer.str(std::string());
                log_buffer.clear();
            }
        });
    }
#endif

    _requestedResourcesNames = {frontend_resources::LuaAPI_Req_Name, frontend_resources::MegaMolGraph_Req_Name,
        "RenderNextFrame", frontend_resources::MegaMolGraph_SubscriptionRegistry_Req_Name,
        frontend_resources::FrameStatistics_Req_Name};

    return true;
}

void Profiling_Service::log_graph_event(
    std::string const& parent, std::string const& name, std::string const& comment) {
    if (this->include_graph_events) {
        const auto frames_rendered = static_cast<int64_t>(
            _requestedResourcesReferences[4].getResource<frontend_resources::FrameStatistics>().rendered_frames_count);
        log_buffer << frames_rendered - 1 << ";Graph;" << parent << ";" << name << ";" << comment
                   << ";0;0;GraphEvent;;;" << std::endl;
    }
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

        log_graph_event(call_inst.callPtr->ClassName(), call_inst.callPtr->GetDescriptiveText(), "AddCall");
        return true;
    };

    profiling_manager_subscription.DeleteCall = [&](core::CallInstance_t const& call_inst) {
        auto the_call = call_inst.callPtr.get();
        _perf_man.remove_timers(the_call->cpu_queries);
        if (the_call->GetCapabilities().OpenGLRequired()) {
            _perf_man.remove_timers(the_call->gl_queries);
        }

        log_graph_event("", call_inst.callPtr->GetDescriptiveText(), "DeleteCall");
        return true;
    };

    profiling_manager_subscription.AddModule = [&](core::ModuleInstance_t const& mod_inst) {
        log_graph_event(mod_inst.modulePtr->ClassName(), mod_inst.modulePtr->FullName().PeekBuffer(), "AddModule");
        return true;
    };

    profiling_manager_subscription.DeleteModule = [&](core::ModuleInstance_t const& mod_inst) {
        log_graph_event("", mod_inst.modulePtr->FullName().PeekBuffer(), "DeleteModule");
        return true;
    };

    profiling_manager_subscription.RenameModule = [&](std::string const& old_name, std::string const& new_name,
                                                      core::ModuleInstance_t const& mod_inst) {
        log_graph_event(old_name + "->" + new_name, "", "RenameModule");
        return true;
    };

    profiling_manager_subscription.ParameterChanged = [&](core::param::ParamSlot* const& param,
                                                          std::string const& new_value) {
        log_graph_event(param->Parent()->FullName().PeekBuffer(), param->Name().PeekBuffer(),
            "'ParamValueChanged=" + new_value + "'");
        return true;
    };

    profiling_manager_subscription.DisableEntryPoint = [&](core::ModuleInstance_t const& module_instance) {
        log_graph_event(module_instance.modulePtr->FullName().PeekBuffer(), "", "DisableEntryPoint");
        return true;
    };

    profiling_manager_subscription.EnableEntryPoint = [&](core::ModuleInstance_t const& module_instance) {
        log_graph_event(module_instance.modulePtr->FullName().PeekBuffer(), "", "EnableEntryPoint");
        return true;
    };


    megamolgraph_subscription.subscribe(profiling_manager_subscription);
#endif

    fill_lua_callbacks();
}

void Profiling_Service::close() {
#ifdef MEGAMOL_USE_PROFILING
    if (log_file.is_open()) {
        // flush rest of log
        log_file << log_buffer.rdbuf();
        log_file.close();
    }
#endif
#ifdef MEGAMOL_USE_NVPERF
    nvperf.Reset();
#endif
}

static const char* const sl_innerframe = "InnerFrame";

void Profiling_Service::updateProvidedResources() {
#ifdef MEGAMOL_USE_TRACY
    FrameMarkStart(sl_innerframe);
#endif
    _perf_man.startFrame(
        _requestedResourcesReferences[4].getResource<frontend_resources::FrameStatistics>().rendered_frames_count);
#ifdef MEGAMOL_USE_NVPERF
    nvperf.OnFrameStart();
#endif
}

void Profiling_Service::resetProvidedResources() {
    _perf_man.endFrame();
#ifdef MEGAMOL_USE_TRACY
    FrameMarkEnd(sl_innerframe);
#endif
#ifdef MEGAMOL_USE_NVPERF
    nvperf.OnFrameEnd();
#endif
}

void Profiling_Service::fill_lua_callbacks() {
    auto luaApi = _requestedResourcesReferences[0].getResource<core::LuaAPI*>();

    auto& graph = const_cast<core::MegaMolGraph&>(_requestedResourcesReferences[1].getResource<core::MegaMolGraph>());
    auto& render_next_frame = _requestedResourcesReferences[2].getResource<std::function<bool()>>();

    luaApi->RegisterCallback(
        "mmSetProfilingLogging", "(bool on)", [&](bool on) -> void { this->profiling_logging.active = on; });

    luaApi->RegisterCallback("mmGenerateCameraScenes",
        "(string entrypoint, string camera_path_pattern, uint num_samples)",
        [&graph, &luaApi](std::string entrypoint, std::string camera_path_pattern, int num_samples) -> std::string {
            const auto entry = graph.FindModule(entrypoint);
            if (!entry)
                luaApi->ThrowError("could not find entrypoint");
            auto view = std::dynamic_pointer_cast<core::view::AbstractViewInterface>(entry);
            if (!view)
                luaApi->ThrowError("requested entrypoint is not a view");
            auto cam_func = megamol::core::utility::GetCamScenesFunctional(camera_path_pattern);
            if (!cam_func)
                luaApi->ThrowError("could not request camera path pattern");
            auto camera_samples = megamol::core::utility::SampleCameraScenes(view, cam_func, num_samples);
            if (camera_samples.empty())
                luaApi->ThrowError("could not sample camera");
            return camera_samples;
        });


    luaApi->RegisterCallback("mmProfile", "(string entrypoint, string cameras, unsigned int num_frames, bool pretty)",
        [&graph, &render_next_frame, &luaApi](
            std::string entrypoint, std::string cameras, int num_frames, bool pretty) -> std::string {
            auto entry = graph.FindModule(entrypoint);
            if (!entry)
                luaApi->ThrowError("could not find entrypoint");
            auto view = std::dynamic_pointer_cast<core::view::AbstractViewInterface>(entry);
            if (!view)
                luaApi->ThrowError("requested entrypoint is not a view");

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

            return sstr.str();
        });


#ifdef MEGAMOL_USE_NVPERF
    luaApi->RegisterCallback("mmNVPerfInit",
        "(string outpath)", {[&](std::string const& outpath) -> void {
            if (!nvperf.IsCollectingReport()) {
                nvperf.Reset();
                nvperf.InitializeReportGenerator();
                nvperf.SetFrameLevelRangeName("Frame");
                // nvperf.SetNumNestingLevels(3);
                nvperf.outputOptions.directoryName = outpath;
                nvperf.StartCollectionOnNextFrame();
            }
        });
#endif

}

} // namespace megamol::frontend
