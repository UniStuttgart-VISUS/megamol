#include "Profiling_Service.hpp"

#include "mmcore/MegaMolGraph.h"
#include "mmcore/view/AbstractView.h"

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

    _requestedResourcesNames = {"RegisterLuaCallbacks", "MegaMolGraph", "RenderNextFrame"};

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

    auto& graph = const_cast<core::MegaMolGraph&>(_requestedResourcesReferences[1].getResource<core::MegaMolGraph>());
    auto& render_next_frame = _requestedResourcesReferences[2].getResource<std::function<bool()>>();


    callbacks.add<frontend_resources::LuaCallbacksCollection::StringResult, std::string, std::string, int>(
        "mmGenerateCameraPositions", "(string entrypoint, string camera_path_pattern, uint num_samples)",
        {[&graph](std::string entrypoint, std::string camera_path_pattern,
             int num_samples) -> frontend_resources::LuaCallbacksCollection::StringResult {
            auto entry = graph.FindModule(entrypoint);
            if (!entry)
                return frontend_resources::LuaCallbacksCollection::Error{"could not find entrypoint"};
            auto view = std::dynamic_pointer_cast<core::view::AbstractView>(entry);
            if (!view)
                return frontend_resources::LuaCallbacksCollection::Error{"requested entrypoint is not a view"};
            auto camera_samples = view->SampleCameraScenes(camera_path_pattern, num_samples);
            if (camera_samples.empty())
                return frontend_resources::LuaCallbacksCollection::Error{"could not sample camera"};
            return frontend_resources::LuaCallbacksCollection::StringResult{camera_samples};
        }});


    callbacks.add<frontend_resources::LuaCallbacksCollection::StringResult, std::string, std::string, int>("mmProfile",
        "(string entrypoint, string cameras, unsigned int num_frames)",
        {[&graph, &render_next_frame](std::string entrypoint, std::string cameras,
             int num_frames) -> frontend_resources::LuaCallbacksCollection::StringResult {
            auto entry = graph.FindModule(entrypoint);
            if (!entry)
                return frontend_resources::LuaCallbacksCollection::Error{"could not find entrypoint"};
            auto view = std::dynamic_pointer_cast<core::view::AbstractView>(entry);
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
            sstr << "Total Number of Frames: " << tot_num_frames
                 << "; Elapsed Time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
                 << "; Time per Frame (ms): " << time_per_frame;

            return frontend_resources::LuaCallbacksCollection::StringResult{sstr.str()};
        }});


    auto& register_callbacks =
        _requestedResourcesReferences[0]
            .getResource<std::function<void(frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}

} // namespace frontend
} // namespace megamol
