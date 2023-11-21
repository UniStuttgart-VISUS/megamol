/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#include "Lua_Service_Wrapper.hpp"

#include "CommandRegistry.h"
#include "FrameStatistics.h"
#include "GUIState.h"
#include "GlobalValueStore.h"
#include "LuaRemoteConnectionsBroker.h"
#include "RuntimeConfig.h"
#include "Screenshots.h"
#include "WindowManipulation.h"
#include "vislib/UTF8Encoder.h"

// local logging wrapper for your convenience until central MegaMol logger established
#include "GUIRegisterWindow.h"
#include "LuaApiResource.h"
#include "PerformanceManager.h"
#include "mmcore/utility/buildinfo/BuildInfo.h"
#include "mmcore/utility/log/Log.h"

static void log(const char* text) {
    const std::string msg = "Lua_Service_Wrapper: " + std::string(text) + "\n";
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log(std::string text) {
    log(text.c_str());
}

static std::shared_ptr<bool> open_version_notification = std::make_shared<bool>(true);
static const std::string version_mismatch_title = "Version Check";
static const std::string version_mismatch_notification = "Warning: MegaMol version does not match version in project!";

namespace {
// used to abort a service callback if we are already inside a service wrapper callback
struct RecursionGuard {
    int& state;

    RecursionGuard(int& s) : state{s} {
        state++;
    }

    ~RecursionGuard() {
        state--;
    }

    bool abort() {
        return state > 1;
    }
};
} // namespace


namespace megamol::frontend {

Lua_Service_Wrapper::Lua_Service_Wrapper() {
    // init members to default states
}

Lua_Service_Wrapper::~Lua_Service_Wrapper() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
    open_version_notification.reset();
}

bool Lua_Service_Wrapper::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

#define luaAPI (*m_config.lua_api_ptr)

bool Lua_Service_Wrapper::init(const Config& config) {
    using megamol::core::utility::log::Log;
    if (!config.lua_api_ptr) {
        log("failed initialization because LuaAPI is nullptr");
        return false;
    }

    m_config = config;

    m_setScriptPath_resource = [&](std::string const& script_path) -> void { luaAPI.SetScriptPath(script_path); };

    luaApi_resource = config.lua_api_ptr;

    this->m_providedResourceReferences = {
        {"LuaScriptPaths", m_scriptpath_resource},
        {frontend_resources::LuaAPI_Req_Name, luaApi_resource},
        {"SetScriptPath", m_setScriptPath_resource},
    };

    this->m_requestedResourcesNames = {"FrontendResourcesList",
        "GLFrontbufferToPNG_ScreenshotTrigger",    // for screenshots
        "FrameStatistics",                         // for LastFrameTime
        "optional<WindowManipulation>",            // for Framebuffer resize
        "optional<GUIState>",                      // propagate GUI state and visibility
        frontend_resources::MegaMolGraph_Req_Name, // LuaAPI manipulates graph
        "RenderNextFrame",                         // LuaAPI can render one frame
        "GlobalValueStore",                        // LuaAPI can read and set global values
        frontend_resources::CommandRegistry_Req_Name, "optional<GUIRegisterWindow>", "RuntimeConfig",
#ifdef MEGAMOL_USE_PROFILING
        frontend_resources::performance::PerformanceManager_Req_Name
#endif
    }; //= {"ZMQ_Context"};

    *open_version_notification = false;

    try {
        m_network_host = std::make_unique<megamol::frontend::LuaRemoteConnectionsBroker>(
            m_config.host_address, m_config.retry_socket_port);
        log("initialized successfully");
    } catch (std::exception const& ex) {
        Log::DefaultLog.WriteError("Failed to start lua host: %s", ex.what());
        return false;
    }

    return true;
}

void Lua_Service_Wrapper::close() {
    m_config = {}; // default to nullptr

    m_network_host.reset();
}

std::vector<FrontendResource>& Lua_Service_Wrapper::getProvidedResources() {
    return m_providedResourceReferences;
}

const std::vector<std::string> Lua_Service_Wrapper::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void Lua_Service_Wrapper::setRequestedResources(std::vector<FrontendResource> resources) {
    // TODO: do something with ZMQ resource we get here
    m_requestedResourceReferences = resources;

    fill_frontend_resources_callbacks();
    fill_graph_manipulation_callbacks();

    //luaAPI.AddCallbacks(frontend_resource_callbacks);

    auto maybe_gui_window_request_resource =
        resources[9].getOptionalResource<megamol::frontend_resources::GUIRegisterWindow>();
    if (maybe_gui_window_request_resource.has_value()) {
        auto& gui_window_request_resource = maybe_gui_window_request_resource.value().get();
        gui_window_request_resource.register_notification(
            version_mismatch_title, std::weak_ptr<bool>(open_version_notification), version_mismatch_notification);
    }
}

// -------- main loop callbacks ---------

#define recursion_guard                           \
    RecursionGuard rg{m_service_recursion_depth}; \
    if (rg.abort())                               \
        return;

void Lua_Service_Wrapper::updateProvidedResources() {
    // during the previous frame module parameters of the graph may have changed.
    // submit the queued parameter changes to graph subscribers before other services do their thing
    auto& graph = const_cast<megamol::core::MegaMolGraph&>(
        m_requestedResourceReferences[5].getResource<megamol::core::MegaMolGraph>());
    graph.Broadcast_graph_subscribers_parameter_changes();

    recursion_guard;
    // we want lua to be the first thing executed in main loop
    // so we do all the lua work here

    m_scriptpath_resource.lua_script_paths.clear();
    m_scriptpath_resource.lua_script_paths.push_back(luaAPI.GetScriptPath());

    bool need_to_shutdown = false; // e.g. mmQuit should set this to true

    // fetch Lua requests from ZMQ queue, execute, and give back result
    if (m_network_host != nullptr && !m_network_host->RequestQueueEmpty()) {
        auto lua_requests = std::move(m_network_host->GetRequestQueue());
        std::string result;
        while (!lua_requests.empty()) {
            auto& request = lua_requests.front();

            auto res = luaAPI.RunString(request.request, "magic_remote_request");
            if (res.valid())
                request.answer_promise.set_value(core::LuaAPI::TypeToString(res));
            else
                request.answer_promise.set_value("error: " + luaAPI.GetError(res));

            lua_requests.pop();
            result.clear();
        }
    }

    // LuaAPI sets shutdown request of this service via direct callback
    // -> see Lua_Service_Wrapper::setRequestedResources()
    //if (need_to_shutdown)
    //    this->setShutdown();
}

void Lua_Service_Wrapper::digestChangedRequestedResources() {
    recursion_guard;
}

void Lua_Service_Wrapper::resetProvidedResources() {
    recursion_guard;
}

void Lua_Service_Wrapper::preGraphRender() {
    recursion_guard;

    // this gets called right before the graph is told to render something
    // e.g. you can start a start frame timer here

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls the equivalent of ::mmcRenderView(hView, &renderContext)
    // which leads to view.Render()
}

void Lua_Service_Wrapper::postGraphRender() {
    recursion_guard;

    // the graph finished rendering and you may more stuff here
    // e.g. end frame timer
    // update window name
    // swap buffers, glClear
}


void Lua_Service_Wrapper::fill_frontend_resources_callbacks() {
    luaApi_resource->RegisterCallback("mmGetMegaMolExecutableDirectory",
        "()\n\tReturns the directory of the running MegaMol executable.", [&]() -> std::string {
            auto& path = m_requestedResourceReferences[10]
                             .getResource<frontend_resources::RuntimeConfig>()
                             .megamol_executable_directory;
            return path;
        });

    luaApi_resource->RegisterCallback(
        "mmListResources", "()\n\tReturn a list of available resources in the frontend.", [&]() -> std::string {
            auto resources_list =
                m_requestedResourceReferences[0].getResource<std::function<std::vector<std::string>(void)>>()();
            std::ostringstream answer;

            for (auto& resource_name : resources_list) {
                answer << resource_name << std::endl;
            }

            if (resources_list.empty()) {
                answer << "(none)" << std::endl;
            }

            return answer.str();
        });

    luaApi_resource->RegisterCallback("mmScreenshot",
        "(string filename)\n\tSave a screen shot of the GL front buffer under 'filename'.",
        [&](std::string file) -> void {
            auto& framestats = m_requestedResourceReferences[2].getResource<frontend_resources::FrameStatistics>();
            if (framestats.rendered_frames_count == 0) {
                luaApi_resource->ThrowError("error capturing screenshot: no frame rendered yet");
            } else {
                m_requestedResourceReferences[1].getResource<std::function<bool(std::filesystem::path const&)>>()(
                    std::filesystem::u8path(file));
            }
        });

    luaApi_resource->RegisterCallback(
        "mmLastFrameTime", "()\n\tReturns the graph execution time of the last frame in ms.", [&]() -> double {
            auto& frame_statistics =
                m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FrameStatistics>();
            return frame_statistics.last_rendered_frame_time_milliseconds;
        });

    luaApi_resource->RegisterCallback("mmLastRenderedFramesCount",
        "()\n\tReturns the number of rendered frames up until to the last frame.", [&]() -> double {
            auto& frame_statistics =
                m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FrameStatistics>();
            return static_cast<double>(frame_statistics.rendered_frames_count);
        });


    auto maybe_window_manipulation =
        m_requestedResourceReferences[3].getOptionalResource<megamol::frontend_resources::WindowManipulation>();
    if (maybe_window_manipulation.has_value()) {
        frontend_resources::WindowManipulation& window_manipulation =
            const_cast<frontend_resources::WindowManipulation&>(maybe_window_manipulation.value().get());

        luaApi_resource->RegisterCallback("mmSetWindowFramebufferSize",
            "(int width, int height)\n\tSet framebuffer dimensions of window to width x height.",
            [&](int width, int height) -> void {
                if (width <= 0 || height <= 0) {
                    luaApi_resource->ThrowError("framebuffer dimensions must be positive, but given values are: " +
                                                std::to_string(width) + " x " + std::to_string(height));
                }

                window_manipulation.set_framebuffer_size(width, height);
            });

        luaApi_resource->RegisterCallback("mmSetWindowPosition", "(int x, int y)\n\tSet window position to x,y.",
            [&](int x, int y) -> void { window_manipulation.set_window_position(x, y); });

        luaApi_resource->RegisterCallback("mmSetFullscreen",
            "(bool fullscreen)\n\tSet window to fullscreen (or restore).", [&](bool fullscreen) -> void {
                window_manipulation.set_fullscreen(fullscreen
                                                       ? frontend_resources::WindowManipulation::Fullscreen::Maximize
                                                       : frontend_resources::WindowManipulation::Fullscreen::Restore);
            });

        luaApi_resource->RegisterCallback("mmSetVSync", "(bool state)\n\tSet window VSync off (false) or on (true).",
            [&](bool state) -> void { window_manipulation.set_swap_interval(state ? 1 : 0); });
    }

    auto maybe_gui_state =
        m_requestedResourceReferences[4].getOptionalResource<megamol::frontend_resources::GUIState>();
    if (maybe_gui_state.has_value()) {
        auto& gui_resource = maybe_gui_state.value().get();

        luaApi_resource->RegisterCallback("mmSetGUIState", "(string json)\n\tSet GUI state from given 'json' string.",
            [&](std::string json) -> void { gui_resource.provide_gui_state(json); });
        luaApi_resource->RegisterCallback(
            "mmGetGUIState", "()\n\tReturns the GUI state as json string.", [&]() -> std::string {
                auto s = gui_resource.request_gui_state(false);
                return s;
            });

        luaApi_resource->RegisterCallback("mmSetGUIVisible", "(bool state)\n\tShow (true) or hide (false) the GUI.",
            [&](bool show) -> void { gui_resource.provide_gui_visibility(show); });
        luaApi_resource->RegisterCallback(
            "mmGetGUIVisible", "()\n\tReturns whether the GUI is visible (true/false).", [&]() -> std::string {
                const auto visible = gui_resource.request_gui_visibility();
                return visible ? "true" : "false";
            });

        luaApi_resource->RegisterCallback("mmSetGUIScale", "(float scale)\n\tSet GUI scaling factor.",
            [&](float scale) -> void { gui_resource.provide_gui_scale(scale); });
        luaApi_resource->RegisterCallback(
            "mmGetGUIScale", "()\n\tReturns the GUI scaling as float.", [&]() -> std::string {
                const auto scale = gui_resource.request_gui_scale();
                return std::to_string(scale);
            });
    }

    luaApi_resource->RegisterCallback(
        "mmQuit", "()\n\tClose the MegaMol instance.", [&]() -> void { this->setShutdown(); });

    luaApi_resource->RegisterCallback(
        "mmRenderNextFrame", "()\n\tAdvances rendering by one frame by poking the main rendering loop.", [&]() -> void {
            auto& render_next_frame = m_requestedResourceReferences[6].getResource<std::function<bool()>>();
            render_next_frame();
        });

    luaApi_resource->RegisterCallback("mmSetGlobalValue",
        "(string key, string value)\n\tSets a global key-value pair. If the key is already present, overwrites the "
        "value.",
        [&](std::string key, std::string value) -> void {
            auto& global_value_store = const_cast<megamol::frontend_resources::GlobalValueStore&>(
                m_requestedResourceReferences[7].getResource<megamol::frontend_resources::GlobalValueStore>());
            global_value_store.insert(key, value);
        });

    luaApi_resource->RegisterCallback("mmGetGlobalValue",
        "(string key)\n\tReturns the value for the given global key. If no key with that name is known, returns empty "
        "string.",
        [&](std::string key) -> std::string {
            auto& global_value_store =
                m_requestedResourceReferences[7].getResource<megamol::frontend_resources::GlobalValueStore>();
            std::optional<std::string> maybe_value = global_value_store.maybe_get(key);

            if (maybe_value.has_value()) {
                return maybe_value.value();
            }

            // TODO: maybe we want to LuaError?
            return "";
        });

    luaApi_resource->RegisterCallback("mmExecCommand",
        "(string command)\n\tExecutes a command as provided by a hotkey, for example.",
        [&](std::string command) -> void {
            auto& command_registry =
                m_requestedResourceReferences[8].getResource<megamol::frontend_resources::CommandRegistry>();
            command_registry.exec_command(command);
        });

    luaApi_resource->RegisterCallback("mmListCommands", "()\n\tLists the available commands.", [&]() -> std::string {
        auto& command_registry =
            m_requestedResourceReferences[8].getResource<megamol::frontend_resources::CommandRegistry>();
        auto& l = command_registry.list_commands();
        std::string output;
        std::for_each(l.begin(), l.end(),
            [&](const frontend_resources::Command& c) { output = output + c.name + ", " + c.key.ToString() + "\n"; });
        return output;
    });

    luaApi_resource->RegisterCallback("mmCheckVersion",
        "(string version)\n\tChecks whether the running MegaMol corresponds to version.",
        [&](std::string version) -> bool {
            bool version_ok = version == megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH();
            *open_version_notification = (!version_ok && m_config.show_version_notification);
            if (!version_ok) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "Version info in project (%s) does not match MegaMol version (%s)!", version.c_str(),
                    megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH().c_str());
            }
            return version_ok;
        });

    // mmLoadProject ?
    // the ProjectLoader resource immediately executes the file contents as lua code
    // -> what happens if this is done inside a lua callback?

    // template for futher callbacks
    //frontend_resource_callbacks.add<>(
    //    "name",
    //    "()\n\t help",
    //    {[&]() -> VoidResult
    //    {
    //        return VoidResult{};
    //    }});
}

void Lua_Service_Wrapper::fill_graph_manipulation_callbacks() {
    auto& graph = const_cast<megamol::core::MegaMolGraph&>(
        m_requestedResourceReferences[5].getResource<megamol::core::MegaMolGraph>());

    luaApi_resource->RegisterCallback("mmCreateView",
        "(string graphName, string className, string moduleName)\n\tCreate a view module instance of class <className> "
        "called <moduleName>. The view module is registered as graph entry point. <graphName> is ignored.",
        [&](std::string baseName, std::string className, std::string instanceName) -> void {
            if (!graph.CreateModule(className, instanceName)) {
                luaApi_resource->ThrowError(
                    "graph could not create module for: " + baseName + " , " + className + " , " + instanceName);
            }

            if (!graph.SetGraphEntryPoint(instanceName)) {
                luaApi_resource->ThrowError("graph could not set graph entry point for: " + baseName + " , " +
                                            className + " , " + instanceName);
            }
        });

    luaApi_resource->RegisterCallback("mmCreateModule",
        "(string className, string moduleName)\n\tCreate a module instance of class <className> called <moduleName>.",
        [&](std::string className, std::string instanceName) -> void {
            if (!graph.CreateModule(className, instanceName)) {
                luaApi_resource->ThrowError("graph could not create module: " + className + " , " + instanceName);
            }
        });

    luaApi_resource->RegisterCallback(
        "mmDeleteModule", "(string name)\n\tDelete the module called <name>.", [&](std::string moduleName) -> void {
            if (!graph.DeleteModule(moduleName)) {
                luaApi_resource->ThrowError("graph could not delete module: " + moduleName);
            }
        });

    luaApi_resource->RegisterCallback("mmRenameModule",
        "(string oldName, string newName)\n\tRenames the module called <oldname> to <newname>.",
        [&](std::string oldName, std::string newName) -> void {
            if (!graph.RenameModule(oldName, newName)) {
                luaApi_resource->ThrowError("graph could not rename module: " + oldName + " to " + newName);
            }
        });

    luaApi_resource->RegisterCallback("mmCreateCall",
        "(string className, string from, string to)\n\tCreate a call of type <className>, connecting CallerSlot <from> "
        "and CalleeSlot <to>.",
        [&](std::string className, std::string from, std::string to) -> void {
            if (!graph.CreateCall(className, from, to)) {
                luaApi_resource->ThrowError("graph could not create call: " + className + " , " + from + " -> " + to);
            }
        });

    luaApi_resource->RegisterCallback("mmDeleteCall",
        "(string from, string to)\n\tDelete the call connecting CallerSlot <from> and CalleeSlot <to>.",
        [&](std::string from, std::string to) -> void {
            if (!graph.DeleteCall(from, to)) {
                luaApi_resource->ThrowError("graph could not delete call: " + from + " -> " + to);
            }
        });

    luaApi_resource->RegisterCallback("mmCreateChainCall",
        "(string className, string chainStart, string to)\n\tAppend a call of type <className>, connection the "
        "rightmost CallerSlot starting at <chainStart> and CalleeSlot <to>.",
        [&](std::string className, std::string chainStart, std::string to) -> void {
            if (!graph.Convenience().CreateChainCall(className, chainStart, to)) {
                luaApi_resource->ThrowError(
                    "graph could not create chain call: " + className + " , " + chainStart + " -> " + to);
            }
        });

    luaApi_resource->RegisterCallback("mmGetModuleParams",
        "(string name)\n\tReturns a 0x1-separated list of module name and all parameters.\n\tFor each parameter the "
        "name, description, definition, and value are returned.",
        [&](std::string moduleName) -> std::string {
            auto mod = graph.FindModule(moduleName);
            if (mod == nullptr) {
                luaApi_resource->ThrowError("graph could not find module: " + moduleName);
            }

            auto slots = graph.EnumerateModuleParameterSlots(moduleName);
            std::ostringstream answer;
            answer << mod->FullName() << "\1";

            for (auto& ps : slots) {
                answer << ps->Name() << "\1";
                answer << ps->Description() << "\1";
                auto par = ps->Parameter();
                if (par == nullptr) {
                    luaApi_resource->ThrowError(
                        "ParamSlot does not seem to hold a parameter: " + std::string(ps->FullName().PeekBuffer()));
                }
                answer << par->ValueString() << "\1";
            }

            return answer.str();
        });

    luaApi_resource->RegisterCallback("mmGetParamDescription",
        "(string name)\n\tReturn the description of a parameter slot.", [&](std::string paramName) -> std::string {
            core::param::ParamSlot* ps = graph.FindParameterSlot(paramName);
            if (ps == nullptr) {
                luaApi_resource->ThrowError("graph could not find parameter: " + paramName);
            }

            vislib::StringA valUTF8;
            vislib::UTF8Encoder::Encode(valUTF8, ps->Description());

            return valUTF8.PeekBuffer();
        });

    luaApi_resource->RegisterCallback("mmGetParamValue", "(string name)\n\tReturn the value of a parameter slot.",
        [&](std::string paramName) -> std::string {
            const auto* param = graph.FindParameter(paramName);
            if (param == nullptr) {
                luaApi_resource->ThrowError("graph could not find parameter: " + paramName);
            }

            return param->ValueString();
        });

    luaApi_resource->RegisterCallback("mmSetParamValue",
        "(string name, string|number value)\n\tSet the value of a parameter slot.",
        sol::overload(
            [&](std::string paramName, std::string paramValue) -> void {
                if (!graph.SetParameter(paramName, paramValue.c_str())) {
                    luaApi_resource->ThrowError(
                        "parameter could not be set to value: " + paramName + " : " + paramValue);
                }
            },
            [&](std::string paramName, double paramValue) -> void {
                if (!graph.SetParameter(paramName, std::to_string(paramValue))) {
                    luaApi_resource->ThrowError(
                        "parameter could not be set to value: " + paramName + " : " + std::to_string(paramValue));
                }
            }));

    luaApi_resource->RegisterCallback("mmSetParamHighlight",
        "(string name, bool is_highlighted)\n\tHighlight parameter slot.",
        [&](std::string paramName, bool is_highlight) -> void {
            auto param_ptr = graph.FindParameter(paramName);

            if (param_ptr == nullptr) {
                luaApi_resource->ThrowError(
                    "parameter highlight could not be set: " + paramName + " : " + std::to_string(is_highlight));
            }

            param_ptr->SetGUIHighlight(is_highlight);
        });

    luaApi_resource->RegisterCallback("mmCreateParamGroup",
        "(string name, string size)\n\tGenerate a param group that can only be set at once. Sets are queued until size "
        "is reached.",
        [&](std::string groupName) -> void { graph.Convenience().CreateParameterGroup(groupName); });

    luaApi_resource->RegisterCallback("mmSetParamGroupValue",
        "(string groupname, string paramname, string value)\n\tQueue the value of a grouped parameter.",
        [&](std::string paramGroup, std::string paramName, std::string paramValue) -> void {
            auto groupPtr = graph.Convenience().FindParameterGroup(paramGroup);
            if (!groupPtr) {
                luaApi_resource->ThrowError("graph could not find parameter group: " + paramGroup);
            }

            bool queued = groupPtr->QueueParameterValue(paramName, paramValue);
            if (!queued) {
                luaApi_resource->ThrowError(
                    "graph could not queue param group value: " + paramGroup + " , " + paramName + " : " + paramValue);
            }
        });

    luaApi_resource->RegisterCallback("mmApplyParamGroupValues",
        "(string groupname)\n\tApply queued parameter values of group to graph.", [&](std::string paramGroup) -> void {
            auto groupPtr = graph.Convenience().FindParameterGroup(paramGroup);
            if (!groupPtr) {
                luaApi_resource->ThrowError("graph could not apply param group: no such group: " + paramGroup);
            }

            bool applied = groupPtr->ApplyQueuedParameterValues();
            if (!applied) {
                luaApi_resource->ThrowError("graph could not apply param group: some parameter values did not parse.");
            }
        });

    luaApi_resource->RegisterCallback("mmListModules",
        "(string basemodule_or_namespace)\n\tReturn a list of instantiated modules (class id, instance id), starting "
        "from a certain module downstream or inside a namespace.\n\tWill use the graph root if an empty string is "
        "passed.",
        [&](std::string starting_point) -> std::string {
            // actually putting an empty string as an argument on purpose is OK too
            auto modules_list =
                starting_point.empty() ? graph.ListModules() : graph.Convenience().ListModules(starting_point);

            std::ostringstream answer;
            for (auto& module : modules_list) {
                answer << module.modulePtr->ClassName() << ";" << module.modulePtr->Name() << std::endl;
            }

            if (modules_list.empty()) {
                answer << "(none)" << std::endl;
            }

            return answer.str();
        });

    luaApi_resource->RegisterCallback(
        "mmClearGraph", "()\n\tClear the MegaMol Graph from all Modules and Calls", [&]() -> void { graph.Clear(); });

    luaApi_resource->RegisterCallback(
        "mmListCalls", "()\n\tReturn a list of instantiated calls.", [&]() -> std::string {
            std::ostringstream answer;
            auto& calls_list = graph.ListCalls();
            for (auto& call : calls_list) {
                answer << call.callPtr->ClassName() << ";" << call.callPtr->PeekCallerSlot()->Parent()->Name() << ","
                       << call.callPtr->PeekCalleeSlot()->Parent()->Name() << ";"
                       << call.callPtr->PeekCallerSlot()->Name() << "," << call.callPtr->PeekCalleeSlot()->Name()
                       << std::endl;
            }

            if (calls_list.empty()) {
                answer << "(none)" << std::endl;
            }

            return answer.str();
        });

    luaApi_resource->RegisterCallback("mmSetGraphEntryPoint",
        "(string moduleName)\n\tSet active graph entry point to one specific module.",
        [&](std::string moduleName) -> void {
            auto res = graph.SetGraphEntryPoint(moduleName);
            if (!res) {
                luaApi_resource->ThrowError("Could not set graph entry point " + moduleName);
            }
        });
    luaApi_resource->RegisterCallback("mmRemoveGraphEntryPoint",
        "(string moduleName)\n\tRemove active graph entry point from one specific module.",
        [&](std::string moduleName) -> void {
            auto res = graph.RemoveGraphEntryPoint(moduleName);
            if (!res) {
                luaApi_resource->ThrowError("Could not remove graph entry point " + moduleName);
            }
        });
    luaApi_resource->RegisterCallback(
        "mmRemoveAllGraphEntryPoints", "\n\tRemove any and all active graph entry points.", [&]() -> void {
            for (auto& m : graph.ListModules()) {
                if (m.isGraphEntryPoint) {
                    graph.RemoveGraphEntryPoint(m.modulePtr->FullName().PeekBuffer());
                }
            }
        });


#ifdef MEGAMOL_USE_PROFILING
    luaApi_resource->RegisterCallback("mmListModuleTimers", "(string name)\n\tList the registered timers of a module.",
        [&](std::string name) -> std::string {
            auto perf_manager = const_cast<megamol::frontend_resources::performance::PerformanceManager*>(
                &this->m_requestedResourceReferences[11]
                     .getResource<megamol::frontend_resources::performance::PerformanceManager>());
            std::stringstream output;
            auto m = graph.FindModule(name);
            if (m) {
                auto timers = perf_manager->lookup_timers(m.get());
                for (auto& t : timers) {
                    auto& timer = perf_manager->lookup_config(t);
                    output << t << ": " << timer.name << " ("
                           << megamol::frontend_resources::performance::query_api_string(timer.api) << ")" << std::endl;
                }
            }
            return output.str();
        });
    luaApi_resource->RegisterCallback("mmSetTimerComment",
        "(int handle, string comment)\n\tSet a transient comment for a timer; will show up in profiling log.",
        [&](int handle, std::string comment) -> void {
            auto perf_manager = const_cast<megamol::frontend_resources::performance::PerformanceManager*>(
                &this->m_requestedResourceReferences[11]
                     .getResource<megamol::frontend_resources::performance::PerformanceManager>());
            perf_manager->set_transient_comment(handle, comment);
        });
#endif
    // TODO
    //const auto fun = [&answer](Module* mod) {
    //    AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
    //    for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se;
    //         ++si) {
    //        const auto slot = dynamic_cast<CallerSlot*>((*si).get());
    //        if (slot) {
    //            const Call* c = const_cast<CallerSlot*>(slot)->CallAs<Call>();
    //            if (c != nullptr) {
    //                answer << c->ClassName() << ";" << c->PeekCallerSlot()->Parent()->Name() << ","
    //                       << c->PeekCalleeSlot()->Parent()->Name() << ";" << c->PeekCallerSlot()->Name() << ","
    //                       << c->PeekCalleeSlot()->Name() << std::endl;
    //            }
    //        }
    //    }
    //};

    //if (n == 1) {
    //    const auto starting_point = luaL_checkstring(L, 1);
    //    if (!std::string(starting_point).empty()) {
    //        this->coreInst->EnumModulesNoLock(starting_point, fun);
    //    } else {
    //        this->coreInst->EnumModulesNoLock(nullptr, fun);
    //    }
    //} else {
    //    this->coreInst->EnumModulesNoLock(nullptr, fun);
    //}


    // template for futher callbacks
    //callbacks.add<>(
    //    "name",
    //    "()\n\t help",
    //    {[&]() -> VoidResult
    //    {
    //        return VoidResult{};
    //    }});


    //    callbacks.add<StringResult>(
    //        "mmListParameters",
    //        "(string baseModule_or_namespace)"
    //            "\n\tReturn all parameters, their type and value, starting from a certain module downstream or inside a namespace."
    //            "\n\tWill use the graph root if an empty string is passed.",
    //        {[&]() -> StringResult
    //        {
    //            return StringResult{"mmListParameters currently not implemented!"};
    //
    //            std::ostringstream answer;
    //
    //            // TODO
    //
    //            //const auto fun = [&answer](Module* mod) {
    //            //    AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
    //            //    for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se;
    //            //         ++si) {
    //            //        const auto slot = dynamic_cast<param::ParamSlot*>((*si).get());
    //            //        if (slot) {
    //            //            answer << slot->FullName() << "\1" << slot->Parameter()->ValueString() << "\1";
    //            //        }
    //            //    }
    //            //};
    //
    //            //if (n == 1) {
    //            //    const auto starting_point = luaL_checkstring(L, 1);
    //            //    if (!std::string(starting_point).empty()) {
    //            //        this->coreInst->EnumModulesNoLock(starting_point, fun);
    //            //    } else {
    //            //        this->coreInst->EnumModulesNoLock(nullptr, fun);
    //            //    }
    //            //} else {
    //            //    this->coreInst->EnumModulesNoLock(nullptr, fun);
    //            //}
    //
    //            lua_pushstring(L, answer.str().c_str());
    //        }});


    // #define MMC_LUA_MMQUERYMODULEGRAPH "mmQueryModuleGraph"
    //    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::QueryModuleGraph>(MMC_LUA_MMQUERYMODULEGRAPH, "()\n\tShow the instantiated modules and their children.");
    // int mmQueryModuleGraph(lua_State* L) {
    //
    //     std::ostringstream answer;
    //
    //     // TODO
    //
    //     // queryModules(answer, anoc);
    //     //std::vector<AbstractNamedObjectContainer::const_ptr_type> anoStack;
    //     //anoStack.push_back(anoc);
    //     //while (!anoStack.empty()) {
    //     //    anoc = anoStack.back();
    //     //    anoStack.pop_back();
    //
    //     //    if (anoc) {
    //     //        const auto m = Module::dynamic_pointer_cast(anoc);
    //     //        answer << (m != nullptr ? "Module:    " : "Container: ") << anoc.get()->FullName() << std::endl;
    //     //        if (anoc.get()->Parent() != nullptr) {
    //     //            answer << "Parent:    " << anoc.get()->Parent()->FullName() << std::endl;
    //     //        } else {
    //     //            answer << "Parent:    none" << std::endl;
    //     //        }
    //     //        const char* cn = nullptr;
    //     //        if (m != nullptr) {
    //     //            cn = m->ClassName();
    //     //        }
    //     //        answer << "Class:     " << ((cn != nullptr) ? cn : "unknown") << std::endl;
    //     //        answer << "Children:  ";
    //     //        auto it_end = anoc->ChildList_End();
    //     //        int numChildren = 0;
    //     //        for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
    //     //            AbstractNamedObject::const_ptr_type ano = *it;
    //     //            AbstractNamedObjectContainer::const_ptr_type anoc =
    //     //                std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
    //     //            if (anoc) {
    //     //                if (numChildren == 0) {
    //     //                    answer << std::endl;
    //     //                }
    //     //                answer << anoc.get()->FullName() << std::endl;
    //     //                numChildren++;
    //     //            }
    //     //        }
    //     //        for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
    //     //            AbstractNamedObject::const_ptr_type ano = *it;
    //     //            AbstractNamedObjectContainer::const_ptr_type anoc =
    //     //                std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
    //     //            if (anoc) {
    //     //                anoStack.push_back(anoc);
    //     //            }
    //     //        }
    //     //        if (numChildren == 0) {
    //     //            answer << "none" << std::endl;
    //     //        }
    //     //    }
    //     //}
    //
    //     lua_pushstring(L, answer.str().c_str());
    //     return 1;
    // }

    // "mmListInstantiations",
    // "()\n\tReturn a list of instantiation names",
    // int mmListInstantiations(lua_State* L) {
    //
    //     std::ostringstream answer;
    //
    //     // TODO
    //
    //     //AbstractNamedObject::const_ptr_type ano = this->coreInst->ModuleGraphRoot();
    //     //AbstractNamedObjectContainer::const_ptr_type anor =
    //     //    std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
    //     //if (!ano) {
    //     //    luaApiInterpreter_.ThrowError(MMC_LUA_MMLISTINSTANTIATIONS ": no root");
    //     //    return 0;
    //     //}
    //
    //
    //     //if (anor) {
    //     //    const auto it_end = anor->ChildList_End();
    //     //    for (auto it = anor->ChildList_Begin(); it != it_end; ++it) {
    //     //        if (!dynamic_cast<const Module*>(it->get())) {
    //     //            AbstractNamedObjectContainer::const_ptr_type anoc =
    //     //                std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(*it);
    //     //            answer << anoc->FullName() << std::endl;
    //     //            // TODO: the immediate child view should be it, generally
    //     //        }
    //     //    }
    //     //}
    //
    //     lua_pushstring(L, answer.str().c_str());
    //     return 1;
    // }
}


} // namespace megamol::frontend
