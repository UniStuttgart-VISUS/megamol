/*
 * Lua_Service_Wrapper.cpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

// TODO: we need this #define because inclusion of LuaHostService.h leads to windows header inclusion errors.
// this stems from linking ZMQ via CMake now being PUBLIC in the core lib. i dont know how to solve this "the right way".
#define _WINSOCKAPI_
#include "Lua_Service_Wrapper.hpp"

#include "LuaRemoteConnectionsBroker.h"

#include "CommandRegistry.h"
#include "FrameStatistics.h"
#include "GUIState.h"
#include "GlobalValueStore.h"
#include "RuntimeConfig.h"
#include "Screenshots.h"
#include "WindowManipulation.h"
#include "vislib/UTF8Encoder.h"


// local logging wrapper for your convenience until central MegaMol logger established
#include "GUIRegisterWindow.h"
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


namespace megamol {
namespace frontend {

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
#define m_network_host \
    reinterpret_cast<megamol::frontend_resources::LuaRemoteConnectionsBroker*>(m_network_host_pimpl.get())

bool Lua_Service_Wrapper::init(const Config& config) {
    if (!config.lua_api_ptr) {
        log("failed initialization because LuaAPI is nullptr");
        return false;
    }

    m_config = config;

    m_executeLuaScript_resource = [&](std::string const& script) -> std::tuple<bool, std::string> {
        std::string result_str;
        bool result_b = luaAPI.RunString(script, result_str);
        return {result_b, result_str};
    };

    m_setScriptPath_resource = [&](std::string const& script_path) -> void { luaAPI.SetScriptPath(script_path); };

    m_registerLuaCallbacks_resource = [&](megamol::frontend_resources::LuaCallbacksCollection const& callbacks) {
        luaAPI.AddCallbacks(callbacks);
    };

    this->m_providedResourceReferences = {
        {"LuaScriptPaths", m_scriptpath_resource},
        {"ExecuteLuaScript", m_executeLuaScript_resource},
        {"SetScriptPath", m_setScriptPath_resource},
        {"RegisterLuaCallbacks", m_registerLuaCallbacks_resource},
    };

    this->m_requestedResourcesNames = {"FrontendResourcesList",
        "GLFrontbufferToPNG_ScreenshotTrigger", // for screenshots
        "FrameStatistics",                      // for LastFrameTime
        "optional<WindowManipulation>",         // for Framebuffer resize
        "optional<GUIState>",                   // propagate GUI state and visibility
        "MegaMolGraph",                         // LuaAPI manipulates graph
        "RenderNextFrame",                      // LuaAPI can render one frame
        "GlobalValueStore",                     // LuaAPI can read and set global values
        frontend_resources::CommandRegistry_Req_Name, "optional<GUIRegisterWindow>", "RuntimeConfig",
#ifdef PROFILING
        frontend_resources::PerformanceManager_Req_Name
#endif
    }; //= {"ZMQ_Context"};

    *open_version_notification = false;

    m_network_host_pimpl =
        std::unique_ptr<void, std::function<void(void*)>>(new megamol::frontend_resources::LuaRemoteConnectionsBroker{},
            [](void* ptr) { delete reinterpret_cast<megamol::frontend_resources::LuaRemoteConnectionsBroker*>(ptr); });

    bool host_ok = m_network_host->spawn_connection_broker(m_config.host_address, m_config.retry_socket_port);

    if (host_ok) {
        log("initialized successfully");
    } else {
        log("failed to start lua host");
    }

    return host_ok;
}

void Lua_Service_Wrapper::close() {
    m_config = {}; // default to nullptr

    m_network_host->close();
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

    using megamol::frontend_resources::LuaCallbacksCollection;
    LuaCallbacksCollection frontend_resource_callbacks;

    fill_frontend_resources_callbacks(&frontend_resource_callbacks);
    fill_graph_manipulation_callbacks(&frontend_resource_callbacks);

    luaAPI.AddCallbacks(frontend_resource_callbacks);

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
    recursion_guard;
    // we want lua to be the first thing executed in main loop
    // so we do all the lua work here

    m_scriptpath_resource.lua_script_paths.clear();
    m_scriptpath_resource.lua_script_paths.push_back(luaAPI.GetScriptPath());

    bool need_to_shutdown = false; // e.g. mmQuit should set this to true

    // fetch Lua requests from ZMQ queue, execute, and give back result
    if (!m_network_host->request_queue.empty()) {
        auto lua_requests = std::move(m_network_host->get_request_queue());
        std::string result;
        while (!lua_requests.empty()) {
            auto& request = lua_requests.front();

            luaAPI.RunString(request.request, result);
            request.answer_promise.get().set_value(result);

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


void Lua_Service_Wrapper::fill_frontend_resources_callbacks(void* callbacks_collection_ptr) {
    using megamol::frontend_resources::LuaCallbacksCollection;
    using Error = megamol::frontend_resources::LuaCallbacksCollection::Error;
    using StringResult = megamol::frontend_resources::LuaCallbacksCollection::StringResult;
    using VoidResult = megamol::frontend_resources::LuaCallbacksCollection::VoidResult;
    using DoubleResult = megamol::frontend_resources::LuaCallbacksCollection::DoubleResult;
    using BoolResult = megamol::frontend_resources::LuaCallbacksCollection::BoolResult;

    auto& callbacks = *reinterpret_cast<LuaCallbacksCollection*>(callbacks_collection_ptr);

    callbacks.add<StringResult>(
        "mmGetMegaMolExecutableDirectory", "()\n\tReturns the directory of the running MegaMol executable.", {[&]() {
            auto& path = m_requestedResourceReferences[10]
                             .getResource<frontend_resources::RuntimeConfig>()
                             .megamol_executable_directory;
            return StringResult{path};
        }});

    callbacks.add<StringResult>(
        "mmListResources", "()\n\tReturn a list of available resources in the frontend.", {[&]() -> StringResult {
            auto resources_list =
                m_requestedResourceReferences[0].getResource<std::function<std::vector<std::string>(void)>>()();
            std::ostringstream answer;

            for (auto& resource_name : resources_list) {
                answer << resource_name << std::endl;
            }

            if (resources_list.empty()) {
                answer << "(none)" << std::endl;
            }

            return StringResult{answer.str().c_str()};
        }});

    callbacks.add<VoidResult, std::string>("mmScreenshot",
        "(string filename)\n\tSave a screen shot of the GL front buffer under 'filename'.",
        {[&](std::string file) -> VoidResult {
            m_requestedResourceReferences[1].getResource<std::function<bool(std::filesystem::path const&)>>()(
                std::filesystem::u8path(file));
            return VoidResult{};
        }});

    callbacks.add<DoubleResult>(
        "mmLastFrameTime", "()\n\tReturns the graph execution time of the last frame in ms.", {[&]() -> DoubleResult {
            auto& frame_statistics =
                m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FrameStatistics>();
            return DoubleResult{frame_statistics.last_rendered_frame_time_milliseconds};
        }});

    callbacks.add<DoubleResult>("mmLastRenderedFramesCount",
        "()\n\tReturns the number of rendered frames up until to the last frame.", {[&]() -> DoubleResult {
            auto& frame_statistics =
                m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FrameStatistics>();
            return DoubleResult{static_cast<double>(frame_statistics.rendered_frames_count)};
        }});


    auto maybe_window_manipulation =
        m_requestedResourceReferences[3].getOptionalResource<megamol::frontend_resources::WindowManipulation>();
    if (maybe_window_manipulation.has_value()) {
        frontend_resources::WindowManipulation& window_manipulation =
            const_cast<frontend_resources::WindowManipulation&>(maybe_window_manipulation.value().get());

        callbacks.add<VoidResult, int, int>("mmSetWindowFramebufferSize",
            "(int width, int height)\n\tSet framebuffer dimensions of window to width x height.",
            {[&](int width, int height) -> VoidResult {
                if (width <= 0 || height <= 0) {
                    return Error{"framebuffer dimensions must be positive, but given values are: " +
                                 std::to_string(width) + " x " + std::to_string(height)};
                }

                window_manipulation.set_framebuffer_size(width, height);
                return VoidResult{};
            }});

        callbacks.add<VoidResult, int, int>(
            "mmSetWindowPosition", "(int x, int y)\n\tSet window position to x,y.", {[&](int x, int y) -> VoidResult {
                window_manipulation.set_window_position(x, y);
                return VoidResult{};
            }});

        callbacks.add<VoidResult, bool>("mmSetFullscreen",
            "(bool fullscreen)\n\tSet window to fullscreen (or restore).", {[&](bool fullscreen) -> VoidResult {
                window_manipulation.set_fullscreen(fullscreen
                                                       ? frontend_resources::WindowManipulation::Fullscreen::Maximize
                                                       : frontend_resources::WindowManipulation::Fullscreen::Restore);
                return VoidResult{};
            }});

        callbacks.add<VoidResult, bool>(
            "mmSetVSync", "(bool state)\n\tSet window VSync off (false) or on (true).", {[&](bool state) -> VoidResult {
                window_manipulation.set_swap_interval(state ? 1 : 0);
                return VoidResult{};
            }});
    }

    auto maybe_gui_state =
        m_requestedResourceReferences[4].getOptionalResource<megamol::frontend_resources::GUIState>();
    if (maybe_gui_state.has_value()) {
        auto& gui_resource = maybe_gui_state.value().get();

        callbacks.add<VoidResult, std::string>("mmSetGUIState",
            "(string json)\n\tSet GUI state from given 'json' string.", {[&](std::string json) -> VoidResult {
                gui_resource.provide_gui_state(json);
                return VoidResult{};
            }});
        callbacks.add<StringResult>(
            "mmGetGUIState", "()\n\tReturns the GUI state as json string.", {[&]() -> StringResult {
                auto s = gui_resource.request_gui_state(false);
                return StringResult{s};
            }});

        callbacks.add<VoidResult, bool>(
            "mmSetGUIVisible", "(bool state)\n\tShow (true) or hide (false) the GUI.", {[&](bool show) -> VoidResult {
                gui_resource.provide_gui_visibility(show);
                return VoidResult{};
            }});
        callbacks.add<StringResult>(
            "mmGetGUIVisible", "()\n\tReturns whether the GUI is visible (true/false).", {[&]() -> StringResult {
                const auto visible = gui_resource.request_gui_visibility();
                return StringResult{visible ? "true" : "false"};
            }});

        callbacks.add<VoidResult, float>(
            "mmSetGUIScale", "(float scale)\n\tSet GUI scaling factor.", {[&](float scale) -> VoidResult {
                gui_resource.provide_gui_scale(scale);
                return VoidResult{};
            }});
        callbacks.add<StringResult>("mmGetGUIScale", "()\n\tReturns the GUI scaling as float.", {[&]() -> StringResult {
            const auto scale = gui_resource.request_gui_scale();
            return StringResult{std::to_string(scale)};
        }});
    }

    callbacks.add<VoidResult>("mmQuit", "()\n\tClose the MegaMol instance.", {[&]() -> VoidResult {
        this->setShutdown();
        return VoidResult{};
    }});

    callbacks.add<VoidResult>("mmRenderNextFrame",
        "()\n\tAdvances rendering by one frame by poking the main rendering loop.", {[&]() -> VoidResult {
            auto& render_next_frame = m_requestedResourceReferences[6].getResource<std::function<bool()>>();
            render_next_frame();
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string>("mmSetGlobalValue",
        "(string key, string value)\n\tSets a global key-value pair. If the key is already present, overwrites the "
        "value.",
        {[&](std::string key, std::string value) -> VoidResult {
            auto& global_value_store = const_cast<megamol::frontend_resources::GlobalValueStore&>(
                m_requestedResourceReferences[7].getResource<megamol::frontend_resources::GlobalValueStore>());
            global_value_store.insert(key, value);
            return VoidResult{};
        }});

    callbacks.add<StringResult, std::string>("mmGetGlobalValue",
        "(string key)\n\tReturns the value for the given global key. If no key with that name is known, returns empty "
        "string.",
        {[&](std::string key) -> StringResult {
            auto& global_value_store =
                m_requestedResourceReferences[7].getResource<megamol::frontend_resources::GlobalValueStore>();
            std::optional<std::string> maybe_value = global_value_store.maybe_get(key);

            if (maybe_value.has_value()) {
                return StringResult{maybe_value.value()};
            }

            // TODO: maybe we want to LuaError?
            return StringResult{""};
        }});

    callbacks.add<VoidResult, std::string>("mmExecCommand",
        "(string command)\n\tExecutes a command as provided by a hotkey, for example.",
        {[&](std::string command) -> VoidResult {
            auto& command_registry =
                m_requestedResourceReferences[8].getResource<megamol::frontend_resources::CommandRegistry>();
            command_registry.exec_command(command);
            return VoidResult{};
        }});

    callbacks.add<StringResult>("mmListCommands", "()\n\tLists the available commands.", {[&]() -> StringResult {
        auto& command_registry =
            m_requestedResourceReferences[8].getResource<megamol::frontend_resources::CommandRegistry>();
        auto& l = command_registry.list_commands();
        std::string output;
        std::for_each(l.begin(), l.end(),
            [&](const frontend_resources::Command& c) { output = output + c.name + ", " + c.key.ToString() + "\n"; });
        return StringResult{output};
    }});

    callbacks.add<BoolResult, std::string>("mmCheckVersion",
        "(string version)\n\tChecks whether the running MegaMol corresponds to version.",
        {[&](std::string version) -> BoolResult {
            bool version_ok = version == megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH();
            *open_version_notification = (!version_ok && m_config.show_version_notification);
            if (!version_ok) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "Version info in project (%s) does not match MegaMol version (%s)!", version.c_str(),
                    megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH().c_str());
            }
            return BoolResult(version_ok);
        }});

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

void Lua_Service_Wrapper::fill_graph_manipulation_callbacks(void* callbacks_collection_ptr) {
    using megamol::frontend_resources::LuaCallbacksCollection;
    using Error = megamol::frontend_resources::LuaCallbacksCollection::Error;
    using StringResult = megamol::frontend_resources::LuaCallbacksCollection::StringResult;
    using VoidResult = megamol::frontend_resources::LuaCallbacksCollection::VoidResult;
    using DoubleResult = megamol::frontend_resources::LuaCallbacksCollection::DoubleResult;

    auto& callbacks = *reinterpret_cast<LuaCallbacksCollection*>(callbacks_collection_ptr);
    auto& graph = const_cast<megamol::core::MegaMolGraph&>(
        m_requestedResourceReferences[5].getResource<megamol::core::MegaMolGraph>());

    callbacks.add<VoidResult, std::string, std::string, std::string>("mmCreateView",
        "(string graphName, string className, string moduleName)\n\tCreate a view module instance of class <className> "
        "called <moduleName>. The view module is registered as graph entry point. <graphName> is ignored.",
        {[&](std::string baseName, std::string className, std::string instanceName) -> VoidResult {
            if (!graph.CreateModule(className, instanceName)) {
                return Error{
                    "graph could not create module for: " + baseName + " , " + className + " , " + instanceName};
            }

            if (!graph.SetGraphEntryPoint(instanceName)) {
                return Error{"graph could not set graph entry point for: " + baseName + " , " + className + " , " +
                             instanceName};
            }

            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string>("mmCreateModule",
        "(string className, string moduleName)\n\tCreate a module instance of class <className> called <moduleName>.",
        {[&](std::string className, std::string instanceName) -> VoidResult {
            if (!graph.CreateModule(className, instanceName)) {
                return Error{"graph could not create module: " + className + " , " + instanceName};
            }
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string>("mmDeleteModule", "(string name)\n\tDelete the module called <name>.",
        {[&](std::string moduleName) -> VoidResult {
            if (!graph.DeleteModule(moduleName)) {
                return Error{"graph could not delete module: " + moduleName};
            }
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string>("mmRenameModule", "(string oldName, string newName)\n\tRenames the module called <oldname> to <newname>.",
        {[&](std::string oldName, std::string newName) -> VoidResult {
            if (!graph.RenameModule(oldName, newName)) {
                return Error{"graph could not rename module: " + oldName + " to " + newName};
            }
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string, std::string>("mmCreateCall",
        "(string className, string from, string to)\n\tCreate a call of type <className>, connecting CallerSlot <from> "
        "and CalleeSlot <to>.",
        {[&](std::string className, std::string from, std::string to) -> VoidResult {
            if (!graph.CreateCall(className, from, to)) {
                return Error{"graph could not create call: " + className + " , " + from + " -> " + to};
            }
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string>("mmDeleteCall",
        "(string from, string to)\n\tDelete the call connecting CallerSlot <from> and CalleeSlot <to>.",
        {[&](std::string from, std::string to) -> VoidResult {
            if (!graph.DeleteCall(from, to)) {
                return Error{"graph could not delete call: " + from + " -> " + to};
            }
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string, std::string>("mmCreateChainCall",
        "(string className, string chainStart, string to)\n\tAppend a call of type <className>, connection the "
        "rightmost CallerSlot starting at <chainStart> and CalleeSlot <to>.",
        {[&](std::string className, std::string chainStart, std::string to) -> VoidResult {
            if (!graph.Convenience().CreateChainCall(className, chainStart, to)) {
                return Error{"graph could not create chain call: " + className + " , " + chainStart + " -> " + to};
            }
            return VoidResult{};
        }});

    callbacks.add<StringResult, std::string>("mmGetModuleParams",
        "(string name)\n\tReturns a 0x1-separated list of module name and all parameters.\n\tFor each parameter the "
        "name, description, definition, and value are returned.",
        {[&](std::string moduleName) -> StringResult {
            auto mod = graph.FindModule(moduleName);
            if (mod == nullptr) {
                return Error{"graph could not find module: " + moduleName};
            }

            auto slots = graph.EnumerateModuleParameterSlots(moduleName);
            std::ostringstream answer;
            answer << mod->FullName() << "\1";

            for (auto& ps : slots) {
                answer << ps->Name() << "\1";
                answer << ps->Description() << "\1";
                auto par = ps->Parameter();
                if (par.IsNull()) {
                    return Error{
                        "ParamSlot does not seem to hold a parameter: " + std::string(ps->FullName().PeekBuffer())};
                }
                answer << par->ValueString() << "\1";
            }

            return StringResult{answer.str()};
        }});

    callbacks.add<StringResult, std::string>("mmGetParamDescription",
        "(string name)\n\tReturn the description of a parameter slot.", {[&](std::string paramName) -> StringResult {
            core::param::ParamSlot* ps = graph.FindParameterSlot(paramName);
            if (ps == nullptr) {
                return Error{"graph could not find parameter: " + paramName};
            }

            vislib::StringA valUTF8;
            vislib::UTF8Encoder::Encode(valUTF8, ps->Description());

            return StringResult{valUTF8.PeekBuffer()};
        }});

    callbacks.add<StringResult, std::string>("mmGetParamValue",
        "(string name)\n\tReturn the value of a parameter slot.", {[&](std::string paramName) -> StringResult {
            const auto* param = graph.FindParameter(paramName);
            if (param == nullptr) {
                return Error{"graph could not find parameter: " + paramName};
            }

            return StringResult{param->ValueString()};
        }});

    callbacks.add<VoidResult, std::string, std::string>("mmSetParamValue",
        "(string name, string value)\n\tSet the value of a parameter slot.",
        {[&](std::string paramName, std::string paramValue) -> VoidResult {
            auto* param = graph.FindParameter(paramName);
            if (param == nullptr) {
                return Error{"graph could not find parameter: " + paramName};
            }

            if (!param->ParseValue(paramValue.c_str())) {
                return Error{"parameter could not be set to value: " + paramName + " : " + paramValue};
            }

            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string>("mmCreateParamGroup",
        "(string name, string size)\n\tGenerate a param group that can only be set at once. Sets are queued until size "
        "is reached.",
        {[&](std::string groupName) -> VoidResult {
            graph.Convenience().CreateParameterGroup(groupName);
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string, std::string>("mmSetParamGroupValue",
        "(string groupname, string paramname, string value)\n\tQueue the value of a grouped parameter.",
        {[&](std::string paramGroup, std::string paramName, std::string paramValue) -> VoidResult {
            auto groupPtr = graph.Convenience().FindParameterGroup(paramGroup);
            if (!groupPtr) {
                return Error{"graph could not find parameter group: " + paramGroup};
            }

            bool queued = groupPtr->QueueParameterValue(paramName, paramValue);
            if (!queued) {
                return Error{
                    "graph could not queue param group value: " + paramGroup + " , " + paramName + " : " + paramValue};
            }
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string>("mmApplyParamGroupValues",
        "(string groupname)\n\tApply queued parameter values of group to graph.",
        {[&](std::string paramGroup) -> VoidResult {
            auto groupPtr = graph.Convenience().FindParameterGroup(paramGroup);
            if (!groupPtr) {
                return Error{"graph could not apply param group: no such group: " + paramGroup};
            }

            bool applied = groupPtr->ApplyQueuedParameterValues();
            if (!applied) {
                return Error{"graph could not apply param group: some parameter values did not parse."};
            }
            return VoidResult{};
        }});

    callbacks.add<StringResult, std::string>("mmListModules",
        "(string basemodule_or_namespace)\n\tReturn a list of instantiated modules (class id, instance id), starting "
        "from a certain module downstream or inside a namespace.\n\tWill use the graph root if an empty string is "
        "passed.",
        {[&](std::string starting_point) -> StringResult {
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

            return StringResult{answer.str().c_str()};
        }});

    callbacks.add<VoidResult>(
        "mmClearGraph", "()\n\tClear the MegaMol Graph from all Modules and Calls", {[&]() -> VoidResult {
            graph.Clear();
            return VoidResult{};
        }});

    callbacks.add<StringResult>("mmListCalls", "()\n\tReturn a list of instantiated calls.", {[&]() -> StringResult {
        std::ostringstream answer;
        auto& calls_list = graph.ListCalls();
        for (auto& call : calls_list) {
            answer << call.callPtr->ClassName() << ";" << call.callPtr->PeekCallerSlot()->Parent()->Name() << ","
                   << call.callPtr->PeekCalleeSlot()->Parent()->Name() << ";" << call.callPtr->PeekCallerSlot()->Name()
                   << "," << call.callPtr->PeekCalleeSlot()->Name() << std::endl;
        }

        if (calls_list.empty()) {
            answer << "(none)" << std::endl;
        }

        return StringResult{answer.str().c_str()};
    }});
#ifdef PROFILING
    callbacks.add<StringResult, std::string>("mmListModuleTimers",
        "(string name)\n\tList the registered timers of a module.", {[&](std::string name) -> StringResult {
            auto perf_manager = const_cast<megamol::frontend_resources::PerformanceManager*>(
                &this->m_requestedResourceReferences[11]
                     .getResource<megamol::frontend_resources::PerformanceManager>());
            std::stringstream output;
            auto m = graph.FindModule(name);
            if (m) {
                auto timers = perf_manager->lookup_timers(m.get());
                for (auto& t : timers) {
                    auto& timer = perf_manager->lookup_config(t);
                    output << t << ": " << timer.name << " ("
                           << megamol::frontend_resources::PerformanceManager::query_api_string(timer.api) << ")"
                           << std::endl;
                }
            }
            return StringResult{output.str()};
        }});
    callbacks.add<VoidResult, int, std::string>("mmSetTimerComment",
        "(int handle, string comment)\n\tSet a transient comment for a timer; will show up in profiling log.",
        {[&](int handle, std::string comment) -> VoidResult {
            auto perf_manager = const_cast<megamol::frontend_resources::PerformanceManager*>(
                &this->m_requestedResourceReferences[11]
                     .getResource<megamol::frontend_resources::PerformanceManager>());
            perf_manager->set_transient_comment(handle, comment);
            return VoidResult{};
        }});
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


} // namespace frontend
} // namespace megamol
