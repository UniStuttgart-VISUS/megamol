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

#include "mmcore/utility/LuaHostService.h"

#include "Screenshots.h"
#include "FrameStatistics.h"
#include "WindowManipulation.h"
#include "GUI_Resource.h"
#include "GlobalValueStore.h"

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"
static void log(const char* text) {
    const std::string msg = "Lua_Service_Wrapper: " + std::string(text) + "\n";
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}
static void log(std::string text) { log(text.c_str()); }

namespace {
    // used to abort a service callback if we are already inside a service wrapper callback
    struct RecursionGuard {
        int& state;

        RecursionGuard(int& s) : state{s} { state++; }
        ~RecursionGuard() { state--; }
        bool abort() { return state > 1; }
    };
}



namespace megamol {
namespace frontend {

Lua_Service_Wrapper::Lua_Service_Wrapper() {
    // init members to default states
}

Lua_Service_Wrapper::~Lua_Service_Wrapper() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
}

bool Lua_Service_Wrapper::init(void* configPtr) {
    if (configPtr == nullptr) return false;

    return init(*static_cast<Config*>(configPtr));
}

#define luaAPI (*m_config.lua_api_ptr)
#define m_network_host reinterpret_cast<megamol::core::utility::LuaHostNetworkConnectionsBroker*>(m_network_host_pimpl.get())

bool Lua_Service_Wrapper::init(const Config& config) {
    if (!config.lua_api_ptr) {
        log("failed initialization because LuaAPI is nullptr");
        return false;
    }

    m_config = config;

    m_executeLuaScript_resource =
        [&](std::string const& script) -> std::tuple<bool,std::string> {
            std::string result_str;
            bool result_b = luaAPI.RunString(script, result_str);
            return {result_b, result_str};
        };

    m_setScriptPath_resource = [&](std::string const& script_path) -> void {
        luaAPI.SetScriptPath(script_path);
    };

    m_registerLuaCallbacks_resource = [&](megamol::frontend_resources::LuaCallbacksCollection const& callbacks) {
        luaAPI.AddCallbacks(callbacks);
    };

    this->m_providedResourceReferences =
    {
        {"LuaScriptPaths", m_scriptpath_resource},
        {"ExecuteLuaScript", m_executeLuaScript_resource},
        {"SetScriptPath", m_setScriptPath_resource},
        {"RegisterLuaCallbacks", m_registerLuaCallbacks_resource},
    };

    this->m_requestedResourcesNames = 
    {
        "FrontendResourcesList",
        "GLFrontbufferToPNG_ScreenshotTrigger", // for screenshots
        "FrameStatistics", // for LastFrameTime
        "WindowManipulation", // for Framebuffer resize
        "GUIResource", // propagate GUI state and visibility
        "MegaMolGraph", // LuaAPI manipulates graph
        "RenderNextFrame", // LuaAPI can render one frame
        "GlobalValueStore" // LuaAPI can read and set global values
    }; //= {"ZMQ_Context"};

    m_network_host_pimpl = std::unique_ptr<void, std::function<void(void*)>>(
        new megamol::core::utility::LuaHostNetworkConnectionsBroker{},
        [](void* ptr) { delete reinterpret_cast<megamol::core::utility::LuaHostNetworkConnectionsBroker*>(ptr); }
    );

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

    luaAPI.AddCallbacks(frontend_resource_callbacks);
}

// -------- main loop callbacks ---------

#define recursion_guard \
    RecursionGuard rg{m_service_recursion_depth}; \
    if (rg.abort()) return;

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

void Lua_Service_Wrapper::resetProvidedResources() { recursion_guard; }

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

    auto& callbacks = *reinterpret_cast<LuaCallbacksCollection*>(callbacks_collection_ptr);

    callbacks.add<StringResult>(
        "mmListResources",
        "()\n\tReturn a list of available resources in the frontend.",
        std::function{[&]() -> StringResult {
            auto resources_list = m_requestedResourceReferences[0].getResource<std::function<std::vector<std::string>(void)>>()();
            std::ostringstream answer;

            for (auto& resource_name: resources_list) {
                answer << resource_name << std::endl;
            }

            if (resources_list.empty()) {
                answer << "(none)" << std::endl;
            }

            return StringResult{answer.str().c_str()};
        }});

    callbacks.add<VoidResult, std::string>(
        "mmScreenshot",
        "(string filename)\n\tSave a screen shot of the GL front buffer under 'filename'.",
        std::function{[&](std::string file) -> VoidResult
        {
            m_requestedResourceReferences[1].getResource<std::function<bool(std::string const&)> >()(file);
            return VoidResult{};
        }});

    callbacks.add<DoubleResult>(
        "mmLastFrameTime",
        "()\n\tReturns the graph execution time of the last frame in ms.",
        std::function{[&]() -> DoubleResult
        {
            auto& frame_statistics = m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FrameStatistics>();
            return DoubleResult{frame_statistics.last_rendered_frame_time_milliseconds};
        }});

    callbacks.add<VoidResult, int, int>(
        "mmSetFramebufferSize",
        "(int width, int height)\n\tSet framebuffer dimensions to width x height.",
        std::function{[&](int width, int height) -> VoidResult
        {
            if (width <= 0 || height <= 0) {
                return Error {"framebuffer dimensions must be positive, but given values are: " + std::to_string(width) + " x " + std::to_string(height)};
            }

            auto& window_manipulation = m_requestedResourceReferences[3].getResource<megamol::frontend_resources::WindowManipulation>();
            window_manipulation.set_framebuffer_size(width, height);
            return VoidResult{};
        }});

    callbacks.add<VoidResult, int, int>(
        "mmSetWindowPosition",
        "(int x, int y)\n\tSet window position to x,y.",
        std::function{[&](int x, int y) -> VoidResult
        {
            auto& window_manipulation = m_requestedResourceReferences[3].getResource<megamol::frontend_resources::WindowManipulation>();
            window_manipulation.set_window_position(x, y);
            return VoidResult{};
        }});

    callbacks.add<VoidResult, bool>(
        "mmSetFullscreen",
        "(bool fullscreen)\n\tSet window to fullscreen (or restore).",
        std::function{[&](bool fullscreen) -> VoidResult
        {
            auto& window_manipulation = m_requestedResourceReferences[3].getResource<megamol::frontend_resources::WindowManipulation>();
            window_manipulation.set_fullscreen(fullscreen?frontend_resources::WindowManipulation::Fullscreen::Maximize:frontend_resources::WindowManipulation::Fullscreen::Restore);
            return VoidResult{};
        }});

    callbacks.add<VoidResult, bool>(
        "mmSetVSync",
        "(bool state)\n\tSet window VSync off (false) or on (true).",
        std::function{[&](bool state) -> VoidResult
        {
            auto& window_manipulation = m_requestedResourceReferences[3].getResource<megamol::frontend_resources::WindowManipulation>();
            window_manipulation.set_swap_interval(state ? 1 : 0);
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string>(
        "mmSetGUIState",
        "(string json)\n\tSet GUI state from given 'json' string.",
        std::function{[&](std::string json) -> VoidResult
        {
            auto& gui_resource =  m_requestedResourceReferences[4].getResource<megamol::frontend_resources::GUIResource>();
            gui_resource.provide_gui_state(json);
            return VoidResult{};
        }});

    callbacks.add<VoidResult, bool>(
        "mmShowGUI",
        "(bool state)\n\tShow (true) or hide (false) the GUI.",
        std::function{[&](bool show) -> VoidResult
        {
            auto& gui_resource = m_requestedResourceReferences[4].getResource<megamol::frontend_resources::GUIResource>();
            gui_resource.provide_gui_visibility(show);
            return VoidResult{};
        }});

    callbacks.add<VoidResult, float>(
        "mmScaleGUI",
        "(float scale)\n\tSet GUI scaling factor.",
        std::function{[&](float scale) -> VoidResult
        {
            auto& gui_resource = m_requestedResourceReferences[4].getResource<megamol::frontend_resources::GUIResource>();
            gui_resource.provide_gui_scale(scale);
            return VoidResult{};
        }});

    callbacks.add<VoidResult>(
        "mmQuit",
        "()\n\tClose the MegaMol instance.",
        std::function{[&]() -> VoidResult
        {
            this->setShutdown();
            return VoidResult{};
        }});

    callbacks.add<VoidResult>(
        "mmRenderNextFrame",
        "()\n\tAdvances rendering by one frame by poking the main rendering loop.",
        std::function{[&]() -> VoidResult
        {
            auto& render_next_frame = m_requestedResourceReferences[6].getResource<std::function<bool()>>();
            render_next_frame();
            return VoidResult{};
        }});

    callbacks.add<VoidResult, std::string, std::string>(
        "mmSetGlobalValue",
        "(string key, string value)\n\t Sets a global key-value pair. If the key is already present, overwrites the value.",
        std::function{[&](std::string key, std::string value) -> VoidResult
        {
            auto& global_value_store = const_cast<megamol::frontend_resources::GlobalValueStore&>(m_requestedResourceReferences[7].getResource<megamol::frontend_resources::GlobalValueStore>());
            global_value_store.insert(key, value);
            return VoidResult{};
        }});

    callbacks.add<StringResult, std::string>(
        "mmGetGlobalValue",
        "(string key)\n\t Returns the value for the given global key. If no key with that name is known, returns empty string.",
        std::function{[&](std::string key) -> StringResult
        {
            auto& global_value_store = m_requestedResourceReferences[7].getResource<megamol::frontend_resources::GlobalValueStore>();
            std::optional<std::string> maybe_value = global_value_store.maybe_get(key);

            if (maybe_value.has_value()) {
                return StringResult{maybe_value.value()};
            }

            // TODO: maybe we want to LuaError?
            return StringResult{""};
        }});

    // mmLoadProject ?
    // the ProjectLoader resource immediately executes the file contents as lua code
    // -> what happens if this is done inside a lua callback?

    // template for futher callbacks
    //frontend_resource_callbacks.add<>(
    //    "name",
    //    "()\n\t help",
    //    std::function{[&]() -> VoidResult
    //    {
    //        return VoidResult{};
    //    }});
}









}


} // namespace frontend
} // namespace megamol
