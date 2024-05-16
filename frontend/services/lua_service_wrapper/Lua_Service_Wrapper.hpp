/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "AbstractFrontendService.hpp"
#include "CommonTypes.h"
#include "LuaRemoteConnectionsBroker.h"
#include "ScriptPaths.h"
#include "mmcore/LuaAPI.h"

namespace megamol::frontend {

// the Lua Service Wrapper wraps the LuaAPI for use as a frontend service
// the main problem this wrapper addresses is the requirement that Lua scripts
// want to issue a "render frame" (mmFlush, mmRenderNextFrame) command that needs to be executed as a callback in the Lua context,
// the flush has to advance the rendering for one frame, which is the job of the main loop
// but since Lua is itself part of the main loop there arises the problem of Lua recursively calling itself
//
// thus the LuaAPI has two conflicting goals:
// 1) we want it to be a frontend service like all other services
// 2) it (sometimes) needs to take control of the megamol main loop and issue a series of frame flushes from inside a
// Lua callback, without recursively calling itself
//
// this wrapper solves this issue by recognizing when it is executed from inside a Lua callback
// if executed from the normal main loop, the wrapper executes lua
// if executed from inside the lua frame flush callback, it does not call the wrapped LuaAPI object, thus eliminating
// risk of recursive Lua calls
//
// also, the wrapper manages communication with the lua remote console by
// receiving lua requests and executing them via the wrapped LuaAPI object
// the multithreaded ZMQ networking logic is implemented in LuaHostService.h in the core
class Lua_Service_Wrapper final : public AbstractFrontendService {
public:
    struct Config {
        std::string host_address;
        megamol::core::LuaAPI* lua_api_ptr =
            nullptr; // lua api object that will be used/called by the service wrapper only one level deep
        int retry_socket_port = 0;
        bool show_version_notification = true;
    };

    // sometimes somebody wants to know the name of the service
    std::string serviceName() const override {
        return "Lua_Service_Wrapper";
    }

    // constructor should not take arguments, actual object initialization deferred until init()
    Lua_Service_Wrapper();
    ~Lua_Service_Wrapper() override;
    // your service will be constructed and destructed, but not copy-constructed or move-constructed
    // so no need to worry about copy or move constructors.

    // init service with input config data, e.g. init GLFW with OpenGL and open window with certain decorations/hints
    // if init() fails return false (this will terminate program execution), on success return true
    bool init(const Config& config);
    bool init(void* configPtr) override;
    void close() override;

    std::vector<FrontendResource>& getProvidedResources() override;
    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    void updateProvidedResources() override;
    void digestChangedRequestedResources() override;
    void resetProvidedResources() override;
    void preGraphRender() override;
    void postGraphRender() override;

    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:
    Config m_config;

    int m_service_recursion_depth = 0;

    std::unique_ptr<megamol::frontend::LuaRemoteConnectionsBroker> m_network_host;

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;

    megamol::frontend_resources::ScriptPaths m_scriptpath_resource;
    std::function<void(std::string const&)> m_setScriptPath_resource;

    core::LuaAPI* luaApi_resource;

    void fill_frontend_resources_callbacks();
    void fill_graph_manipulation_callbacks();
};

} // namespace megamol::frontend
