/*
 * Lua_Service_Wrapper.cpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "Lua_Service_Wrapper.hpp"

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

    this->m_providedResourceReferences = {};

    this->m_requestedResourcesNames; //= {"ZMQ_Context"};

    log("initialized successfully");

    return true;
}

void Lua_Service_Wrapper::close() {
    m_config = {}; // default to nullptr
}

std::vector<ModuleResource>& Lua_Service_Wrapper::getProvidedResources() {
    return m_providedResourceReferences; // empty
}

const std::vector<std::string> Lua_Service_Wrapper::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void Lua_Service_Wrapper::setRequestedResources(std::vector<ModuleResource> resources) {
    // TODO: do something with ZMQ resource we get here
    m_requestedResourceReferences = resources;
}

// -------- main loop callbacks ---------

#define recursion_guard \
    RecursionGuard rg{m_service_recursion_depth}; \
    if (rg.abort()) return;

void Lua_Service_Wrapper::updateProvidedResources() {
    recursion_guard;
    // we want lua to be the first thing executed in main loop
    // so we do all the lua work here

    bool need_to_shutdown = false; // e.g. mmQuit should set this to true

    need_to_shutdown |= luaAPI.getShutdown();

    if (need_to_shutdown) this->setShutdown();
}

void Lua_Service_Wrapper::digestChangedRequestedResources() { recursion_guard; }

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

// this will be removed soon and is only here for compatibility reasons
const void* Lua_Service_Wrapper::getSharedDataPtr() const { return nullptr; }


} // namespace frontend
} // namespace megamol
