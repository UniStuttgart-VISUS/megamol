/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ProjectLoader_Service.hpp"

#include <fstream>
#include <sstream>

#include "LuaApiResource.h"
#include "Window_Events.h"
#include "mmcore/LuaAPI.h"
#include "mmcore/utility/String.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"
#include "mmcore/utility/log/Log.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

static void log(const char* text) {
    const std::string msg = "ProjectLoader_Service: " + std::string(text);
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}
static void log(std::string text) {
    log(text.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = "ProjectLoader_Service: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

namespace megamol::frontend {

ProjectLoader_Service::ProjectLoader_Service() {
    // init members to default states
}

ProjectLoader_Service::~ProjectLoader_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
}

bool ProjectLoader_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool ProjectLoader_Service::init(const Config& config) {

    m_loader.load_filename = [&](std::filesystem::path const& filename) { return this->load_file(filename); };

    this->m_providedResourceReferences = {{"ProjectLoader", m_loader}};

    this->m_requestedResourcesNames = {frontend_resources::LuaAPI_Req_Name, "SetScriptPath", "optional<WindowEvents>"};

    log("initialized successfully");
    return true;
}

bool ProjectLoader_Service::load_file(std::filesystem::path const& filename) const {
#ifdef MEGAMOL_USE_TRACY
    ZoneScopedNC("ProjectLoadFile", 0xFF0000);
#endif

    // file loaders
    const auto load_lua = [](std::filesystem::path const& filename, std::string& script) -> bool {
        std::ifstream input(filename, std::ios::in);

        if (input.fail())
            return false;

        std::ostringstream buffer;
        buffer << input.rdbuf();
        script = buffer.str();
        return true;
    };

    const auto load_png = [](std::filesystem::path const& filename, std::string& script) -> bool {
        script = megamol::core::utility::graphics::ScreenShotComments::GetProjectFromPNG(filename);

        if (script.empty())
            return false;

        return true;
    };

    // do we support the file type?
    std::vector<std::tuple<const char*, std::function<bool(std::filesystem::path const&, std::string&)>>>
        supported_filetypes = {std::make_tuple(".lua", load_lua), std::make_tuple(".png", load_png)};

    auto found_it = std::find_if(supported_filetypes.begin(), supported_filetypes.end(), [&](auto& item) {
        const std::string suffix = std::get<0>(item);
        return megamol::core::utility::string::EndsWithAsciiCaseInsensitive(filename.generic_string(), suffix);
    });

    if (found_it == supported_filetypes.end()) {
        log_error("unsupported file type: " + filename.extension().generic_string());
        return false;
    }

    // extract script from file
    std::string script;
    auto& file_opener = std::get<1>(*found_it);
    bool file_ok = file_opener(filename, script);

    if (!file_ok) {
        log_error("error opening file: " + filename.generic_string());
        return false;
    }

    // run lua
    auto luaApi = m_requestedResourceReferences[0].getResource<core::LuaAPI*>();

    // TODO: remove this resource from Lua when project-centric structure is in place
    using SetScriptPath = std::function<void(std::string const&)>;
    const SetScriptPath& set_script_path = m_requestedResourceReferences[1].getResource<SetScriptPath>();

    set_script_path(filename.generic_string());

    auto result = luaApi->RunString(script);

    if (!result.valid()) {
        log_error("failed to load file " + filename.generic_string() + "\n\t" + luaApi->GetError(result));
        set_script_path("");
        return false;
    }

    log("loaded file " + filename.generic_string());
    return true;
}

void ProjectLoader_Service::close() {}

std::vector<FrontendResource>& ProjectLoader_Service::getProvidedResources() {
    return m_providedResourceReferences;
}

const std::vector<std::string> ProjectLoader_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void ProjectLoader_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    this->m_requestedResourceReferences = resources;
}

void ProjectLoader_Service::updateProvidedResources() {}

void ProjectLoader_Service::digestChangedRequestedResources() {
    // if a project file that containts mmRenderNextFrame is dropped into the window
    // we get into a recursion here where the RenderNextFrame calls back into
    // this digestion function - from where we again execute the projet file and call RenderNextFrame
    // break this recursion
    if (m_digestion_recursion)
        return;

    // execute lua files dropped into megamol window
    auto maybe_window_events =
        this->m_requestedResourceReferences[2].getOptionalResource<frontend_resources::WindowEvents>();
    if (!maybe_window_events.has_value()) {
        return;
    }

    m_digestion_recursion = true;

    auto& window_events = const_cast<frontend_resources::WindowEvents&>(maybe_window_events.value().get());

    // in mmRenderNextFrame recursion the dropped paths get cleared. remember them.
    auto possible_files = window_events.dropped_path_events;

    for (auto& events : possible_files)
        events.erase(std::remove_if(events.begin(), events.end(),
                         [&](auto& filename) -> bool { return this->load_file(filename); }),
            events.end());

    possible_files.erase(std::remove_if(possible_files.begin(), possible_files.end(),
                             [&](auto& events) -> bool { return events.empty(); }),
        possible_files.end());

    // restore, this gets cleared by the service outside of the recursion again
    window_events.dropped_path_events = possible_files;

    m_digestion_recursion = false;
}

void ProjectLoader_Service::resetProvidedResources() {}

void ProjectLoader_Service::preGraphRender() {
    // this gets called right before the graph is told to render something
    // e.g. you can start a start frame timer here

    // rendering via MegaMol View is called after this function finishes
    // in the end this calls the equivalent of ::mmcRenderView(hView, &renderContext)
    // which leads to view.Render()
}

void ProjectLoader_Service::postGraphRender() {
    // the graph finished rendering and you may more stuff here
    // e.g. end frame timer
    // update window name
    // swap buffers, glClear
}


} // namespace megamol::frontend
