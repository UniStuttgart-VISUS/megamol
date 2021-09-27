/*
 * ImagePresentation_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

// search/replace ImagePresentation_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
#include "ImagePresentation_Service.hpp"

#include "WindowManipulation.h"
#include "Framebuffer_Events.h"
#include "GUIState.h"

#include "ImageWrapper_to_GLTexture.h"
#include "ImagePresentation_Sinks.hpp"

#include "LuaCallbacksCollection.h"

#include "mmcore/view/AbstractView_EventConsumption.h"

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

static const std::string service_name = "ImagePresentation_Service: ";
static void log(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}

namespace megamol {
namespace frontend {

ImagePresentation_Service::ImagePresentation_Service() {
    // init members to default states
}

ImagePresentation_Service::~ImagePresentation_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does 
}

bool ImagePresentation_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool ImagePresentation_Service::init(const Config& config) {

    m_entry_points_registry_resource.add_entry_point =    [&](std::string name, void* module_raw_ptr)   -> bool { return add_entry_point(name, module_raw_ptr); };
    m_entry_points_registry_resource.remove_entry_point = [&](std::string name)                         -> bool { return remove_entry_point(name); };
    m_entry_points_registry_resource.rename_entry_point = [&](std::string oldName, std::string newName) -> bool { return rename_entry_point(oldName, newName); };
    m_entry_points_registry_resource.clear_entry_points = [&]() { clear_entry_points(); };

    m_presentation_sinks.push_back(
        {"GLFW Window Presentation Sink", [&](auto const& images) { this->present_images_to_glfw_window(images); }});

    this->m_providedResourceReferences =
    {
          {"ImagePresentationEntryPoints", m_entry_points_registry_resource} // used by MegaMolGraph to set entry points
    };

    this->m_requestedResourcesNames =
    {
          "FrontendResources" // std::vector<FrontendResource>
        , "WindowManipulation"
        , "FramebufferEvents"
        , "GUIState"
        , "RegisterLuaCallbacks"
    };

    m_framebuffer_size_from_resource_handler = [&]() -> std::pair<unsigned int, unsigned int> {
        return {m_window_framebuffer_size.first, m_window_framebuffer_size.second};
    };

    log("initialized successfully");
    return true;
}

void ImagePresentation_Service::close() {
}

std::vector<FrontendResource>& ImagePresentation_Service::getProvidedResources() {
    return m_providedResourceReferences;
}

const std::vector<std::string> ImagePresentation_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void ImagePresentation_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    this->m_requestedResourceReferences = resources;

    m_frontend_resources_ptr = & m_requestedResourceReferences[0].getResource<std::vector<megamol::frontend::FrontendResource>>();

    auto& framebuffer_events = m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FramebufferEvents>();
    m_window_framebuffer_size = {framebuffer_events.previous_state.width, framebuffer_events.previous_state.height};

    fill_lua_callbacks();
}
#define m_frontend_resources (*m_frontend_resources_ptr)

void ImagePresentation_Service::updateProvidedResources() {
}

void ImagePresentation_Service::digestChangedRequestedResources() {
    auto& framebuffer_events = m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FramebufferEvents>();

    if (framebuffer_events.size_events.size()) {
        m_window_framebuffer_size = {framebuffer_events.size_events.back().width, framebuffer_events.size_events.back().height};
    }
}

void ImagePresentation_Service::resetProvidedResources() {
}

void ImagePresentation_Service::preGraphRender() {
}

void ImagePresentation_Service::postGraphRender() {
}

void ImagePresentation_Service::RenderNextFrame() {
    for (auto& entry : m_entry_points) {

        entry.entry_point_data->update();

        entry.execute(entry.modulePtr, entry.entry_point_resources, entry.execution_result_image);
    }
}

void ImagePresentation_Service::PresentRenderedImages() {
    // pull result images into separate list
    static std::vector<ImageWrapper> wrapped_images;
    wrapped_images.clear();
    wrapped_images.reserve(m_entry_points.size());

    // rendering results are presented in order of execution of entry points
    for (auto& entry : m_entry_points) {
        wrapped_images.push_back(entry.execution_result_image);
    }

    for (auto& sink: m_presentation_sinks) {
        sink.present_images(wrapped_images);
    }
}

// clang-format off
using FrontendResource = megamol::frontend::FrontendResource;
using EntryPointExecutionCallback = megamol::frontend::ImagePresentation_Service::EntryPointExecutionCallback;

using EntryPointInitFunctions =
std::tuple<
    // rendering execution function
    EntryPointExecutionCallback,
    // get requested resources function
    std::function<std::vector<std::string>()>
>;
// clang-format on
static EntryPointInitFunctions get_init_execute_resources(void* ptr) {
    if (auto module_ptr = static_cast<megamol::core::Module*>(ptr); module_ptr != nullptr) {
        if (auto view_ptr = dynamic_cast<megamol::core::view::AbstractView*>(module_ptr); view_ptr != nullptr) {
            return EntryPointInitFunctions{
                std::function{megamol::core::view::view_rendering_execution},
                std::function{megamol::core::view::get_view_runtime_resources_requests},
            };
        }
    }

    log_error("Fatal Error setting Graph Entry Point callback functions. Unknown Entry Point type.");
    throw std::exception();
}

namespace {
    struct ViewRenderInputs : public ImagePresentation_Service::RenderInputsUpdate {
        // individual inputs used by view for rendering of next frame
        megamol::frontend_resources::RenderInput render_input;

        // sets (local) fbo resolution of render_input from various sources
        std::function<std::pair<unsigned int, unsigned int>()> render_input_framebuffer_size_handler;

        void update() override {
            auto fbo_size = render_input_framebuffer_size_handler();
            render_input.local_view_framebuffer_resolution = {fbo_size.first, fbo_size.second};
        }
    };
} // namespace
#define accessViewRenderInput(unique_ptr) (*static_cast<ViewRenderInputs*>(unique_ptr.get()))

std::tuple<
    std::vector<FrontendResource>,
    std::unique_ptr<ImagePresentation_Service::RenderInputsUpdate>
> ImagePresentation_Service::map_resources(std::vector<std::string> const& requests) {

    std::vector<FrontendResource> resources;

    // this unique_data/reindering input handler thing is a bit convoluted but the idea is that we want to give the view (or any other type of entry point)
    // the ability to get updated with newest frame data, e.g. framebuffer size
    // but we also want to maintain the "empty" handler throughout the code path to avoid checking for a null ptr
    auto unique_data = std::make_unique<RenderInputsUpdate>();

    for (auto& request: requests) {
        auto find_it = std::find_if(m_frontend_resources.begin(), m_frontend_resources.end(),
            [&](FrontendResource const& resource) { return resource.getIdentifier() == request; });
        bool found_request = find_it != m_frontend_resources.end();

        // intercept view requests for individual rendering inputs
        // which are not global resources but managed for each entry point individually
        if (request == "ViewRenderInput") {
            unique_data = std::make_unique<ViewRenderInputs>();
            accessViewRenderInput(unique_data).render_input_framebuffer_size_handler = m_framebuffer_size_from_resource_handler;

            // wrap render input for view in locally handled resource
            resources.push_back({request, accessViewRenderInput(unique_data).render_input});
            continue;
        }

        if (!found_request) {
            log_error("could not find requested resource " + request);
            return {};
        }

        resources.push_back(*find_it);
    }

    return {resources, std::move(unique_data)};
}

bool ImagePresentation_Service::add_entry_point(std::string name, void* module_raw_ptr) {
    auto [execute_etry, entry_resource_requests] = get_init_execute_resources(module_raw_ptr);

    auto resource_requests = entry_resource_requests();
    auto [resources, unique_data] = map_resources(resource_requests);

    if (resources.empty() && !resource_requests.empty()) {
        log_error("could not assign resources requested by entry point " + name + ". Entry point not created.");
        return false;
    }

    m_entry_points.push_back(GraphEntryPoint{
        name,
        module_raw_ptr,
        resources,
        std::move(unique_data), // render inputs and their update
        execute_etry,
        {name} // image
        });

    auto& entry_point = m_entry_points.back();

    return true;
}

bool ImagePresentation_Service::remove_entry_point(std::string name) {

    m_entry_points.remove_if([&](auto& entry) { return entry.moduleName == name; });

    return true;
}

bool ImagePresentation_Service::rename_entry_point(std::string oldName, std::string newName) {

    auto entry_it = std::find_if(m_entry_points.begin(), m_entry_points.end(),
        [&](auto& entry) { return entry.moduleName == oldName; });

    if (entry_it == m_entry_points.end()) {
        return false;
    }

    entry_it->moduleName = newName;

    return true;
}

bool ImagePresentation_Service::clear_entry_points() {
    m_entry_points.clear();

    return true;
}

void ImagePresentation_Service::present_images_to_glfw_window(std::vector<ImageWrapper> const& images) {
    static auto& window_manipulation       = m_requestedResourceReferences[1].getResource<megamol::frontend_resources::WindowManipulation>();
    static auto& window_framebuffer_events = m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FramebufferEvents>();
    static glfw_window_blit glfw_sink;
    // TODO: glfw_window_blit destuctor gets called after GL context died

    glfw_sink.set_framebuffer_active();

    // glfw sink needs to know current glfw framebuffer size
    auto framebuffer_width = window_framebuffer_events.previous_state.width;
    auto framebuffer_height = window_framebuffer_events.previous_state.height;
    glfw_sink.set_framebuffer_size(framebuffer_width, framebuffer_height);

    for (auto& image: images) {
        static frontend_resources::gl_texture gl_image = image;
        gl_image = image;
        glfw_sink.blit_texture(gl_image.as_gl_handle(), image.size.width, image.size.height);
    }

    // EXPERIMENTAL: until the GUI Service provides rendering of the GUI on its own
    // render UI overlay
    static auto& gui_state = m_requestedResourceReferences[3].getResource<megamol::frontend_resources::GUIState>();
    gui_state.provide_gui_render();

    window_manipulation.swap_buffers();
}

void ImagePresentation_Service::fill_lua_callbacks() {
    using megamol::frontend_resources::LuaCallbacksCollection;
    using Error = megamol::frontend_resources::LuaCallbacksCollection::Error;
    using StringResult = megamol::frontend_resources::LuaCallbacksCollection::StringResult;
    using VoidResult = megamol::frontend_resources::LuaCallbacksCollection::VoidResult;
    using DoubleResult = megamol::frontend_resources::LuaCallbacksCollection::DoubleResult;
    using BoolResult = megamol::frontend_resources::LuaCallbacksCollection::BoolResult;

    LuaCallbacksCollection callbacks;

    callbacks.add<VoidResult, std::string, int, int>(
        "mmSetViewFramebufferSize",
        "(string view, int width, int height)\n\tSet framebuffer dimensions of view to width x height.",
        {[&](std::string view, int width, int height) -> VoidResult
        {
            if (width <= 0 || height <= 0) {
                return Error {"framebuffer dimensions must be positive, but given values are: " + std::to_string(width) + " x " + std::to_string(height)};
            }

            auto entry_it = std::find_if(m_entry_points.begin(), m_entry_points.end(),
            [&](GraphEntryPoint& entry) {
                return entry.moduleName == view;
            });

            if (entry_it == m_entry_points.end()) {
                return Error {"no view found with name: " + view};
            }

            accessViewRenderInput(entry_it->entry_point_data).render_input_framebuffer_size_handler =
            [=]() -> std::pair<unsigned int, unsigned int> {
                return {width, height};
            };

            return VoidResult{};
        }});

    auto& register_callbacks = m_requestedResourceReferences[4].getResource<std::function<void(megamol::frontend_resources::LuaCallbacksCollection const&)>>();
    register_callbacks(callbacks);
}


} // namespace frontend
} // namespace megamol
