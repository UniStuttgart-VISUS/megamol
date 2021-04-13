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

#include "ImageWrapper_to_GLTexture.h"
#include "ImagePresentation_Sinks.hpp"

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
}
#define m_frontend_resources (*m_frontend_resources_ptr)

void ImagePresentation_Service::updateProvidedResources() {
}

void ImagePresentation_Service::digestChangedRequestedResources() {

}

void ImagePresentation_Service::resetProvidedResources() {
}

void ImagePresentation_Service::preGraphRender() {
}

void ImagePresentation_Service::postGraphRender() {
}

void ImagePresentation_Service::RenderNextFrame() {
    for (auto& entry : m_entry_points) {
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
    // set inital state function
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
                std::function{megamol::core::view::view_init_rendering_state},
                std::function{megamol::core::view::get_gl_view_runtime_resources_requests}
            };
        }
    }

    log_error("Fatal Error setting Graph Entry Point callback functions. Unknown Entry Point type.");
    throw std::exception();
}

std::vector<FrontendResource> ImagePresentation_Service::map_resources(std::vector<std::string> const& requests) {
    std::vector<FrontendResource> result;

    for (auto& request: requests) {
        auto find_it = std::find_if(m_frontend_resources.begin(), m_frontend_resources.end(),
            [&](FrontendResource const& resource) { return resource.getIdentifier() == request; });
        bool found_request = find_it != m_frontend_resources.end();


        if (!found_request) {
            log_error("could not find requested resource " + request);
            return {};
        }

        result.push_back(*find_it);
    }

    return result;
}

bool ImagePresentation_Service::add_entry_point(std::string name, void* module_raw_ptr) {
    auto [execute_etry, init_entry, entry_resource_requests] = get_init_execute_resources(module_raw_ptr);

    auto resource_requests = entry_resource_requests();
    auto resources = map_resources(resource_requests);

    if (resources.empty() && !resource_requests.empty()) {
        log_error("could not assign resources requested by entry point " + name + ". Entry point not created.");
        return false;
    }

    m_entry_points.push_back(GraphEntryPoint{
        name,
        module_raw_ptr,
        resources,
        execute_etry,
        {name} // image
        });

    auto& entry_point = m_entry_points.back();

    if (!init_entry(entry_point.modulePtr, entry_point.entry_point_resources, entry_point.execution_result_image)) {
        log_error("init function for entry point " + entry_point.moduleName + " failed. Entry point not created.");
        m_entry_points.pop_back();
        return false;
    }

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

    // glfw sink needs to know current glfw framebuffer size
    auto framebuffer_width = window_framebuffer_events.previous_state.width;
    auto framebuffer_height = window_framebuffer_events.previous_state.height;
    glfw_sink.set_framebuffer_size(framebuffer_width, framebuffer_height);

    for (auto& image: images) {
        static frontend_resources::gl_texture gl_image = image;
        glfw_sink.blit_texture(gl_image.as_gl_handle(), image.size.width, image.size.height);
    }

    window_manipulation.swap_buffers();
}

} // namespace frontend
} // namespace megamol
