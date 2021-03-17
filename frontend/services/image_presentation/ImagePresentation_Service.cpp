/*
 * ImagePresentation_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

// search/replace ImagePresentation_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
#include "ImagePresentation_Service.hpp"

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

using ImageWrapper = megamol::frontend_resources::ImageWrapper;

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

    m_image_registry_resource.pimpl = &m_images; // pimpl access happens in frontend resources ImageWrapper.cpp

    m_entry_points_registry_resource.add_entry_point =    [&](std::string name, void* module_raw_ptr)   -> bool { return add_entry_point(name, module_raw_ptr); };
    m_entry_points_registry_resource.remove_entry_point = [&](std::string name)                         -> bool { return remove_entry_point(name); };
    m_entry_points_registry_resource.rename_entry_point = [&](std::string oldName, std::string newName) -> bool { return rename_entry_point(oldName, newName); };
    m_entry_points_registry_resource.clear_entry_points = [&]() { clear_entry_points(); };

    this->m_providedResourceReferences =
    {
          {"ImageRegistry", m_image_registry_resource} // mostly we use this resource, but other services like Screenshots may access it too
        , {"ImagePresentationEntryPoints", m_entry_points_registry_resource} // used by MegaMolGraph to set entry points
    };

    this->m_requestedResourcesNames =
    {
        "FrontendResources" // std::vector<FrontendResource>
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

    distribute_changed_resources_to_entry_points();

    if (false)
        this->setShutdown();
}

void ImagePresentation_Service::resetProvidedResources() {
}

void ImagePresentation_Service::preGraphRender() {
}

void ImagePresentation_Service::postGraphRender() {
}

void ImagePresentation_Service::RenderNextFrame() {
    for (auto& entry : m_entry_points) {
        entry.execute(entry.modulePtr, entry.entry_point_resources, entry.execution_result_image.get());
    }
}

void ImagePresentation_Service::PresentRenderedImages() {
}

// clang-format off
using FrontendResource = megamol::frontend::FrontendResource;
using EntryPointExecutionCallback = std::function<bool(void*, std::vector<FrontendResource> const&, ImageWrapper&)>;

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
    throw std::exception("Fatal Error setting Graph Entry Point callback functions");
}

std::vector<FrontendResource> ImagePresentation_Service::map_resources(std::vector<std::string> const& requests) {
    std::vector<FrontendResource> result;

    for (auto& request: requests) {
        auto find_it = std::find_if(m_frontend_resources.begin(), m_frontend_resources.end(),
            [&](FrontendResource const& resource) { return resource.getIdentifier() == request; });
        bool found_request = find_it != m_frontend_resources.end();

        // intercept our special-needs resources and treat specially
        auto fbo_events_name = std::string{"FramebufferEvents"};
        if (request == fbo_events_name) {
            // each entry point gets its own framebuffer size
            // what we do here is add a temporary object that gets switches for the real deal later on
            //
            // WARNING: we now create a dingling reference to the temp object - which is really stupid
            // but we need to keep the requested resource at exactly this place in the result array
            // and the resource object that is going to be actually used later does not exist after we filled this resource array
            // so we need this temp variable
            megamol::frontend_resources::FramebufferEvents temp_fbo_events;

            // we mark the resource name with # at the end to later recognize it as an individual resource
            result.push_back({fbo_events_name + '#', temp_fbo_events});
            continue;
        }

        if (!found_request) {
            log_error("could not find requested resource " + request);
            return {};
        }

        result.push_back(*find_it);
    }

    return result;
}

void ImagePresentation_Service::remap_individual_resources(GraphEntryPoint& entry) {
    auto with_hash = [&](std::string const& in) { return in + '#'; };

    for (auto& resource : entry.entry_point_resources) {
        auto& name = resource.getIdentifier();
        if (name.back() == '#') {
            // switch over possible individual resources
            // that we need to remap/replace with references to data/objects held by us

            auto fbo_events = "FramebufferEvents";
            if (name == with_hash(fbo_events)) {
                resource = {fbo_events, entry.framebuffer_events};
            }
        }
    }
}

template <typename ResourceType>
static
const ResourceType* maybeGetResource(std::string const& name, std::vector<FrontendResource> const& resources) {
    auto resource_it = std::find_if(resources.begin(), resources.end(),
        [&](FrontendResource const& resource) {
            return resource.getIdentifier() == name;
        });
    bool no_resource = resource_it == resources.end();

    if (no_resource)
        return nullptr;

    return &resource_it->getResource<ResourceType>();
}

void ImagePresentation_Service::distribute_changed_resources_to_entry_points() {
    auto maybe_window_fbo_events_ptr = maybeGetResource<megamol::frontend_resources::FramebufferEvents>("WindowFramebufferEvents", m_frontend_resources);

    for (auto& entry : m_entry_points) {
        if (maybe_window_fbo_events_ptr && entry.framebuffer_events_source == GraphEntryPoint::FboEventsSource::WindowSize) {
            for (auto& size: maybe_window_fbo_events_ptr->size_events) {
                entry.framebuffer_events.size_events.push_back({size.width, size.height});
            }
        } else {
            log_error("entry point " + entry.moduleName + " demands Fbo Size events to be fed from WindowFramebufferEvents resource. "
                      "\nBut i could not find that resource. Requesting to shut down MegaMol.");
            this->setShutdown();
        }
    }
}

void ImagePresentation_Service::set_individual_entry_point_resources_defaults(GraphEntryPoint& entry) {
    auto maybe_window_fbo_events_ptr = maybeGetResource<megamol::frontend_resources::FramebufferEvents>("WindowFramebufferEvents", m_frontend_resources);

    if (entry.framebuffer_events_source == GraphEntryPoint::FboEventsSource::WindowSize && maybe_window_fbo_events_ptr) {
        entry.framebuffer_events.previous_state = maybe_window_fbo_events_ptr->previous_state;
        entry.framebuffer_events.size_events = maybe_window_fbo_events_ptr->size_events;
    } else {
        log_error("entry point " + entry.moduleName + " demands Fbo Size initial values from WindowFramebufferEvents resource. "
                  "\nBut i could not find that resource. Requesting to shut down MegaMol.");
        this->setShutdown();
    }
}

bool ImagePresentation_Service::add_entry_point(std::string name, void* module_raw_ptr) {
    auto& image = m_image_registry_resource.make(name);

    auto [execute_etry, init_entry, entry_resource_requests] = get_init_execute_resources(module_raw_ptr);

    auto resource_requests = entry_resource_requests();
    auto resources = map_resources(resource_requests);

    if (resources.empty() && !resource_requests.empty()) {
        log_error("could not assign resources requested by entry point " + name + ". Entry point not created.");
        return false;
    }

    // TODO: making entry points with non-window fbo size how?
    const auto fbo_events_source = m_default_fbo_events_source;

    m_entry_points.push_back(GraphEntryPoint{
        name,
        module_raw_ptr,
        resources,
        fbo_events_source,
        megamol::frontend_resources::FramebufferEvents{},
        execute_etry,
        image
        });

    auto& entry_point = m_entry_points.back();

    remap_individual_resources(entry_point);
    set_individual_entry_point_resources_defaults(entry_point);

    if (!init_entry(entry_point.modulePtr, entry_point.entry_point_resources, entry_point.execution_result_image)) {
        log_error("init function for entry point " + entry_point.moduleName + " failed. Entry point not created.");
        m_entry_points.pop_back();
        return false;
    }

    return true;
}

bool ImagePresentation_Service::remove_entry_point(std::string name) {
    if (!m_image_registry_resource.remove(name))
        return false;

    m_entry_points.remove_if([&](auto& entry) { return entry.moduleName == name; });

    return true;
}

bool ImagePresentation_Service::rename_entry_point(std::string oldName, std::string newName) {
    if (!m_image_registry_resource.rename(oldName, newName))
        return false;

    auto entry_it = std::find_if(m_entry_points.begin(), m_entry_points.end(),
        [&](auto& entry) { return entry.moduleName == oldName; });

    if (entry_it == m_entry_points.end()) {
        return false;
    }

    entry_it->moduleName = newName;

    return true;
}

bool ImagePresentation_Service::clear_entry_points() {
    for (auto& entry : m_entry_points) {
        m_image_registry_resource.remove(entry.moduleName);
    }
    m_entry_points.clear();

    return true;
}


} // namespace frontend
} // namespace megamol
