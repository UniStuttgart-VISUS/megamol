/*
 * ImagePresentation_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ImagePresentation_Service.hpp"

#include "WindowManipulation.h"
#include "GUIState.h"

#include "OpenGL_Context.h"
#include "ImageWrapper_to_GLTexture.hpp"
#include "ImagePresentation_Sinks.hpp"

#include "LuaCallbacksCollection.h"
#include "RenderInput.h"

#include <any>
#include <utility>
#include <filesystem>

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

    m_entry_points_registry_resource.add_entry_point =    [&](std::string name, EntryPointRenderFunctions const& entry_point)   -> bool { return add_entry_point(name, entry_point); };
    m_entry_points_registry_resource.remove_entry_point = [&](std::string name)                         -> bool { return remove_entry_point(name); };
    m_entry_points_registry_resource.rename_entry_point = [&](std::string oldName, std::string newName) -> bool { return rename_entry_point(oldName, newName); };
    m_entry_points_registry_resource.clear_entry_points = [&]() { clear_entry_points(); };

    this->m_providedResourceReferences =
    {
          {"ImagePresentationEntryPoints", m_entry_points_registry_resource} // used by MegaMolGraph to set entry points
        , {"EntryPointToPNG_ScreenshotTrigger", m_entrypointToPNG_trigger}
        , {"FramebufferEvents", m_global_framebuffer_events}
    };

    this->m_requestedResourcesNames =
    {
          "FrontendResources" // std::vector<FrontendResource>
        , "optional<WindowManipulation>"
        , "FramebufferEvents"
        , "optional<GUIState>" // TODO: unused?
        , "RegisterLuaCallbacks"
        , "optional<OpenGL_Context>"
        , "ImageWrapperToPNG_ScreenshotTrigger"
    };

    m_framebuffer_size_handler = [&]() -> UintPair {
        return {m_window_framebuffer_size.first, m_window_framebuffer_size.second};
    };

    m_viewport_tile_handler =
        [&]() -> ViewportTile
        {
            return {m_framebuffer_size_handler(), {0.0, 0.0}, {1.0, 1.0}};
        };

    if (config.local_viewport_tile.has_value()) {
        auto value = config.local_viewport_tile.value();

        auto global_size = value.global_framebuffer_resolution;
        auto tile_start = value.tile_start_pixel;
        auto tile_size = value.tile_resolution;

        auto diff = [](auto const& point, auto const& size) -> DoublePair {
            return {point.first / static_cast<double>(size.first), point.second / static_cast<double>(size.second)};
        };

        DoublePair start = diff(tile_start, global_size);
        DoublePair end = diff(UintPair{tile_start.first+tile_size.first,tile_start.second+tile_size.second}, global_size);

        m_framebuffer_size_handler = [=]() -> UintPair {
            return {tile_size.first, tile_size.second};
        };

        m_viewport_tile_handler =
            [=]() -> ViewportTile
            {
                return {{global_size.first, global_size.second}, start, end};
            };
    }

    // if framebuffer size is set via CLI
    // overwrite the framebuffer size handler maybe set by the tile CLI option (see above)
    if (config.local_framebuffer_resolution.has_value()) {
        auto value = config.local_framebuffer_resolution.value();

        m_framebuffer_size_handler = [=]() -> UintPair {
            return {value.first, value.second};
        };
    }

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

    m_frontend_resources_lookup = {m_requestedResourceReferences[0].getResource<std::vector<megamol::frontend::FrontendResource>>()};

    auto& framebuffer_events = m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FramebufferEvents>();
    m_window_framebuffer_size = {framebuffer_events.previous_state.width, framebuffer_events.previous_state.height};

    add_glfw_sink();

    fill_lua_callbacks();
}

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
    // before presenting to the sinks, we need to apply the latest global fbo size changes
    // this way sinks can access the current fbo size as previous_state
    m_global_framebuffer_events.clear();

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

namespace {
    struct ViewRenderInputs : public ImagePresentation_Service::RenderInputsUpdate {
        static constexpr const char* Name = "ViewRenderInputs";

        // individual inputs used by view for rendering of next frame
        megamol::frontend_resources::RenderInput render_input;

        // sets (local) fbo resolution of render_input from various sources
        std::function<ImagePresentation_Service::UintPair()> render_input_framebuffer_size_handler;
        std::function<ImagePresentation_Service::ViewportTile()> render_input_tile_handler;

        void update() override {
            auto fbo_size = render_input_framebuffer_size_handler();
            render_input.local_view_framebuffer_resolution = {fbo_size.first, fbo_size.second};

            auto tile = render_input_tile_handler();
            render_input.global_framebuffer_resolution = {tile.global_resolution.first, tile.global_resolution.second};
            render_input.local_tile_relative_begin = {tile.tile_start_normalized.first, tile.tile_start_normalized.second};
            render_input.local_tile_relative_end = {tile.tile_end_normalized.first, tile.tile_end_normalized.second};
        }

        FrontendResource get_resource() override{
            return {Name, render_input};
        };
    };
#define accessViewRenderInput(unique_ptr) (*static_cast<ViewRenderInputs*>(unique_ptr.get()))

    // this is a somewhat improvised factory to abstract away instantiation of different RenderInputUpdate types
    struct RenderInputsFactory {
        std::tuple<std::string, std::unique_ptr<ImagePresentation_Service::RenderInputsUpdate>>
            get(std::string const& request) {
            auto renderinputs = std::make_unique<ImagePresentation_Service::RenderInputsUpdate>();

            if (request == ViewRenderInputs::Name) {
                renderinputs = std::make_unique<ViewRenderInputs>();
                accessViewRenderInput(renderinputs).render_input_framebuffer_size_handler = fromservice<std::function<ImagePresentation_Service::UintPair()>>(0);
                accessViewRenderInput(renderinputs).render_input_tile_handler = fromservice<std::function<ImagePresentation_Service::ViewportTile()>>(1);

                return {ViewRenderInputs::Name, std::move(renderinputs)};
            }
        }

        template <typename T>
        T const& fromservice(size_t index) {
            return *std::any_cast<T*>(service_data[index]);
        }

        std::vector<std::any> service_data; // holding ptrs to image presentation members/resources we need to pass to the stuff the factory produces
    };
} // namespace

std::tuple<
    bool, // success
    std::vector<FrontendResource>, // resources
    std::unique_ptr<ImagePresentation_Service::RenderInputsUpdate> // unique_data for entry point
>
ImagePresentation_Service::map_resources(std::vector<std::string> const& requests) {
    static RenderInputsFactory renderinputs_factory{
        // the factory needs to pass certain data/handlers to the RenderInput structs/implementations
        {
              {&m_framebuffer_size_handler}
            , {&m_viewport_tile_handler}
        }};

    // this unique_data/reindering input handler thing is a bit convoluted but the idea is that we want to give the view (or any other type of entry point)
    // the ability to get updated with newest frame data, e.g. framebuffer size
    // but we also want to maintain the "empty" handler throughout the code path to avoid checking for a null ptr
    auto unique_data = std::make_unique<RenderInputsUpdate>();

    bool success = false;
    std::vector<FrontendResource> resources;

    auto handle_resource_requests = [&](auto from, auto to) -> bool
    {
        if (from == to)
            return true;

        std::vector<std::string> requests{from, to};
        auto [lookup_success, lookup_resources] = m_frontend_resources_lookup.get_requested_resources(requests);
        resources.insert(resources.end(), lookup_resources.begin(), lookup_resources.end());

        return lookup_success;
    };

    // find resource request for input update of this entry point
    // currently there is only one such request
    // we then split up the requests into the ones before and after the request for the input update
    // and look up those resource requests in the frontend resources
    // the input update resource is not known in the frontend, so we need to fiddle a bit here
    success = requests.empty();
    for (auto request_it = requests.begin(); request_it != requests.end(); request_it++) {
        if (auto [name, result_unique_ptr] = renderinputs_factory.get(*request_it); result_unique_ptr != nullptr) {
            unique_data = std::move(result_unique_ptr);
            success = true;

            success &= handle_resource_requests(requests.begin(), request_it);

            // put the resource for input update at position where it was requested
            resources.emplace_back(unique_data->get_resource());

            success &= handle_resource_requests(request_it+1, requests.end());

            break;
        }
    }

    if (!success) {
        log_error("could not find a requested resource for an entry point");
    }

    return {success, resources, std::move(unique_data)};
}

bool ImagePresentation_Service::add_entry_point(std::string name, EntryPointRenderFunctions const& entry_point) {
    auto [module_ptr, execute_etry, entry_resource_requests] = entry_point;

    auto resource_requests = entry_resource_requests();

    auto [success, resources, unique_data] = map_resources(resource_requests);

    if (!success) {
        log_error("could not assign resources requested by entry point " + name + ". Entry point not created.");
        return false;
    }

    m_entry_points.push_back(GraphEntryPoint{
        name,
        module_ptr,
        resources,
        std::move(unique_data), // render inputs and their update
        execute_etry,
        {name} // image
        });

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

void ImagePresentation_Service::add_glfw_sink() {
    auto maybe_gl_context = m_requestedResourceReferences[5].getOptionalResource<megamol::frontend_resources::OpenGL_Context>();

    if (maybe_gl_context == std::nullopt) {
        log_warning("no GLFW OpenGL_Context resource available");
        return;
    }

    m_presentation_sinks.push_back(
        {"GLFW Window Presentation Sink", [&](auto const& images) { this->present_images_to_glfw_window(images); }});
}

void ImagePresentation_Service::present_images_to_glfw_window(std::vector<ImageWrapper> const& images) {
    static auto const& window_manipulation       = m_requestedResourceReferences[1].getOptionalResource<megamol::frontend_resources::WindowManipulation>().value().get();
    static auto const& window_framebuffer_events = m_requestedResourceReferences[2].getResource<megamol::frontend_resources::FramebufferEvents>();
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
            [=]() -> UintPair {
                return {width, height};
            };

            return VoidResult{};
        }});

    auto handle_screenshot = [&](std::string const& entrypoint, std::string file) -> megamol::frontend_resources::LuaCallbacksCollection::VoidResult {
        if (m_entry_points.empty())
            return Error{"no views registered as entry points. nothing to write as screenshot into "};

        auto find_it = std::find_if(m_entry_points.begin(), m_entry_points.end(),
            [&](GraphEntryPoint const& elem) { return elem.moduleName == entrypoint; });

        if (find_it == m_entry_points.end())
            return Error{"error writing screenshot into file " + file + ". no such entry point: " + entrypoint};

        auto& entry_result_image = find_it->execution_result_image;

        auto& triggerscreenshot = m_requestedResourceReferences[6].getResource<std::function<bool(ImageWrapper const&, std::filesystem::path const&)>>();
        bool trigger_ok = triggerscreenshot(entry_result_image, file);

        if (!trigger_ok)
            return Error{"error writing screenshot for entry point " + entrypoint + " into file " + file};

        return VoidResult{};
    };

    m_entrypointToPNG_trigger = [handle_screenshot](std::string const& entrypoint, std::string const& file) -> bool {
        auto result = handle_screenshot(entrypoint, file);

        if (!result.exit_success) {
            log_warning(result.exit_reason);
            return false;
        }

        return true;
    };

    callbacks.add<VoidResult, std::string, std::string>(
        "mmScreenshotEntryPoint",
        "(string entrypoint, string filename)\n\tSave a screen shot of entry point view as 'filename'",
        {[handle_screenshot](std::string entrypoint, std::string filename) -> VoidResult {
            return handle_screenshot(entrypoint, filename);
        }});

    auto& register_callbacks = m_requestedResourceReferences[4].getResource<std::function<void(megamol::frontend_resources::LuaCallbacksCollection const&)>>();

    register_callbacks(callbacks);
}


} // namespace frontend
} // namespace megamol
