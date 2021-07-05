/*
 * AbstractView_EventConsumption.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/AbstractView_EventConsumption.h"

#include "Framebuffer_Events.h"
#include "KeyboardMouse_Events.h"
#include "Window_Events.h"
#include "IOpenGL_Context.h"

#include <chrono>

namespace megamol {
namespace core {
namespace view {

using namespace megamol::frontend_resources;

// shorthand notation to unpack a FrontendResource to some type. 
// if the type is present in the resource is made available as an 'events' variable in the if statemtnt.
// note that when using this macro there is no visible opening bracket { for the if statements because it is hidden inside the macro
#define GET_RESOURCE(TYPENAME) \
    { \
        TYPENAME const& events = resource.getResource<TYPENAME>();


void view_consume_keyboard_events(AbstractView& view, megamol::frontend::FrontendResource const& resource) {
    GET_RESOURCE(KeyboardEvents)//{
        for (auto& e : events.key_events)
            view.OnKey(std::get<0>(e), std::get<1>(e), std::get<2>(e));

        for (auto& e : events.codepoint_events)
            view.OnChar(e);
    }
}

void view_consume_mouse_events(AbstractView& view, megamol::frontend::FrontendResource const& resource) {
    GET_RESOURCE(MouseEvents)//{
        for (auto& e : events.buttons_events) 
            view.OnMouseButton(std::get<0>(e), std::get<1>(e), std::get<2>(e));

        for (auto& e : events.position_events)
            view.OnMouseMove(std::get<0>(e), std::get<1>(e));

        for (auto& e : events.scroll_events)
            view.OnMouseScroll(std::get<0>(e), std::get<1>(e));

        //for (auto& e: events.enter_events) {}
    }
}

void view_consume_window_events(AbstractView& view, megamol::frontend::FrontendResource const& resource) {
    GET_RESOURCE(WindowEvents)//{
        events.is_focused_events;
    }
}

void view_consume_framebuffer_events(AbstractView& view, megamol::frontend::FrontendResource const& resource) {
    GET_RESOURCE(FramebufferEvents)//{
        for (auto& e: events.size_events)
            view.Resize(static_cast<unsigned int>(e.width), static_cast<unsigned int>(e.height));
    }
}

// this is a weird place to measure passed program time, but we do it here so we satisfy _mmcRenderViewContext and nobody else needs to know
static std::chrono::high_resolution_clock::time_point render_view_context_timer_start;

void view_poke_rendering(AbstractView& view, megamol::frontend_resources::ImageWrapper& result_image) {
    static bool started_timer = false;
    if (!started_timer) {
        render_view_context_timer_start = std::chrono::high_resolution_clock::now();
        started_timer = true;
    }

    const auto render = [&]() {
        _mmcRenderViewContext dummyRenderViewContext = {}; // zero initialization. the struct is currently only used for time propagation.

        const double time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - render_view_context_timer_start)
            .count() / static_cast<double>(1000);

            // this is missing!
            // auto core = view->View()->GetCoreInstance();
            // core->SetFrameID(frameID);
            dummyRenderViewContext.InstanceTime = time;
            dummyRenderViewContext.Time = view.DefaultTime(time);

        view.Render(dummyRenderViewContext);
        result_image = view.GetRenderingResult();
    };
    
    render();
}

std::vector<std::string> get_gl_view_runtime_resources_requests() {
    return {"KeyboardEvents", "MouseEvents", "WindowEvents", "FramebufferEvents"};
}

bool view_rendering_execution(
      void* module_ptr
    , std::vector<megamol::frontend::FrontendResource> const& resources
    , megamol::frontend_resources::ImageWrapper& result_image
) {
    megamol::core::view::AbstractView* view_ptr =
        dynamic_cast<megamol::core::view::AbstractView*>(static_cast<megamol::core::Module*>(module_ptr));

    if (!view_ptr) {
        std::cout << "error. module is not a view module. could not use as rendering entry point." << std::endl;
        return false;
    }
    
    megamol::core::view::AbstractView& view = *view_ptr;
    
    // resources are in order of initial requests from get_gl_view_runtime_resources_requests()
    megamol::core::view::view_consume_keyboard_events(view, resources[0]);
    megamol::core::view::view_consume_mouse_events(view, resources[1]);
    megamol::core::view::view_consume_window_events(view, resources[2]);
    megamol::core::view::view_consume_framebuffer_events(view, resources[3]);
    megamol::core::view::view_poke_rendering(view, result_image);
    
    return true;
}

bool view_init_rendering_state(
      void* module_ptr
    , std::vector<megamol::frontend::FrontendResource> const& resources
    , megamol::frontend_resources::ImageWrapper& result_image
) {
    megamol::core::view::AbstractView* view_ptr =
        dynamic_cast<megamol::core::view::AbstractView*>(static_cast<megamol::core::Module*>(module_ptr));

    if (!view_ptr) {
        std::cout << "error. module is not a view module. could not use as rendering entry point." << std::endl;
        return false;
    }
    
    megamol::core::view::AbstractView& view = *view_ptr;

    // fake resize events for view to consume
    auto& framebuffer_events = const_cast<megamol::frontend_resources::FramebufferEvents&>(resources[3].getResource<megamol::frontend_resources::FramebufferEvents>());
    auto& framebuffer_size = framebuffer_events.previous_state;
    framebuffer_events.size_events.push_back(framebuffer_size);
    
    auto& window_events = const_cast<megamol::frontend_resources::WindowEvents&>(resources[2].getResource<megamol::frontend_resources::WindowEvents>());
    auto& window_width = window_events.previous_state.width;
    auto& window_height = window_events.previous_state.height;
    if(window_width <= 1 || window_height <= 1) {
        window_width = framebuffer_size.width;
        window_height = framebuffer_size.height;
    }
    window_events.size_events.push_back({window_width, window_height});
    
    auto& mouse_events = const_cast<megamol::frontend_resources::MouseEvents&>(resources[1].getResource<megamol::frontend_resources::MouseEvents>());
    mouse_events.position_events.push_back({
        mouse_events.previous_state.x_cursor_position,
        mouse_events.previous_state.y_cursor_position});

    // resources are in order of initial requests from get_gl_view_runtime_resources_requests()
    megamol::core::view::view_consume_mouse_events(view, resources[1]);
    megamol::core::view::view_consume_window_events(view, resources[2]);
    megamol::core::view::view_consume_framebuffer_events(view, resources[3]);

    return true;
}

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

