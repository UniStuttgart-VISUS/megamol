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

using namespace megamol::module_resources;

// shorthand notation to unpack a ModuleResource to some type. 
// if the type is present in the resource is made available as an 'events' variable in the if statemtnt.
// note that when using this macro there is no visible opening bracket { for the if statements because it is hidden inside the macro
#define GET_RESOURCE(TYPENAME) \
    { \
		TYPENAME const& events = resource.getResource<TYPENAME>();


void view_consume_keyboard_events(AbstractView& view, megamol::frontend::ModuleResource const& resource) {
    GET_RESOURCE(KeyboardEvents)//{
		for (auto& e : events.key_events)
			view.OnKey(std::get<0>(e), std::get<1>(e), std::get<2>(e));

		for (auto& e : events.codepoint_events)
			view.OnChar(e);
	}
}

void view_consume_mouse_events(AbstractView& view, megamol::frontend::ModuleResource const& resource) {
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

void view_consume_window_events(AbstractView& view, megamol::frontend::ModuleResource const& resource) {
    GET_RESOURCE(WindowEvents)//{
		events.is_focused_events;
	}
}

void view_consume_framebuffer_events(AbstractView& view, megamol::frontend::ModuleResource const& resource) {
    GET_RESOURCE(FramebufferEvents)//{
		for (auto& e: events.size_events)
			view.Resize(static_cast<unsigned int>(e.width), static_cast<unsigned int>(e.height));
    }
}

// this is a weird place to measure passed program time, but we do it here so we satisfy _mmcRenderViewContext and nobody else needs to know
static std::chrono::high_resolution_clock::time_point render_view_context_timer_start;

void view_poke_rendering(AbstractView& view, megamol::frontend::ModuleResource const& resource) {
    megamol::module_resources::IOpenGL_Context const * maybe_opengl = nullptr;

    if (resource.getIdentifier() == "IOpenGL_Context")
		maybe_opengl = &resource.getResource<megamol::module_resources::IOpenGL_Context>();
	
	static bool started_timer = false;
	if (!started_timer) {
		render_view_context_timer_start = std::chrono::high_resolution_clock::now();
        started_timer = true;
	}

	const auto render = [&]() {
		_mmcRenderViewContext dummyRenderViewContext; // doesn't do anything, really

        const double time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - render_view_context_timer_start)
            .count() / static_cast<double>(1000);

        dummyRenderViewContext.InstanceTime = time;

		view.Render(dummyRenderViewContext);
	};

    if (maybe_opengl) {
		maybe_opengl->activate(); // makes GL context current

		render();

		maybe_opengl->close();
	} else {
		render();
	}
}

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

