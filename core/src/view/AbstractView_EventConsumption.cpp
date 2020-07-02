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
#include "OpenGL_Context.h"

namespace megamol {
namespace core {
namespace view {


using namespace megamol::input_events;

// shorthand notation to unpack a RenderResource to some type. 
// if the type is present in the resource is made available as an 'events' variable in the if statemtnt.
// note that when using this macro there is no visible opening bracket { for the if statements because it is hidden inside the macro
#define IF(TYPENAME) \
    auto optional_resource = resource.getResource<TYPENAME>();    \
    if (optional_resource.has_value()) {                          \
		TYPENAME const& events = optional_resource.value().get();


void view_consume_keyboard_events(AbstractView& view, megamol::render_api::RenderResource const& resource) {
//    auto optional_resource = resource.getResource<KeyboardEvents>();
//    if (optional_resource.has_value()) {
//		const KeyboardEvents& events = optional_resource.value().get();
//
//		for (auto& e : events.key_events)
//			view.OnKey(std::get<0>(e), std::get<1>(e), std::get<2>(e));
//    }
    IF(KeyboardEvents)//{
		for (auto& e : events.key_events)
			view.OnKey(std::get<0>(e), std::get<1>(e), std::get<2>(e));

		for (auto& e : events.codepoint_events)
			view.OnChar(e);
	}
}

void view_consume_mouse_events(AbstractView& view, megamol::render_api::RenderResource const& resource) {
    IF(MouseEvents)//{
		for (auto& e : events.buttons_events) 
			view.OnMouseButton(std::get<0>(e), std::get<1>(e), std::get<2>(e));

		for (auto& e : events.position_events)
			view.OnMouseMove(std::get<0>(e), std::get<1>(e));

		for (auto& e : events.scroll_events)
			view.OnMouseScroll(std::get<0>(e), std::get<1>(e));

		//for (auto& e: events.enter_events) {}
	}
}

void view_consume_window_events(AbstractView& view, megamol::render_api::RenderResource const& resource) {
    IF(WindowEvents)//{
		events.is_focused_events;
	}
}

void view_consume_framebuffer_events(AbstractView& view, megamol::render_api::RenderResource const& resource) {
    IF(FramebufferEvents)//{
		for (auto& e: events.size_events)
			view.Resize(static_cast<unsigned int>(e.width), static_cast<unsigned int>(e.height));
    }
}

void view_poke_rendering(AbstractView& view, megamol::render_api::RenderResource const& resource) {
    auto optional_resource = resource.getResource<megamol::input_events::IOpenGL_Context>();

	const auto render = [&]() {
		_mmcRenderViewContext dummyRenderViewContext; // doesn't do anything, really
		view.Render(dummyRenderViewContext);
	};

    if (optional_resource.has_value()) {
		megamol::input_events::IOpenGL_Context const& gl_context = optional_resource.value().get();
		gl_context.activate(); // makes GL context current

		render();

		gl_context.close();
	} else {
		render();
	}
}

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

