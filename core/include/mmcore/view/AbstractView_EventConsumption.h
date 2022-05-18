/*
 * AbstractView_EventConsumption.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractView.h"
#include "FrontendResource.h"

#include "ImageWrapper.h"
#include "RenderInput.h"

namespace megamol {
namespace core {
namespace view {

// these functions implement passing rendering resources or events from the frontend to the AbstractView
// they get called by the MegaMolGraph when a View module got spawned in the graph
// note that starting the actual rendering of a frame is handled via a callback to 'view_poke_rendering'
// and from the perspective of the MegaMolGreaph does not look different from consumption of rendering resources or input events

void view_consume_keyboard_events(AbstractView& view, megamol::frontend::FrontendResource const& resource);
void view_consume_mouse_events(AbstractView& view, megamol::frontend::FrontendResource const& resource);
void view_consume_window_events(AbstractView& view, megamol::frontend::FrontendResource const& resource);
void view_poke_rendering(AbstractView& view, megamol::frontend_resources::RenderInput const& render_input,
    megamol::frontend_resources::ImageWrapper& result_image);

// to do this right we should be able to as a view object which runtime resources it expects (keyboard inputs, gl context)
// and just pass those resources to the view when rendering a frame
// until we implement the 'optimal' approach, this is the best we can do
std::vector<std::string> get_view_runtime_resources_requests();

bool view_rendering_execution(void* module_ptr, std::vector<megamol::frontend::FrontendResource> const& resources,
    megamol::frontend_resources::ImageWrapper& result_image);

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
