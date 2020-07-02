/*
 * AbstractView_EventConsumption.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractView.h"
#include "RenderResource.h"

namespace megamol {
namespace core {
namespace view {

	// these functions implement passing rendering resources or events from the frontend to the AbstractView
	// they get called by the MegaMolGraph when a View module got spawned in the graph
	// note that starting the actual rendering of a frame is handled via a callback to 'view_poke_rendering'
	// and from the perspective of the MegaMolGreaph does not look different from consumption of rendering resources or input events

	MEGAMOLCORE_API void view_consume_keyboard_events(AbstractView& view, megamol::render_api::RenderResource const& resource);
	MEGAMOLCORE_API void view_consume_mouse_events(AbstractView& view, megamol::render_api::RenderResource const& resource);
	MEGAMOLCORE_API void view_consume_window_events(AbstractView& view, megamol::render_api::RenderResource const& resource);
	MEGAMOLCORE_API void view_consume_framebuffer_events(AbstractView & view, megamol::render_api::RenderResource const& resource);
	MEGAMOLCORE_API void view_poke_rendering(AbstractView & view, megamol::render_api::RenderResource const& resource);

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

