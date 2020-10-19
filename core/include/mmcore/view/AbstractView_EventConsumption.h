/*
 * AbstractView_EventConsumption.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractView.h"
#include "FrontendResource.h"

namespace megamol {
namespace core {
namespace view {

    // these functions implement passing rendering resources or events from the frontend to the AbstractView
    // they get called by the MegaMolGraph when a View module got spawned in the graph
    // note that starting the actual rendering of a frame is handled via a callback to 'view_poke_rendering'
    // and from the perspective of the MegaMolGreaph does not look different from consumption of rendering resources or input events

    MEGAMOLCORE_API void view_consume_keyboard_events(AbstractView& view, megamol::frontend::FrontendResource const& resource);
    MEGAMOLCORE_API void view_consume_mouse_events(AbstractView& view, megamol::frontend::FrontendResource const& resource);
    MEGAMOLCORE_API void view_consume_window_events(AbstractView& view, megamol::frontend::FrontendResource const& resource);
    MEGAMOLCORE_API void view_consume_framebuffer_events(AbstractView & view, megamol::frontend::FrontendResource const& resource);
    MEGAMOLCORE_API void view_poke_rendering(AbstractView& view);//, megamol::frontend::FrontendResource const& resource);

    // to do this right we should be able to as a view object which runtime resources it expects (keyboard inputs, gl context)
    // and just pass those resources to the view when rendering a frame
    // until we implement the 'optimal' approach, this is the best we can do
    MEGAMOLCORE_API std::vector<std::string> get_gl_view_runtime_resources_requests();
    MEGAMOLCORE_API bool view_rendering_execution(megamol::core::Module::ptr_type module_ptr, std::vector<megamol::frontend::FrontendResource> const& resources);
    // before rendering the first frame views need to know the current framebuffer size
    // because they may have beed added to the graph after the initial framebuffer size event, we need this init callback to give them that info
    MEGAMOLCORE_API bool view_init_rendering_state(megamol::core::Module::ptr_type module_ptr, std::vector<megamol::frontend::FrontendResource> const& resources);

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

