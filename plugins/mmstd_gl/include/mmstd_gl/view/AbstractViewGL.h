/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "OpenGL_Context.h"
#include "mmstd/view/AbstractView.h"

namespace megamol::mmstd_gl::view {
class AbstractViewGL : public core::view::AbstractView {
public:
    using core::view::AbstractView::AbstractView;

    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        AbstractView::requested_lifetime_resources(req);
        req.require<frontend_resources::OpenGL_Context>(); // GL modules should request the GL context resource
    }
};
} // namespace megamol::mmstd_gl::view
