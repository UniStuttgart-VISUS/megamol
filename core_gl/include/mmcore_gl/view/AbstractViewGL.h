/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/view/AbstractView.h"

namespace megamol::core_gl::view {
class AbstractViewGL : public core::view::AbstractView {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = AbstractView::requested_lifetime_resources();
        resources.emplace_back("OpenGL_Context"); // GL modules should request the GL context resource
        return resources;
    }
};
} // namespace megamol::core_gl::view
