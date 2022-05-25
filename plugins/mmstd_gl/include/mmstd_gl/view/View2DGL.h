/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glowl/FramebufferObject.hpp>

#include "mmcore/view/BaseView.h"
#include "mmcore/view/CameraControllers.h"
#include "mmcore_gl/view/AbstractViewGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"

namespace megamol::core_gl::view {

/**
 * Base class of rendering graph calls
 */
class View2DGL : public core::view::BaseView<CallRenderViewGL, core::view::Camera2DController, AbstractViewGL> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "View2DGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "2D View Module";
    }

    /** Ctor. */
    View2DGL();

    /** Dtor. */
    ~View2DGL() override;

    /**
     * ...
     */
    ImageWrapper Render(double time, double instanceTime) override;

    ImageWrapper GetRenderingResult() const override;

    /**
     * Resizes the framebuffer object and calls base class function that sets camera aspect ratio if applicable.
     *
     * @param width The new width.
     * @param height The new height.
     */
    void Resize(unsigned int width, unsigned int height) override;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;
};
} // namespace megamol::core_gl::view
