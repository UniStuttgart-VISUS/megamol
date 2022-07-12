/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/view/CPUFramebuffer.h"
#include "mmcore/view/CameraControllers.h"
#include "mmstd/renderer/CallRenderView.h"
#include "mmstd/view/AbstractView.h"
#include "mmstd/view/BaseView.h"

namespace megamol::core::view {

inline constexpr auto cpu_fbo_resize = [](std::shared_ptr<CPUFramebuffer>& fbo, int width, int height) -> void {
    fbo->width = width;
    fbo->height = height;
    // TODO reallocate buffer?
};

class View3D : public view::BaseView<CallRenderView, Camera3DController, AbstractView> {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "View3D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "View 3D module";
    }

    /** Ctor. */
    View3D();

    /** Dtor. */
    virtual ~View3D();

    /**
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual ImageWrapper Render(double time, double instanceTime) override;

    ImageWrapper GetRenderingResult() const override;

    /**
     * Resizes the framebuffer object and calls base class function that sets camera aspect ratio if applicable.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height) override;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create();
};

} // namespace megamol::core::view
