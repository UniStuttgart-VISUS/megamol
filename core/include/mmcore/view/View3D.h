/*
 * View3D.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/AbstractView3D.h"

#include "mmcore/view/CameraControllers.h"
#include "mmcore/view/CameraParameterSlots.h"

namespace megamol {
namespace core {
namespace view {

    inline constexpr auto cpu_fbo_resize = [](std::shared_ptr<CPUFramebuffer>& fbo, int width,
                                              int height) -> void {
        fbo->width = width;
        fbo->height = height;
        // TODO reallocate buffer?
    };

class MEGAMOLCORE_API View3D : public view::AbstractView3D<CPUFramebuffer, cpu_fbo_resize, Camera3DController, Camera3DParameters> {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "View3D"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "View 3D module"; }

    /** Ctor. */
    View3D(void);

    /** Dtor. */
    virtual ~View3D(void);

      /**
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual ImageWrapper Render(double time, double instanceTime, bool present_fbo) override;

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    virtual void ResetView();

};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

