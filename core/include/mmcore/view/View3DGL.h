/*
 * View3DGL.h
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "mmcore/view/BaseView.h"
#include "vislib/graphics/Cursor2D.h"

#include "mmcore/view/CallRenderViewGL.h"
#include "mmcore/view/CameraControllers.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/FramebufferObject.hpp"

namespace megamol {
namespace core {
namespace view {

class MEGAMOLCORE_API View3DGL : public view::BaseView<CallRenderViewGL, Camera3DController> {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "View3DGL"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "New and improved 3D View Module"; }

    /** Ctor. */
    View3DGL(void);

    /** Dtor. */
    virtual ~View3DGL(void);

    virtual ImageWrapper Render(double time, double instanceTime, bool present_fbo) override;

    ImageWrapper GetRenderingResult() const override;

    /**
     * Resizes the framebuffer object and calls base class function that sets camera aspect ratio if applicable.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height) override;

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);
};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

