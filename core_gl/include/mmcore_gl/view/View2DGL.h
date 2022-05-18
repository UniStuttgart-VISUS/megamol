/*
 * View2DGL.h
 *
 * Copyright (C) 2008 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEW2DGL_H_INCLUDED
#define MEGAMOLCORE_VIEW2DGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/BaseView.h"
#include "mmcore/view/TimeControl.h"

#include "mmcore/view/CameraControllers.h"
#include "mmcore_gl/view/AbstractViewGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"

#include <glowl/FramebufferObject.hpp>

namespace megamol {
namespace core_gl {
namespace view {

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
    static const char* ClassName(void) {
        return "View2DGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "2D View Module";
    }

    /** Ctor. */
    View2DGL(void);

    /** Dtor. */
    virtual ~View2DGL(void);

    /**
     * ...
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
    virtual bool create(void);
};
} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEW2DGL_H_INCLUDED */
