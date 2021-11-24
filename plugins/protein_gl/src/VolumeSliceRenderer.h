/*
 * VolumeSliceRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS).
 * All Rights reserved.
 */

#ifndef MEGAMOLCORE_VOLSLICERENDERER_H_INCLUDED
#define MEGAMOLCORE_VOLSLICERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"
#include "protein/VolumeSliceCall.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"

namespace megamol {
namespace protein_gl {
/**
 * Protein Renderer class
 */
class VolumeSliceRenderer : public core_gl::view::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "VolumeSliceRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers volume slice renderings.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    VolumeSliceRenderer(void);

    /** dtor */
    ~VolumeSliceRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

private:
    /**********************************************************************
     * 'render'-functions
     **********************************************************************/

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(megamol::core_gl::view::CallRender2DGL& call);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core_gl::view::CallRender2DGL& call);

    /**********************************************************************
     * variables
     **********************************************************************/

    /** caller slot */
    core::CallerSlot volDataCallerSlot;

    // shader for volume slice rendering
    vislib_gl::graphics::gl::GLSLShader volumeSliceShader;

    // the mouse position
    vislib::math::Vector<float, 3> mousePos;
};

} // namespace protein_gl
} // namespace megamol

#endif // MEGAMOLCORE_VOLSLICERENDERER_H_INCLUDED
