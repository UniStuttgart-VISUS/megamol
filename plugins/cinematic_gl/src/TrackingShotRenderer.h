/*
 * TrackingShotRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_CINEMATIC_TRACKINGSHOTRENDERER_H_INCLUDED
#define MEGAMOL_CINEMATIC_TRACKINGSHOTRENDERER_H_INCLUDED
#pragma once


#include "mmcore/CallerSlot.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "cinematic_gl/CinematicUtils.h"
#include "cinematic_gl/KeyframeManipulators.h"


namespace megamol {
namespace cinematic_gl {

/**
 * Tracking shot rendering.
 */
class TrackingShotRenderer : public core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Gets the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "TrackingShotRenderer";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Renders the tracking shot and passes the render call to another renderer.";
    }

    /**
     * Gets whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /** Ctor. */
    TrackingShotRenderer(void);

    /** Dtor. */
    virtual ~TrackingShotRenderer(void);

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
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(megamol::core_gl::view::CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core_gl::view::CallRender3DGL& call);

    /**
     * The mouse button pressed/released callback.
     */
    virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods) final override;

    /**
     * The mouse movement callback.
     */
    virtual bool OnMouseMove(double x, double y) final override;

private:
    /**********************************************************************
     * variables
     **********************************************************************/

    KeyframeManipulators manipulators;
    CinematicUtils utils;
    float mouseX;
    float mouseY;
    GLuint texture;
    bool manipulatorGrabbed;
    unsigned int interpolSteps;
    bool showHelpText;
    float lineWidth;

    bool skipped_first_mouse_interact; // XXX TODO Find bug why this is necessary

    /**********************************************************************
     * callbacks
     **********************************************************************/

    core::CallerSlot keyframeKeeperSlot;

    /**********************************************************************
     * parameters
     **********************************************************************/

    core::param::ParamSlot stepsParam; // Amount of interpolation steps between keyframes
    core::param::ParamSlot toggleHelpTextParam;
};

} // namespace cinematic_gl
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_TRACKINGSHOTRENDERER_H_INCLUDED
