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
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "cinematic_gl/CinematicUtils.h"
#include "cinematic_gl/KeyframeManipulators.h"


namespace megamol {
namespace cinematic_gl {

/**
 * Tracking shot rendering.
 */
class TrackingShotRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Gets the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TrackingShotRenderer";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Renders the tracking shot and passes the render call to another renderer.";
    }

    /**
     * Gets whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    TrackingShotRenderer();

    /** Dtor. */
    ~TrackingShotRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

    /**
     * The mouse button pressed/released callback.
     */
    bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods) final;

    /**
     * The mouse movement callback.
     */
    bool OnMouseMove(double x, double y) final;

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
