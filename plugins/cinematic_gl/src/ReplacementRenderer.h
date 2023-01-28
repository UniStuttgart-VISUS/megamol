/*
 * ReplacementRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_CINEMATIC_REPLACEMENTRENDERER_H_INCLUDED
#define MEGAMOL_CINEMATIC_REPLACEMENTRENDERER_H_INCLUDED
#pragma once


#include "cinematic_gl/CinematicUtils.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/renderer/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "vislib/math/Cuboid.h"


namespace megamol::cinematic_gl {

/*
 * Replacement rendering.
 */
class ReplacementRenderer : public megamol::core::view::RendererModule<mmstd_gl::CallRender3DGL, mmstd_gl::ModuleGL> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ReplacementRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers replacement rendering.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    ReplacementRenderer();

    /** Dtor. */
    ~ReplacementRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'release'.
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
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    /**********************************************************************
     * variables
     **********************************************************************/

    enum KeyAssignment {
        KEY_ASSIGN_NONE,
        KEY_ASSIGN_1,
        KEY_ASSIGN_2,
        KEY_ASSIGN_3,
        KEY_ASSIGN_4,
        KEY_ASSIGN_5,
        KEY_ASSIGN_6,
        KEY_ASSIGN_7,
        KEY_ASSIGN_8,
        KEY_ASSIGN_9,
        KEY_ASSIGN_0
    };

    bool draw_replacement;
    CinematicUtils utils;
    vislib::math::Cuboid<float> bbox;

    /**********************************************************************
     * parameters
     **********************************************************************/

    core::param::ParamSlot replacementRenderingParam;
    core::param::ParamSlot toggleReplacementParam;
    core::param::ParamSlot alphaParam;
};

} // namespace megamol::cinematic_gl

#endif // MEGAMOL_CINEMATIC_REPLACEMENTRENDERER_H_INCLUDED
