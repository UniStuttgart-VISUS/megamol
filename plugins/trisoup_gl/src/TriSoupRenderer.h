/*
 * TriSoupRenderer.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib/math/Cuboid.h"
#include "vislib/memutils.h"


namespace megamol::trisoup_gl {


/**
 * Renderer for tri-mesh data
 */
class TriSoupRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TriSoupRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for tri-mesh data";
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
    TriSoupRenderer();

    /** Dtor. */
    ~TriSoupRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

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
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    /** The slot to fetch the data */
    core::CallerSlot getDataSlot;

    /** The slot to fetch the volume data */
    core::CallerSlot getVolDataSlot;

    /** The call for light sources */
    core::CallerSlot getLightsSlot;

    /** Flag whether or not to show vertices */
    core::param::ParamSlot showVertices;

    /** Flag whether or not use lighting for the surface */
    core::param::ParamSlot lighting;

    /** The rendering style for the front surface */
    core::param::ParamSlot surFrontStyle;

    /** The rendering style for the back surface */
    core::param::ParamSlot surBackStyle;

    /** The Triangle winding rule */
    core::param::ParamSlot windRule;

    /** The Triangle color */
    core::param::ParamSlot colorSlot;
};


} // namespace megamol::trisoup_gl
