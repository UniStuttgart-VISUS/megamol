//
// ComparativeMolSurfaceRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#if (defined(WITH_CUDA) && (WITH_CUDA))

#ifndef MMPROTEINPLUGIN_COMPARATIVEMOLSURFACERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_COMPARATIVEMOLSURFACERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer3DModuleDS.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"

#include "vislib/GLSLShader.h"

namespace megamol {
namespace protein {

class ComparativeMolSurfaceRenderer : public core::view::Renderer3DModuleDS {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "ComparativeMolSurfaceRenderer";
    }

    /** Ctor. */
    ComparativeMolSurfaceRenderer(void);

    /** Dtor. */
    virtual ~ComparativeMolSurfaceRenderer(void);

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Offers comparative rendering of two molecular surfaces.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        if (!vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()) {
            return false;
        }
        return true;
    }

protected:

    /**
     * Implementation of 'create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get capabilities callback. The module should set the members
     * of 'call' to tell the caller its capabilities.
     *
     * @param  call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetCapabilities(core::Call& call);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param  call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * Call callback to get the mapped vertex positions
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVtxData(core::Call& call);

    /**
     * Call callback to get the mapped vertex positions
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVtxExtent(core::Call& call);

    /**
     * Call callback to get the GVF volume texture
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVolData(core::Call& call);

    /**
     * Call callback to get the GVF volume texture
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVolExtent(core::Call& call);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

private:

    /* Data caller/callee slots */

    /// Caller slot for incoming vertex data (representing the mapped surface)
    core::CallerSlot vtxInputSlot;

    /// Callee slot for outgoing vertex data (representing the source mesh)
    core::CalleeSlot vtxOutputSlot;

    /// Callee slot to output the initial external forces (representing the
    /// target shape)
    core::CalleeSlot volOutputSlot;

    /// Caller slot for input molecule #1
    core::CallerSlot molDataSlot1;

    /// Caller slot for input molecule #2
    core::CallerSlot molDataSlot2;

};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_COMPARATIVEMOLSURFACERENDERER_H_INCLUDED
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
