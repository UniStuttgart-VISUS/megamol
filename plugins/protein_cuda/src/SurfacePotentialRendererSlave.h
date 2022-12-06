//
// SurfaceSurfacePotentialRendererSlave.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 31, 2013
//     Author: scharnkn
//

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/CallerSlot.h"
#include "mmstd/renderer/CallRender3D.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
//#include "vislib_vector_typedefs.h"
#include "VBODataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace protein_cuda {

class SurfacePotentialRendererSlave : public mmstd_gl::Renderer3DModuleGL {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SurfacePotentialRendererSlave";
    }


    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers rendering of molecular surfaces textured by electrostatic potential";
    }


    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }


    /** Ctor. */
    SurfacePotentialRendererSlave(void);


    /** Dtor. */
    virtual ~SurfacePotentialRendererSlave(void);


protected:
    /**
     * Implementation of 'create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);


    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param  call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call);


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
    virtual bool Render(mmstd_gl::CallRender3DGL& call);

    /**
     * Renders the isosurface using different rendering modes and surface
     * colorings.
     *
     * @param c The data call containing the vbo data
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderSurface(VBODataCall* c);


private:
    /// Caller slot to obtaing vbo data
    core::CallerSlot vboSlot;

    /// Transparency factor for the surface
    core::param::ParamSlot surfAlphaSclSlot;

    /// The data sets bounding boxes
    core::BoundingBoxes bbox;

    /// Shader implementing per pixel lighting
    std::unique_ptr<glowl::GLSLProgram> pplSurfaceShader;
};

} // namespace protein_cuda
} // namespace megamol
