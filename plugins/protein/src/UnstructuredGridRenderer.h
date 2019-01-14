//
// UnstructuredGridRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 25, 2013
// Author     : scharnkn
//

#ifndef MMPROTEINPLUGIN_UNSTRUCTUREDGRIDRENDERER__H_INCLUDED
#define MMPROTEINPLUGIN_UNSTRUCTUREDGRIDRENDERER__H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "SphereDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace protein {


class UnstructuredGridRenderer : public megamol::core::view::Renderer3DModule {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "UnstructuredGridRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Offers sphere renderings.";
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
    UnstructuredGridRenderer(void);

    /** Dtor. */
    virtual ~UnstructuredGridRenderer(void);

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

private:

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents( megamol::core::Call& call);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render( megamol::core::Call& call);

    /// The data caller slot
    megamol::core::CallerSlot dataCallerSlot;

    /// Parameter for sphere radius
    core::param::ParamSlot sphereRadSlot;

    /// camera information
    vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

    /// Shader for the spheres (raycasting view)
    vislib::graphics::gl::GLSLShader sphereShader;

    /// Shader for the cylinders (raycasting view)
    vislib::graphics::gl::GLSLShader cylinderShader;

    /// Position attribute location for GLSL-Shader
    GLint attribLocInParams;

    /// Orientation attribute location for GLSL-Shader
    GLint attribLocQuatC;

    /// Color attribute location for GLSL-Shader
    GLint attribLocColor1;

    /// Color attribute location for GLSL-Shader
    GLint attribLocColor2;
};


} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_UNSTRUCTUREDGRIDRENDERER_H_INCLUDED
