/*
 * SimplestSphereRenderer_2.h
 *
 * Copyright (C) 2018 by Karsten Schatz
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MM101PLG_SIMPLESTSPHERERENDERER_2_H_INCLUDED
#define MM101PLG_SIMPLESTSPHERERENDERER_2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/nextgen/Renderer3DModule_2.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"

namespace megamol {
namespace megamol101 {

/**
 * Renders incoming spheres to the screen, either using GL_POINTS or more
 * sophisticated shaders.
 */
class SimplestSphereRenderer_2 : public core::nextgen::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * TUTORIAL: Mandatory method for every module or call that states the name of the class.
     * This name should be unique.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "SimplestSphereRenderer_2"; }

    /**
     * Answer a human readable description of this module.
     *
     * TUTORIAL: Mandatory method for every module or call that returns a description.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renders a set of incoming spheres"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * TUTORIAL: Mandatory method for every module.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Constructor. */
    SimplestSphereRenderer_2(void);

    /** Destructor. */
    virtual ~SimplestSphereRenderer_2(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * TUTORIAL: The overwritten version of this method gets called right after an object of this class has been
     * instantiated. Shader compilation should be done inside this method.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     *
     * TUTORIAL: The overwritten version of this method gets called on destruction of this object.
     * Necessary cleanup should be done inside this method.
     */
    virtual void release(void);

private:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * TUTORIAL: This method computes the extents of the rendered data, namely the bounding box and other relevant
     *  values, and writes them into the calling call.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::nextgen::CallRender3D_2& call);

    /**
     * The Open GL Render callback.
     *
     * TUTORIAL: Mandatory method for each renderer. It performs the main OpenGL rendering.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::nextgen::CallRender3D_2& call);

    /** The input data slot. */
    core::CallerSlot sphereDataSlot;

    /** The vertex buffer object for the rendered vertices. */
    GLuint vbo;

    /** The vertex array for the rendered vertices. */
    GLuint va;

    /** The data hash of the most recent rendered data */
    SIZE_T lastDataHash;

    /** The simple shader for the drawing of GL_POINTS */
    vislib::graphics::gl::GLSLShader simpleShader;

    /** The pretty shader that draws spheres*/
    vislib::graphics::gl::GLSLGeometryShader sphereShader;

    /** Slot for the scaling factor of the pointsize*/
    core::param::ParamSlot sizeScalingSlot;

    /** Slot for the switch between GL_POINTS and spheres */
    core::param::ParamSlot sphereModeSlot;
};

} /* end namespace megamol101 */
} /* end namespace megamol */

#endif /* MM101PLG_SIMPLESTSPHERERENDERER_H_INCLUDED */
