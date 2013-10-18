//
// StreamlineRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 11, 2013
//     Author: scharnkn
//

#if (defined(WITH_CUDA) && (WITH_CUDA))

#ifndef MMPROTEINPLUGIN_STREAMLINERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_STREAMLINERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif // (defined(_MSC_VER) && (_MSC_VER > 1000))

#include "view/Renderer3DModuleDS.h"
#include "CallerSlot.h"
#include "CudaDevArr.h"
#include "view/CallRender3D.h"
#include "VTIDataCall.h"
#include "VBODataCall.h"
#include "param/ParamSlot.h"

typedef vislib::math::Vector<int, 3> Vec3i;

typedef unsigned int uint;

namespace megamol {
namespace protein {

class StreamlineRenderer : public core::view::Renderer3DModuleDS {

public:

    /** CTor */
    StreamlineRenderer(void);

    /** DTor */
    virtual ~StreamlineRenderer(void);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "StreamlineRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Renders streamlines based on a given array of seepoints.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

protected:

    enum StreamlineShading {UNIFORM=0, TEXTURE};

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Creates a vertex buffer object of size s
     *
     * @param vbo    The vertex buffer object
     * @param size   The size of the vertex buffer object
     * @param target The target enum, can either be GL_ARRAY_BUFFER or
     *               GL_ELEMENT_ARRAY_BUFFER
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool createVbo(GLuint* vbo, size_t s, GLuint target);

    /**
     * Destroys the vertex buffer object 'vbo'
     *
     * @param vbo    The vertex buffer object
     * @param target The target enum, can either be GL_ARRAY_BUFFER or
     *               GL_ELEMENT_ARRAY_BUFFER
     */
    void destroyVbo(GLuint* vbo, GLuint target);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * The get capabilities callback. The module should set the members
     * of 'call' to tell the caller its capabilities.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetCapabilities(core::Call& call);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * Open GL Render call.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

private:

    /**
     * Calculate the gradient field of the given volume data
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool computeGradient(VTIDataCall *vtiCall);

    /**
     * Calculate the streamlines
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool computeStreamlines(VBODataCall *vboCall, VTIDataCall *vtiCall);

    /**
     * Renders a streamline bundle around the manually set seed point.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderStreamlineBundleManual();

    /**
     * Renders a streamline bundle around the manually set seed point.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderStreamlines();

    /**
     * Update all parameters and set boolean flags accordingly.
     */
    void updateParams();


    /* Data caller slots */

    /// Caller slot for vertex data
    megamol::core::CallerSlot vertexDataCallerSlot;

    /// Caller slot volume data
    megamol::core::CallerSlot volumeDataCallerSlot;


    /* Streamline integration parameters */

    /// Parameter for streamline maximum steps
    core::param::ParamSlot streamlineMaxStepsSlot;
    unsigned int streamlineMaxSteps;

    /// Parameter to set the step size for streamline integration
    core::param::ParamSlot streamlineStepSlot;
    float streamlineStep;

    /// Parameter to set the epsilon for stream line terminations
    core::param::ParamSlot streamlineEpsSlot;
    float streamlineEps;


    /* Field data */

    /// The volume data (device memory)
    CudaDevArr<float> scalarField_D;

    /// The volumes gradient (device memory)
    CudaDevArr<float> gradientField_D;

    /// The bounding boxes (union of both data sets)
    core::BoundingBoxes bbox;


    /* Streamline vertex positions */

    /// Cuda graphics ressource associated with streamline vbo
    struct cudaGraphicsResource *vboResource[2];

    /// Vertex buffer object for triangle indices of surface #0
    GLuint vbo;

    // Current size of the vbo in byte
    size_t vboSize;


    /* Boolean flags */

    /// Triggers recomputation of the gradient field
    bool triggerComputeGradientField;

    /// Triggers recomputation of the streamlines
    bool triggerComputeStreamlines;

};


} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_STREAMLINERENDERER_H_INCLUDED
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
