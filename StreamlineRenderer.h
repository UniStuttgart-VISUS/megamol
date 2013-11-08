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
#include "CUDAStreamlines.h"
#include "vislib/GLSLGeometryShader.h"
#include "vislib/GLSLShader.h"

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
     * Fills up the seed point array based on given clipping plane z values
     * and matching isovalues.
     *
     * @param vti                                 The data call with the density
     *                                            values
     * @param zClip0, zClip1, zClip2, zClip3      The clipping plane z values
     * @param isoval0, isoval1, isoval2, isoval3  The iso values
     */
    void genSeedPoints(VTIDataCall *vti,
            float zClip0, float zClip1, float zClip2, float zClip3,
            float isoval0, float isoval1, float isoval2, float isoval3);

    /**
     * Samples the field at a given position using linear interpolation.
     *
     * @param The data call with the grid information
     * @param pos The position
     * @return The sampled value of the field
     */
    float sampleFieldAtPosTrilin(VTIDataCall *vtiCall, float3 pos, float *field_D);

    /**
     * Update all parameters and set boolean flags accordingly.
     */
    void updateParams();


    /* Data caller slots */

    /// Caller slot volume data
    megamol::core::CallerSlot fieldDataCallerSlot;


    /* Streamline integration parameters */

    /// Parameter for streamline maximum steps
    core::param::ParamSlot nStreamlinesSlot;
    unsigned int nStreamlines;

    /// Parameter for streamline maximum steps
    core::param::ParamSlot streamlineMaxStepsSlot;
    unsigned int streamlineMaxSteps;

    /// Parameter to set the step size for streamline integration
    core::param::ParamSlot streamlineStepSlot;
    float streamlineStep;

    /// Parameter to set the epsilon for stream line terminations
    core::param::ParamSlot streamlineEpsSlot;
    float streamlineEps;

    /// Parameter to set the trickness of the streamtubes
    core::param::ParamSlot streamtubesThicknessSlot;
    float streamtubesThickness;

    /// Parameter for minimum color
    core::param::ParamSlot minColSlot;
    float minCol;

    /// Parameter for maximum color
    core::param::ParamSlot maxColSlot;
    float maxCol;


    /* Field data */

    /// The bounding boxes (union of both data sets)
    core::BoundingBoxes bbox;

    /// The streamlines object
    CUDAStreamlines strLines;

    /// Array with sedd points
    vislib::Array<float> seedPoints;


    /* Boolean flags */

    /// Triggers recomputation of the gradient field
    bool triggerComputeGradientField;

    /// Triggers recomputation of the streamlines
    bool triggerComputeStreamlines;


    /* Rendering */

    // Shader for stream tubes
    vislib::graphics::gl::GLSLGeometryShader tubeShader;

    // Shader for illuminated streamlines
    vislib::graphics::gl::GLSLGeometryShader illumShader;

};


} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_STREAMLINERENDERER_H_INCLUDED
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
