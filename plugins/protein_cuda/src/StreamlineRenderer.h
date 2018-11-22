//
// StreamlineRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 11, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_STREAMLINERENDERER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_STREAMLINERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif // (defined(_MSC_VER) && (_MSC_VER > 1000))

#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/CallerSlot.h"
#include "CudaDevArr.h"
#include "mmcore/view/CallRender3D.h"
#include "protein_calls/VTIDataCall.h"
#include "VBODataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "CUDAStreamlines.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/GLSLShader.h"

typedef vislib::math::Vector<int, 3> Vec3i;
typedef unsigned int uint;

namespace megamol {
namespace protein_cuda {

class StreamlineRenderer : public core::view::Renderer3DModuleDS {

public:

    enum RenderModes {NONE=0, LINES, ILLUMINATED_LINES, TUBES};

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

    /**
     * Callback called when the clipping plane is requested.
     *
     * @param call The calling call
     *
     * @return 'true' on success
     */
    bool requestPlane(core::Call& call);

private:

    /**
     * Fills up the seed point array based on given clipping plane z values
     * and matching isovalues.
     *
     * @param vti     The data call with the density values
     * @param zClip   The clipping plane z values
     * @param isoval  The iso values
     */
	void genSeedPoints(protein_calls::VTIDataCall *vti, float zClip, float isoval);

    /**
     * Samples the field at a given position using linear interpolation.
     *
     * @param The data call with the grid information
     * @param pos The position
     * @return The sampled value of the field
     */
	float sampleFieldAtPosTrilin(protein_calls::VTIDataCall *vtiCall, float3 pos, float *field_D);

    /**
     * Update all parameters and set boolean flags accordingly.
     */
    void updateParams();


    /* Data caller slots */

    /// Caller slot volume data
    megamol::core::CallerSlot fieldDataCallerSlot;

    /// Caller slot for clipping plane
    megamol::core::CalleeSlot getClipPlaneSlot;


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
    core::param::ParamSlot seedClipZSlot;
    float seedClipZ;

    /// Parameter to set the epsilon for stream line terminations
    core::param::ParamSlot seedIsoSlot;
    float seedIso;


    /* Streamlines rendering parameters */

    /// Parameter to set the trickness of the streamtubes
    core::param::ParamSlot renderModeSlot;
    RenderModes renderMode;

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

    /// The uniform color for surface #1
    static const Vec3f uniformColor;

    /// Variable to store the last position of the clipping plane
    vislib::math::Plane<float> clipPlane;
};


} // end namespace protein_cuda
} // end namespace megamol

#endif // MMPROTEINCUDAPLUGIN_STREAMLINERENDERER_H_INCLUDED
