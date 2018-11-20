//
// ComparativeFieldTopologyRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Feb 07, 2013
//     Author: scharnkn
//

#include "stdafx.h"
#include "ComparativeFieldTopologyRenderer.h"

#include <cmath>
#include <ctime>

#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"

#include "vislib/math/Matrix.h"
#include "vislib/sys/Log.h"
#include "vislib/math/mathfunctions.h"

#include "protein_calls/VTIDataCall.h"
#include "protein_calls/Interpol.h"
#include "VecField3f.h"
#include "CUDAFieldTopology.cuh"
#include <GL/glu.h>

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"

bool myisnan(double x) { return x != x; } // uses Nan != Nan
bool myisinf(double x) { return !myisnan(x) && myisnan(x - x); } // uses inf-inf=NaN

using namespace megamol;
using namespace megamol::protein_cuda;
using namespace megamol::core;


/*
 * ComparativeFieldTopologyRenderer::ComparativeFieldTopologyRenderer
 */
ComparativeFieldTopologyRenderer::ComparativeFieldTopologyRenderer(void) : view::Renderer3DModule(),
        /* Data caller slots */
        dataCallerSlot0("getdata0", "Connects the renderer with the first data source"),
        dataCallerSlot1("getdata1", "Connects the renderer with the second data source"),
        /* Parameters related to fog */
        fogZSlot("fog::startZ", "Change the starting distance for fog"),
        /* Parameters for arrow glyph rendering */
        arrowRadSclSlot("arrows::arrowRad", "Radius of arrow glyphs"),
        arrowLenSclSlot("arrows::arrowScl", "(Linear) scaling for arrow glyphs"),
        arrowFilterXMaxSlot("arrows::XMax", "The maximum position in x-direction"),
        arrowFilterYMaxSlot("arrows::YMax", "The maximum position in y-direction"),
        arrowFilterZMaxSlot("arrows::ZMax", "The maximum position in z-direction"),
        arrowFilterXMinSlot("arrows::XMin", "The minimum position in x-direction"),
        arrowFilterYMinSlot("arrows::YMin", "The minimum position in y-direction"),
        arrowFilterZMinSlot("arrows::ZMin", "The minimum position in z-direction"),
        /* Parameters for critical point analysis */
        critPointsSphereSclSlot("critpoints::scale", "(Linear) scaling for the sphere radius"),
        critPointsMaxBisectionsSlot("critpoints::maxBisect", "The maximum number of bisections"),
        critPointsNewtonMaxStepsSlot("critpoints::newtonMaxSteps", "The maximum number of Newton iterations"),
        critPointsNewtonStepsizeSlot("critpoints::newtonStep", "The stepSize of the Newton iteration"),
        critPointsNewtonEpsSlot("critpoints::newtonEps", "The epsilon of the Newton iteration"),
        critPointsShowAllSlot("critpoints::showAll", "Toggle rendering of all critpoints vs. critpints in roi"),
        /* Parameters for streamlines */
        streamlineMaxStepsSlot("streamlines::maxSteps", "Maximum number of steps for streamlines"),
        streamlineShadingSlot("streamlines::shading", "Shading of the streamlines"),
        streamBundleRadSlot("streamlines::bundleRad", "Radius of the streamline bundle"),
        streamBundleResSlot("streamlines::bundleRes", "Resolution of the streamline bundle"),
        streamBundlePhiSlot("streamlines::bundlePhi", "Stepsize of the streamline bundle"),
        streamlineEpsSlot("streamlines::epsilon", "Epsilon for streamline termination"),
        streamlineStepsizeSlot("streamlines::step", "Stepsize for streamline integration"),
        toggleStreamlinesSlot("streamlines::toggle", "Toggle the rendering of the streamlines"),
        streamlinesShowAllSlot("streamlines::showAll", "Show all streamlines"),
        /* Parameters for manually set streamline seed point */
        streamBundleSeedXSlot("manualSeed::posX", "The x-coord of the streamline seedpoint"),
        streamBundleSeedYSlot("manualSeed::posY", "The y-coord of the streamline seedpoint"),
        streamBundleSeedZSlot("manualSeed::posZ", "The z-coord of the streamline seedpoint"),
        streamlineMaxStepsManualSlot("manualSeed::maxSteps", "The maximum step number"),
        streamlineShadingManualSlot("manualSeed::shading", "Shading mode for the streamlines"),
        streamBundleRadManualSlot("manualSeed::rad", "The radius of the streamline bundle"),
        streamBundleResManualSlot("manualSeed::res", "The resolution of the streamline bundle"),
        streamBundlePhiManualSlot("manualSeed::phi", "The size of the streamline bundle"),
        streamlineEpsManualSlot("manualSeed::eps", "The minimum vector length for streamline integration"),
        streamlineStepsizeManualSlot("manualSeed::step", "The stepsize for the streamlines"),
        toggleStreamlinesManualSlot("manualSeed::show", "Toggle the rendering of the manual streamline bundle"),
        /* Parameters for finding regions of interest */
        roiMaxDistSlot("roi::maxDist", "Maximum distance between two critical points to be neighbours"),
        /* Parameters for debugging purposes */
        texMinValSlot("texslice::min", "Minimum texture value"),
        texMaxValSlot("texslice::max", "Maximum texture value"),
        texPosZSlot("texslice::posZ", "Texture slice position"),
        /* Electrostatics */
        potentialTex0(0), potentialTex1(0),
        /* Boolean flags */
        recalcEfield(true), recalcStreamlines(true), recalcCritPoints(true),
        recalcStreamlinesManualSeed(true), recalcNeighbours(true), recalcArrowData(true),
        /* Misc */
        dataHash(0) {


    /* Data caller slots */

    // The slot for the first data call
    this->dataCallerSlot0.SetCompatibleCall<protein_calls::VTIDataCallDescription>();
    this->MakeSlotAvailable (&this->dataCallerSlot0);

    // The slot for the second data call
    this->dataCallerSlot1.SetCompatibleCall<protein_calls::VTIDataCallDescription>();
    this->MakeSlotAvailable (&this->dataCallerSlot1);


    /* Parameters related to fog */

    // Parameter for fog starting point
    this->fogZ = 0.1f;
    this->fogZSlot.SetParameter(new core::param::FloatParam(this->fogZ, 0.0f, 0.99f));
    this->MakeSlotAvailable(&this->fogZSlot);


    /* Parameters for arrow glyph rendering */

    // Param for the radius for the arrow glyphs
    this->arrowRadScl = 0.001f;
    this->arrowRadSclSlot.SetParameter(new core::param::FloatParam(this->arrowRadScl, 0.0f));
    this->MakeSlotAvailable(&this->arrowRadSclSlot);

    // Param for the scaling of the arrow glyphs
    this->arrowLenScl = 0.001f;
    this->arrowLenSclSlot.SetParameter(new core::param::FloatParam(this->arrowLenScl, 0.0f));
    this->MakeSlotAvailable(&this->arrowLenSclSlot);

    // Param for maximum x value
    this->arrowFilterXMax = 1.0f;
    this->arrowFilterXMaxSlot.SetParameter(new core::param::FloatParam(this->arrowFilterXMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->arrowFilterXMaxSlot);

    // Param for maximum y value
    this->arrowFilterYMax = 1.0f;
    this->arrowFilterYMaxSlot.SetParameter(new core::param::FloatParam(this->arrowFilterYMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->arrowFilterYMaxSlot);

    // Param for maximum z value
    this->arrowFilterZMax = 1.0f;
    this->arrowFilterZMaxSlot.SetParameter(new core::param::FloatParam(this->arrowFilterZMax, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->arrowFilterZMaxSlot);

    // Param for minimum x value
    this->arrowFilterXMin = -1.0f;
    this->arrowFilterXMinSlot.SetParameter(new core::param::FloatParam(this->arrowFilterXMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->arrowFilterXMinSlot);

    // Param for minimum y value
    this->arrowFilterYMin = -1.0f;
    this->arrowFilterYMinSlot.SetParameter(new core::param::FloatParam(this->arrowFilterYMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->arrowFilterYMinSlot);

    // Param for minimum z value
    this->arrowFilterZMin = -1.0f;
    this->arrowFilterZMinSlot.SetParameter(new core::param::FloatParam(this->arrowFilterZMin, -1000.0f, 1000.0f));
    this->MakeSlotAvailable(&this->arrowFilterZMinSlot);


    /* Parameters for critical point analysis */

    // Param for the scaling of the arrow glyphs
    this->critPointsSphereScl = 0.001f;
    this->critPointsSphereSclSlot.SetParameter(new core::param::FloatParam(this->critPointsSphereScl, 0.0f));
    this->MakeSlotAvailable(&this->critPointsSphereSclSlot);

    // Parameter to determine the maximum number of bisections
    this->critPointsMaxBisections = 10;
    this->critPointsMaxBisectionsSlot.SetParameter(new core::param::IntParam(this->critPointsMaxBisections, 0));
    this->MakeSlotAvailable(&this->critPointsMaxBisectionsSlot);

    // Parameter for maximum number of Newton iterations
    this->critPointsNewtonMaxSteps = 10;
    this->critPointsNewtonMaxStepsSlot.SetParameter(new core::param::IntParam(this->critPointsNewtonMaxSteps, 0));
    this->MakeSlotAvailable(&this->critPointsNewtonMaxStepsSlot);

    // Parameter for the stepsize of the Newton iteration
    this->critPointsNewtonStepsize = 0.2f;
    this->critPointsNewtonStepsizeSlot.SetParameter(new core::param::FloatParam(this->critPointsNewtonStepsize, 0.0f));
    this->MakeSlotAvailable(&this->critPointsNewtonStepsizeSlot);

    // Parameter for the epsilon for the Newton iteration
    this->critPointsNewtonEps = 0.2f;
    this->critPointsNewtonEpsSlot.SetParameter(new core::param::FloatParam(this->critPointsNewtonEps, 0.0f));
    this->MakeSlotAvailable(&this->critPointsNewtonEpsSlot);

    // Param slot to determine whether all critpints are to be rendered
    this->critPointsShowAll = false;
    this->critPointsShowAllSlot.SetParameter(new core::param::BoolParam(this->critPointsShowAll));
    this->MakeSlotAvailable(&this->critPointsShowAllSlot);


    /* Parameters for streamlines */

    // Param for maximum steps of streamline drawing
    this->streamlineMaxSteps = 50;
    this->streamlineMaxStepsSlot.SetParameter(new core::param::IntParam(this->streamlineMaxSteps, 0));
    this->MakeSlotAvailable(&this->streamlineMaxStepsSlot);

    // Param for the shading mode of the streamlines
    this->streamlineShading = POTENTIAL;
    param::EnumParam *ss = new core::param::EnumParam(this->streamlineShading);
    ss->SetTypePair(UNIFORM, "Uniform");
    ss->SetTypePair(POTENTIAL, "Potential");
    this->streamlineShadingSlot << ss;
    this->MakeSlotAvailable(&this->streamlineShadingSlot);

    // Param for the radius of the streamline bundle
    this->streamBundleRad = 0.1f;
    this->streamBundleRadSlot.SetParameter(new core::param::FloatParam(this->streamBundleRad, 0.0f));
    this->MakeSlotAvailable(&this->streamBundleRadSlot);

    // Param for the resolution of the streamline bundle
    this->streamBundleRes = 1;
    this->streamBundleResSlot.SetParameter(new core::param::IntParam(this->streamBundleRes, 1));
    this->MakeSlotAvailable(&this->streamBundleResSlot);

    // Param for the stepSize of the streamline bundle
    this->streamBundlePhi = 0.4f;
    this->streamBundlePhiSlot.SetParameter(new core::param::FloatParam(this->streamBundlePhi, 0.01f, 3.14f));
    this->MakeSlotAvailable(&this->streamBundlePhiSlot);

    // Param for the epsilon for streamline termination
    this->streamlineEps = 0.0f;
    this->streamlineEpsSlot.SetParameter(new core::param::FloatParam(this->streamlineEps, 0.0f));
    this->MakeSlotAvailable(&this->streamlineEpsSlot);

    // Param for the stepsize in streamline integration
    this->streamlineStepsize = 0.5f;
    this->streamlineStepsizeSlot.SetParameter(new core::param::FloatParam(this->streamlineStepsize, 0.0f));
    this->MakeSlotAvailable(&this->streamlineStepsizeSlot);

    // Parameter to toggle rendering of streamlines
    this->toggleStreamlines = false;
    this->toggleStreamlinesSlot.SetParameter(new core::param::BoolParam(this->toggleStreamlines));
    this->MakeSlotAvailable(&this->toggleStreamlinesSlot);

    // Parameter to toggle rendering of all streamlines
    this->streamlinesShowAll = false;
    this->streamlinesShowAllSlot.SetParameter(new core::param::BoolParam(this->streamlinesShowAll));
    this->MakeSlotAvailable(&this->streamlinesShowAllSlot);


    /* Parameters for manually set streamline seed point */

    // Param for x-coord of streamline seed point
    this->streamBundleSeedX = 0.0f;
    this->streamBundleSeedXSlot.SetParameter(new core::param::FloatParam(this->streamBundleSeedX));
    this->MakeSlotAvailable(&this->streamBundleSeedXSlot);

    // Param for y-coord of streamline seed point
    this->streamBundleSeedY = 0.0f;
    this->streamBundleSeedYSlot.SetParameter(new core::param::FloatParam(this->streamBundleSeedY));
    this->MakeSlotAvailable(&this->streamBundleSeedYSlot);

    // Param for z-coord of streamline seed point
    this->streamBundleSeedZ = 0.0f;
    this->streamBundleSeedZSlot.SetParameter(new core::param::FloatParam(this->streamBundleSeedZ));
    this->MakeSlotAvailable(&this->streamBundleSeedZSlot);

    // Parameter for streamline maximum steps
    this->streamlineMaxStepsManual = 50;
    this->streamlineMaxStepsManualSlot.SetParameter(new core::param::IntParam(this->streamlineMaxStepsManual, 0));
    this->MakeSlotAvailable(&this->streamlineMaxStepsManualSlot);

    // Parameter to determine streamline shading
    this->streamlineShadingManual = UNIFORM;
    param::EnumParam *sm = new core::param::EnumParam(this->streamlineShadingManual);
    sm->SetTypePair(UNIFORM, "Uniform");
    sm->SetTypePair(POTENTIAL, "Potential");
    this-> streamlineShadingManualSlot << sm;
    this->MakeSlotAvailable(&this-> streamlineShadingManualSlot);

    // Parameter to set the radius of the streamline bundle
    this->streamBundleRadManual = 0.1f;
    this->streamBundleRadManualSlot.SetParameter(new core::param::FloatParam(this->streamBundleRadManual, 0.0f));
    this->MakeSlotAvailable(&this->streamBundleRadManualSlot);

    // Parameter to set the resolution of the streamline bundle
    this->streamBundleResManual = 1;
    this->streamBundleResManualSlot.SetParameter(new core::param::IntParam(this->streamBundleResManual, 1));
    this->MakeSlotAvailable(&this->streamBundleResManualSlot);

    // Parameter to set the step size of the streamline bundle
    this->streamBundlePhiManual = 0.7f;
    this->streamBundlePhiManualSlot.SetParameter(new core::param::FloatParam(this->streamBundlePhiManual, 0.0f, 3.14f));
    this->MakeSlotAvailable(&this->streamBundlePhiManualSlot);

    // Parameter to set the epsilon for stream line terminations
    this->streamlineEpsManual = 0.0f;
    this->streamlineEpsManualSlot.SetParameter(new core::param::FloatParam(this->streamlineEpsManual, 0.0f));
    this->MakeSlotAvailable(&this->streamlineEpsManualSlot);

    // Parameter to set the stepsize for streamline integration
    this->streamlineStepsizeManual = 0.5f;
    this->streamlineStepsizeManualSlot.SetParameter(new core::param::FloatParam(this->streamlineStepsizeManual, 0.01f));
    this->MakeSlotAvailable(&this->streamlineStepsizeManualSlot);

    // Parameter to toggle rendering of streamlines based on manual seed
    this->toggleStreamlinesManual = false;
    this->toggleStreamlinesManualSlot.SetParameter(new core::param::BoolParam(this->toggleStreamlinesManual));
    this->MakeSlotAvailable(&this->toggleStreamlinesManualSlot);


    /* Parameters for finding regions of interest */

    // Maximum distance between sinks/sources in different data sets
    this->roiMaxDist = 1.0f;
    this->roiMaxDistSlot.SetParameter(new core::param::FloatParam(this->roiMaxDist, 0.0f));
    this->MakeSlotAvailable(&this->roiMaxDistSlot);


    /* Parameters for debugging purposes */

    // Minimum texture value
    this->texMinVal = -1.0f;
    this->texMinValSlot.SetParameter(new core::param::FloatParam(this->texMinVal));
    this->MakeSlotAvailable(&this->texMinValSlot);

    // Maximum texture value
    this->texMaxVal = 1.0f;
    this->texMaxValSlot.SetParameter(new core::param::FloatParam(this->texMaxVal));
    this->MakeSlotAvailable(&this->texMaxValSlot);

    // Position of texture slice
    this->texPosZ = 0.0f;
    this->texPosZSlot.SetParameter(new core::param::FloatParam(this->texPosZ));
    this->MakeSlotAvailable(&this->texPosZSlot);

}


/*
 * ComparativeFieldTopologyRenderer::~ComparativeFieldTopologyRenderer
 */
ComparativeFieldTopologyRenderer::~ComparativeFieldTopologyRenderer(void) {
    this->Release();
}


/*
 * ComparativeFieldTopologyRenderer::create
 */
bool ComparativeFieldTopologyRenderer::create(void) {

    // Load shaders
    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }
    if(!ogl_IsVersionGEQ(2,0) || !isExtAvailable("GL_EXT_texture3D") || !isExtAvailable("GL_ARB_multitexture")) {
            return false;
    }

    // Load shader sources
    ShaderSource vertSrc, fragSrc, geomSrc;

    core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) return false;

    // Load arrow shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::arrowVertexGeom", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for arrow shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::arrowGeom", geomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for arrow shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::arrowFragmentGeom", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for arrow shader");
        return false;
    }
    this->arrowShader.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count());
    this->arrowShader.Link();

    // Load sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::sphereVertexGeom", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::sphereGeom", geomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::sphereFragmentGeom", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for sphere shader");
        return false;
    }
    this->sphereShader.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count());
    this->sphereShader.Link();

    // Load streamline shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::streamline::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source: streamline shader", this->ClassName());
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::streamline::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load fragment shader source:  streamline shader", this->ClassName());
        return false;
    }
    try {
        if(!this->streamlineShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
	catch (vislib::Exception &e){
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load slice shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::slice::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source: slice shader", this->ClassName());
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::slice::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load fragment shader source:  slice shader", this->ClassName());
        return false;
    }
    try {
        if(!this->sliceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
	catch (vislib::Exception &e){
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Init random number generator
    srand((unsigned)time(0));

    return true;
}


/*
 * ComparativeFieldTopologyRenderer::release
 */
void ComparativeFieldTopologyRenderer::release(void) {
    this->arrowShader.Release();
    this->sphereShader.Release();
    this->streamlineShader.Release();
    this->sliceShader.Release();
    if(glIsTexture(this->potentialTex0)) glDeleteTextures(1, &this->potentialTex0);
    if(glIsTexture(this->potentialTex1)) glDeleteTextures(1, &this->potentialTex1);
}


/*
 * ComparativeFieldTopologyRenderer::GetExtents
 */
bool ComparativeFieldTopologyRenderer::GetExtents(core::Call& call) {
//    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
//    if (cr3d == NULL) return false;
//
//    vislib::math::Cuboid<float> bbox0, bbox1;
//
//    // Get bounding box of first data set
//    protein_cuda::VTIDataCall *cmd0 =
//                this->dataCallerSlot0.CallAs<protein_cuda::VTIDataCall>();
//    if(cmd0 == NULL) return false;
//    if(!(*cmd0)(protein_cuda::VTIDataCall::CallForGetExtent)) return false;
//    bbox0.Set(cmd0->GetOrigin().X(),
//            cmd0->GetOrigin().Y(),
//            cmd0->GetOrigin().Z(),
//            cmd0->GetOrigin().X() + (cmd0->GetGridsize().X()-1)*cmd0->GetSpacing().X(),
//            cmd0->GetOrigin().Y() + (cmd0->GetGridsize().Y()-1)*cmd0->GetSpacing().Y(),
//            cmd0->GetOrigin().Z() + (cmd0->GetGridsize().Z()-1)*cmd0->GetSpacing().Z());
//
//    // Get bounding box of first data set
//    protein_cuda::VTIDataCall *cmd1 =
//                this->dataCallerSlot1.CallAs<protein_cuda::VTIDataCall>();
//    if(cmd1 == NULL) return false;
//    if(!(*cmd1)(protein_cuda::VTIDataCall::CallForGetExtent)) return false;
//    bbox1.Set(cmd0->GetOrigin().X(),
//            cmd1->GetOrigin().Y(),
//            cmd1->GetOrigin().Z(),
//            cmd1->GetOrigin().X() + (cmd1->GetGridsize().X()-1)*cmd1->GetSpacing().X(),
//            cmd1->GetOrigin().Y() + (cmd1->GetGridsize().Y()-1)*cmd1->GetSpacing().Y(),
//            cmd1->GetOrigin().Z() + (cmd1->GetGridsize().Z()-1)*cmd1->GetSpacing().Z());
//
//    this->bbox = bbox0;
//    this->bbox.Union(bbox1);
//
//    /*printf("bbox %f %f %f %f %f %f\n", cmd->GetOrigin().X(),
//            cmd->GetOrigin().Y(), cmd->GetOrigin().Z(),
//            cmd->GetOrigin().X() + (cmd->GetGridsize().X()-1)*cmd->GetSpacing().X(),
//            cmd->GetOrigin().Y() + (cmd->GetGridsize().Y()-1)*cmd->GetSpacing().Y(),
//            cmd->GetOrigin().Z() + (cmd->GetGridsize().Z()-1)*cmd->GetSpacing().Z());*/
//
//    float scale;
//    if(!vislib::math::IsEqual(this->bbox.LongestEdge(), 0.0f) ) {
//        scale = 2.0f / this->bbox.LongestEdge();
//    } else {
//        scale = 1.0f;
//    }
//    cr3d->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
//    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
//    cr3d->SetTimeFramesCount(1);

    return true;
}


/*
 * ComparativeFieldTopologyRenderer::Render
 */
bool ComparativeFieldTopologyRenderer::Render(core::Call& call) {

    using namespace vislib::math;
    using namespace vislib::sys;

    // Get render call from view3d
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if(cr3d == NULL) return false;

    // Get call time
    float callTime = cr3d->Time();

    // Get data calls
    protein_calls::VTIDataCall *cmd0 = this->dataCallerSlot0.CallAs<protein_calls::VTIDataCall>();
    if(cmd0 == NULL) return false;
    protein_calls::VTIDataCall *cmd1 = this->dataCallerSlot1.CallAs<protein_calls::VTIDataCall>();
    if(cmd1 == NULL) return false;

    // Get extents
    if(!(*cmd0)(protein_calls::VTIDataCall::CallForGetExtent)) return false;
    cmd0->SetCalltime(callTime); // Set call time
    if(!(*cmd1)(protein_calls::VTIDataCall::CallForGetExtent)) return false;
    cmd1->SetCalltime(callTime); // Set call time

    // Load data containing potential maps
    if(!(*cmd0)(protein_calls::VTIDataCall::CallForGetData)) return false;
    if(!(*cmd1)(protein_calls::VTIDataCall::CallForGetData)) return false;




    vislib::math::Cuboid<float> bbox0, bbox1;

    // Get bounding box of first data set
    bbox0.Set(cmd0->GetOrigin().X(),
            cmd0->GetOrigin().Y(),
            cmd0->GetOrigin().Z(),
            cmd0->GetOrigin().X() + (cmd0->GetGridsize().X()-1)*cmd0->GetSpacing().X(),
            cmd0->GetOrigin().Y() + (cmd0->GetGridsize().Y()-1)*cmd0->GetSpacing().Y(),
            cmd0->GetOrigin().Z() + (cmd0->GetGridsize().Z()-1)*cmd0->GetSpacing().Z());

    // Get bounding box of first data set
    bbox1.Set(cmd0->GetOrigin().X(),
            cmd1->GetOrigin().Y(),
            cmd1->GetOrigin().Z(),
            cmd1->GetOrigin().X() + (cmd1->GetGridsize().X()-1)*cmd1->GetSpacing().X(),
            cmd1->GetOrigin().Y() + (cmd1->GetGridsize().Y()-1)*cmd1->GetSpacing().Y(),
            cmd1->GetOrigin().Z() + (cmd1->GetGridsize().Z()-1)*cmd1->GetSpacing().Z());

    this->bbox = bbox0;
    this->bbox.Union(bbox1);

    /*printf("bbox %f %f %f %f %f %f\n", cmd->GetOrigin().X(),
            cmd->GetOrigin().Y(), cmd->GetOrigin().Z(),
            cmd->GetOrigin().X() + (cmd->GetGridsize().X()-1)*cmd->GetSpacing().X(),
            cmd->GetOrigin().Y() + (cmd->GetGridsize().Y()-1)*cmd->GetSpacing().Y(),
            cmd->GetOrigin().Z() + (cmd->GetGridsize().Z()-1)*cmd->GetSpacing().Z());*/

    float scale;
    if(!vislib::math::IsEqual(this->bbox.LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.LongestEdge();
    } else {
        scale = 1.0f;
    }
    cr3d->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(1);


    if((cmd0->GetGridsize().GetX() == 0)||
            (cmd0->GetGridsize().GetY() == 0)||
            (cmd0->GetGridsize().GetZ() == 0)) {
        return true;
    }

    if((cmd1->GetGridsize().GetX() == 0)||
            (cmd1->GetGridsize().GetY() == 0)||
            (cmd1->GetGridsize().GetZ() == 0)) {
        return true;
    }

    // Check whether one of the data sets has been changed
    if(cmd0->DataHash() != this->dataHash) {
        this->dataHash = cmd0->DataHash();
        this->recalcEfield = true;
        this->recalcNeighbours = true;
        this->recalcStreamlines = true;
        this->recalcCritPoints = true;
        if(glIsTexture(this->potentialTex0)) glDeleteTextures(1, &this->potentialTex0);
    }
    if(cmd1->DataHash() != this->dataHash) {
        this->dataHash = cmd1->DataHash();
        this->recalcEfield = true;
        this->recalcNeighbours = true;
        this->recalcStreamlines = true;
        this->recalcCritPoints = true;
        if(glIsTexture(this->potentialTex1)) glDeleteTextures(1, &this->potentialTex1);
    }

    // Get camera information
    this->cameraInfo =  dynamic_cast<core::view::CallRender3D*>(&call)->GetCameraParameters();

    // Update parameters
    this->updateParams();

    // (Re-)calculate the electric fields if necessary
    if(this->recalcEfield) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: (Re-)calculate the electrostatic field ...",
                this->ClassName()); // DEBUG
        time_t t = clock(); // DEBUG
        this->calcElectrostaticField(cmd0, this->efield0, this->egradfield0);
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #0 done (%f s)",
                this->ClassName(),
                this->efield0.GetCritPointCount(),
                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
        t = clock(); // DEBUG
        this->calcElectrostaticField(cmd1, this->efield1, this->egradfield1);
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #1 done (%f s)",
                this->ClassName(),
                this->efield0.GetCritPointCount(),
                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
        this->recalcEfield = false;
        this->recalcCritPoints = true;
        this->recalcStreamlines = true;
        this->recalcStreamlinesManualSeed = true;
    }

//    // Find and classify critical points
//    if(this->recalcCritPoints) {
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Searching for critical points in the electric field ...",
//                this->ClassName()); // DEBUG
//        time_t t = clock(); // DEBUG
//
//        this->efield0.SearchCritPointsCUDA(
//                this->critPointsNewtonMaxSteps,
//                this->critPointsNewtonStepsize,
//                this->critPointsNewtonEps);
//
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #0 done, %u null points found (%f s)",
//                this->ClassName(),
//                this->efield0.GetCritPointCount(),
//                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
//        t = clock(); // DEBUG
//
//        this->efield1.SearchCritPointsCUDA(
//                this->critPointsNewtonMaxSteps,
//                this->critPointsNewtonStepsize,
//                this->critPointsNewtonEps);
//
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #1 done, %u null points found (%f s)",
//                this->ClassName(),        // Set attribute pointers
//                this->efield1.GetCritPointCount(),
//                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
//        t = clock(); // DEBUG
//
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Searching for critical points in the gradient field ...",
//                this->ClassName()); // DEBUG
//        t = clock(); // DEBUG
//
//        this->egradfield0.SearchCritPointsCUDA(
//                this->critPointsNewtonMaxSteps,
//                this->critPointsNewtonStepsize,
//                this->critPointsNewtonEps);
//
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #0 done, %u null points found (%f s)",
//                this->ClassName(),
//                this->egradfield0.GetCritPointCount(),
//                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
//        t = clock(); // DEBUG
//
//        this->egradfield1.SearchCritPointsCUDA(
//                this->critPointsNewtonMaxSteps,
//                this->critPointsNewtonStepsize,
//                this->critPointsNewtonEps);
//
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #1 done, %u null points found (%f s)",
//                this->ClassName(),
//                this->egradfield1.GetCritPointCount(),
//                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
//
//        this->recalcCritPoints = false;
//    }
//
//    if(this->recalcNeighbours) {
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Searching for null points without neighbours ...",
//                this->ClassName()); // DEBUG
//        time_t t = clock(); // DEBUG
//        this->findNeighbours();
//        unsigned int i = 0;
//        for(unsigned int cnt = 0; cnt < this->neighbours0.Count(); cnt++) {
//            if(this->neighbours0[cnt].Count() == 0) i++;
//        }
//        for(unsigned int cnt = 0; cnt < this->neighbours1.Count(); cnt++) {
//            if(this->neighbours1[cnt].Count() == 0) i++;
//        }
//        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... done, %u sinks/sources/saddles without neighbour inside dist %f found (%f s)",
//                this->ClassName(),
//                this->egradfield1.GetCritPointCount(), i,
//                this->roiMaxDist,
//                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
//        this->recalcNeighbours = false;
//    }

    /* RENDERING */

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    // Compute scale factor and scale world
//    float scale;
//    if(!vislib::math::IsEqual(this->bbox.LongestEdge(), 0.0f) ) {
//        scale = 2.0f /this->bbox.LongestEdge();
//    } else {
//        scale = 1.0f;
//    }
    glScalef(scale, scale, scale);

    // Setup OpenGL
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glLineWidth(1.0);

    // Prepare stuff for glyph ray casting
    this->viewportStuff[0] = this->cameraInfo->TileRect().Left();
    this->viewportStuff[1] = this->cameraInfo->TileRect().Bottom();
    this->viewportStuff[2] = this->cameraInfo->TileRect().Width();
    this->viewportStuff[3] = this->cameraInfo->TileRect().Height();
    if(this->viewportStuff[2] < 1.0f) this->viewportStuff[2] = 1.0f;
    if(this->viewportStuff[3] < 1.0f) this->viewportStuff[3] = 1.0f;
    this->viewportStuff[2] = 2.0f / this->viewportStuff[2];
    this->viewportStuff[3] = 2.0f / this->viewportStuff[3];
    glGetFloatv(GL_MODELVIEW_MATRIX, this->modelMatrix);
    glGetFloatv(GL_PROJECTION_MATRIX, this->projMatrix);
    glGetLightfv(GL_LIGHT0, GL_POSITION, this->lightPos);

    // Prepare potential texture if necessary
    glEnable(GL_TEXTURE_3D);
    if (!glIsTexture(this->potentialTex0)) {
        glGenTextures(1, &this->potentialTex0);
        glBindTexture(GL_TEXTURE_3D, this->potentialTex0);
        glTexImage3DEXT(GL_TEXTURE_3D, 0, GL_RGBA32F, cmd0->GetGridsize().X(),
                cmd0->GetGridsize().Y(), cmd0->GetGridsize().Z(), 0, GL_ALPHA,
                GL_FLOAT, (float*)(cmd0->GetPointDataByIdx(0, 0)));
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_3D, 0);
    }
    if (!glIsTexture(this->potentialTex1)) {
        glGenTextures(1, &this->potentialTex1);
        glBindTexture(GL_TEXTURE_3D, this->potentialTex1);
        glTexImage3DEXT(GL_TEXTURE_3D, 0, GL_RGBA32F, cmd1->GetGridsize().X(),
                cmd1->GetGridsize().Y(), cmd1->GetGridsize().Z(), 0, GL_ALPHA,
                GL_FLOAT, (float*)(cmd0->GetPointDataByIdx(0, 0)));
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_3D, 0);
    }

    // Render field vectors using arrow glyphs
    this->renderFieldArrows(cmd0);

    // Render critical points using sphere glyphs
//    if (!this->renderCritPointsSpheres()) {
//        return false;
//    }

//    // Render streamlines in region of interest
//    if(!this->renderStreamlinesRoi(cmd0, cmd1)) return false;

    // Render streamlines based on manually set seed point
    if (!this->renderStreamlineBundleManual()) {
        return false;
    }

    glPopMatrix();

    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glDisable(GL_TEXTURE_3D);

    // Check for opengl error
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::Render: glError %s\n",
                this->ClassName(),
                gluErrorString(err));
        return false;
    }

    return true;
}


/*
 * ComparativeFieldTopologyRenderer::calcElectrostaticField
 */
void ComparativeFieldTopologyRenderer::calcElectrostaticField(protein_calls::VTIDataCall *cmd,
        VecField3f &efield, VecField3f &egradfield) {

    float *efieldBuff = new float[cmd->GetGridsize().X()*
                                  cmd->GetGridsize().Y()*
                                  cmd->GetGridsize().Z()*3];
#pragma omp parallel for
     for(int z = 0; z < cmd->GetGridsize().Z(); z++) {
         for(int y = 0; y < cmd->GetGridsize().Y(); y++) {
             for(int x = 0; x < cmd->GetGridsize().X(); x++) {

                 unsigned int cnt = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x;

                 // x component
                 if(x == 0) { // Forward differences
                     efieldBuff[3*cnt+0] = (this->GetPotentialAt(cmd, x+1, y, z)-this->GetPotentialAt(cmd, x, y, z))/cmd->GetSpacing().X();
                 }
                 else if (x == cmd->GetGridsize().X()-1) { // Backward differences
                     efieldBuff[3*cnt+0] = (this->GetPotentialAt(cmd, x, y, z)-this->GetPotentialAt(cmd, x-1, y, z))/cmd->GetSpacing().X();
                 }
                 else { // Central differences
                     efieldBuff[3*cnt+0] = (this->GetPotentialAt(cmd, x+1, y, z)-this->GetPotentialAt(cmd, x-1, y, z))/(2.0f*cmd->GetSpacing().X());
                 }

                 // y component
                 if(y == 0) { // Forward differences
                     efieldBuff[3*cnt+1] = (this->GetPotentialAt(cmd, x, y+1, z)-this->GetPotentialAt(cmd, x, y, z))/cmd->GetSpacing().Y();
                 }
                 else if (y == cmd->GetGridsize().Y()-1) { // Backward differences
                     efieldBuff[3*cnt+1] = (this->GetPotentialAt(cmd, x, y, z)-this->GetPotentialAt(cmd, x, y-1, z))/cmd->GetSpacing().Y();
                 }
                 else { // Central differences
                     efieldBuff[3*cnt+1] = (this->GetPotentialAt(cmd, x, y+1, z)-this->GetPotentialAt(cmd, x, y-1, z))/(2.0f*cmd->GetSpacing().Y());
                 }

                 // z component
                 if(z == 0) { // Forward differences
                     efieldBuff[3*cnt+2] = (this->GetPotentialAt(cmd, x, y, z+1)-this->GetPotentialAt(cmd, x, y, z))/cmd->GetSpacing().Z();
                 }
                 else if (z == cmd->GetGridsize().Z()-1) { // Backward differences
                     efieldBuff[3*cnt+2] = (this->GetPotentialAt(cmd, x, y, z)-this->GetPotentialAt(cmd, x, y, z-1))/cmd->GetSpacing().Z();
                 }
                 else { // Central differences
                     efieldBuff[3*cnt+2] = (this->GetPotentialAt(cmd, x, y, z+1)-this->GetPotentialAt(cmd, x, y, z-1))/(2.0f*cmd->GetSpacing().Z());
                 }

                 //printf("Grad1 %f %f %f\n", efieldBuff[3*cnt+0],
                 //        efieldBuff[3*cnt+1], efieldBuff[3*cnt+2]);

                 // Normalize the field vector
                 /*float mag = sqrt(efieldBuff[3*cnt+0]*efieldBuff[3*cnt+0] +
                                  efieldBuff[3*cnt+1]*efieldBuff[3*cnt+1] +
                                  efieldBuff[3*cnt+2]*efieldBuff[3*cnt+2]);
                 if(mag > 0.0f) {
                     efieldBuff[3*cnt+0] /= mag;
                     efieldBuff[3*cnt+1] /= mag;
                     efieldBuff[3*cnt+2] /= mag;
                 }*/

                 /*if(!((x == 0)||(x == cmd->GetGridsize().X()-1)||
                         (y == 0)||(y == cmd->GetGridsize().Y()-1)||
                         (z == 0)||(z == cmd->GetGridsize().Z()-1))) {
                     if((z >= 9)&&(z <= 11)) {
                         printf("------------------------------------------\n");
                         printf("potential: x(%f %f %f) y(%f %f %f) z(%f %f %f)\n",
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x+1],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x-1],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y+1)+x],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y-1)+x],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*(z+1)+y)+x],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x],
                                 cmd->GetPotential()[cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*(z-1)+y)+x]
                         );

                         printf("(%u %u %u) (%f %f %f)\n", x, y, z,
                                 efieldBuff[3*cnt+0],
                                 efieldBuff[3*cnt+1],
                                 efieldBuff[3*cnt+2]);
                     }
                 }*/
             }
         }
     }

     float *efieldBuffTmp = new float[cmd->GetGridsize().X()*
                                   cmd->GetGridsize().Y()*
                                   cmd->GetGridsize().Z()*3];

     memcpy(efieldBuffTmp, efieldBuff, cmd->GetGridsize().X()*
                                   cmd->GetGridsize().Y()*
                                   cmd->GetGridsize().Z()*3*sizeof(float));

#pragma omp parallel for
     for(int z = 0; z < cmd->GetGridsize().Z(); z++) {
         for(int y = 0; y < cmd->GetGridsize().Y(); y++) {
             for(int x = 0; x < cmd->GetGridsize().X(); x++) {

                 unsigned int cnt = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x;

                 // X component
                 if(myisnan(efieldBuff[3*cnt+0])||myisinf(efieldBuff[3*cnt+0])) {
                     unsigned int left = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x-1;
                     unsigned int middle = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x;
                     unsigned int right = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x+1;
                     // Use forward/backward differences or set to zero if sink/source is hit
                     if(!(myisinf(cmd->GetPointDataByIdx(0, 0)[left]))) {
                         efieldBuffTmp[3*middle+0] =
                                 ((float*)(cmd->GetPointDataByIdx(0, 0))[middle]-
                                 (float*)(cmd->GetPointDataByIdx(0, 0))[left])/cmd->GetSpacing().X();
                     }
                     else if(!(myisinf(cmd->GetPointDataByIdx(0, 0)[right]))) {
                         efieldBuffTmp[3*middle+0] = ((float*)(cmd->GetPointDataByIdx(0, 0))[right]-
                                 (float*)(cmd->GetPointDataByIdx(0, 0)))/cmd->GetSpacing().X();
                     }
                     else {
                         efieldBuffTmp[3*middle+0] = 0.0f;
                     }
                 }

                 // Y component
                 if(myisnan(efieldBuff[3*cnt+1])||myisinf(efieldBuff[3*cnt+1])) {
                     unsigned int down = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y-1)+x;
                     unsigned int middle = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x;
                     unsigned int up = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y+1)+x;
                     // Use forward/backward differences or set to zero if sink/source is hit
                     if(!(myisinf(cmd->GetPointDataByIdx(0, 0)[down]))) {
                         efieldBuffTmp[3*middle+1] = (cmd->GetPointDataByIdx(0, 0)[middle]-
                                 cmd->GetPointDataByIdx(0, 0)[down])/cmd->GetSpacing().Y();
                     }
                     else if(!(myisinf(cmd->GetPointDataByIdx(0, 0)[up]))) {
                         efieldBuffTmp[3*middle+1] = (cmd->GetPointDataByIdx(0, 0)[up]-
                                 cmd->GetPointDataByIdx(0, 0)[middle])/cmd->GetSpacing().Y();
                     }
                     else {
                         efieldBuffTmp[3*middle+1] = 0.0f;
                     }
                 }

                 // Z component
                 if(myisnan(efieldBuff[3*cnt+2])||myisinf(efieldBuff[3*cnt+2])) {
                     unsigned int back = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*(z-1)+y)+x;
                     unsigned int middle = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x;
                     unsigned int front = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*(z+1)+y)+x;
                     // Use forward/backward differences or set to zero if sink/source is hit
                     if(!(myisinf(cmd->GetPointDataByIdx(0, 0)[back]))) {
                         efieldBuffTmp[3*middle+2] =
                                 (cmd->GetPointDataByIdx(0, 0)[middle]-
                                  cmd->GetPointDataByIdx(0, 0)[back])/cmd->GetSpacing().Z();
                     }
                     else if(!(myisinf(cmd->GetPointDataByIdx(0, 0)[front]))) {
                         efieldBuffTmp[3*middle+2] =
                                 (cmd->GetPointDataByIdx(0, 0)[front]-
                                  cmd->GetPointDataByIdx(0, 0)[middle])/cmd->GetSpacing().Z();
                     }
                     else {
                         efieldBuffTmp[3*middle+2] = 0.0f;
                     }
                 }

                 /*printf("(%u %u %u) (%f %f %f)\n", x, y, z,
                         efieldBuffTmp[3*cnt+0],
                         efieldBuffTmp[3*cnt+1],
                         efieldBuffTmp[3*cnt+2]);*/

                 // Normalize the field vector
                 /*float mag = sqrt(efieldBuffTmp[3*cnt+0]*efieldBuffTmp[3*cnt+0] +
                                  efieldBuffTmp[3*cnt+1]*efieldBuffTmp[3*cnt+1] +
                                  efieldBuffTmp[3*cnt+2]*efieldBuffTmp[3*cnt+2]);
                 if(mag > 0.0f) {
                     efieldBuffTmp[3*cnt+0] /= mag;
                     efieldBuffTmp[3*cnt+1] /= mag;
                     efieldBuffTmp[3*cnt+2] /= mag;
                 }*/
             }
         }
     }


     /*for(int i = 0; i <  cmd->GetGridsize().X()*
     cmd->GetGridsize().Y()*
     cmd->GetGridsize().Z(); i++) {
         efieldBuffTmp[3*i+0] = 3*i+0;
         efieldBuffTmp[3*i+1] = 3*i+1;
         efieldBuffTmp[3*i+2] = 3*i+2;
     }*/

     // Init vector field (makes deep copy of the data)
     efield.SetData(efieldBuffTmp,
             cmd->GetGridsize().X(), cmd->GetGridsize().Y(), cmd->GetGridsize().Z(),
             cmd->GetSpacing().X(), cmd->GetSpacing().Y(), cmd->GetSpacing().Z(),
             cmd->GetOrigin().X(), cmd->GetOrigin().Y(), cmd->GetOrigin().Z());

     //printf("Gridsize %u %u %u\n", cmd->GetGridsize().X(), cmd->GetGridsize().Y(), cmd->GetGridsize().Z());
     //printf("Spacing %f %f %f\n", cmd->GetSpacing().X(), cmd->GetSpacing().Y(), cmd->GetSpacing().Z());
     //printf("Origin %f %f %f\n", cmd->GetOrigin().X(), cmd->GetOrigin().Y(), cmd->GetOrigin().Z());

     // DEBUG
     /*for(int x = 0; x < cmd->GetGridsize().X(); x++) {
         for(int y = 0; y < cmd->GetGridsize().Y(); y++) {
             for(int z = 0; z < cmd->GetGridsize().Z(); z++) {
                 printf("EField: (%f %f %f)\n", this->efield.GetAt(x,y,z).X(),
                         this->efield.GetAt(x,y,z).Y(),this->efield.GetAt(x,y,z).Z());
             }
         }
     }*/


     // Compute the gradient of the electric field

     float *efieldGradBuff = new float[cmd->GetGridsize().X()*
                                   cmd->GetGridsize().Y()*
                                   cmd->GetGridsize().Z()*3];
#pragma omp parallel for
     for(int z = 0; z < cmd->GetGridsize().Z(); z++) {
         for(int y = 0; y < cmd->GetGridsize().Y(); y++) {
             for(int x = 0; x < cmd->GetGridsize().X(); x++) {

                 unsigned int cnt = cmd->GetGridsize().X()*(cmd->GetGridsize().Y()*z+y)+x;

                 // x component
                 if(x == 0) { // Forward differences
                     efieldGradBuff[3*cnt+0] = (efield.GetAt(x+1, y, z).Norm()-efield.GetAt(x, y, z).Norm())/cmd->GetSpacing().X();
                 }
                 else if (x == cmd->GetGridsize().X()-1) { // Backward differences
                     efieldGradBuff[3*cnt+0] = (efield.GetAt(x, y, z).Norm()-efield.GetAt(x-1, y, z).Norm())/cmd->GetSpacing().X();
                 }
                 else { // Central differences
                     efieldGradBuff[3*cnt+0] = (efield.GetAt(x+1, y, z).Norm()-efield.GetAt(x-1, y, z).Norm())/(2.0f*cmd->GetSpacing().X());
                 }

                 // y component
                 if(y == 0) { // Forward differences
                     efieldGradBuff[3*cnt+1] = (efield.GetAt(x, y+1, z).Norm()-efield.GetAt(x, y, z).Norm())/cmd->GetSpacing().Y();
                 }
                 else if (y == cmd->GetGridsize().Y()-1) { // Backward differences
                     efieldGradBuff[3*cnt+1] = (efield.GetAt(x, y, z).Norm()-efield.GetAt(x, y-1, z).Norm())/cmd->GetSpacing().Y();
                 }
                 else { // Central differences
                     efieldGradBuff[3*cnt+1] = (efield.GetAt(x, y+1, z).Norm()-efield.GetAt(x, y-1, z).Norm())/(2.0f*cmd->GetSpacing().Y());
                 }

                 // z component
                 if(z == 0) { // Forward differences
                     efieldGradBuff[3*cnt+2] = (efield.GetAt(x, y, z+1).Norm()-efield.GetAt(x, y, z).Norm())/cmd->GetSpacing().Z();
                 }
                 else if (z == cmd->GetGridsize().Z()-1) { // Backward differences
                     efieldGradBuff[3*cnt+2] = (efield.GetAt(x, y, z).Norm()-efield.GetAt(x, y, z-1).Norm())/cmd->GetSpacing().Z();
                 }
                 else { // Central differences
                     efieldGradBuff[3*cnt+2] = (efield.GetAt(x, y, z+1).Norm()-efield.GetAt(x, y, z-1).Norm())/(2.0f*cmd->GetSpacing().Z());
                 }


                 // Normalize the field vector
                 /*float mag = sqrt(efieldGradBuff[3*cnt+0]*efieldGradBuff[3*cnt+0] +
                         efieldGradBuff[3*cnt+1]*efieldGradBuff[3*cnt+1] +
                         efieldGradBuff[3*cnt+2]*efieldGradBuff[3*cnt+2]);
                 if(mag > 0.0f) {
                     efieldGradBuff[3*cnt+0] /= mag;
                     efieldGradBuff[3*cnt+1] /= mag;
                     efieldGradBuff[3*cnt+2] /= mag;
                 }*/
             }
         }
     }

     // Init vector field (makes deep copy of the data)
     egradfield.SetData(efieldGradBuff,
             cmd->GetGridsize().X(), cmd->GetGridsize().Y(), cmd->GetGridsize().Z(),
             cmd->GetSpacing().X(), cmd->GetSpacing().Y(), cmd->GetSpacing().Z(),
             cmd->GetOrigin().X(), cmd->GetOrigin().Y(), cmd->GetOrigin().Z());

     delete[] efieldBuff;
     delete[] efieldBuffTmp;
     delete[] efieldGradBuff;
}


/*
 * ComparativeFieldTopologyRenderer::renderCritPointsSpheres
 */
bool ComparativeFieldTopologyRenderer::renderCritPointsSpheres() {
    using namespace vislib;
    using namespace vislib::math;
    using namespace vislib::sys;

    // Render spheres for all critical points found
    vislib::Array<float> spherePos, sphereCol;

    // DEBUG Streamline seed point
    if(this->toggleStreamlinesManual) {
        spherePos.Add(this->streamBundleSeedX);
        spherePos.Add(this->streamBundleSeedY);
        spherePos.Add(this->streamBundleSeedZ);
        spherePos.Add(this->critPointsSphereScl/100.0f);
        sphereCol.Add(0.0f);
        sphereCol.Add(1.0f);
        sphereCol.Add(0.0f);
    }
    // DEBUG

    for(int cnt = 0; cnt < static_cast<int>(this->efield0.GetCritPointCount()); cnt++) {

        if((this->neighbours0[cnt].Count() == 0)||
                this->critPointsShowAllSlot.Param<core::param::BoolParam>()->Value()) {
            /*Vector<unsigned int, 3> cellId = this->efield0.GetCritPoint(cnt).GetCellId();
            printf("Streamlines ending in cell # %u: %u\n", cnt,
                    this->cellEndpoints1[cmd0->GetGridsize().X()*(cmd0->GetGridsize().Y()*cellId.Z()+cellId.Y())+cellId.X()].Count());*/

        // Sphere color
        if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SOURCE) {
            // Sphere positions
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(0.0f);
            sphereCol.Add(0.0f);
            sphereCol.Add(1.0f);
            //spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/100000.0f*abs(cmd0->GetPotentialAtWSTrilinear(this->efield0.GetCritPoint(cnt).GetPos())));
            spherePos.Add(this->critPointsSphereScl/100.0f);
        }
        else if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SINK) {
            sphereCol.Add(1.0f);
            sphereCol.Add(0.0f);
            sphereCol.Add(0.0f);
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            //spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/100000.0f*cmd0->GetPotentialAtWSTrilinear(this->efield0.GetCritPoint(cnt).GetPos()));
            spherePos.Add(this->critPointsSphereScl/100.0f);
        }
 /*       else if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::REPELLING_SADDLE) {
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(0.5f);
            sphereCol.Add(0.7f);
            sphereCol.Add(1.0f);
            spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/150.0f);
        }
        else if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(1.0f);
            sphereCol.Add(0.6f);
            sphereCol.Add(0.7f);
            spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/150.0f);
        }
        else { // Type is unknown
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(1.0f);
            sphereCol.Add(1.0f);
            sphereCol.Add(1.0f);
            spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/150.0f);
        }*/
        }
    }



    for(int cnt = 0; cnt < static_cast<int>(this->efield1.GetCritPointCount()); cnt++) {

        if((this->neighbours1[cnt].Count() == 0)||
                this->critPointsShowAllSlot.Param<core::param::BoolParam>()->Value()) {



        // Sphere color
        if(this->efield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SOURCE) {

            //printf("source\n");

            // Sphere positions
            spherePos.Add(this->efield1.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield1.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield1.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(0.5f);
            sphereCol.Add(0.7f);
            sphereCol.Add(1.0f);
            //spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/100000.0f*abs(cmd1->GetPotentialAtWSTrilinear(this->efield1.GetCritPoint(cnt).GetPos())));
            spherePos.Add(this->critPointsSphereScl/100.0f);
        }
        else if(this->efield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SINK) {

            //printf("sink\n");
            sphereCol.Add(1.0f);
            sphereCol.Add(0.6f);
            sphereCol.Add(0.7f);
            spherePos.Add(this->efield1.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield1.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield1.GetCritPoint(cnt).GetPos().Z());
            //spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/100000.0f*cmd1->GetPotentialAtWSTrilinear(this->efield1.GetCritPoint(cnt).GetPos()));
            spherePos.Add(this->critPointsSphereScl/100.0f);
        }
 /*       else if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::REPELLING_SADDLE) {
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(0.5f);
            sphereCol.Add(0.7f);
            sphereCol.Add(1.0f);
            spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/150.0f);
        }
        else if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(1.0f);
            sphereCol.Add(0.6f);
            sphereCol.Add(0.7f);
            spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/150.0f);
        }
        else { // Type is unknown
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->efield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(1.0f);
            sphereCol.Add(1.0f);
            sphereCol.Add(1.0f);
            spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/150.0f);
        }*/
        }
    }


    for(int cnt = 0; cnt < static_cast<int>(this->egradfield0.GetCritPointCount()); cnt++) {

        /*if(this->egradfield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {
            // Sphere positions
            spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(1.0f);
            sphereCol.Add(0.0f);
            sphereCol.Add(1.0f);
            spherePos.Add(this->critPointsSphereScl/250.0f);
        }*/

        /*if(this->egradfield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::REPELLING_SADDLE) {
             // Sphere positions
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
             sphereCol.Add(0.0f);
             sphereCol.Add(1.0f);
             sphereCol.Add(0.0f);
             spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/250.0f);
         }
         else if(this->egradfield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {
             // Sphere positions
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
             sphereCol.Add(1.0f);
             sphereCol.Add(0.0f);
             sphereCol.Add(1.0f);
             spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/250.0f);
         }
         else {
             // Sphere positions
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
             sphereCol.Add(1.0f);
             sphereCol.Add(1.0f);
             sphereCol.Add(1.0f);
             spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/250.0f);
         }*/
    }


    for(int cnt = 0; cnt < static_cast<int>(this->egradfield1.GetCritPointCount()); cnt++) {

        /*if(this->egradfield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {
            // Sphere positions
            spherePos.Add(this->egradfield1.GetCritPoint(cnt).GetPos().X());
            spherePos.Add(this->egradfield1.GetCritPoint(cnt).GetPos().Y());
            spherePos.Add(this->egradfield1.GetCritPoint(cnt).GetPos().Z());
            sphereCol.Add(0.0f);
            sphereCol.Add(1.0f);
            sphereCol.Add(1.0f);
            spherePos.Add(this->critPointsSphereScl/250.0f);
        }*/

        /*if(this->egradfield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::REPELLING_SADDLE) {
             // Sphere positions
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
             sphereCol.Add(0.0f);
             sphereCol.Add(1.0f);
             sphereCol.Add(0.0f);
             spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/250.0f);
         }
         else if(this->egradfield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {
             // Sphere positions
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
             sphereCol.Add(1.0f);
             sphereCol.Add(0.0f);
             sphereCol.Add(1.0f);
             spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/250.0f);
         }
         else {
             // Sphere positions
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
             spherePos.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
             sphereCol.Add(1.0f);
             sphereCol.Add(1.0f);
             sphereCol.Add(1.0f);
             spherePos.Add(this->sphereRadSclParam.Param<core::param::FloatParam>()->Value()/250.0f);
         }*/
    }

    // Enable sphere shader
    this->sphereShader.Enable();

    // Set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, this->viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
    glUniformMatrix4fvARB(this->sphereShader.ParameterLocation("modelview"), 1, false, this->modelMatrix);
    glUniformMatrix4fvARB(this->sphereShader.ParameterLocation("proj"), 1, false, this->projMatrix);
    glUniform4fvARB(this->sphereShader.ParameterLocation("lightPos"), 1, this->lightPos);
    glUniform3fARB(this->sphereShader.ParameterLocation( "zValues"), this->fogZ,
            this->cameraInfo->NearClip(), this->cameraInfo->FarClip());

    // Vertex attributes
    GLint vertexPos = glGetAttribLocation(this->sphereShader, "vertex");
    GLint vertexColor = glGetAttribLocation(this->sphereShader, "color");

    // Enable arrays for attributes
    glEnableVertexAttribArray(vertexPos);
    glEnableVertexAttribArray(vertexColor);

    // Set attribute pointers
    glVertexAttribPointer(vertexPos, 4, GL_FLOAT, GL_FALSE, 0, spherePos.PeekElements());
    glVertexAttribPointer(vertexColor, 3, GL_FLOAT, GL_FALSE, 0, sphereCol.PeekElements());

    // Draw points
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(spherePos.Count())/4);

    // Disable arrays for attributes
    glDisableVertexAttribArray(vertexPos);
    glDisableVertexAttribArray(vertexColor);

    // Disable sphere shader
    this->sphereShader.Disable();

    // Check for opengl error
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s::Render: glError %s\n",
                this->ClassName(),
                gluErrorString(err));
        return false;
    }

    return true;
}


/*
 * ComparativeFieldTopologyRenderer::renderFieldArrows
 */
bool ComparativeFieldTopologyRenderer::renderFieldArrows(const protein_calls::VTIDataCall *cmd) {
    using namespace vislib;
    using namespace vislib::math;
    using namespace vislib::sys;

    int arrowCnt = static_cast<int>(cmd->GetGridsize().X()*cmd->GetGridsize().Y()*cmd->GetGridsize().Z());

    if(this->recalcArrowData) {

        this->arrowData.SetCount(arrowCnt*4);
        this->arrowDataPos.SetCount(arrowCnt*3);
        this->arrowCol.SetCount(arrowCnt*3);
        this->gridPos.SetCount(arrowCnt*3);

#pragma omp parallel for
        for(int cnt = 0; cnt < arrowCnt; cnt++) {
            this->gridPos[cnt][0] = cmd->GetOrigin().X() + (cnt%cmd->GetGridsize().X())*cmd->GetSpacing().X();
            this->gridPos[cnt][1] = cmd->GetOrigin().Y() + ((cnt/cmd->GetGridsize().X())%cmd->GetGridsize().Y())*cmd->GetSpacing().Y();
            this->gridPos[cnt][2] = cmd->GetOrigin().Z() + ((cnt/cmd->GetGridsize().X())/cmd->GetGridsize().Y())*cmd->GetSpacing().Z();
        }

 #pragma omp parallel for
         for(int cnt = 0; cnt < arrowCnt; cnt++) {

             Vector<float, 3> e(this->efield0.PeekBuff()[3*cnt+0],
                                this->efield0.PeekBuff()[3*cnt+1],
                                this->efield0.PeekBuff()[3*cnt+2]);

             e.Normalise();

             // Arrow color
             this->arrowCol[cnt*3+0] = 1.0f;
             this->arrowCol[cnt*3+1] = 1.0f;
             this->arrowCol[cnt*3+2] = 1.0f;

             // Setup vector
             this->arrowData[4*cnt+0] = this->gridPos[cnt][0] + e.X()*this->arrowLenScl*0.5f/100.0f;
             this->arrowData[4*cnt+1] = this->gridPos[cnt][1] + e.Y()*this->arrowLenScl*0.5f/100.0f;
             this->arrowData[4*cnt+2] = this->gridPos[cnt][2] + e.Z()*this->arrowLenScl*0.5f/100.0f;
             this->arrowData[4*cnt+3] = 1.0f;

             // Set position of the arrow
             this->arrowDataPos[3*cnt+0] = this->gridPos[cnt][0] - e.X()*this->arrowLenScl*0.5f/100.0f;
             this->arrowDataPos[3*cnt+1] = this->gridPos[cnt][1] - e.Y()*this->arrowLenScl*0.5f/100.0f;
             this->arrowDataPos[3*cnt+2] = this->gridPos[cnt][2] - e.Z()*this->arrowLenScl*0.5f/100.0f;
         }


         // Write idx array
         this->arrowVisIdx.Clear();
         this->arrowVisIdx.SetCapacityIncrement(1000);
         for(int cnt = 0; cnt < arrowCnt; cnt++) {

             if((this->gridPos[cnt][0] > this->arrowFilterXMax)||
                (this->gridPos[cnt][0] < this->arrowFilterXMin))
                 continue;

             if((this->gridPos[cnt][1] > this->arrowFilterYMax)||
                (this->gridPos[cnt][1] < this->arrowFilterYMin))
                 continue;

             if((this->gridPos[cnt][2] > this->arrowFilterZMax)||
                (this->gridPos[cnt][2] < this->arrowFilterZMin))
                 continue;

             this->arrowVisIdx.Add(static_cast<int>(cnt));
         }

         this->recalcArrowData = false;
     }


     // Actual rendering

     this->arrowShader.Enable();

     glUniform4fvARB(this->arrowShader.ParameterLocation("viewAttr"), 1, viewportStuff);
     glUniform3fvARB(this->arrowShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
     glUniform3fvARB(this->arrowShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
     glUniform3fvARB(this->arrowShader.ParameterLocation("camUp" ), 1, cameraInfo->Up().PeekComponents());
     glUniform1fARB(this->arrowShader.ParameterLocation("radScale"), this->arrowRadScl/100.0f);
     glUniformMatrix4fvARB(this->arrowShader.ParameterLocation("modelview"), 1, false, this->modelMatrix);
     glUniformMatrix4fvARB(this->arrowShader.ParameterLocation("proj"), 1, false, this->projMatrix);
     glUniform4fvARB(this->arrowShader.ParameterLocation("lightPos"), 1, this->lightPos);

     // Get attribute locations
     GLint attribPos0 = glGetAttribLocationARB(this->arrowShader, "pos0");
     GLint attribPos1 = glGetAttribLocationARB(this->arrowShader, "pos1");
     GLint attribColor = glGetAttribLocationARB(this->arrowShader, "color");

     // Enable arrays for attributes
     glEnableVertexAttribArrayARB(attribPos0);
     glEnableVertexAttribArrayARB(attribPos1);
     glEnableVertexAttribArrayARB(attribColor);

     // Set attribute pointers
     glVertexAttribPointerARB(attribPos0, 4, GL_FLOAT, GL_FALSE, 0, this->arrowData.PeekElements());
     glVertexAttribPointerARB(attribPos1, 3, GL_FLOAT, GL_FALSE, 0, this->arrowDataPos.PeekElements());
     glVertexAttribPointerARB(attribColor, 3, GL_FLOAT, GL_FALSE, 0, this->arrowCol.PeekElements());

     // Draw points
     glDrawElements(GL_POINTS, static_cast<GLsizei>(arrowVisIdx.Count()),
             GL_UNSIGNED_INT, arrowVisIdx.PeekElements());

     // Disable arrays for attributes
     glDisableVertexAttribArrayARB(attribPos0);
     glDisableVertexAttribArrayARB(attribPos1);
     glDisableVertexAttribArrayARB(attribColor);

     this->arrowShader.Disable();

     // Check for opengl error
     GLenum err = glGetError();
     if(err != GL_NO_ERROR) {
         Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                 "%s::Render: glError %s\n",
                 this->ClassName(),
                 gluErrorString(err));
         return false;
     }

     return true;
}


/*
 * ComparativeFieldTopologyRenderer::findNeighbours
 */
void ComparativeFieldTopologyRenderer::findNeighbours() {

    using namespace vislib;
    using namespace vislib::math;


    this->neighbours0.SetCount(this->efield0.GetCritPointCount());
    this->neighbours1.SetCount(this->efield1.GetCritPointCount());
    for(unsigned int cnt = 0; cnt < this->neighbours0.Count(); cnt++) {
        this->neighbours0[cnt].Clear();
    }
    for(unsigned int cnt = 0; cnt < this->neighbours1.Count(); cnt++) {
        this->neighbours1[cnt].Clear();
    }

    for(unsigned int cnt = 0; cnt < this->neighbours0.Count(); cnt++) {
        Vector<float, 3> pos0 = this->efield0.GetCritPoint(cnt).GetPos();
        VecField3f::CritPoint::Type t0 = this->efield0.GetCritPoint(cnt).GetType();
        for(unsigned int c = 0; c < this->efield1.GetCritPointCount(); c++) {
            if(this->efield1.GetCritPoint(c).GetType() == t0) {
                if((this->efield1.GetCritPoint(c).GetPos()-pos0).Norm() <= this->roiMaxDist) {
                    this->neighbours0[cnt].Add(&this->efield1.GetCritPoint(c));
                    this->neighbours1[c].Add(&this->efield0.GetCritPoint(cnt));
                }
            }
        }
    }
}


/*
 * ComparativeFieldTopologyRenderer::renderStreamlineBundleManual
 */
bool ComparativeFieldTopologyRenderer::renderStreamlineBundleManual() {
    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    // (Re-)calculate streamlines if necessary
    if(this->recalcStreamlinesManualSeed) {

        time_t t = clock(); // DEBUG

        this->streamlinesManualSeed0.Clear();
        Vector<float, 3> seedPt = Vector<float, 3>(
                this->streamBundleSeedX, this->streamBundleSeedY, this->streamBundleSeedZ);

        Vector<unsigned int, 3> cellSeed = this->efield0.GetCellId(seedPt);
        if(this->efield0.IsValidGridpos(seedPt)) {

            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Integrating streamlines using RK4 (manual seed point) ...",
                    this->ClassName()); // DEBUG

            // Compute bundle of streamlines around the manually set seed point
            float rad = this->streamBundleRadManual;
            unsigned int res = this->streamBundleResManual;
            float step = rad / static_cast<float>(res);

            Vector<float, 3> tangent = this->efield0.GetAtTrilin(seedPt.X(), seedPt.Y(), seedPt.Z());
            tangent.Normalise();
            Vector<float, 3> normal(-tangent.Y()/tangent.X(), 1.0f, 0.0f);
            normal.Normalise();
            float alpha = this->streamBundlePhiManual; // Rotation angle
            Matrix<float, 3, COLUMN_MAJOR> rotMat =  Matrix<float, 3, COLUMN_MAJOR>(
                    // Row #1
                    tangent.X()*tangent.X()*(1.0f - cos(alpha))+cos(alpha),
                    tangent.X()*tangent.Y()*(1.0f - cos(alpha))-tangent.Z()*sin(alpha),
                    tangent.X()*tangent.Z()*(1.0f - cos(alpha))+tangent.Y()*sin(alpha),
                    // Row #2
                    tangent.Y()*tangent.X()*(1.0f - cos(alpha))+tangent.Z()*sin(alpha),
                    tangent.Y()*tangent.Y()*(1.0f - cos(alpha))+cos(alpha),
                    tangent.Y()*tangent.Z()*(1.0f - cos(alpha))-tangent.X()*sin(alpha),
                    // Row #3
                    tangent.Z()*tangent.X()*(1.0f - cos(alpha))-tangent.Y()*sin(alpha),
                    tangent.Z()*tangent.Y()*(1.0f - cos(alpha))+tangent.X()*sin(alpha),
                    tangent.Z()*tangent.Z()*(1.0f - cos(alpha))+cos(alpha));

            for(float s = step; s <= rad; s += step) {
                for(float a = 0; a < 2*M_PI; a += alpha) {
                    normal = rotMat*normal;

                    this->streamlinesManualSeed0.Add(new Streamline());
                    this->streamlinesManualSeed0.Last()->IntegrateRK4(
                            seedPt+normal*s,      // Starting point
                            this->efield0,                                                // Vector field
                            this->streamlineMaxStepsManual, // Maximum length
                            this->streamlineStepsizeManual/10.0f, // Step size
                            this->streamlineEpsManual,        // Epsilon
                            Streamline::BIDIRECTIONAL);
                }
            }
            // Center
            this->streamlinesManualSeed0.Add(new Streamline());
            this->streamlinesManualSeed0.Last()->IntegrateRK4(
                    seedPt,      // Starting point
                    this->efield0,                                                // Vector field
                    this->streamlineMaxStepsManual, // Maximum length
                    this->streamlineStepsizeManual/10.0f, // Step size
                    this->streamlineEpsManual,        // Epsilon
                    Streamline::BIDIRECTIONAL);

            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #0 done, number of streamlines %u (%f s)",
                    this->ClassName(),
                    this->streamlinesManualSeed0.Count(),
                    (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
            t = clock(); // DEBUG

            this->streamlinesManualSeed1.Clear();
            tangent = this->efield1.GetAtTrilin(seedPt.X(), seedPt.Y(), seedPt.Z());
            tangent.Normalise();
            normal = Vector<float, 3>(-tangent.Y()/tangent.X(), 1.0f, 0.0f);
            normal.Normalise();
            rotMat =  Matrix<float, 3, COLUMN_MAJOR>(
                    // Row #1
                    tangent.X()*tangent.X()*(1.0f - cos(alpha))+cos(alpha),
                    tangent.X()*tangent.Y()*(1.0f - cos(alpha))-tangent.Z()*sin(alpha),
                    tangent.X()*tangent.Z()*(1.0f - cos(alpha))+tangent.Y()*sin(alpha),
                    // Row #2
                    tangent.Y()*tangent.X()*(1.0f - cos(alpha))+tangent.Z()*sin(alpha),
                    tangent.Y()*tangent.Y()*(1.0f - cos(alpha))+cos(alpha),
                    tangent.Y()*tangent.Z()*(1.0f - cos(alpha))-tangent.X()*sin(alpha),
                    // Row #3
                    tangent.Z()*tangent.X()*(1.0f - cos(alpha))-tangent.Y()*sin(alpha),
                    tangent.Z()*tangent.Y()*(1.0f - cos(alpha))+tangent.X()*sin(alpha),
                    tangent.Z()*tangent.Z()*(1.0f - cos(alpha))+cos(alpha));

            for(float s = step; s <= rad; s += step) {
                for(float a = 0; a < 2*M_PI; a += alpha) {
                    normal = rotMat*normal;

                    this->streamlinesManualSeed1.Add(new Streamline());
                    this->streamlinesManualSeed1.Last()->IntegrateRK4(
                            seedPt+normal*s,                      // Starting point
                            this->efield1,                        // Vector field
                            this->streamlineMaxStepsManual,       // Maximum length
                            this->streamlineStepsizeManual/10.0f, // Step size
                            this->streamlineEpsManual,            // Epsilon
                            Streamline::BIDIRECTIONAL);
                }
            }
            // Center
            this->streamlinesManualSeed1.Add(new Streamline());
            this->streamlinesManualSeed1.Last()->IntegrateRK4(
                    seedPt,                               // Starting point
                    this->efield1,                        // Vector field
                    this->streamlineMaxStepsManual,       // Maximum length
                    this->streamlineStepsizeManual/10.0f, // Step size
                    this->streamlineEpsManual,            // Epsilon
                    Streamline::BIDIRECTIONAL);

            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... #1 done, number of streamlines %u (%f s)",
                    this->ClassName(),
                    this->streamlinesManualSeed1.Count(),
                    (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG

            this->recalcStreamlinesManualSeed = false;
        }
    }

    // Render streamlines using GLSL shadersGLuint
    if(this->toggleStreamlinesManual) {

        this->streamlineShader.Enable();

        // Set shader variables
        glUniformMatrix4fvARB(this->streamlineShader.ParameterLocation("modelview"), 1, false, this->modelMatrix);
        glUniformMatrix4fvARB(this->streamlineShader.ParameterLocation("proj"), 1, false, this->projMatrix);
        glUniform4fvARB(this->streamlineShader.ParameterLocation("lightPos"), 1, this->lightPos);
        glUniform4fvARB(this->streamlineShader.ParameterLocation("viewAttr"), 1, this->viewportStuff);
        glUniform1iARB(this->streamlineShader.ParameterLocation("shading"), static_cast<int>(this->streamlineShadingManual));
        glUniform1fARB(this->streamlineShader.ParameterLocation("fogZ"), this->fogZ);
        glUniform1fARB(this->streamlineShader.ParameterLocation("minPot"), this->texMinVal);
        glUniform1fARB(this->streamlineShader.ParameterLocation("maxPot"), this->texMaxVal);
        glUniform1iARB(this->streamlineShader.ParameterLocation("potentialTex"), 0);

        // Vertex attributes
        GLuint vertexPos = glGetAttribLocationARB(this->streamlineShader, "vertex");
        GLuint vertexTangent = glGetAttribLocationARB(this->streamlineShader, "tangent");
        GLuint vertexTC = glGetAttribLocationARB(this->streamlineShader, "tc");

        // Enable arrays for attributes
        glEnableVertexAttribArrayARB(vertexPos);
        glEnableVertexAttribArrayARB(vertexTangent);
        glEnableVertexAttribArrayARB(vertexTC);

        // Vector field #0
        glUniform1iARB(this->streamlineShader.ParameterLocation("vecFieldIdx"), 0);
        glActiveTextureARB(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, this->potentialTex0);
        for (unsigned int cnt = 0; cnt < this->streamlinesManualSeed0.Count(); cnt++) {
            // Set attribute pointers
            glVertexAttribPointerARB(vertexPos, 3, GL_FLOAT, GL_FALSE, 0, this->streamlinesManualSeed0[cnt]->PeekVertexArr());
            glVertexAttribPointerARB(vertexTangent, 3, GL_FLOAT, GL_FALSE, 0, this->streamlinesManualSeed0[cnt]->PeekTangentArr());
            glVertexAttribPointerARB(vertexTC, 3, GL_FLOAT, GL_FALSE, 0, this->streamlinesManualSeed0[cnt]->PeekTexCoordArr());
            // Draw points
            glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(this->streamlinesManualSeed0[cnt]->GetLength()));
        }

        glUniform1iARB(this->streamlineShader.ParameterLocation("vecFieldIdx"), 1);
        glActiveTextureARB(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, this->potentialTex1);
        for (unsigned int cnt = 0; cnt < this->streamlinesManualSeed1.Count(); cnt++) {
            // Set attribute pointers
            glVertexAttribPointerARB(vertexPos, 3, GL_FLOAT, GL_FALSE, 0, this->streamlinesManualSeed1[cnt]->PeekVertexArr());
            glVertexAttribPointerARB(vertexTangent, 3, GL_FLOAT, GL_FALSE, 0, this->streamlinesManualSeed1[cnt]->PeekTangentArr());
            glVertexAttribPointerARB(vertexTC, 3, GL_FLOAT, GL_FALSE, 0, this->streamlinesManualSeed1[cnt]->PeekTexCoordArr());
            // Draw points
            glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(this->streamlinesManualSeed1[cnt]->GetLength()));
        }

        // Disable arrays for attributes
        glDisableVertexAttribArrayARB(vertexPos);
        glDisableVertexAttribArrayARB(vertexTangent);
        glDisableVertexAttribArrayARB(vertexTC);

        this->streamlineShader.Disable();

        // Check for opengl error
        GLenum err = glGetError();
        if(err != GL_NO_ERROR) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: glError in 'renderStreamlineBundleManual' %s \n",
                    this->ClassName(),
                    gluErrorString(err));
            return false;
        }
    }

    return true;
}


/*
 * ComparativeFieldTopologyRenderer::renderStreamlinesRoi
 */
bool ComparativeFieldTopologyRenderer::renderStreamlinesRoi(
        const protein_calls::VTIDataCall *cmd0,
        const protein_calls::VTIDataCall *cmd1) {
    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    // (Re-)calculate streamlines if necessary
    if(this->recalcStreamlines) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Searching for endpoints of streamline bundles ...",
                this->ClassName()); // DEBUG
        time_t t = clock(); // DEBUG
        this->seedPosStart0.Clear();
        this->seedPosStart0.SetCapacityIncrement(10000);
        for(int cnt = 0; cnt < static_cast<int>(this->egradfield0.GetCritPointCount()); cnt++) {
            if(this->egradfield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {

                // Compute bundle of streamlines around the seed point
                float rad = this->streamBundleRad;
                unsigned int res = this->streamBundleRes;
                float step = rad / static_cast<float>(res);
                Vector<float, 3> tangent = this->efield0.GetAtTrilin(this->egradfield0.GetCritPoint(cnt).GetPos());
                tangent.Normalise();
                Vector<float, 3> normal(-tangent.Y()/tangent.X(), 1.0f, 0.0f);
                normal.Normalise();
                Vector<float, 3> p;
                float alpha = this->streamBundlePhi; // Rotation angle
                Matrix<float, 3, COLUMN_MAJOR> rotMat =  Matrix<float, 3, COLUMN_MAJOR>(
                        // Row #1
                        tangent.X()*tangent.X()*(1.0f - cos(alpha))+cos(alpha),
                        tangent.X()*tangent.Y()*(1.0f - cos(alpha))-tangent.Z()*sin(alpha),
                        tangent.X()*tangent.Z()*(1.0f - cos(alpha))+tangent.Y()*sin(alpha),
                        // Row #2
                        tangent.Y()*tangent.X()*(1.0f - cos(alpha))+tangent.Z()*sin(alpha),
                        tangent.Y()*tangent.Y()*(1.0f - cos(alpha))+cos(alpha),
                        tangent.Y()*tangent.Z()*(1.0f - cos(alpha))-tangent.X()*sin(alpha),
                        // Row #3
                        tangent.Z()*tangent.X()*(1.0f - cos(alpha))-tangent.Y()*sin(alpha),
                        tangent.Z()*tangent.Y()*(1.0f - cos(alpha))+tangent.X()*sin(alpha),
                        tangent.Z()*tangent.Z()*(1.0f - cos(alpha))+cos(alpha));
                for(float s = step; s <= rad; s += step) {
                    for(float a = 0; a < 2*M_PI; a += alpha) {
                        normal = rotMat*normal;
                        p = this->egradfield0.GetCritPoint(cnt).GetPos()+normal*s;
                        if(this->efield0.IsValidGridpos(p)) {
                            this->seedPosStart0.Add(p.X());
                            this->seedPosStart0.Add(p.Y());
                            this->seedPosStart0.Add(p.Z());
                        }
                    }
                }

                // Add seedpoint
                 this->seedPosStart0.Add(this->egradfield0.GetCritPoint(cnt).GetPos().X());
                 this->seedPosStart0.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Y());
                 this->seedPosStart0.Add(this->egradfield0.GetCritPoint(cnt).GetPos().Z());
            }
        }

        // Compute streamline endpoints for field #0
        float *seedPointEnd0 = new float[this->seedPosStart0.Count()];
        memcpy(seedPointEnd0, this->seedPosStart0.PeekElements(), sizeof(float)*this->seedPosStart0.Count());


        SetGridParams(make_uint3(this->efield0.GetDim().X(), this->efield0.GetDim().Y(),this->efield0.GetDim().Z()),
                      make_float3(this->efield0.GetOrg().X(), this->efield0.GetOrg().Y(),this->efield0.GetOrg().Z()),
                      make_float3(this->bbox.GetRight(), this->bbox.GetTop(), this->bbox.GetFront()),
                      make_float3(this->efield0.GetSpacing().X(), this->efield0.GetSpacing().Y(),this->efield0.GetSpacing().Z()));
        SetStreamlineParams(this->streamlineStepsize/10.0f, this->streamlineMaxSteps);
        SetNumberOfPos(static_cast<uint>(this->seedPosStart0.Count())/3);

        // Compute positions using CUDA (integrating forward)
        UpdatePositionRK4(this->efield0.PeekBuff(),
                make_uint3(this->efield0.GetDim().X(),
                        this->efield0.GetDim().Y(),
                        this->efield0.GetDim().Z()),
                seedPointEnd0, static_cast<uint>(this->seedPosStart0.Count())/3,
                this->streamlineMaxSteps);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... CUDA #0 (forward) done (%f s)",
                this->ClassName(),
                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
        t = clock(); // DEBUG

        this->seedCellEnd0.SetCount(this->seedPosStart0.Count());
#pragma omp parallel for
        for(int cnt = 0; cnt < static_cast<int>(this->seedPosStart0.Count()/3); cnt++) {
            this->seedCellEnd0[3*cnt+0] = static_cast<unsigned int>((seedPointEnd0[3*cnt+0]-this->efield0.GetOrg().X())/this->efield0.GetSpacing().X());
            this->seedCellEnd0[3*cnt+1] = static_cast<unsigned int>((seedPointEnd0[3*cnt+1]-this->efield0.GetOrg().Y())/this->efield0.GetSpacing().Y());
            this->seedCellEnd0[3*cnt+2] = static_cast<unsigned int>((seedPointEnd0[3*cnt+2]-this->efield0.GetOrg().Z())/this->efield0.GetSpacing().Z());
        }

        // Compute positions using CUDA (integrating backward)
        float *seedPointEndBackward0 = new float[this->seedPosStart0.Count()];
        memcpy(seedPointEndBackward0, this->seedPosStart0.PeekElements(), sizeof(float)*this->seedPosStart0.Count());
        UpdatePositionRK4(this->efield0.PeekBuff(),
                make_uint3(this->efield0.GetDim().X(),
                        this->efield0.GetDim().Y(),
                        this->efield0.GetDim().Z()),
                seedPointEnd0,
                static_cast<uint>(this->seedPosStart0.Count())/3,
                this->streamlineMaxSteps,
                true);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... CUDA #0 (backward) done (%f s)",
                this->ClassName(),
                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
        t = clock(); // DEBUG

        this->seedCellEndBackward0.SetCount(this->seedPosStart0.Count());
#pragma omp parallel for
        for(int cnt = 0; cnt < static_cast<int>(this->seedPosStart0.Count()/3); cnt++) {
            this->seedCellEndBackward0[3*cnt+0] = static_cast<unsigned int>((seedPointEnd0[3*cnt+0]-this->efield0.GetOrg().X())/this->efield0.GetSpacing().X());
            this->seedCellEndBackward0[3*cnt+1] = static_cast<unsigned int>((seedPointEnd0[3*cnt+1]-this->efield0.GetOrg().Y())/this->efield0.GetSpacing().Y());
            this->seedCellEndBackward0[3*cnt+2] = static_cast<unsigned int>((seedPointEnd0[3*cnt+2]-this->efield0.GetOrg().Z())/this->efield0.GetSpacing().Z());
        }

        this->seedPosStart1.Clear();
        this->seedPosStart1.SetCapacityIncrement(10000);
        for(int cnt = 0; cnt < static_cast<int>(this->egradfield1.GetCritPointCount()); cnt++) {
            if(this->egradfield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::ATTRACTING_SADDLE) {

                // Add seedpoint
                this->seedPosStart1.Add(this->egradfield1.GetCritPoint(cnt).GetPos().X());
                this->seedPosStart1.Add(this->egradfield1.GetCritPoint(cnt).GetPos().Y());
                this->seedPosStart1.Add(this->egradfield1.GetCritPoint(cnt).GetPos().Z());
                // Compute bundle of streamlines around the seed point
                float rad = this->streamBundleRad;
                unsigned int res = this->streamBundleRes;
                float step = rad / static_cast<float>(res);
                Vector<float, 3> tangent = this->efield1.GetAtTrilin(this->egradfield1.GetCritPoint(cnt).GetPos());
                tangent.Normalise();
                Vector<float, 3> normal(-tangent.Y()/tangent.X(), 1.0f, 0.0f);
                normal.Normalise();
                Vector<float, 3> p;
                float alpha = this->streamBundlePhi; // Rotation angle
                Matrix<float, 3, COLUMN_MAJOR> rotMat =  Matrix<float, 3, COLUMN_MAJOR>(
                        // Row #1
                        tangent.X()*tangent.X()*(1.0f - cos(alpha))+cos(alpha),
                        tangent.X()*tangent.Y()*(1.0f - cos(alpha))-tangent.Z()*sin(alpha),
                        tangent.X()*tangent.Z()*(1.0f - cos(alpha))+tangent.Y()*sin(alpha),
                        // Row #2
                        tangent.Y()*tangent.X()*(1.0f - cos(alpha))+tangent.Z()*sin(alpha),
                        tangent.Y()*tangent.Y()*(1.0f - cos(alpha))+cos(alpha),
                        tangent.Y()*tangent.Z()*(1.0f - cos(alpha))-tangent.X()*sin(alpha),
                        // Row #3
                        tangent.Z()*tangent.X()*(1.0f - cos(alpha))-tangent.Y()*sin(alpha),
                        tangent.Z()*tangent.Y()*(1.0f - cos(alpha))+tangent.X()*sin(alpha),
                        tangent.Z()*tangent.Z()*(1.0f - cos(alpha))+cos(alpha));
                for(float s = 0.0f; s <= rad; s += step) {
                    for(float a = 0; a < 2*M_PI; a += alpha) {
                        normal = rotMat*normal;
                        p = this->egradfield1.GetCritPoint(cnt).GetPos()+normal*s;
                        if(this->efield1.IsValidGridpos(p)) {
                            this->seedPosStart1.Add(p.X());
                            this->seedPosStart1.Add(p.Y());
                            this->seedPosStart1.Add(p.Z());
                        }
                    }
                }
            }
        }

        float *seedPointEnd1 = new float[this->seedPosStart1.Count()];
        memcpy(seedPointEnd1, this->seedPosStart1.PeekElements(), sizeof(float)*this->seedPosStart1.Count());
        SetGridParams(make_uint3(this->efield1.GetDim().X(), this->efield1.GetDim().Y(),this->efield1.GetDim().Z()),
                      make_float3(this->efield1.GetOrg().X(), this->efield1.GetOrg().Y(),this->efield1.GetOrg().Z()),
                      make_float3(this->bbox.GetRight(), this->bbox.GetTop(), this->bbox.GetFront()),
                      make_float3(this->efield1.GetSpacing().X(), this->efield1.GetSpacing().Y(),this->efield1.GetSpacing().Z()));
        SetStreamlineParams(this->streamlineStepsize/10.0f, this->streamlineMaxSteps);
        SetNumberOfPos(static_cast<uint>(this->seedPosStart1.Count())/3);
        // Compute positions using CUDA
        UpdatePositionRK4(this->efield1.PeekBuff(),
                make_uint3(this->efield1.GetDim().X(),
                           this->efield1.GetDim().Y(),
                           this->efield1.GetDim().Z()),
                seedPointEnd1, static_cast<uint>(this->seedPosStart1.Count())/3, this->streamlineMaxSteps);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... CUDA #1 (forward) done (%f s)",
                this->ClassName(),
                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
        t = clock(); // DEBUG

        this->seedCellEnd1.SetCount(this->seedPosStart1.Count());
#pragma omp parallel for
        for(int cnt = 0; cnt < static_cast<int>(this->seedPosStart1.Count()/3); cnt++) {
            this->seedCellEnd1[3*cnt+0] = static_cast<unsigned int>((seedPointEnd1[3*cnt+0]-this->efield1.GetOrg().X())/this->efield1.GetSpacing().X());
            this->seedCellEnd1[3*cnt+1] = static_cast<unsigned int>((seedPointEnd1[3*cnt+1]-this->efield1.GetOrg().Y())/this->efield1.GetSpacing().Y());
            this->seedCellEnd1[3*cnt+2] = static_cast<unsigned int>((seedPointEnd1[3*cnt+2]-this->efield1.GetOrg().Z())/this->efield1.GetSpacing().Z());
            //printf("Ending cell in #1 %u %u %u\n", this->seedCellEnd1[3*cnt+0], this->seedCellEnd1[3*cnt+1], this->seedCellEnd1[3*cnt+2]);
        }

        // Compute positions using CUDA (integrating backward)
        memcpy(seedPointEnd1, this->seedPosStart1.PeekElements(), sizeof(float)*this->seedPosStart1.Count());
        UpdatePositionRK4(this->efield1.PeekBuff(),
                make_uint3(this->efield1.GetDim().X(),
                           this->efield1.GetDim().Y(),
                           this->efield1.GetDim().Z()),
                seedPointEnd1, static_cast<uint>(this->seedPosStart1.Count())/3, this->streamlineMaxSteps,
                true);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... CUDA #1 (backward) done (%f s)",
                this->ClassName(),
                (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG
        t = clock(); // DEBUG

        this->seedCellEndBackward1.SetCount(this->seedPosStart1.Count());
#pragma omp parallel for
        for(int cnt = 0; cnt < static_cast<int>(this->seedPosStart1.Count()/3); cnt++) {
            this->seedCellEndBackward1[3*cnt+0] = static_cast<unsigned int>((seedPointEnd1[3*cnt+0]-this->efield1.GetOrg().X())/this->efield1.GetSpacing().X());
            this->seedCellEndBackward1[3*cnt+1] = static_cast<unsigned int>((seedPointEnd1[3*cnt+1]-this->efield1.GetOrg().Y())/this->efield1.GetSpacing().Y());
            this->seedCellEndBackward1[3*cnt+2] = static_cast<unsigned int>((seedPointEnd1[3*cnt+2]-this->efield1.GetOrg().Z())/this->efield1.GetSpacing().Z());
            //printf("Ending cell in #1 %u %u %u\n", this->seedCellEndBackward1[3*cnt+0], this->seedCellEndBackward1[3*cnt+1], this->seedCellEndBackward1[3*cnt+2]);
        }

        delete[] seedPointEnd1;

        // Fill endpoint arrays
        this->cellEndpoints0.SetCount(cmd0->GetGridsize().X()*cmd0->GetGridsize().Y()*cmd0->GetGridsize().Z());
        for(unsigned int cnt = 0; cnt < this->cellEndpoints0.Count(); cnt++) {
            this->cellEndpoints0[cnt].Clear();
        }
        for(unsigned int cnt = 0; cnt < this->seedCellEnd0.Count()/3; cnt++) {
            this->cellEndpoints0[cmd0->GetGridsize().X()*(cmd0->GetGridsize().Y()*
                    this->seedCellEnd0[3*cnt+2]+
                    this->seedCellEnd0[3*cnt+1])+
                    this->seedCellEnd0[3*cnt+0]].Add(cnt);
        }
        for(unsigned int cnt = 0; cnt < this->seedCellEndBackward0.Count()/3; cnt++) {
            this->cellEndpoints0[cmd0->GetGridsize().X()*(cmd0->GetGridsize().Y()*
                    this->seedCellEndBackward0[3*cnt+2]
                     +this->seedCellEndBackward0[3*cnt+1])
                     +this->seedCellEndBackward0[3*cnt+0]].Add(cnt);
        }

        this->cellEndpoints1.SetCount(cmd1->GetGridsize().X()*cmd1->GetGridsize().Y()*cmd1->GetGridsize().Z());
        for(unsigned int cnt = 0; cnt < this->cellEndpoints1.Count(); cnt++) {
            this->cellEndpoints1[cnt].Clear();
        }
        for(unsigned int cnt = 0; cnt < this->seedCellEnd1.Count()/3; cnt++) {
            this->cellEndpoints1[cmd1->GetGridsize().X()*(cmd1->GetGridsize().Y()*
                    this->seedCellEnd1[3*cnt+2]+
                    this->seedCellEnd1[3*cnt+1])+
                    this->seedCellEnd1[3*cnt+0]].Add(cnt);
        }
        for(unsigned int cnt = 0; cnt < this->seedCellEndBackward1.Count()/3; cnt++) {

            this->cellEndpoints1[cmd1->GetGridsize().X()*(cmd1->GetGridsize().Y()*
                    this->seedCellEndBackward1[3*cnt+2]+
                    this->seedCellEndBackward1[3*cnt+1])+
                    this->seedCellEndBackward1[3*cnt+0]].Add(cnt);
        }

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Integrating streamlines in regions of interest using RK4 ...",
                this->ClassName()); // DEBUG

        // Integrate streamlines in field #0
        bool *usedSeedPoints = new bool[this->seedCellEnd0.Count()/3];
        for(unsigned int cnt = 0; cnt < this->seedCellEnd0.Count()/3; cnt++) {
            usedSeedPoints[cnt] = false;
        }
        this->streamlines0.SetCount(0);
        for(int cnt = 0; cnt < static_cast<int>(this->efield0.GetCritPointCount()); cnt++) {

            if((this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SOURCE) ||
                    (this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SINK)) {
                if((this->neighbours0[cnt].Count() == 0)||(this->streamlinesShowAll)) {

                    Vector<unsigned int, 3> cellId =
                            this->efield0.GetCritPoint(cnt).GetCellId();
                    unsigned int cellIdx =
                            cmd0->GetGridsize().X()*
                            (cmd0->GetGridsize().Y()*
                            cellId.Z()+cellId.Y())+cellId.X();

                    /*if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SOURCE) {
                        printf("Field #0 Type SOURCE, cell %u %u %u, pos %f %f %f, ",
                                this->efield0.GetCritPoint(cnt).GetCellId().X(),
                                this->efield0.GetCritPoint(cnt).GetCellId().Y(),
                                this->efield0.GetCritPoint(cnt).GetCellId().Z(),
                                this->efield0.GetCritPoint(cnt).GetPos().X(),
                                this->efield0.GetCritPoint(cnt).GetPos().Y(),
                                this->efield0.GetCritPoint(cnt).GetPos().Z());
                    }

                    if(this->efield0.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SINK) {
                        printf("Field #0 Type SINK, cell %u %u %u, pos %f %f %f, ",
                                this->efield0.GetCritPoint(cnt).GetCellId().X(),
                                this->efield0.GetCritPoint(cnt).GetCellId().Y(),
                                this->efield0.GetCritPoint(cnt).GetCellId().Z(),
                                this->efield0.GetCritPoint(cnt).GetPos().X(),
                                this->efield0.GetCritPoint(cnt).GetPos().Y(),
                                this->efield0.GetCritPoint(cnt).GetPos().Z());
                    }
                    printf("streamlines %u\n", this->cellEndpoints0[cellIdx].Count());*/

                    for(unsigned int e = 0; e < this->cellEndpoints0[cellIdx].Count(); e++) {
                        //printf("idx %u, ", this->cellEndpoints0[cellIdx][e]);
                        // Check whether this streamline has already been integrated
                        if(!usedSeedPoints[this->cellEndpoints0[cellIdx][e]]) {
                            vislib::math::Vector<float, 3> posStart;
                            posStart.SetX(this->seedPosStart0[3*this->cellEndpoints0[cellIdx][e]+0]);
                            posStart.SetY(this->seedPosStart0[3*this->cellEndpoints0[cellIdx][e]+1]);
                            posStart.SetZ(this->seedPosStart0[3*this->cellEndpoints0[cellIdx][e]+2]);
                            this->streamlines0.Add(new Streamline());
                            this->streamlines0.Last()->IntegrateRK4(
                                    posStart,                       // Starting point
                                    this->efield0,                  // Vector field
                                    this->streamlineMaxSteps,       // Maximum length
                                    this->streamlineStepsize/10.0f, // Step size
                                    this->streamlineEps,            // Epsilon
                                    Streamline::BIDIRECTIONAL);
                            usedSeedPoints[this->cellEndpoints0[cellIdx][e]] = true;
                        }
                    }
                   // printf("\n");
                }
            }
        }
        delete[] usedSeedPoints;

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... done, %u streamlines in #0 (%f s)",
                 this->ClassName(),
                 (double(clock()-t)/double(CLOCKS_PER_SEC)),
                 this->streamlines0.Count()); // DEBUG
        t = clock(); // DEBUG

        // Integrate streamlines in field #1
        usedSeedPoints = new bool[this->seedCellEnd1.Count()/3];
        for(unsigned int cnt = 0; cnt < this->seedCellEnd1.Count()/3; cnt++) {
            usedSeedPoints[cnt] = false;
        }
        this->streamlines1.SetCount(0);
        for(int cnt = 0; cnt < static_cast<int>(this->efield1.GetCritPointCount()); cnt++) {

            if((this->efield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SOURCE) ||
                    (this->efield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SINK)) {
                if((this->neighbours1[cnt].Count() == 0)||(this->streamlinesShowAll)) {

                    Vector<unsigned int, 3> cellId =
                            this->efield1.GetCritPoint(cnt).GetCellId();
                    unsigned int cellIdx =
                            cmd1->GetGridsize().X()*
                            (cmd1->GetGridsize().Y()*
                            cellId.Z()+cellId.Y())+cellId.X();

                    /*if(this->efield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SOURCE) {
                        printf("Field #1 Type SOURCE, cell %u %u %u, pos %f %f %f, ",
                                this->efield1.GetCritPoint(cnt).GetCellId().X(),
                                this->efield1.GetCritPoint(cnt).GetCellId().Y(),
                                this->efield1.GetCritPoint(cnt).GetCellId().Z(),
                                this->efield1.GetCritPoint(cnt).GetPos().X(),
                                this->efield1.GetCritPoint(cnt).GetPos().Y(),
                                this->efield1.GetCritPoint(cnt).GetPos().Z());
                    }

                    if(this->efield1.GetCritPoint(cnt).GetType() == VecField3f::CritPoint::SINK) {
                        printf("Field #1 Type SINK, cell %u %u %u, pos %f %f %f, ",
                                this->efield1.GetCritPoint(cnt).GetCellId().X(),
                                this->efield1.GetCritPoint(cnt).GetCellId().Y(),
                                this->efield1.GetCritPoint(cnt).GetCellId().Z(),
                                this->efield1.GetCritPoint(cnt).GetPos().X(),
                                this->efield1.GetCritPoint(cnt).GetPos().Y(),
                                this->efield1.GetCritPoint(cnt).GetPos().Z());
                    }
                    printf("streamlines %u\n", this->cellEndpoints1[cellIdx].Count());*/

                    for(unsigned int e = 0; e < this->cellEndpoints1[cellIdx].Count(); e++) {
                        //printf("idx %u, ", this->cellEndpoints1[cellIdx][e]);
                        // Check whether this streamline has already been integrated
                        if(!usedSeedPoints[this->cellEndpoints1[cellIdx][e]]) {
                            vislib::math::Vector<float, 3> posStart;
                            posStart.SetX(this->seedPosStart1[3*this->cellEndpoints1[cellIdx][e]+0]);
                            posStart.SetY(this->seedPosStart1[3*this->cellEndpoints1[cellIdx][e]+1]);
                            posStart.SetZ(this->seedPosStart1[3*this->cellEndpoints1[cellIdx][e]+2]);
                            this->streamlines1.Add(new Streamline());
                            this->streamlines1.Last()->IntegrateRK4(
                                    posStart,                       // Starting point
                                    this->efield1,                  // Vector field
                                    this->streamlineMaxSteps,       // Maximum length
                                    this->streamlineStepsize/10.0f, // Step size
                                    this->streamlineEps,            // Epsilon
                                    Streamline::BIDIRECTIONAL);
                            usedSeedPoints[this->cellEndpoints1[cellIdx][e]] = true;
                        }
                    }
                   // printf("\n");
                }
            }
        }
        delete[] usedSeedPoints;

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: ... done, %u streamlines in #1 (%f s)",
                 this->ClassName(),
                 (double(clock()-t)/double(CLOCKS_PER_SEC)),
                 this->streamlines1.Count()); // DEBUG

        this->recalcStreamlines = false;

        delete[] seedPointEnd0;
        delete[] seedPointEndBackward0;
    }

    // Render streamlines using GLSL shaders
    if(toggleStreamlines) {
        this->streamlineShader.Enable();

        // Set shader variables
        glUniformMatrix4fvARB(this->streamlineShader.ParameterLocation("modelview"), 1, false, this->modelMatrix);
        glUniformMatrix4fvARB(this->streamlineShader.ParameterLocation("proj"), 1, false, this->projMatrix);
        glUniform4fvARB(this->streamlineShader.ParameterLocation("lightPos"), 1, this->lightPos);
        glUniform4fvARB(this->streamlineShader.ParameterLocation("viewAttr"), 1, this->viewportStuff);
        glUniform1iARB(this->streamlineShader.ParameterLocation("shading"), static_cast<int>(this->streamlineShading));
        glUniform1fARB(this->streamlineShader.ParameterLocation("fogZ"), this->fogZ);
        glUniform1fARB(this->streamlineShader.ParameterLocation("minPot"), this->texMinVal);
        glUniform1fARB(this->streamlineShader.ParameterLocation("maxPot"), this->texMaxVal);
        glUniform1iARB(this->streamlineShader.ParameterLocation("potentialTex"), 0);

        // Vertex attributes
        GLuint vertexPos = glGetAttribLocationARB(this->streamlineShader, "vertex");
        GLuint vertexTangent = glGetAttribLocationARB(this->streamlineShader, "tangent");
        GLuint vertexTC = glGetAttribLocationARB(this->streamlineShader, "tc");

        // Enable arrays for attributes
        glEnableVertexAttribArrayARB(vertexPos);
        glEnableVertexAttribArrayARB(vertexTangent);
        glEnableVertexAttribArrayARB(vertexTC);

        // Vector field #0
        glUniform1iARB(this->streamlineShader.ParameterLocation("vecFieldIdx"), 0);
        glActiveTextureARB(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, this->potentialTex0);
        printf("Rendering %u streamlines\n", this->streamlines0.Count());
        for (unsigned int cnt = 0; cnt < this->streamlines0.Count(); cnt++) {
            // Set attribute pointers
            glVertexAttribPointerARB(vertexPos, 3, GL_FLOAT, GL_FALSE, 0, this->streamlines0[cnt]->PeekVertexArr());
            glVertexAttribPointerARB(vertexTangent, 3, GL_FLOAT, GL_FALSE, 0, this->streamlines0[cnt]->PeekTangentArr());
            glVertexAttribPointerARB(vertexTC, 3, GL_FLOAT, GL_FALSE, 0, this->streamlines0[cnt]->PeekTexCoordArr());
            // Draw points
            glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(this->streamlines0[cnt]->GetLength()));
        }

        glUniform1iARB(this->streamlineShader.ParameterLocation("vecFieldIdx"), 1);
        glActiveTextureARB(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, this->potentialTex1);
        for (unsigned int cnt = 0; cnt < this->streamlines1.Count(); cnt++) {
            // Set attribute pointers
            glVertexAttribPointerARB(vertexPos, 3, GL_FLOAT, GL_FALSE, 0, this->streamlines1[cnt]->PeekVertexArr());
            glVertexAttribPointerARB(vertexTangent, 3, GL_FLOAT, GL_FALSE, 0, this->streamlines1[cnt]->PeekTangentArr());
            glVertexAttribPointerARB(vertexTC, 3, GL_FLOAT, GL_FALSE, 0, this->streamlines1[cnt]->PeekTexCoordArr());
            // Draw points
            glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(this->streamlines1[cnt]->GetLength()));
        }

        // Disable arrays for attributes
        glDisableVertexAttribArrayARB(vertexPos);
        glDisableVertexAttribArrayARB(vertexTangent);
        glDisableVertexAttribArrayARB(vertexTC);

        this->streamlineShader.Disable();

        // Check for opengl error
        GLenum err = glGetError();
        if(err != GL_NO_ERROR) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: glError in 'renderStreamlinesRoi' %s \n",
                    this->ClassName(),
                    gluErrorString(err));
            return false;
        }
    }

    return true;
}


/*
 * ComparativeFieldTopologyRenderer::updateParams
 */
void ComparativeFieldTopologyRenderer::updateParams() {

    /* Parameters related to fog */

    // Param for the radius for the arrow glyphs
    if(this->fogZSlot.IsDirty()) {
        this->fogZ = this->fogZSlot.Param<core::param::FloatParam>()->Value();
        this->fogZSlot.ResetDirty();
    }


    /* Parameters for arrow glyph rendering */

    // Param for the radius for the arrow glyphs
    if(this->arrowRadSclSlot.IsDirty()) {
        this->arrowRadScl = this->arrowRadSclSlot.Param<core::param::FloatParam>()->Value();
        this->arrowRadSclSlot.ResetDirty();
    }

    // Param for the scaling of the arrow glyphs
    if(this->arrowLenSclSlot.IsDirty()) {
        this->arrowLenScl = this->arrowLenSclSlot.Param<core::param::FloatParam>()->Value();
        this->arrowLenSclSlot.ResetDirty();
        this->recalcArrowData = true;
    }

    // Param for maximum x value
    if(this->arrowFilterXMaxSlot.IsDirty()) {
        this->arrowFilterXMax = this->arrowFilterXMaxSlot.Param<core::param::FloatParam>()->Value();
        this->arrowFilterXMaxSlot.ResetDirty();
        this->recalcArrowData = true;
    }

    // Param for maximum y value
    if(this->arrowFilterYMaxSlot.IsDirty()) {
        this->arrowFilterYMax = this->arrowFilterYMaxSlot.Param<core::param::FloatParam>()->Value();
        this->arrowFilterYMaxSlot.ResetDirty();
        this->recalcArrowData = true;
    }

    // Param for maximum z value
    if(this->arrowFilterZMaxSlot.IsDirty()) {
        this->arrowFilterZMax = this->arrowFilterZMaxSlot.Param<core::param::FloatParam>()->Value();
        this->arrowFilterZMaxSlot.ResetDirty();
        this->recalcArrowData = true;
    }

    // Param for minimum x value
    if(this->arrowFilterXMinSlot.IsDirty()) {
        this->arrowFilterXMin = this->arrowFilterXMinSlot.Param<core::param::FloatParam>()->Value();
        this->arrowFilterXMinSlot.ResetDirty();
        this->recalcArrowData = true;
    }

    // Param for minimum y value
    if(this->arrowFilterYMinSlot.IsDirty()) {
        this->arrowFilterYMin = this->arrowFilterYMinSlot.Param<core::param::FloatParam>()->Value();
        this->arrowFilterYMinSlot.ResetDirty();
        this->recalcArrowData = true;
    }

    // Param for minimum z value
    if(this->arrowFilterZMinSlot.IsDirty()) {
        this->arrowFilterZMin = this->arrowFilterZMinSlot.Param<core::param::FloatParam>()->Value();
        this->arrowFilterZMinSlot.ResetDirty();
        this->recalcArrowData = true;
    }


    /* Parameters for critical point analysis */

    // Parameter for sphere radius
    if(this->critPointsSphereSclSlot.IsDirty()) {
        this->critPointsSphereScl = this->critPointsSphereSclSlot.Param<core::param::FloatParam>()->Value();
        this->critPointsSphereSclSlot.ResetDirty();
    }

    // Parameter to determine the maximum number of bisections
    if(this->critPointsMaxBisectionsSlot.IsDirty()) {
        this->critPointsMaxBisections = this->critPointsMaxBisectionsSlot.Param<core::param::IntParam>()->Value();
        this->critPointsMaxBisectionsSlot.ResetDirty();
        this->recalcCritPoints = true;
    }

    // Parameter for maximum number of Newton iterations
    if(this->critPointsNewtonMaxStepsSlot.IsDirty()) {
        this->critPointsNewtonMaxSteps = this->critPointsNewtonMaxStepsSlot.Param<core::param::IntParam>()->Value();
        this->critPointsNewtonMaxStepsSlot.ResetDirty();
        this->recalcCritPoints = true;
    }

    // Parameter for the stepsize of the Newton iteration
    if(this->critPointsNewtonStepsizeSlot.IsDirty()) {
        this->critPointsNewtonStepsize = this->critPointsNewtonStepsizeSlot.Param<core::param::FloatParam>()->Value();
        this->critPointsNewtonStepsizeSlot.ResetDirty();
        this->recalcCritPoints = true;
    }

    // Parameter for the epsilon for the Newton iteration
    if(this->critPointsNewtonEpsSlot.IsDirty()) {
        this->critPointsNewtonEps = this->critPointsNewtonEpsSlot.Param<core::param::FloatParam>()->Value();
        this->critPointsNewtonEpsSlot.ResetDirty();
        this->recalcCritPoints = true;
    }

    // Param slot to determine whether all critpoints are to be shown
    if(this->critPointsShowAllSlot.IsDirty()) {
        this->critPointsShowAll = this->critPointsShowAllSlot.Param<core::param::BoolParam>()->Value();
        this->critPointsShowAllSlot.ResetDirty();
    }


    /* Parameters for streamlines */

    // Param for maximum steps of streamline drawing
    if(this->streamlineMaxStepsSlot.IsDirty()) {
        this->streamlineMaxSteps = this->streamlineMaxStepsSlot.Param<core::param::IntParam>()->Value();
        this->streamlineMaxStepsSlot.ResetDirty();
        this->recalcStreamlines = true;
    }

    // Param for the shading mode of the streamlines
    if(this->streamlineShadingSlot.IsDirty()) {
        this->streamlineShading = static_cast<StreamlineShading>(this->streamlineShadingSlot.Param<core::param::EnumParam>()->Value());
        this->streamlineShadingSlot.ResetDirty();
    }

    // Param for the radius of the streamline bundle
    if(this->streamBundleRadSlot.IsDirty()) {
        this->streamBundleRad = this->streamBundleRadSlot.Param<core::param::FloatParam>()->Value();
        this->streamBundleRadSlot.ResetDirty();
        this->recalcStreamlines = true;
    }

    // Param for the resolution of the streamline bundle
    if(this->streamBundleResSlot.IsDirty()) {
        this->streamBundleRes = this->streamBundleResSlot.Param<core::param::IntParam>()->Value();
        this->streamBundleResSlot.ResetDirty();
        this->recalcStreamlines = true;
    }

    // Param for the stepSize of the streamline bundle
    if(this->streamBundlePhiSlot.IsDirty()) {
        this->streamBundlePhi = this->streamBundlePhiSlot.Param<core::param::FloatParam>()->Value();
        this->streamBundlePhiSlot.ResetDirty();
        this->recalcStreamlines = true;
    }

    // Param for the epsilon for streamline termination
    if(this->streamlineEpsSlot.IsDirty()) {
        this->streamlineEps = this->streamlineEpsSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineEpsSlot.ResetDirty();
        this->recalcStreamlines = true;
    }

    // Param for the stepsize in streamline integration
    if(this->streamlineStepsizeSlot.IsDirty()) {
        this->streamlineStepsize = this->streamlineStepsizeSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineStepsizeSlot.ResetDirty();
        this->recalcStreamlines = true;
    }

    // Parameter to toggle rendering of streamlines
    if(this->toggleStreamlinesSlot.IsDirty()) {
        this->toggleStreamlines = this->toggleStreamlinesSlot.Param<core::param::BoolParam>()->Value();
        this->toggleStreamlinesSlot.ResetDirty();
    }

    // Parameter to toggle rendering of all streamlines
    if(this->streamlinesShowAllSlot.IsDirty()) {
        this->streamlinesShowAll = this->streamlinesShowAllSlot.Param<core::param::BoolParam>()->Value();
        this->streamlinesShowAllSlot.ResetDirty();
        this->recalcStreamlines = true;
    }


    /* Parameters for manually set streamline seed point */

    // Parameter for x coord of streamline seed point
    if(this->streamBundleSeedXSlot.IsDirty()) {
        this->streamBundleSeedX = this->streamBundleSeedXSlot.Param<core::param::FloatParam>()->Value();
        this->streamBundleSeedXSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter for y coord of streamline seed point
    if(this->streamBundleSeedYSlot.IsDirty()) {
        this->streamBundleSeedY = this->streamBundleSeedYSlot.Param<core::param::FloatParam>()->Value();
        this->streamBundleSeedYSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter for z coord of streamline seed point
    if(this->streamBundleSeedZSlot.IsDirty()) {
        this->streamBundleSeedZ = this->streamBundleSeedZSlot.Param<core::param::FloatParam>()->Value();
        this->streamBundleSeedZSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter for streamline maximum steps
    if(this->streamlineMaxStepsManualSlot.IsDirty()) {
        this->streamlineMaxStepsManual = this->streamlineMaxStepsManualSlot.Param<core::param::IntParam>()->Value();
        this->streamlineMaxStepsManualSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter to determine streamline shading
    if(this->streamlineShadingManualSlot.IsDirty()) {
        this->streamlineShadingManual = static_cast<StreamlineShading>(this->streamlineShadingManualSlot.Param<core::param::EnumParam>()->Value());
        this->streamlineShadingManualSlot.ResetDirty();
    }

    // Parameter to set the radius of the streamline bundle
    if(this->streamBundleRadManualSlot.IsDirty()) {
        this->streamBundleRadManual = this->streamBundleRadManualSlot.Param<core::param::FloatParam>()->Value();
        this->streamBundleRadManualSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter to set the resolution of the streamline bundle
    if(this->streamBundleResManualSlot.IsDirty()) {
        this->streamBundleResManual = this->streamBundleResManualSlot.Param<core::param::IntParam>()->Value();
        this->streamBundleResManualSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter to set the step size of the streamline bundle
    if(this->streamBundlePhiManualSlot.IsDirty()) {
        this->streamBundlePhiManual = this->streamBundlePhiManualSlot.Param<core::param::FloatParam>()->Value();
        this->streamBundlePhiManualSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter to set the epsilon for stream line terminations
    if(this->streamlineEpsManualSlot.IsDirty()) {
        this->streamlineEpsManual = this->streamlineEpsManualSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineEpsManualSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter to set the stepsize for streamline integration
    if(this->streamlineStepsizeManualSlot.IsDirty()) {
        this->streamlineStepsizeManual = this->streamlineStepsizeManualSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineStepsizeManualSlot.ResetDirty();
        this->recalcStreamlinesManualSeed = true;
    }

    // Parameter to toggle rendering of streamlines based on manual seed
    if(this->toggleStreamlinesManualSlot.IsDirty()) {
        this->toggleStreamlinesManual = this->toggleStreamlinesManualSlot.Param<core::param::BoolParam>()->Value();
        this->toggleStreamlinesManualSlot.ResetDirty();
    }


    /* Parameters for finding regions of interest */

    // Param slot for maximum euclidean distance between critpoints
    if(this->roiMaxDistSlot.IsDirty()) {
        this->roiMaxDist = this->roiMaxDistSlot.Param<core::param::FloatParam>()->Value();
        this->roiMaxDistSlot.ResetDirty();
        this->recalcNeighbours = true;
        this->recalcStreamlines = true;
    }


    /* Parameters for debugging purposes */

    // Minimum texture value
    if(this->texMinValSlot.IsDirty()) {
        this->texMinVal = this->texMinValSlot.Param<core::param::FloatParam>()->Value();
        this->texMinValSlot.ResetDirty();
    }

    // Maximum texture value
    if(this->texMaxValSlot.IsDirty()) {
        this->texMaxVal = this->texMaxValSlot.Param<core::param::FloatParam>()->Value();
        this->texMaxValSlot.ResetDirty();
    }

    // Position of texture slice
    if(this->texPosZSlot.IsDirty()) {
        this->texPosZ = this->texPosZSlot.Param<core::param::FloatParam>()->Value();
        this->texPosZSlot.ResetDirty();
    }
}
