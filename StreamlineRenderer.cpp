//
// StreamlineRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 11, 2013
//     Author: scharnkn
//

#include "stdafx.h"
#include "StreamlineRenderer.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))

#include "VBODataCall.h"
#include "VTIDataCall.h"
#include "cuda_error_check.h"
#include "ogl_error_check.h"
#include "CUDAFieldTopology.cuh"
#include "Interpol.h"

#include "vislib/GLSLShader.h"
#include "vislib/Cuboid.h"
#include "vislib/Vector.h"
#include "vislib/mathfunctions.h"

#include "CoreInstance.h"
#include "param/FloatParam.h"
#include "param/IntParam.h"

#include <cuda_gl_interop.h>
#include <cstdlib>

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core;

typedef vislib::math::Vector<float, 3> Vec3f;


/*
 * StreamlineRenderer::StreamlineRenderer
 */
StreamlineRenderer::StreamlineRenderer(void) : Renderer3DModuleDS(),
        /* Caller slots */
        fieldDataCallerSlot("getFieldData", "Connects the renderer with the field data"),
        /* Streamline integration parameters */
        nStreamlinesSlot("streamlines::nStreamlines", "Set the number of streamlines"),
        streamlineMaxStepsSlot("streamlines::nSteps", "Set the number of steps for streamline integration"),
        streamlineStepSlot("streamlines::step","Set stepsize for the streamline integration"),
        streamlineEpsSlot("streamlines::eps","Set epsilon for the termination of the streamline integration"),
        streamtubesThicknessSlot("streamlines::tubesScl","The scale factor for the streamtubes thickness"),
        minColSlot("streamlines::minCol","Minimum color value"),
        maxColSlot("streamlines::maxCol","Maximum color value"),
        triggerComputeGradientField(true), triggerComputeStreamlines(true) {

    // Data caller for volume data
    this->fieldDataCallerSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->fieldDataCallerSlot);


    /* Streamline integration parameters */

    // Set the number of streamlines
    this->nStreamlines = 10;
    this->nStreamlinesSlot.SetParameter(new core::param::IntParam(this->nStreamlines, 0));
    this->MakeSlotAvailable(&this->nStreamlinesSlot);

    // Set the number of steps for streamline integration
    this->streamlineMaxSteps = 10;
    this->streamlineMaxStepsSlot.SetParameter(new core::param::IntParam(this->streamlineMaxSteps, 0));
    this->MakeSlotAvailable(&this->streamlineMaxStepsSlot);

    // Set the step size for streamline integration
    this->streamlineStep = 1.0f;
    this->streamlineStepSlot.SetParameter(new core::param::FloatParam(this->streamlineStep, 0.1f));
    this->MakeSlotAvailable(&this->streamlineStepSlot);

    // Set the step size for streamline integration
    this->streamlineEps = 0.01f;
    this->streamlineEpsSlot.SetParameter(new core::param::FloatParam(this->streamlineEps, 0.0f));
    this->MakeSlotAvailable(&this->streamlineEpsSlot);

    // Set the streamtubes thickness slot
    this->streamtubesThickness = 1.0f;
    this->streamtubesThicknessSlot.SetParameter(new core::param::FloatParam(this->streamtubesThickness, 0.0f));
    this->MakeSlotAvailable(&this->streamtubesThicknessSlot);

    // Set the streamtubes min color
    this->minCol = -1.0f;
    this->minColSlot.SetParameter(new core::param::FloatParam(this->minCol));
    this->MakeSlotAvailable(&this->minColSlot);

    // Set the streamtubes max color
    this->maxCol = 1.0f;
    this->maxColSlot.SetParameter(new core::param::FloatParam(this->maxCol));
    this->MakeSlotAvailable(&this->maxColSlot);
}


/*
 * StreamlineRenderer::~StreamlineRenderer
 */
StreamlineRenderer::~StreamlineRenderer(void) {
    this->Release();
}


/*
 * StreamlineRenderer::create
 */
bool StreamlineRenderer::create(void) {

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Init extensions
    if (!glh_init_extensions(
            "GL_ARB_vertex_shader \
            GL_ARB_vertex_program \
            GL_ARB_shader_objects \
            GL_EXT_gpu_shader4 \
            GL_EXT_geometry_shader4 \
            GL_EXT_bindable_uniform \
            GL_VERSION_2_0 \
            GL_ARB_draw_buffers \
            GL_ARB_copy_buffer \
            GL_ARB_vertex_buffer_object")) {
        return false;
    }
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }
    if (!vislib::graphics::gl::GLSLGeometryShader::InitialiseExtensions()) {
        return false;
    }


    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable( GL_VERTEX_PROGRAM_TWO_SIDE);

    ShaderSource vertSrc;
    ShaderSource fragSrc;
    ShaderSource geomSrc;

    // Load the shader source for the tube shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("streamlines::tube::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cartoon shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("streamlines::tube::geometry", geomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for tube shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("streamlines::tube::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for cartoon shader");
        return false;
    }
    this->tubeShader.Compile(vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count());
    this->tubeShader.SetProgramParameter(GL_GEOMETRY_INPUT_TYPE_EXT , GL_LINES_ADJACENCY);
    this->tubeShader.SetProgramParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    this->tubeShader.SetProgramParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 200);
    if (!this->tubeShader.Link()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to link the streamtube shader");
        return false;
    }

    // Load the shader source for the illuminated streamlines
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("streamlines::illuminated::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for illuminated streamlines");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("streamlines::illuminated::geometry", geomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for illuminated streamlines");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("streamlines::illuminated::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for illuminated streamlines");
        return false;
    }
    this->illumShader.Compile(vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count());

    this->illumShader.SetProgramParameter(GL_GEOMETRY_INPUT_TYPE_EXT , GL_LINES_ADJACENCY);
    this->illumShader.SetProgramParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_LINE_STRIP);
    this->illumShader.SetProgramParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 200);
    if (!this->illumShader.Link()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to link the illuminated streamlines shader");
        return false;
    }

    return true;
}



/*
 * StreamlineRenderer::release
 */
void StreamlineRenderer::release(void) {
    this->tubeShader.Release();
    this->illumShader.Release();
}


/*
 * StreamlineRenderer::GetCapabilities
 */
bool StreamlineRenderer::GetCapabilities(core::Call& call) {

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    cr3d->SetCapabilities(core::view::AbstractCallRender3D::CAP_RENDER |
                          core::view::AbstractCallRender3D::CAP_LIGHTING |
                          core::view::AbstractCallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * StreamlineRenderer::GetExtents
 */
bool StreamlineRenderer::GetExtents(core::Call& call) {

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    // Extent of volume data

    VTIDataCall *vtiCall = this->fieldDataCallerSlot.CallAs<VTIDataCall>();
    if (vtiCall == NULL) {
        return false;
    }

    if (!(*vtiCall)(VTIDataCall::CallForGetExtent)) {
         return false;
    }

    vtiCall->SetCalltime(cr3d->Time());
    vtiCall->SetFrameID(static_cast<int>(cr3d->Time()));
    if (!(*vtiCall)(VTIDataCall::CallForGetData)) {
         return false;
    }

    float scale;
    this->bbox.SetObjectSpaceBBox(vtiCall->GetWholeExtent());
    if(!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    //this->bbox.MakeScaledWorld(scale);
    cr3d->AccessBoundingBoxes() = vtiCall->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(vtiCall->FrameCount());

    return true;
}


/*
 * StreamlineRenderer::Render
 */
bool StreamlineRenderer::Render(core::Call& call) {

    // Update parameters
    this->updateParams();

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    VTIDataCall *vtiCall = this->fieldDataCallerSlot.CallAs<VTIDataCall>();
    if (vtiCall == NULL) {
        return false;
    }

    // Get volume data
    vtiCall->SetCalltime(cr3d->Time());
    if (!(*vtiCall)(VTIDataCall::CallForGetData)) {
         return false;
    }


    float scale;

    // (Re)compute streamlines if necessary
    if (this->triggerComputeStreamlines) {

        float zHeight = (vtiCall->GetGridsize().GetZ()-1)*vtiCall->GetSpacing().GetZ();
        this->genSeedPoints(vtiCall,
                zHeight*0.1, zHeight*0.3, zHeight*0.5, zHeight*0.8, // clipping planes
                0.3, 0.5, 0.7, 0.9); // Isovalues

        //printf("height: %f, clip %f %f %f %f\n", zHeight, zHeight*0.2, zHeight*0.4, zHeight*0.6,zHeight*0.8);

        if (!this->strLines.InitStreamlines(this->streamlineMaxSteps,
                this->nStreamlines*4, CUDAStreamlines::BIDIRECTIONAL)) {
            return false;
        }

        // Integrate streamlines
        if (!this->strLines.IntegrateRK4(
                this->seedPoints.PeekElements(),
                this->streamlineStep,
                (float*)vtiCall->GetPointDataByIdx(1, 0), // TODO Do not hardcode array
                make_int3(vtiCall->GetGridsize().GetX(),
                        vtiCall->GetGridsize().GetY(),
                        vtiCall->GetGridsize().GetZ()),
                make_float3(vtiCall->GetOrigin().GetX(),
                        vtiCall->GetOrigin().GetY(),
                        vtiCall->GetOrigin().GetZ()),
                make_float3(vtiCall->GetSpacing().GetX(),
                                vtiCall->GetSpacing().GetY(),
                                vtiCall->GetSpacing().GetZ()))) {
            return false;
        }

        // Sample the density field to the alpha component
        if (!this->strLines.SampleScalarFieldToAlpha(
                (float*)vtiCall->GetPointDataByIdx(0, 0), // TODO do not hardcode array
                make_int3(vtiCall->GetGridsize().GetX(),
                        vtiCall->GetGridsize().GetY(),
                        vtiCall->GetGridsize().GetZ()),
                        make_float3(vtiCall->GetOrigin().GetX(),
                                vtiCall->GetOrigin().GetY(),
                                vtiCall->GetOrigin().GetZ()),
                                make_float3(vtiCall->GetSpacing().GetX(),
                                        vtiCall->GetSpacing().GetY(),
                                        vtiCall->GetSpacing().GetZ()))) {
            return false;
        }

        // Set RGB color value
//        if (!this->strLines.SetUniformRGBColor(make_float3(1.0, 0.0, 1.0))) {
//            return false;
//        }

        if (!this->strLines.SampleVecFieldToRGB(
                (float*)vtiCall->GetPointDataByIdx(1, 0), // TODO do not hardcode array
                make_int3(vtiCall->GetGridsize().GetX(),
                                        vtiCall->GetGridsize().GetY(),
                                        vtiCall->GetGridsize().GetZ()),
                                        make_float3(vtiCall->GetOrigin().GetX(),
                                                vtiCall->GetOrigin().GetY(),
                                                vtiCall->GetOrigin().GetZ()),
                                                make_float3(vtiCall->GetSpacing().GetX(),
                                                        vtiCall->GetSpacing().GetY(),
                                                        vtiCall->GetSpacing().GetZ()))) {
            return false;
        }

        this->triggerComputeStreamlines = false;
    }


    glPushMatrix();

    // Compute scale factor and scale world
    if( !vislib::math::IsEqual( this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);


    // Render streamlines
    glDisable(GL_LINE_SMOOTH);
    glColor3f(0.0f, 1.0f, 1.0f);

    this->tubeShader.Enable();
    glUniform1fARB(this->tubeShader.ParameterLocation("streamTubeThicknessScl"), this->streamtubesThickness);
    glUniform1fARB(this->tubeShader.ParameterLocation("minColTexValue"), this->minCol); // TODO Parameter?
    glUniform1fARB(this->tubeShader.ParameterLocation("maxColTexValue"), this->maxCol); // TODO Paremeter?

    if (!this->strLines.RenderLineStrip()) {
        return false;
    }

    this->tubeShader.Disable();

    glPopMatrix();

    return true;
}


/*
 * StreamlineRenderer::genSeedPoints
 */
void StreamlineRenderer::genSeedPoints(
        VTIDataCall *vti,
        float zClip0, float zClip1, float zClip2, float zClip3,
        float isoval0, float isoval1, float isoval2, float isoval3) {


    float posZ[4];
    posZ[0]= vti->GetOrigin().GetZ() + zClip0; // Start above the lower boundary
    posZ[1]= vti->GetOrigin().GetZ() + zClip1; // Start above the lower boundary
    posZ[2]= vti->GetOrigin().GetZ() + zClip2; // Start above the lower boundary
    posZ[3]= vti->GetOrigin().GetZ() + zClip3; // Start above the lower boundary


//    printf("clip %f %f %f %f\n", zClip0, zClip1, zClip2, zClip3);



    float xMax = vti->GetOrigin().GetX() + vti->GetSpacing().GetX()*(vti->GetGridsize().GetX()-1);
    float yMax = vti->GetOrigin().GetY() + vti->GetSpacing().GetY()*(vti->GetGridsize().GetY()-1);
    //float zMax = vti->GetOrigin().GetZ() + vtiCall->GetSpacing().GetZ()*(vtiCall->GetGridsize().GetZ()-1);
    float xMin = vti->GetOrigin().GetX();
    float yMin = vti->GetOrigin().GetY();

    // Initialize random seed
    srand (time(NULL));
    this->seedPoints.SetCount(0);
    //for (size_t cnt = 0; cnt < this->nStreamlines; ++cnt) {
    while (this->seedPoints.Count()/3 < this->nStreamlines) {
        Vec3f pos;
        pos.SetX(vti->GetOrigin().GetX() + (float(rand() % 10000)/10000.0f)*(xMax-xMin));
        pos.SetY(vti->GetOrigin().GetY() + (float(rand() % 10000)/10000.0f)*(yMax-yMin));
        pos.SetZ(posZ[0]);
//            printf("Random pos %f %f %f\n", pos.GetX(), pos.GetY(), pos.GetZ());

        float sample = this->sampleFieldAtPosTrilin(
                vti, make_float3(pos.GetX(),
                        pos.GetY(), pos.GetZ()), (float*)vti->GetPointDataByIdx(0, 0));

        // Sample density value
        if (vislib::math::Abs(sample - isoval0) < 0.05) {
            this->seedPoints.Add(pos.GetX());
            this->seedPoints.Add(pos.GetY());
            this->seedPoints.Add(pos.GetZ());
        }
    }

    //for (size_t cnt = 0; cnt < this->nStreamlines; ++cnt) {
    while (this->seedPoints.Count()/3 < 2*this->nStreamlines) {
        Vec3f pos;
        pos.SetX(vti->GetOrigin().GetX() + (float(rand() % 10000)/10000.0f)*(xMax-xMin));
        pos.SetY(vti->GetOrigin().GetY() + (float(rand() % 10000)/10000.0f)*(yMax-yMin));
        pos.SetZ(posZ[1]);
//            printf("Random pos %f %f %f\n", pos.GetX(), pos.GetY(), pos.GetZ());

        float sample = this->sampleFieldAtPosTrilin(
                vti, make_float3(pos.GetX(),
                        pos.GetY(), pos.GetZ()), (float*)vti->GetPointDataByIdx(0, 0));

        // Sample density value
        if (vislib::math::Abs(sample - isoval1) < 0.05) {
            this->seedPoints.Add(pos.GetX());
            this->seedPoints.Add(pos.GetY());
            this->seedPoints.Add(pos.GetZ());
        }
    }

    //for (size_t cnt = 0; cnt < this->nStreamlines; ++cnt) {
    while (this->seedPoints.Count()/3 < 3*this->nStreamlines) {
        Vec3f pos;
        pos.SetX(vti->GetOrigin().GetX() + (float(rand() % 10000)/10000.0f)*(xMax-xMin));
        pos.SetY(vti->GetOrigin().GetY() + (float(rand() % 10000)/10000.0f)*(yMax-yMin));
        pos.SetZ(posZ[2]);
//            printf("Random pos %f %f %f\n", pos.GetX(), pos.GetY(), pos.GetZ());

        float sample = this->sampleFieldAtPosTrilin(
                vti, make_float3(pos.GetX(),
                        pos.GetY(), pos.GetZ()), (float*)vti->GetPointDataByIdx(0, 0));

        // Sample density value
        if (vislib::math::Abs(sample - isoval2) < 0.05) {
            this->seedPoints.Add(pos.GetX());
            this->seedPoints.Add(pos.GetY());
            this->seedPoints.Add(pos.GetZ());
        }
    }

    //for (size_t cnt = 0; cnt < this->nStreamlines; ++cnt) {
    while (this->seedPoints.Count()/3 < 4*this->nStreamlines) {
        Vec3f pos;
        pos.SetX(vti->GetOrigin().GetX() + (float(rand() % 10000)/10000.0f)*(xMax-xMin));
        pos.SetY(vti->GetOrigin().GetY() + (float(rand() % 10000)/10000.0f)*(yMax-yMin));
        pos.SetZ(posZ[3]);
//            printf("Random pos %f %f %f\n", pos.GetX(), pos.GetY(), pos.GetZ());

        float sample = this->sampleFieldAtPosTrilin(
                vti, make_float3(pos.GetX(),
                        pos.GetY(), pos.GetZ()), (float*)vti->GetPointDataByIdx(0, 0));

        // Sample density value
        if (vislib::math::Abs(sample - isoval3) < 0.05) {
            this->seedPoints.Add(pos.GetX());
            this->seedPoints.Add(pos.GetY());
            this->seedPoints.Add(pos.GetZ());
        }
    }

}


/*
 * StreamlineRenderer::sampleFieldAtPosTrilin
 */
float StreamlineRenderer::sampleFieldAtPosTrilin(VTIDataCall *vtiCall, float3 pos, float *field_D) {

    int3 c;
    float3 f;

    int3 gridSize_D = make_int3(vtiCall->GetGridsize().GetX(),
            vtiCall->GetGridsize().GetY(),
            vtiCall->GetGridsize().GetZ());
    float3 gridOrg_D =  make_float3(vtiCall->GetOrigin().GetX(),
            vtiCall->GetOrigin().GetY(),
            vtiCall->GetOrigin().GetZ());
    float3 gridDelta_D = make_float3(vtiCall->GetSpacing().GetX(),
            vtiCall->GetSpacing().GetY(),
            vtiCall->GetSpacing().GetZ());

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f.x = (pos.x-gridOrg_D.x)/gridDelta_D.x;
    f.y = (pos.y-gridOrg_D.y)/gridDelta_D.y;
    f.z = (pos.z-gridOrg_D.z)/gridDelta_D.z;
    c.x = (int)(f.x);
    c.y = (int)(f.y);
    c.z = (int)(f.z);
    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma

    c.x = vislib::math::Clamp(c.x, int(0), gridSize_D.x-2);
    c.y = vislib::math::Clamp(c.y, int(0), gridSize_D.y-2);
    c.z = vislib::math::Clamp(c.z, int(0), gridSize_D.z-2);

    // Get values at corners of current cell
    float s[8];
    s[0] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+0))+c.x+0];
    s[1] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+0))+c.x+1];
    s[2] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+1))+c.x+0];
    s[3] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+0) + (c.y+1))+c.x+1];
    s[4] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+0))+c.x+0];
    s[5] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+0))+c.x+1];
    s[6] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+1))+c.x+0];
    s[7] = field_D[gridSize_D.x*(gridSize_D.y*(c.z+1) + (c.y+1))+c.x+1];

    // Use trilinear interpolation to sample the volume
   return Interpol::Trilin(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], f.x, f.y, f.z);
}


/*
 * StreamlineRenderer::updateParams
 */
void StreamlineRenderer::updateParams() {

    /* Streamline integration parameters */

    // Set the number of steps for streamline integration
    if (this->nStreamlinesSlot.IsDirty()) {
        this->nStreamlines = this->nStreamlinesSlot.Param<core::param::IntParam>()->Value();
        this->nStreamlinesSlot.ResetDirty();
        this->triggerComputeStreamlines = true;
    }

    // Set the number of steps for streamline integration
    if (this->streamlineMaxStepsSlot.IsDirty()) {
        this->streamlineMaxSteps = this->streamlineMaxStepsSlot.Param<core::param::IntParam>()->Value();
        this->streamlineMaxStepsSlot.ResetDirty();
        this->triggerComputeStreamlines = true;
    }

    // Set the step size for streamline integration
    if (this->streamlineStepSlot.IsDirty()) {
        this->streamlineStep = this->streamlineStepSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineStepSlot.ResetDirty();
        this->triggerComputeStreamlines = true;
    }

    // Set the epsilon for the streamline termination
    if (this->streamlineEpsSlot.IsDirty()) {
        this->streamlineEps = this->streamlineEpsSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineEpsSlot.ResetDirty();
        this->triggerComputeStreamlines = true;
    }

    // Set the streamtubes thickness slot
    if (this->streamtubesThicknessSlot.IsDirty()) {
        this->streamtubesThickness = this->streamtubesThicknessSlot.Param<core::param::FloatParam>()->Value();
        this->streamtubesThicknessSlot.ResetDirty();
    }

    // Set minimum color value
    if (this->minColSlot.IsDirty()) {
        this->minCol = this->minColSlot.Param<core::param::FloatParam>()->Value();
        this->minColSlot.ResetDirty();
    }

    // Set maximum color value
    if (this->maxColSlot.IsDirty()) {
        this->maxCol = this->maxColSlot.Param<core::param::FloatParam>()->Value();
        this->maxColSlot.ResetDirty();
    }

}


#endif // (defined(WITH_CUDA) && (WITH_CUDA))
