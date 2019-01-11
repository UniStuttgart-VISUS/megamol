//
// UnstructuredGridRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 25, 2013
// Author     : scharnkn
//

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "UnstructuredGridRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include "VTKLegacyDataCallUnstructuredGrid.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * protein::UnstructuredGridRenderer::UnstructuredGridRenderer (CTOR)
 */
protein::UnstructuredGridRenderer::UnstructuredGridRenderer(void) : Renderer3DModule (),
    dataCallerSlot( "getData", "Connects the rendering with the data storage"),
    sphereRadSlot("sphereRad", "The sphere radius scale factor") {

    this->dataCallerSlot.SetCompatibleCall<VTKLegacyDataCallUnstructuredGridDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    this->sphereRadSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->sphereRadSlot);
}


/*
 * protein::UnstructuredGridRenderer::~UnstructuredGridRenderer (DTOR)
 */
protein::UnstructuredGridRenderer::~UnstructuredGridRenderer(void)  {
    this->Release ();
}


/*
 * protein::UnstructuredGridRenderer::release
 */
void protein::UnstructuredGridRenderer::release(void) {

}


/*
 * protein::UnstructuredGridRenderer::create
 */
bool protein::UnstructuredGridRenderer::create(void)
{
    if (isExtAvailable("GL_ARB_vertex_program") == 0) {
        return false;
    }
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    // Load sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load cylinder shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cylinder shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::cylinderFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cylinder shader");
        return false;
    }
    try {
        if (!this->cylinderShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create cylinder shader: %s\n", e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * protein::UnstructuredGridRenderer::GetExtents
 */
bool protein::UnstructuredGridRenderer::GetExtents(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    VTKLegacyDataCallUnstructuredGrid *dc =
            this->dataCallerSlot.CallAs<VTKLegacyDataCallUnstructuredGrid>();
    if (dc == NULL ) {
        return false;
    }
    if (!(*dc)(VTKLegacyDataCallUnstructuredGrid::CallForGetExtent)) {
        return false;
    }

    float scale;
    if( !vislib::math::IsEqual(dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = dc->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(dc->FrameCount());

    return true;
}

/*
 * protein::UnstructuredGridRenderer::Render
 */
bool protein::UnstructuredGridRenderer::Render(Call& call) {
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    VTKLegacyDataCallUnstructuredGrid *dc =
            this->dataCallerSlot.CallAs<VTKLegacyDataCallUnstructuredGrid>();
    if (dc == NULL) {
        return false;
    }

    unsigned int cnt;

    dc->SetFrameID(static_cast<int>(callTime));
    if (!(*dc)(SphereDataCall::CallForGetData)) {
        return false;
    }

    float *pos0 = new float[dc->GetNumberOfPoints()*3];
    memcpy(pos0, dc->PeekPoints(), dc->GetNumberOfPoints()*3*sizeof( float));

    if ((static_cast<int>(callTime) + 1) < int(dc->FrameCount())) {
        dc->SetFrameID(static_cast<int>( callTime) + 1);
    } else {
        dc->SetFrameID(0);
    }
    if (!(*dc)(VTKLegacyDataCallUnstructuredGrid::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    float *pos1 = new float[dc->GetNumberOfPoints()*3];
    memcpy(pos1, dc->PeekPoints(), dc->GetNumberOfPoints()*3*sizeof(float));

    float *posInter = new float[dc->GetNumberOfPoints()*4];
    float inter = callTime - static_cast<float>(static_cast<int>( callTime));
    float threshold = vislib::math::Min(dc->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        vislib::math::Min(dc->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
        dc->AccessBoundingBoxes().ObjectSpaceBBox().Depth()))*0.5f;
    for (cnt = 0; cnt < dc->GetNumberOfPoints(); ++cnt ) {
        if (std::sqrt(
                std::pow(pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                std::pow(pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                std::pow(pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
            posInter[4*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
            posInter[4*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
            posInter[4*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
            posInter[4*cnt+3] = this->sphereRadSlot.Param<core::param::FloatParam>()->Value();
        } else if( inter < 0.5 ) {
            posInter[4*cnt+0] = pos0[3*cnt+0];
            posInter[4*cnt+1] = pos0[3*cnt+1];
            posInter[4*cnt+2] = pos0[3*cnt+2];
            posInter[4*cnt+3] = this->sphereRadSlot.Param<core::param::FloatParam>()->Value();
        } else {
            posInter[4*cnt+0] = pos1[3*cnt+0];
            posInter[4*cnt+1] = pos1[3*cnt+1];
            posInter[4*cnt+2] = pos1[3*cnt+2];
            posInter[4*cnt+3] = this->sphereRadSlot.Param<core::param::FloatParam>()->Value();
        }
    }

    glPushMatrix();

    float scale;
    if( !vislib::math::IsEqual(dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / dc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    glScalef( scale, scale, scale);

    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    // enable sphere shader
    this->sphereShader.Enable();
    glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // draw points
    // set vertex and color pointers and draw them
    glVertexPointer(4, GL_FLOAT, 0, posInter);
    //glColorPointer( 3, GL_UNSIGNED_BYTE, 0, sphere->SphereColors() );
    //glColorPointer( 3, GL_FLOAT, 0, this->colors.PeekElements() );
    glDrawArrays(GL_POINTS, 0, (GLsizei)dc->GetNumberOfPoints());
    // disable sphere shader
    //glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    this->sphereShader.Disable();

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    glDisable(GL_DEPTH_TEST);

    glPopMatrix();

    return true;
}
