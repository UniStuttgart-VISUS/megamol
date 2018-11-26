/*
 * SphereRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "SphereRenderer.h"
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

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max


/*
 * protein::SphereRenderer::SphereRenderer (CTOR)
 */
protein::SphereRenderer::SphereRenderer(void) : Renderer3DModule (), 
    sphereDataCallerSlot( "getData", "Connects the sphere rendering with data storage"),
    coloringModeParam( "coloringMode", "Coloring Mode"),
    minValueParam( "minValue", "Minimum vlue for gradient coloring"),
    maxValueParam( "maxValue", "Maximum vlue for gradient coloring"){
    this->sphereDataCallerSlot.SetCompatibleCall<SphereDataCallDescription>();
    this->MakeSlotAvailable( &this->sphereDataCallerSlot);

    
    // --- set the coloring mode ---
    param::EnumParam *cm = new param::EnumParam(int(COLOR_TYPE));
    cm->SetTypePair( COLOR_TYPE, "Type");
    cm->SetTypePair( COLOR_CHARGE, "Charge");
    this->coloringModeParam << cm;
    this->MakeSlotAvailable( &this->coloringModeParam);

    // --- set the min value for gradient coloring ---
    this->minValueParam.SetParameter(new param::FloatParam(-1.0f));
    this->MakeSlotAvailable( &this->minValueParam);
    // --- set the max value for gradient coloring ---
    this->maxValueParam.SetParameter(new param::FloatParam( 1.0f));
    this->MakeSlotAvailable( &this->maxValueParam);
}


/*
 * protein::SphereRenderer::~SphereRenderer (DTOR)
 */
protein::SphereRenderer::~SphereRenderer(void)  {
    this->Release ();
}


/*
 * protein::SphereRenderer::release
 */
void protein::SphereRenderer::release(void) {

}


/*
 * protein::SphereRenderer::create
 */
bool protein::SphereRenderer::create(void)
{
    if (!isExtAvailable( "GL_ARB_vertex_program") ) {
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
 * protein::SphereRenderer::GetExtents
 */
bool protein::SphereRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    SphereDataCall *sphere = this->sphereDataCallerSlot.CallAs<SphereDataCall>();
    if( sphere == NULL ) return false;
    if (!(*sphere)(SphereDataCall::CallForGetExtent)) return false;

    float scale;
    if( !vislib::math::IsEqual( sphere->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / sphere->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = sphere->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld( scale);
    cr3d->SetTimeFramesCount( sphere->FrameCount());

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::SphereRenderer::Render
 */
bool protein::SphereRenderer::Render(Call& call)
{
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    SphereDataCall *sphere = this->sphereDataCallerSlot.CallAs<SphereDataCall>();
    if( sphere == NULL) return false;

    unsigned int cnt;

    sphere->SetFrameID(static_cast<int>( callTime));
    if (!(*sphere)(SphereDataCall::CallForGetData)) return false;

    float *pos0 = new float[sphere->SphereCount() * 4];
    memcpy( pos0, sphere->Spheres(), sphere->SphereCount() * 4 * sizeof( float));

    if( ( static_cast<int>( callTime) + 1) < int(sphere->FrameCount()) ) 
        sphere->SetFrameID(static_cast<int>( callTime) + 1);
    else
        sphere->SetFrameID( 0);
    if (!(*sphere)(SphereDataCall::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    float *pos1 = new float[sphere->SphereCount() * 4];
    memcpy( pos1, sphere->Spheres(), sphere->SphereCount() * 4 * sizeof( float));

    float *posInter = new float[sphere->SphereCount() * 4];
    float inter = callTime - static_cast<float>(static_cast<int>( callTime));
    float threshold = vislib::math::Min( sphere->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        vislib::math::Min( sphere->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
        sphere->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.5f;
    for( cnt = 0; cnt < sphere->SphereCount(); ++cnt ) {
        if( std::sqrt( std::pow( pos0[4*cnt+0] - pos1[4*cnt+0], 2) +
                std::pow( pos0[4*cnt+1] - pos1[4*cnt+1], 2) +
                std::pow( pos0[4*cnt+2] - pos1[4*cnt+2], 2) ) < threshold ) {
            posInter[4*cnt+0] = (1.0f - inter) * pos0[4*cnt+0] + inter * pos1[4*cnt+0];
            posInter[4*cnt+1] = (1.0f - inter) * pos0[4*cnt+1] + inter * pos1[4*cnt+1];
            posInter[4*cnt+2] = (1.0f - inter) * pos0[4*cnt+2] + inter * pos1[4*cnt+2];
            posInter[4*cnt+3] = (1.0f - inter) * pos0[4*cnt+3] + inter * pos1[4*cnt+3];
        } else if( inter < 0.5 ) {
            posInter[4*cnt+0] = pos0[4*cnt+0];
            posInter[4*cnt+1] = pos0[4*cnt+1];
            posInter[4*cnt+2] = pos0[4*cnt+2];
            posInter[4*cnt+3] = pos0[4*cnt+3];
        } else {
            posInter[4*cnt+0] = pos1[4*cnt+0];
            posInter[4*cnt+1] = pos1[4*cnt+1];
            posInter[4*cnt+2] = pos1[4*cnt+2];
            posInter[4*cnt+3] = pos1[4*cnt+3];
        }
    }

    // compute colors
    this->ComputeColors( sphere, 
        static_cast<ColoringMode>(int(this->coloringModeParam.Param<param::EnumParam>()->Value())));

    // TODO: ---------- render ----------

    // -- draw  --
    glPushMatrix();

    float scale;
    if( !vislib::math::IsEqual( sphere->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / sphere->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
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
    glEnableClientState(GL_COLOR_ARRAY);
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // draw points
    // set vertex and color pointers and draw them
    glVertexPointer( 4, GL_FLOAT, 0, posInter);
    //glColorPointer( 3, GL_UNSIGNED_BYTE, 0, sphere->SphereColors() );
    glColorPointer( 3, GL_FLOAT, 0, this->colors.PeekElements() );
    glDrawArrays( GL_POINTS, 0, sphere->SphereCount());
    // disable sphere shader
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    this->sphereShader.Disable();

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    glDisable(GL_DEPTH_TEST);

    glPopMatrix();

    return true;
}


/*
 * Compute the color array.
 */
void SphereRenderer::ComputeColors( const SphereDataCall *sphere, ColoringMode mode) {
    unsigned int cnt;
    // resize color array
    this->colors.SetCount( sphere->SphereCount() * 3);
    // check color mode
    if( mode == COLOR_TYPE ) {
        for( cnt = 0; cnt < sphere->SphereCount(); ++cnt ) {
            this->colors[3*cnt+0] = float(sphere->SphereColors()[3*cnt+0]) / 255.0f;
            this->colors[3*cnt+1] = float(sphere->SphereColors()[3*cnt+1]) / 255.0f;
            this->colors[3*cnt+2] = float(sphere->SphereColors()[3*cnt+2]) / 255.0f;
        }
    } else if( mode == COLOR_CHARGE ) {
        // set charge colors
        vislib::math::Vector<float, 3> colMax( 250.0f/255.0f,  94.0f/255.0f,  82.0f/255.0f);
        vislib::math::Vector<float, 3> colMid( 250.0f/255.0f, 250.0f/255.0f, 250.0f/255.0f);
        vislib::math::Vector<float, 3> colMin(  37.0f/255.0f, 136.0f/255.0f, 195.0f/255.0f);
        vislib::math::Vector<float, 3> col;
        // get charge range
        //float min( sphere->MinimumCharge() );
        float min( this->minValueParam.Param<param::FloatParam>()->Value());
        //float max( sphere->MaximumCharge() );
        float max( this->maxValueParam.Param<param::FloatParam>()->Value());
        float mid( ( max - min)/2.0f + min );
        float val;
        // loop over all spheres
        for( cnt = 0; cnt < sphere->SphereCount(); ++cnt ) {
            if( min == max ) {
                this->colors[3*cnt+0] = colMid.X();
                this->colors[3*cnt+1] = colMid.Y();
                this->colors[3*cnt+2] = colMid.Z();
                continue;
            }
            // get charge value
            val = sphere->SphereCharges()[cnt];
            if( val > max ) val = max;
            else if( val < min ) val = min;
            // assign color
            if( val < mid ) {
                // below middle value --> blend between min and mid color
                col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( val - min );
                this->colors[3*cnt+0] = col.X();
                this->colors[3*cnt+1] = col.Y();
                this->colors[3*cnt+2] = col.Z();
            } else if( val > mid ) {
                // above middle value --> blend between max and mid color
                col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( val - mid );
                this->colors[3*cnt+0] = col.X();
                this->colors[3*cnt+1] = col.Y();
                this->colors[3*cnt+2] = col.Z();
            } else {
                // middle value --> assign mid color
                this->colors[3*cnt+0] = colMid.X();
                this->colors[3*cnt+1] = colMid.Y();
                this->colors[3*cnt+2] = colMid.Z();
            }
        }
    }
}
