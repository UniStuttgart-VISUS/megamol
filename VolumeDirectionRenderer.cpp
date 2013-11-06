/*
 * VolumeDirectionRenderer.cpp
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "VolumeDirectionRenderer.h"
#include "CoreInstance.h"
#include "utility/ShaderSourceFactory.h"
#include "utility/ColourParser.h"
#include "param/FloatParam.h"
#include "param/IntParam.h"
#include "param/BoolParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include <GL/glu.h>
#include <omp.h>



using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * protein::VolumeDirectionRenderer::VolumeDirectionRenderer (CTOR)
 */
VolumeDirectionRenderer::VolumeDirectionRenderer(void) : Renderer3DModuleDS (),
    vtiDataCallerSlot("getData", "Connects the arrow rendering with volume data storage"),
    xResParam("numArrXAxis", "Number of arrows along X axis"),
    yResParam("numArrYAxis", "Number of arrows along Y axis"),
    zResParam("numArrZAxis", "Number of arrows along Z axis"),
    arrowCount(0), triggerArrowComputation(true)
{
    this->vtiDataCallerSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable( &this->vtiDataCallerSlot);
    
    this->xRes = this->yRes = this->zRes = 5;
    this->xResParam.SetParameter(new param::IntParam(this->xRes, 0));
    this->MakeSlotAvailable( &this->xResParam);
    this->yResParam.SetParameter(new param::IntParam(this->yRes, 0));
    this->MakeSlotAvailable( &this->yResParam);
    this->zResParam.SetParameter(new param::IntParam(this->zRes, 0));
    this->MakeSlotAvailable( &this->zResParam);
}


/*
 * protein::VolumeDirectionRenderer::~VolumeDirectionRenderer (DTOR)
 */
VolumeDirectionRenderer::~VolumeDirectionRenderer(void)  {
    this->Release();
}


/*
 * protein::VolumeDirectionRenderer::release
 */
void VolumeDirectionRenderer::release(void) {
    this->arrowShader.Release();
}


/*
 * protein::VolumeDirectionRenderer::create
 */
bool VolumeDirectionRenderer::create(void) {
    if( !glh_init_extensions( "GL_VERSION_2_0") )
        return false;

    if ( !vislib::graphics::gl::GLSLShader::InitialiseExtensions() )
        return false;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;
    
    // Load alternative arrow shader (uses geometry shader)
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("arrow::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for arrow shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("arrow::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for arrow shader");
        return false;
    }
    this->arrowShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());

    return true;
}


/*
 * protein::VolumeDirectionRenderer::GetCapabilities
 */
bool VolumeDirectionRenderer::GetCapabilities(Call& call) {
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::AbstractCallRender3D::CAP_RENDER
            | view::AbstractCallRender3D::CAP_LIGHTING
            | view::AbstractCallRender3D::CAP_ANIMATION );

    return true;
}


/*
 * protein::VolumeDirectionRenderer::GetExtents
 */
bool VolumeDirectionRenderer::GetExtents(Call& call) {
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    VTIDataCall *vti = this->vtiDataCallerSlot.CallAs<VTIDataCall>();
    if( vti == NULL ) return false;
    if (!(*vti)(VTIDataCall::CallForGetExtent)) return false;

    float scale;
    if( !vislib::math::IsEqual( vti->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / vti->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = vti->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld( scale);
    cr3d->SetTimeFramesCount( vti->FrameCount());

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::VolumeDirectionRenderer::Render
 */
bool VolumeDirectionRenderer::Render(Call& call) {
    // cast the call to Render3D
    view::AbstractCallRender3D *cr = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if( cr == NULL ) return false;

    // get camera information
    this->cameraInfo = cr->GetCameraParameters();

    float callTime = cr->Time();

    // get pointer to MolecularDataCall
    VTIDataCall *vti = this->vtiDataCallerSlot.CallAs<VTIDataCall>();
    if( vti == NULL) return false;
    
    // set call time
    vti->SetCalltime(callTime);
    // set frame ID
    vti->SetFrameID(static_cast<int>( callTime));
    // try to call for data
    if (!(*vti)(VTIDataCall::CallForGetData)) return false;
    
    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( vti->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / vti->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

    // ---------- update parameters ----------
    this->UpdateParameters(vti);
    
    // ---------- prepare data ---------- 
    if( this->triggerArrowComputation ) {
        arrowCount = xRes * yRes * zRes;
        this->vertexArray.SetCount( arrowCount * 4);
        this->colorArray.SetCount( arrowCount * 3);
        this->dirArray.SetCount( arrowCount * 3);
        
        float deltaX = vti->AccessBoundingBoxes().ObjectSpaceBBox().Width() / static_cast<float>(xRes+1);
        float deltaY = vti->AccessBoundingBoxes().ObjectSpaceBBox().Height() / static_cast<float>(yRes+1);
        float deltaZ = vti->AccessBoundingBoxes().ObjectSpaceBBox().Depth() / static_cast<float>(zRes+1);
        
        float xPos = vti->AccessBoundingBoxes().ObjectSpaceBBox().Left();
        float yPos = vti->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
        float zPos = vti->AccessBoundingBoxes().ObjectSpaceBBox().Back();

        unsigned int numPieces = vti->GetNumberOfPieces();
        unsigned int numArrays = vti->GetArrayCntOfPiecePointData(0);
        unsigned int arr0Size = vti->GetPiecePointArraySize(0, 0); 
        unsigned int arr1Size = vti->GetPiecePointArraySize(1, 0);
        unsigned int arr0Type = vti->GetPiecePointArrayType(0, 0);
        unsigned int arr1Type = vti->GetPiecePointArrayType(1, 0);
        unsigned int arr0NC = vti->GetPointDataArrayNumberOfComponents(0, 0);
        unsigned int arr1NC = vti->GetPointDataArrayNumberOfComponents(1, 0);
        const float *arr0data = (const float*)(vti->GetPointDataByIdx(0, 0));
        const float *arr1data = (const float*)(vti->GetPointDataByIdx(1, 0));

        unsigned int idx = 0;
        for( unsigned int xIdx = 0; xIdx < xRes; xIdx++) {
            xPos += deltaX;
            for( unsigned int yIdx = 0; yIdx < yRes; yIdx++) {
                yPos += deltaY;
                for( unsigned int zIdx = 0; zIdx < zRes; zIdx++) {
                    zPos += deltaZ;
                    // set position
                    this->vertexArray[idx*4+0] = xPos;
                    this->vertexArray[idx*4+1] = yPos;
                    this->vertexArray[idx*4+2] = zPos;
                    this->vertexArray[idx*4+3] = 1.0f;
                    // set color
                    this->colorArray[idx*3+0] = 1.0f;
                    this->colorArray[idx*3+1] = 0.7f;
                    this->colorArray[idx*3+2] = 0.0f;
                    // set direction
                    this->dirArray[idx*3+0] = 0.0f;
                    this->dirArray[idx*3+1] = 0.0f;
                    this->dirArray[idx*3+2] = 2.0f;
                    idx++;
                }
                zPos = vti->AccessBoundingBoxes().ObjectSpaceBBox().Back();
            }
            yPos = vti->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
        }


        this->triggerArrowComputation = false;
    }

    // ---------- render ---------
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    this->arrowShader.Enable();

    glUniform4fvARB(this->arrowShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->arrowShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(this->arrowShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(this->arrowShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    //this->arrowShader.SetParameter("lengthScale", lengthScale);
    this->arrowShader.SetParameter("lengthScale", 1.0f);
    //this->arrowShader.SetParameter("lengthFilter", lengthFilter);
    this->arrowShader.SetParameter("lengthFilter", 0.0f);

    unsigned int cial = glGetAttribLocationARB(this->arrowShader, "colIdx");
    unsigned int tpal = glGetAttribLocationARB(this->arrowShader, "dir");
    
    //glUniform4fARB(this->arrowShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
    glUniform4fARB(this->arrowShader.ParameterLocation("inConsts1"), -1.0f, 0.0f, 0.0f, 0.0f);

    // colour
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_FLOAT, 0, this->colorArray.PeekElements());

    // radius and position
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, this->vertexArray.PeekElements());

    // direction
    glEnableVertexAttribArrayARB(tpal);
    glVertexAttribPointerARB(tpal, 3, GL_FLOAT, GL_FALSE, 0, this->dirArray.PeekElements());

    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(this->arrowCount));

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableVertexAttribArrayARB(cial);
    glDisableVertexAttribArrayARB(tpal);

    this->arrowShader.Disable();
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    
    // unlock the current frame
    vti->Unlock();
    return true;
}


/*
 * update parameters
 */
void VolumeDirectionRenderer::UpdateParameters(const VTIDataCall *vti) {
    if( this->xResParam.IsDirty() ) {
        this->xRes = this->xResParam.Param<param::IntParam>()->Value();
        this->xResParam.ResetDirty();
        this->triggerArrowComputation = true;
    }
    if( this->yResParam.IsDirty() ) {
        this->yRes = this->yResParam.Param<param::IntParam>()->Value();
        this->yResParam.ResetDirty();
        this->triggerArrowComputation = true;
    }
    if( this->zResParam.IsDirty() ) {
        this->zRes = this->zResParam.Param<param::IntParam>()->Value();
        this->zResParam.ResetDirty();
        this->triggerArrowComputation = true;
    }
}
