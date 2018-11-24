/*
 * DirectVolumeRenderer.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#define _USE_MATH_DEFINES 1
#define STOP_SEGMENTATION

#include "DirectVolumeRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/view/AbstractCallRender.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/glverify.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include <GL/glu.h>
#include <math.h>
#include <time.h>

using namespace megamol;
using namespace megamol::stdplugin;

/*
 * DirectVolumeRenderer::DirectVolumeRenderer (CTOR)
 */
volume::DirectVolumeRenderer::DirectVolumeRenderer (void) : Renderer3DModule (),
        volDataCallerSlot ("getData", "Connects the volume rendering with data storage"),
        secRenCallerSlot ("secRen", "Connects the volume rendering with a secondary renderer"),
        volIsoValueParam("volIsoValue", "Isovalue for isosurface rendering"),
        volIsoOpacityParam("volIsoOpacity", "Opacity of isosurface"),
        volClipPlaneFlagParam("volClipPlane", "Enable volume clipping"),
        volClipPlane0NormParam("clipPlane0Norm", "Volume clipping plane 0 normal"),
        volClipPlane0DistParam("clipPlane0Dist", "Volume clipping plane 0 distance"),
        volClipPlaneOpacityParam("clipPlaneOpacity", "Volume clipping plane opacity"),
        opaqRenWorldScaleParam("opaqRenWorldScale", "World space scaling for opaque renderer"),
        toggleVolBBoxParam("toggleVolBBox", "..."),
        togglePbcXParam("togglePbcX", "..."), 
        togglePbcYParam("togglePbcY", "..."), 
        togglePbcZParam("togglePbcZ", "..."),
        volumeTex(0), currentFrameId(-1), volFBO(0), width(0), height(0), volRayTexWidth(0), 
        volRayTexHeight(0), volRayStartTex(0), volRayLengthTex(0), volRayDistTex(0),
        renderIsometric(true), meanDensityValue(0.0f), isoValue(0.5f), 
        volIsoOpacity(0.4f), volClipPlaneFlag(false), volClipPlaneOpacity(0.4f), hashValVol(-1)
{
    // set caller slot for different data calls
    this->volDataCallerSlot.SetCompatibleCall<core::moldyn::VolumeDataCallDescription>();
    this->MakeSlotAvailable (&this->volDataCallerSlot);

    // set renderer caller slot
    this->secRenCallerSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->secRenCallerSlot);

    // --- set up parameters for isovalues ---
    this->volIsoValueParam.SetParameter(new core::param::FloatParam(this->isoValue));
    this->MakeSlotAvailable(&this->volIsoValueParam);

    // --- set up parameter for isosurface opacity ---
    this->volIsoOpacityParam.SetParameter(new core::param::FloatParam(this->volIsoOpacity, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->volIsoOpacityParam);

    // set default clipping plane
    this->volClipPlane.Clear();
    this->volClipPlane.Add(vislib::math::Vector<double, 4>(0.0, 1.0, 0.0, 0.0));

    // --- set up parameter for volume clipping ---
    this->volClipPlaneFlagParam.SetParameter(new core::param::BoolParam(this->volClipPlaneFlag));
    this->MakeSlotAvailable(&this->volClipPlaneFlagParam);

    // --- set up parameter for volume clipping plane normal ---
    vislib::math::Vector<float, 3> cp0n(
        static_cast<float>(this->volClipPlane[0].PeekComponents()[0]), 
        static_cast<float>(this->volClipPlane[0].PeekComponents()[1]), 
        static_cast<float>(this->volClipPlane[0].PeekComponents()[2]));
    this->volClipPlane0NormParam.SetParameter(new core::param::Vector3fParam(cp0n));
    this->MakeSlotAvailable(&this->volClipPlane0NormParam);

    // --- set up parameter for volume clipping plane distance ---
    float d = static_cast<float>(this->volClipPlane[0].PeekComponents()[3]);
    this->volClipPlane0DistParam.SetParameter(new core::param::FloatParam(d, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->volClipPlane0DistParam);

    // --- set up parameter for clipping plane opacity ---
    this->volClipPlaneOpacityParam.SetParameter(new core::param::FloatParam(this->volClipPlaneOpacity, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->volClipPlaneOpacityParam);

    // --- set up parameter for opaque renderer world scale factor ---
    this->opaqRenWorldScaleParam.SetParameter(new core::param::FloatParam(2.0f, 1.0f));
    this->MakeSlotAvailable(&this->opaqRenWorldScaleParam);    

    // Set up parameter for volume clipping
    this->toggleVolBBoxParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->toggleVolBBoxParam);


    this->togglePbcXParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->togglePbcXParam);
    this->togglePbcYParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->togglePbcYParam);
    this->togglePbcZParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->togglePbcZParam);    
}


/*
 * DirectVolumeRenderer::~DirectVolumeRenderer (DTOR)
 */
volume::DirectVolumeRenderer::~DirectVolumeRenderer (void) {
    this->Release ();
}


/*
 * DirectVolumeRenderer::release
 */
void volume::DirectVolumeRenderer::release (void) {

}


/*
 * DirectVolumeRenderer::create
 */
bool volume::DirectVolumeRenderer::create (void) {
    ASSERT(IsAvailable());
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    // Load ray start shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::rayStartVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray start shader", this->ClassName());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::rayStartFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for ray start shader", this->ClassName());
        return false;
    }
    try {
        if (!this->volRayStartShader.Create (vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception ("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to create ray start shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load ray start eye shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::rayStartEyeVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray start eye shader", this->ClassName());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::rayStartEyeFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for ray start eye shader", this->ClassName());
        return false;
    }
    try {
        if (!this->volRayStartEyeShader.Create (vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception ("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to create ray start eye shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load ray length shader (uses same vertex shader as ray start shader)
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::rayStartVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray length shader", this->ClassName());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::rayLengthFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for ray length shader", this->ClassName());
        return false;
    }
    try {
        if (!this->volRayLengthShader.Create (vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception ("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to create ray length shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load volume rendering shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::volumeVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%: Unable to load vertex shader source for volume rendering shader", this->ClassName());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("volumerenderer::std::volumeFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for volume rendering shader", this->ClassName());
        return false;
    }
    try {
        if (!this->volumeShader.Create (vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception ("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%s: Unable to create volume rendering shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * ProteinRenderer::GetExtents
 */
bool volume::DirectVolumeRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    core::moldyn::VolumeDataCall *volume = this->volDataCallerSlot.CallAs<core::moldyn::VolumeDataCall>();

    float xoff, yoff, zoff;
    vislib::math::Cuboid<float> boundingBox;
    vislib::math::Point<float, 3> bbc;

    // try to call the volume data
    if (!(*volume)(core::moldyn::VolumeDataCall::CallForGetExtent)) return false;
    // get bounding box
    boundingBox = volume->BoundingBox();
    // get the pointer to CallRender3D for the secondary renderer 
    core::view::CallRender3D *cr3dSec = this->secRenCallerSlot.CallAs<core::view::CallRender3D>();
    if (cr3dSec) {
        (*cr3dSec)(core::view::AbstractCallRender::FnGetExtents);
        core::BoundingBoxes &secRenBbox = cr3dSec->AccessBoundingBoxes();
        boundingBox.Union(secRenBbox.ObjectSpaceBBox());
    }

    this->unionBBox = boundingBox;
    bbc = boundingBox.CalcCenter();
    this->bboxCenter = bbc;
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    if (!vislib::math::IsEqual(this->unionBBox.LongestEdge(), 0.0f)) { 
        this->scale = 2.0f / this->unionBBox.LongestEdge();
    } else {
        this->scale = 1.0f;
    }

    core::BoundingBoxes &bbox = cr3d->AccessBoundingBoxes();
    bbox.SetObjectSpaceBBox(this->unionBBox);
    bbox.MakeScaledWorld(this->scale);
//    bbox.SetWorldSpaceBBox(
//        (boundingBox.Left() + xoff) * this->scale,
//        (boundingBox.Bottom() + yoff) * this->scale,
//        (boundingBox.Back() + zoff) * this->scale,
//        (boundingBox.Right() + xoff) * this->scale,
//        (boundingBox.Top() + yoff) * this->scale,
//        (boundingBox.Front() + zoff) * this->scale);
    bbox.SetObjectSpaceClipBox(bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox(bbox.WorldSpaceBBox());

    return true;
}

/*
 * DirectVolumeRenderer::Render
 */
bool volume::DirectVolumeRenderer::Render(core::Call& call) {
    using core::moldyn::VolumeDataCall;
    // cast the call to Render3D
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (!cr3d) return false;

    // get pointer to VolumeDataCall
    VolumeDataCall *volume = this->volDataCallerSlot.CallAs<VolumeDataCall>();
    
    // set frame ID and call data
    if (volume) {
        volume->SetFrameID(static_cast<int>(cr3d->Time()));
        if (!(*volume)(VolumeDataCall::CallForGetData)) {
            return false;
        }
    } else {
        return false;
    }

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    // =============== Query Camera View Dimensions ===============
    if (static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height) {
        this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
    }

    // create the fbo, if necessary
    if (!this->opaqueFBO.IsValid()) {
        this->opaqueFBO.Create(this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, 
            vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }
    // resize the fbo, if necessary
    if (this->opaqueFBO.GetWidth() != this->width || this->opaqueFBO.GetHeight() != this->height) {
        this->opaqueFBO.Create(this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, 
            vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }
  
    // disable the output buffer
    cr3d->DisableOutputBuffer();
    // start rendering to the FBO
    this->opaqueFBO.Enable();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    core::view::CallRender3D *cr3dSec = this->secRenCallerSlot.CallAs<core::view::CallRender3D>();
    if (cr3dSec) {

        // Determine revert scale factor
        float scaleRevert;
        if (!vislib::math::IsEqual(cr3dSec->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
            scaleRevert = this->opaqRenWorldScaleParam.Param<core::param::FloatParam>()->Value()/
                cr3dSec->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        } else {
            scaleRevert = 1.0f;
        }
        scaleRevert = 1.0f / scaleRevert;

        // Setup and call secondary renderer
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        vislib::math::Vector<float, 3> trans(this->unionBBox.Width(), this->unionBBox.Height(), 
            this->unionBBox.Depth());
//        trans *= -this->scale*0.5f;
        glTranslatef(cr3dSec->AccessBoundingBoxes().ObjectSpaceBBox().Left(), 
                cr3dSec->AccessBoundingBoxes().ObjectSpaceBBox().Bottom(), 
                cr3dSec->AccessBoundingBoxes().ObjectSpaceBBox().Back());

        glScalef(this->scale, this->scale, this->scale);
    
        *cr3dSec = *cr3d;

        // Revert scaling done by external renderer in advance
        glScalef(scaleRevert, scaleRevert, scaleRevert);

        (*cr3dSec)(core::view::AbstractCallRender::FnRender);

        glPopMatrix();
    }
    // stop rendering to the FBO
    this->opaqueFBO.Disable();
    // re-enable the output buffer
    cr3d->EnableOutputBuffer();

    // =============== Refresh all parameters ===============
    this->ParameterRefresh(cr3d);
    
    unsigned int cpCnt;
    for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
        glClipPlane(GL_CLIP_PLANE0+cpCnt, this->volClipPlane[cpCnt].PeekComponents());
    }

    // =============== Volume Rendering ===============
    bool retval = false;

    // try to start volume rendering using volume data
    if (volume) {
        retval = this->RenderVolumeData(cr3d, volume);
    }
    
    // unlock the current frame
    if (volume) {
        volume->Unlock();
    }

    // Render volume bbox
    if (this->toggleVolBBoxParam.Param<core::param::BoolParam>()->Value())
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glColor3f(1,1,1);
        glDisable(GL_LIGHTING);
        glDisable(GL_BLEND);
        //glEnable(GL_DEPTH_TEST);
        glLineWidth(1);
        glPushMatrix();
        glTranslatef(volume->BoundingBox().Left()*this->scale, 
                volume->BoundingBox().Bottom()*this->scale, 
                volume->BoundingBox().Back()*this->scale);
        this->DrawBoundingBox(volume->BoundingBox());
        glPopMatrix();
    }
    
    return retval;
}


/*
 * Volume rendering using volume data.
 */
bool volume::DirectVolumeRenderer::RenderVolumeData(core::view::CallRender3D *call, core::moldyn::VolumeDataCall *volume) {
    glEnable (GL_DEPTH_TEST);
    glEnable (GL_VERTEX_PROGRAM_POINT_SIZE);

    // test for volume data
    if (volume->FrameCount() == 0)
        return false;

    glPushMatrix();

    // translate scene for volume ray casting
    this->scale = 2.0f / vislib::math::Max(vislib::math::Max(
        volume->BoundingBox().Width(),volume->BoundingBox().Height()),
        volume->BoundingBox().Depth());
    vislib::math::Vector<float, 3> trans(volume->BoundingBox().GetSize().PeekDimension());
    
//    trans *= -this->scale*0.5f;
//    glTranslatef(trans.GetX(), trans.GetY(), trans.GetZ());
        glTranslatef(volume->BoundingBox().Left()*this->scale, 
                volume->BoundingBox().Bottom()*this->scale, 
                volume->BoundingBox().Back()*this->scale);

    // ------------------------------------------------------------
    // --- Volume Rendering                                     ---
    // --- update & render the volume                           ---
    // ------------------------------------------------------------
    if ((static_cast<int>(volume->FrameID()) != this->currentFrameId) ||
            (this->hashValVol != volume->DataHash()) ||
            this->togglePbcXParam.IsDirty() ||
            this->togglePbcYParam.IsDirty() ||
            this->togglePbcZParam.IsDirty()) 
    {
        this->currentFrameId = static_cast<int>(volume->FrameID());
        this->UpdateVolumeTexture(volume);
        CHECK_FOR_OGL_ERROR();

        this->togglePbcXParam.ResetDirty();
        this->togglePbcYParam.ResetDirty();
        this->togglePbcZParam.ResetDirty();
    }

    // reenable second renderer
    this->opaqueFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();
    
    unsigned int cpCnt;
    if (this->volClipPlaneFlag) {
        for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
            glEnable(GL_CLIP_PLANE0+cpCnt);
        }
    }

    this->RenderVolume(volume->BoundingBox());
    CHECK_FOR_OGL_ERROR();
    
    if (this->volClipPlaneFlag) {
        for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
            glDisable(GL_CLIP_PLANE0+cpCnt);
        }
    }

    glDisable (GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable (GL_DEPTH_TEST);
    
    glPopMatrix();

    return true;
}

/*
 * refresh parameters
 */
void volume::DirectVolumeRenderer::ParameterRefresh(core::view::CallRender3D *call) {
    
    // volume parameters
    if (this->volIsoValueParam.IsDirty()) {
        this->isoValue = this->volIsoValueParam.Param<core::param::FloatParam>()->Value();
        this->volIsoValueParam.ResetDirty();
    }
    if (this->volIsoOpacityParam.IsDirty()) {
        this->volIsoOpacity = this->volIsoOpacityParam.Param<core::param::FloatParam>()->Value();
        this->volIsoOpacityParam.ResetDirty();
    }
    if (this->volClipPlaneFlagParam.IsDirty()) {
        this->volClipPlaneFlag = this->volClipPlaneFlagParam.Param<core::param::BoolParam>()->Value();
        this->volClipPlaneFlagParam.ResetDirty();
    }

    // get current clip plane normal
    vislib::math::Vector<float, 3> cp0n(
        (float)this->volClipPlane[0].PeekComponents()[0],
        (float)this->volClipPlane[0].PeekComponents()[1],
        (float)this->volClipPlane[0].PeekComponents()[2]);
    // get current clip plane distance
    float cp0d = (float)this->volClipPlane[0].PeekComponents()[3];

    // check clip plane normal parameter
    if (this->volClipPlane0NormParam.IsDirty()) {
        // overwrite clip plane normal
        cp0n = this->volClipPlane0NormParam.Param<core::param::Vector3fParam>()->Value();
        // normalize clip plane normal, if necessary and set normalized clip plane normal to parameter
        if (!vislib::math::IsEqual<float>(cp0n.Length(), 1.0f)) {
            cp0n.Normalise();
            this->volClipPlane0NormParam.Param<core::param::Vector3fParam>()->SetValue(cp0n);
        }
        this->volClipPlane0NormParam.ResetDirty();
    }
    // compute maximum extent
    vislib::math::Cuboid<float> bbox(call->AccessBoundingBoxes().WorldSpaceBBox());
    vislib::math::Vector<float, 3> tmpVec;
    float d, maxD, minD;
    // 1
    tmpVec.Set(bbox.GetLeftBottomBack().X(), bbox.GetLeftBottomBack().Y(), bbox.GetLeftBottomBack().Z());
    maxD = minD = cp0n.Dot(tmpVec);
    // 2
    tmpVec.Set(bbox.GetRightBottomBack().X(), bbox.GetRightBottomBack().Y(), bbox.GetRightBottomBack().Z());
    d = cp0n.Dot(tmpVec);
    if (minD > d) minD = d;
    if (maxD < d) maxD = d;
    // 3
    tmpVec.Set(bbox.GetLeftBottomFront().X(), bbox.GetLeftBottomFront().Y(), bbox.GetLeftBottomFront().Z());
    d = cp0n.Dot(tmpVec);
    if (minD > d) minD = d;
    if (maxD < d) maxD = d;
    // 4
    tmpVec.Set(bbox.GetRightBottomFront().X(), bbox.GetRightBottomFront().Y(), bbox.GetRightBottomFront().Z());
    d = cp0n.Dot(tmpVec);
    if (minD > d) minD = d;
    if (maxD < d) maxD = d;
    // 5
    tmpVec.Set(bbox.GetLeftTopBack().X(), bbox.GetLeftTopBack().Y(), bbox.GetLeftTopBack().Z());
    d = cp0n.Dot(tmpVec);
    if (minD > d) minD = d;
    if (maxD < d) maxD = d;
    // 6
    tmpVec.Set(bbox.GetRightTopBack().X(), bbox.GetRightTopBack().Y(), bbox.GetRightTopBack().Z());
    d = cp0n.Dot(tmpVec);
    if (minD > d) minD = d;
    if (maxD < d) maxD = d;
    // 7
    tmpVec.Set(bbox.GetLeftTopFront().X(), bbox.GetLeftTopFront().Y(), bbox.GetLeftTopFront().Z());
    d = cp0n.Dot(tmpVec);
    if (minD > d) minD = d;
    if (maxD < d) maxD = d;
    // 8
    tmpVec.Set(bbox.GetRightTopFront().X(), bbox.GetRightTopFront().Y(), bbox.GetRightTopFront().Z());
    d = cp0n.Dot(tmpVec);
    if (minD > d) minD = d;
    if (maxD < d) maxD = d;
    // check clip plane distance
    if (this->volClipPlane0DistParam.IsDirty()) {
        cp0d = this->volClipPlane0DistParam.Param<core::param::FloatParam>()->Value();
        cp0d = minD + (maxD - minD) * cp0d;
        this->volClipPlane0DistParam.ResetDirty();
    }    

    // set clip plane normal and distance to current clip plane
    this->volClipPlane[0].Set(cp0n.X(), cp0n.Y(), cp0n.Z(), cp0d);

    // check clip plane opacity parameter
    if (this->volClipPlaneOpacityParam.IsDirty()) {
        this->volClipPlaneOpacity = this->volClipPlaneOpacityParam.Param<core::param::FloatParam>()->Value();
        this->volClipPlaneOpacityParam.ResetDirty();
    }

}

/*
 * Create a volume containing the voxel map
 */
void volume::DirectVolumeRenderer::UpdateVolumeTexture(const core::moldyn::VolumeDataCall *volume) {
    // generate volume, if necessary
    if (!glIsTexture(this->volumeTex)) {
        glGenTextures(1, &this->volumeTex);
    }
    // set voxel map to volume texture
    glBindTexture(GL_TEXTURE_3D, this->volumeTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, 
        volume->VolumeDimension().GetWidth(), 
        volume->VolumeDimension().GetHeight(), 
        volume->VolumeDimension().GetDepth(), 0, GL_LUMINANCE, GL_FLOAT, 
        volume->VoxelMap());

    GLint param = GL_LINEAR;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);
//    GLint mode = GL_CLAMP_TO_EDGE;
//    GLint mode = GL_REPEAT;

    if (this->togglePbcXParam.Param<core::param::BoolParam>()->Value())
    {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }
        
    if (this->togglePbcYParam.Param<core::param::BoolParam>()->Value())
    {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    if (this->togglePbcZParam.Param<core::param::BoolParam>()->Value())
    {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }    
    glBindTexture(GL_TEXTURE_3D, 0);
    CHECK_FOR_OGL_ERROR();

    // generate FBO, if necessary
    if (!glIsFramebufferEXT(this->volFBO)) {
        glGenFramebuffersEXT(1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }
    CHECK_FOR_OGL_ERROR();
    
    // scale[i] = 1/extent[i] --- extent = size of the bbox
    this->volScale[0] = 1.0f / (volume->BoundingBox().Width() * this->scale);
    this->volScale[1] = 1.0f / (volume->BoundingBox().Height() * this->scale);
    this->volScale[2] = 1.0f / (volume->BoundingBox().Depth() * this->scale);
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];

    // set volume size
    this->volumeSize = vislib::math::Max<unsigned int>(volume->VolumeDimension().Depth(),
        vislib::math::Max<unsigned int>(volume->VolumeDimension().Height(), volume->VolumeDimension().Width()));
}

/*
 * draw the volume
 */
void volume::DirectVolumeRenderer::RenderVolume(vislib::math::Cuboid<float> boundingbox) {

//    printf("bbox %f %f %f -- %f %f %f\n", 
//            boundingbox.Left(),
//            boundingbox.Bottom(),
//            boundingbox.Back(),
//            boundingbox.Right(),
//            boundingbox.Top(),
//            boundingbox.Front());
    const float stepWidth = 1.0f/ (2.0f * float(this->volumeSize));

    glDisable(GL_BLEND);

    GLint prevFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

    this->RayParamTextures(boundingbox);
    CHECK_FOR_OGL_ERROR();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, prevFBO);

    this->volumeShader.Enable();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUniform4fv(this->volumeShader.ParameterLocation("scaleVol"), 1, this->volScale);
    glUniform4fv(this->volumeShader.ParameterLocation("scaleVolInv"), 1, this->volScaleInv);
    glUniform1f(this->volumeShader.ParameterLocation("stepSize"), stepWidth);
    glUniform1f(this->volumeShader.ParameterLocation("alphaCorrection"), this->volumeSize/256.0f);
    glUniform1i(this->volumeShader.ParameterLocation("numIterations"), 255);
    glUniform2f(this->volumeShader.ParameterLocation("screenResInv"), 1.0f/ float(this->width), 1.0f/ float(this->height));
    // bind depth texture
    glUniform1i(this->volumeShader.ParameterLocation("volumeSampler"), 0);
    glUniform1i(this->volumeShader.ParameterLocation("transferRGBASampler"), 1);
    glUniform1i(this->volumeShader.ParameterLocation("rayStartSampler"), 2);
    glUniform1i(this->volumeShader.ParameterLocation("rayLengthSampler"), 3);

    glUniform1f(this->volumeShader.ParameterLocation("isoValue"), this->isoValue);
    glUniform1f(this->volumeShader.ParameterLocation("isoOpacity"), this->volIsoOpacity);
    glUniform1f(this->volumeShader.ParameterLocation("clipPlaneOpacity"), this->volClipPlaneOpacity);

    // transfer function
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, 0);
    // ray start positions
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, this->volRayStartTex);
    // ray direction and length
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, this->volRayLengthTex);

    // volume texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, this->volumeTex);
    CHECK_FOR_OGL_ERROR();

    // draw a screen-filling quad
    glRectf(-1.0f, -1.0f, 1.0f, 1.0f);
    CHECK_FOR_OGL_ERROR();

    this->volumeShader.Disable();

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    CHECK_FOR_OGL_ERROR();

    // restore depth buffer
    //glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->opaqueFBO.GetDepthTextureID(), 0);
    CHECK_FOR_OGL_ERROR();
}

/*
 * write the parameters of the ray to the textures
 */
void volume::DirectVolumeRenderer::RayParamTextures(vislib::math::Cuboid<float> boundingbox) {

    GLint param = GL_NEAREST;
    GLint mode = GL_CLAMP_TO_EDGE;

    // generate / resize ray start texture for volume ray casting
    if (!glIsTexture(this->volRayStartTex)) {
        glGenTextures(1, &this->volRayStartTex);
        glBindTexture(GL_TEXTURE_2D, this->volRayStartTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    } else if (this->width != this->volRayTexWidth || this->height != this->volRayTexHeight) {
        glBindTexture(GL_TEXTURE_2D, this->volRayStartTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    }
    // generate / resize ray length texture for volume ray casting
    if (!glIsTexture(this->volRayLengthTex)) {
        glGenTextures(1, &this->volRayLengthTex);
        glBindTexture(GL_TEXTURE_2D, this->volRayLengthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    } else if (this->width != this->volRayTexWidth || this->height != this->volRayTexHeight) {
        glBindTexture(GL_TEXTURE_2D, this->volRayLengthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    }
    // generate / resize ray distance texture for volume ray casting
    if (!glIsTexture(this->volRayDistTex)) {
        glGenTextures(1, &this->volRayDistTex);
        glBindTexture(GL_TEXTURE_2D, this->volRayDistTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    } else if (this->width != this->volRayTexWidth || this->height != this->volRayTexHeight) {
        glBindTexture(GL_TEXTURE_2D, this->volRayDistTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    }
    CHECK_FOR_OGL_ERROR();
    glBindTexture(GL_TEXTURE_2D, 0);
    // set vol ray dimensions
    this->volRayTexWidth = this->width;
    this->volRayTexHeight = this->height;

    GLuint db[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->volFBO);
    CHECK_FOR_OGL_ERROR();

    // -------- ray start ------------
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, this->volRayStartTex, 0);
    CHECK_FOR_OGL_ERROR();
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, this->volRayDistTex, 0);
    CHECK_FOR_OGL_ERROR();

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    // draw to two rendertargets (the second does not need to be cleared)
    glDrawBuffers(2, db);

    // draw near clip plane
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glColor4f(0.0f, 0.0f, 0.0f, 0.0f);

    // the shader transforms camera coords back to object space
    this->volRayStartEyeShader.Enable();

    float u = this->cameraInfo->NearClip() * tan(this->cameraInfo->ApertureAngle() * float(vislib::math::PI_DOUBLE) / 360.0f);
    float r = (this->width / this->height)*u;

    glBegin(GL_QUADS);
        //glVertex3f(-r, -u, -_nearClip);
        glVertex3f(-r, -u, -this->cameraInfo->NearClip());
        glVertex3f(r, -u, -this->cameraInfo->NearClip());
        glVertex3f(r,  u, -this->cameraInfo->NearClip());
        glVertex3f(-r,  u, -this->cameraInfo->NearClip());
    glEnd();
    CHECK_FOR_OGL_ERROR();

    this->volRayStartEyeShader.Disable();

    glDrawBuffers(1, db);

    this->volRayStartShader.Enable();

    // ------------ !useSphere && iso -------------
    vislib::math::Vector<float, 3> trans(boundingbox.GetSize().PeekDimension());
    trans *= this->scale*0.5f;
    if (this->renderIsometric) {
        glUniform3f(this->volRayStartShader.ParameterLocation("translate"), 
            0.0f, 0.0f, 0.0f);
    } else {
        glUniform3fv(this->volRayStartShader.ParameterLocation("translate"), 
            1, trans.PeekComponents());
    }

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);

    glEnable(GL_CULL_FACE);

    // draw nearest backfaces
    glCullFace(GL_FRONT);

    // draw bBox
    this->DrawBoundingBox(boundingbox);

    // draw nearest frontfaces
    glCullFace(GL_BACK);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    CHECK_FOR_OGL_ERROR();
    
    // draw bBox
    this->DrawBoundingBox(boundingbox);

    this->volRayStartShader.Disable();

    // --------------------------------
    // -------- ray length ------------
    // --------------------------------
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->volFBO);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, this->volRayLengthTex, 0);
    CHECK_FOR_OGL_ERROR();

    // get clear color
    float clearCol[4];
    glGetFloatv(GL_COLOR_CLEAR_VALUE, clearCol);
    glClearColor(0, 0, 0, 0);
    glClearDepth(0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearDepth(1.0f);
    glDrawBuffers(2, db);
    glClearColor(clearCol[0], clearCol[1], clearCol[2], clearCol[3]);

    this->volRayLengthShader.Enable();

    glUniform1i(this->volRayLengthShader.ParameterLocation("sourceTex"), 0);
    glUniform1i(this->volRayLengthShader.ParameterLocation("depthTex"), 1);
    glUniform2f(this->volRayLengthShader.ParameterLocation("screenResInv"),
        1.0f / float(this->width), 1.0f / float(this->height));
    glUniform2f(this->volRayLengthShader.ParameterLocation("zNearFar"),
        this->cameraInfo->NearClip(), this->cameraInfo->FarClip());

    if (this->renderIsometric) {
        glUniform3f(this->volRayLengthShader.ParameterLocation("translate"), 
            0.0f, 0.0f, 0.0f);
    } else {
        glUniform3fv(this->volRayLengthShader.ParameterLocation("translate"), 
            1, trans.PeekComponents());
    }
    glUniform1f(this->volRayLengthShader.ParameterLocation("scale"),
        this->scale);

    glActiveTexture(GL_TEXTURE1);
    this->opaqueFBO.BindDepthTexture();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->volRayStartTex);

    // draw farthest backfaces
    glCullFace(GL_FRONT);
    glDepthFunc(GL_GREATER);

    // draw bBox
    this->DrawBoundingBox(boundingbox);

    this->volRayLengthShader.Disable();

    glDrawBuffers(1, db);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, 0, 0);

    glDepthFunc(GL_LESS);
    glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);
}

/*
 * Draw the bounding box.
 */
void volume::DirectVolumeRenderer::DrawBoundingBox(vislib::math::Cuboid<float> boundingbox) {

    vislib::math::Vector<float, 3> position(boundingbox.GetSize().PeekDimension());
    position *= this->scale;


    glBegin(GL_QUADS);
    {
        // back side
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, position.GetY(), 0.0f);
        glVertex3f(position.GetX(), position.GetY(), 0.0f);
        glVertex3f(position.GetX(), 0.0f, 0.0f);

        // front side
        glVertex3f(0.0f, 0.0f, position.GetZ());
        glVertex3f(position.GetX(), 0.0f, position.GetZ());
        glVertex3f(position.GetX(), position.GetY(), position.GetZ());
        glVertex3f(0.0f, position.GetY(), position.GetZ());

        // top side
        glVertex3f(0.0f, position.GetY(), 0.0f);
        glVertex3f(0.0f, position.GetY(), position.GetZ());
        glVertex3f(position.GetX(), position.GetY(), position.GetZ());
        glVertex3f(position.GetX(), position.GetY(), 0.0f);

        // bottom side
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(position.GetX(), 0.0f, 0.0f);
        glVertex3f(position.GetX(), 0.0f, position.GetZ());
        glVertex3f(0.0f, 0.0f, position.GetZ());

        // left side
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, position.GetZ());
        glVertex3f(0.0f, position.GetY(), position.GetZ());
        glVertex3f(0.0f, position.GetY(), 0.0f);

        // right side
        glVertex3f(position.GetX(), 0.0f, 0.0f);
        glVertex3f(position.GetX(), position.GetY(), 0.0f);
        glVertex3f(position.GetX(), position.GetY(), position.GetZ());
        glVertex3f(position.GetX(), 0.0f, position.GetZ());
    }
    glEnd();
    CHECK_FOR_OGL_ERROR();

    // draw slice for volume clipping
    if (this->volClipPlaneFlag)
        this->drawClippedPolygon(boundingbox);
}

/*
 * draw the clipped polygon for correct clip plane rendering
 */
void volume::DirectVolumeRenderer::drawClippedPolygon(vislib::math::Cuboid<float> boundingbox) {
    if (!this->volClipPlaneFlag)
        return;

    vislib::math::Vector<float, 3> position(boundingbox.GetSize().PeekDimension());
    position *= this->scale;

    // check for each clip plane
    float vcpd;
    for (int i = 0; i < static_cast<int>(this->volClipPlane.Count()); ++i) {
        slices.setupSingleSlice(this->volClipPlane[i].PeekComponents(), position.PeekComponents());
        float d = 0.0f;
        vcpd = static_cast<float>(this->volClipPlane[i].PeekComponents()[3]);
        glBegin(GL_TRIANGLE_FAN);
        slices.drawSingleSlice(-(-d + vcpd - 0.0001f));
        glEnd();
    }
}

