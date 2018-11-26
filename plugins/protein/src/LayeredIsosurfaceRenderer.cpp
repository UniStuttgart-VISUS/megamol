/*
 * LayeredIsosurfaceRenderer.cpp
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#define _USE_MATH_DEFINES 1

#include "LayeredIsosurfaceRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/CallClipPlane.h"
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
using namespace megamol::core;
using namespace megamol::protein;

/*
 * LayeredIsosurfaceRenderer::LayeredIsosurfaceRenderer (CTOR)
 */
LayeredIsosurfaceRenderer::LayeredIsosurfaceRenderer (void) : Renderer3DModule (),
        volDataCallerSlot ("getData", "Connects the volume rendering with data storage"),
        rendererCallerSlot ("renderer", "Connects the volume rendering with another renderer"),
        clipPlane0Slot("clipPlane0", "Connects the rendering with a clip plane"),
        clipPlane1Slot("clipPlane1", "Connects the rendering with a clip plane"),
        clipPlane2Slot("clipPlane2", "Connects the rendering with a clip plane"),
        volIsoValue0Param("volIsoValue0", "Isovalue 1 for isosurface rendering"),
        volIsoValue1Param("volIsoValue1", "Isovalue 2 for isosurface rendering"),
        volIsoValue2Param("volIsoValue2", "Isovalue 3 for isosurface rendering"),
        volIsoOpacityParam("volIsoOpacity", "Opacity of isosurface"),
        volClipPlaneFlagParam("volClipPlane", "Enable volume clipping"),
        volLicDirSclParam("lic::volLicDirScl", "..."),
        volLicLenParam("lic::volLicLen", "..."),
        volLicContrastStretchingParam("lic::volLicContrast", "Change the contrast of the LIC output image"),
        volLicBrightParam("lic::volLicBright", "..."),
        volLicTCSclParam("lic::volLicTCScl", "Scale factor for texture coordinates."),
        doVolumeRenderingParam("doVolumeRendering", "Turn volume rendering on and off."),
        doVolumeRenderingToggleParam("toggleVolumeRendering", "Turn volume rendering on and off."),
        volumeTex(0), vectorfieldTex(0), currentFrameId(-1), volFBO(0), width(0), height(0), volRayTexWidth(0), 
        volRayTexHeight(0), volRayStartTex(0), volRayLengthTex(0), volRayDistTex(0),
        renderIsometric(true), meanDensityValue(0.0f), isoValue0(0.5f), isoValue1(0.5f), isoValue2(0.5f), 
        volIsoOpacity(0.4f), volClipPlaneFlag(false), randNoiseTex(0), lastHash(-1)
{
    // set caller slot for different data calls
    this->volDataCallerSlot.SetCompatibleCall<core::moldyn::VolumeDataCallDescription>();
	this->volDataCallerSlot.SetCompatibleCall<protein_calls::VTIDataCallDescription>();
    this->MakeSlotAvailable (&this->volDataCallerSlot);

    // set renderer caller slot
    this->rendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot);
    
    // set caller slot for clip plane calls
    this->clipPlane0Slot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable (&this->clipPlane0Slot);
    // set caller slot for clip plane calls
    this->clipPlane1Slot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable (&this->clipPlane1Slot);
    // set caller slot for clip plane calls
    this->clipPlane2Slot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable (&this->clipPlane2Slot);
    
    // --- set up parameters for isovalue 0 ---
    this->volIsoValue0Param.SetParameter(new param::FloatParam(this->isoValue0));
    this->MakeSlotAvailable(&this->volIsoValue0Param);

    // --- set up parameters for isovalue 1 ---
    this->volIsoValue1Param.SetParameter(new param::FloatParam(this->isoValue1));
    this->MakeSlotAvailable(&this->volIsoValue1Param);

    // --- set up parameters for isovalue 2 ---
    this->volIsoValue2Param.SetParameter(new param::FloatParam(this->isoValue2));
    this->MakeSlotAvailable(&this->volIsoValue2Param);

    // --- set up parameter for isosurface opacity ---
    this->volIsoOpacityParam.SetParameter(new param::FloatParam(this->volIsoOpacity, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->volIsoOpacityParam);

    // set default clipping plane
    this->volClipPlane.Clear();
    this->volClipPlane.Add(vislib::math::Vector<float, 4>(1.0, 0.0, 0.0, 1.0));
    this->volClipPlane.Add(vislib::math::Vector<float, 4>(0.0, 1.0, 0.0, 1.0));
    this->volClipPlane.Add(vislib::math::Vector<float, 4>(0.0, 0.0, 1.0, 1.0));

    // --- set up parameter for volume clipping ---
    this->volClipPlaneFlagParam.SetParameter(new param::BoolParam(this->volClipPlaneFlag));
    this->MakeSlotAvailable(&this->volClipPlaneFlagParam);
    
    // Scale the direction vector when computing LIC on isosurface
    this->volLicDirSclParam.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->volLicDirSclParam);

    // Scale the direction vector when computing LIC on isosurface
    this->volLicLenParam.SetParameter(new core::param::IntParam(10, 1));
    this->MakeSlotAvailable(&this->volLicLenParam);

    // Change LIC contrast
    this->volLicContrastStretchingParam.SetParameter(new core::param::FloatParam(0.25f, 0.0f, 0.5f));
    this->MakeSlotAvailable(&this->volLicContrastStretchingParam);

    // Change LIC brightness
    this->volLicBrightParam.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->volLicBrightParam);

    // Volume LIC texture coordinates
    this->volLicTCSclParam.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->volLicTCSclParam);
    
    // --- set up parameter for volume rendering ---
    this->doVolumeRenderingParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->doVolumeRenderingParam);

    // --- set up parameter for volume rendering ---
    this->doVolumeRenderingToggleParam.SetParameter(new param::ButtonParam('V'));
    this->MakeSlotAvailable(&this->doVolumeRenderingToggleParam);
    
}


/*
 * LayeredIsosurfaceRenderer::~LayeredIsosurfaceRenderer (DTOR)
 */
LayeredIsosurfaceRenderer::~LayeredIsosurfaceRenderer (void) {
    this->Release ();
}


/*
 * LayeredIsosurfaceRenderer::release
 */
void LayeredIsosurfaceRenderer::release (void) {

}


/*
 * LayeredIsosurfaceRenderer::create
 */
bool LayeredIsosurfaceRenderer::create (void) {
    if (!ogl_IsVersionGEQ(2,0))
        return false;
    if (!areExtsAvailable("GL_EXT_framebuffer_object GL_ARB_texture_float GL_EXT_gpu_shader4 GL_EXT_bindable_uniform"))
        return false;
    if (!isExtAvailable("GL_ARB_vertex_program"))
        return false;
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
        return false;
    if (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions())
        return false;

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
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("layeredisosurfaces::isosurfaces::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg (Log::LEVEL_ERROR, "%: Unable to load vertex shader source for volume rendering shader", this->ClassName());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ("layeredisosurfaces::isosurfaces::fragment", fragSrc)) {
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

    // initialize 3D texture for shader-based LIC
    this->InitLIC(16);

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/


/*
 * ProteinRenderer::GetExtents
 */
bool LayeredIsosurfaceRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;
    
    float scale, xoff, yoff, zoff;
    vislib::math::Cuboid<float> boundingBox;
    vislib::math::Point<float, 3> bbc;
    
    // SFB-DEMO
    view::CallRender3D *rencr3d = this->rendererCallerSlot.CallAs<view::CallRender3D>();
    unsigned int rencr3dFrames = rencr3d->TimeFramesCount();

    core::moldyn::VolumeDataCall *volume = this->volDataCallerSlot.CallAs<core::moldyn::VolumeDataCall>();
    if( volume ) {
        // try to call the volume data
        if (!(*volume)(core::moldyn::VolumeDataCall::CallForGetExtent)) return false;
        // get bounding box
        boundingBox = volume->BoundingBox();

        bbc = boundingBox.CalcCenter();
        xoff = -bbc.X();
        yoff = -bbc.Y();
        zoff = -bbc.Z();
        if (!vislib::math::IsEqual(boundingBox.LongestEdge(), 0.0f)) { 
            scale = 2.0f / boundingBox.LongestEdge();
        } else {
            scale = 1.0f;
        }
        //cr3d->SetTimeFramesCount(volume->FrameCount());
        // SFB-DEMO
        cr3d->SetTimeFramesCount( vislib::math::Max( volume->FrameCount(), rencr3dFrames));
        
    } else {
		protein_calls::VTIDataCall *vti = this->volDataCallerSlot.CallAs<protein_calls::VTIDataCall>();
        if( vti == NULL ) return false;
        // set call time
        // SFB-DEMO
        float callTime = vislib::math::Min( cr3d->Time(), static_cast<float>(vti->FrameCount()-1));
        vti->SetCalltime(callTime);
        vti->SetFrameID(static_cast<int>(callTime));
        // try to call for extent
		if (!(*vti)(protein_calls::VTIDataCall::CallForGetExtent)) return false;
        // try to call for data
		if (!(*vti)(protein_calls::VTIDataCall::CallForGetData)) return false;
        
        // get bounding box
        boundingBox = vti->AccessBoundingBoxes().ObjectSpaceBBox();

        bbc = boundingBox.CalcCenter();
        xoff = -bbc.X();
        yoff = -bbc.Y();
        zoff = -bbc.Z();
        if (!vislib::math::IsEqual(boundingBox.LongestEdge(), 0.0f)) { 
            scale = 2.0f / boundingBox.LongestEdge();
        } else {
            scale = 1.0f;
        }
        //cr3d->SetTimeFramesCount(vti->FrameCount());
        // SFB-DEMO
        cr3d->SetTimeFramesCount( vislib::math::Max( vti->FrameCount(), rencr3dFrames));
    }

    BoundingBoxes &bbox = cr3d->AccessBoundingBoxes();
    bbox.SetObjectSpaceBBox(boundingBox);
    bbox.MakeScaledWorld( scale);

    return true;
}


/*
 * LayeredIsosurfaceRenderer::Render
 */
bool LayeredIsosurfaceRenderer::Render(Call& call) {

    if( this->doVolumeRenderingToggleParam.IsDirty() ) {
        this->doVolumeRenderingParam.Param<param::BoolParam>()->SetValue( !this->doVolumeRenderingParam.Param<param::BoolParam>()->Value());
        this->doVolumeRenderingToggleParam.ResetDirty();
    }

    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (!cr3d) return false;
    
    // get pointer to core::moldyn::VolumeDataCall
    core::moldyn::VolumeDataCall *volume = this->volDataCallerSlot.CallAs<core::moldyn::VolumeDataCall>();
    // get pointer to core::moldyn::VolumeDataCall
	protein_calls::VTIDataCall *vti = this->volDataCallerSlot.CallAs<protein_calls::VTIDataCall>();
    
    // set frame ID and call data
    if (volume) {
        volume->SetFrameID(static_cast<int>(cr3d->Time()));
        if (!(*volume)(core::moldyn::VolumeDataCall::CallForGetData)) {
            return false;
        }
    } else if (vti) {
        // set call time
        //vti->SetCalltime(cr3d->Time());
        // set frame ID
        //vti->SetFrameID(static_cast<int>(cr3d->Time()));
        // SFB-DEMO
        float callTime = vislib::math::Min( cr3d->Time(), static_cast<float>(vti->FrameCount()-1));
        vti->SetCalltime(callTime);
        vti->SetFrameID(static_cast<int>(callTime));
        // try to call for data
		if (!(*vti)(protein_calls::VTIDataCall::CallForGetData)) return false;
    } else {
        return false;
    }

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    // =============== Query Camera View Dimensions ===============
    if (static_cast<unsigned int>(cameraInfo->TileRect().Width()) != this->width ||
        static_cast<unsigned int>(cameraInfo->TileRect().Height()) != this->height) {
        this->width = static_cast<unsigned int>(cameraInfo->TileRect().Width());
        this->height = static_cast<unsigned int>(cameraInfo->TileRect().Height());
    }
    
    // create the fbo, if necessary
    if (!this->opaqueFBO.IsValid()) {
        this->opaqueFBO.Create(this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }
    // resize the fbo, if necessary
    if (this->opaqueFBO.GetWidth() != this->width || this->opaqueFBO.GetHeight() != this->height) {
        this->opaqueFBO.Create(this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }
    
    // =============== Protein Rendering ===============
    // disable the output buffer
    //cr3d->DisableOutputBuffer();
    // start rendering to the FBO for protein rendering
    this->opaqueFBO.Enable();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Apply scaling based on combined bounding box
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float scale;
    if (!vislib::math::IsEqual(cr3d->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scale = 2.0f/cr3d->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef(scale, scale, scale);

    // TODO
    // get the pointer to CallRender3D (protein renderer)
    view::CallRender3D *rencr3d = this->rendererCallerSlot.CallAs<view::CallRender3D>();
    if (rencr3d) {
        rencr3d->SetCameraParameters(cr3d->GetCameraParameters());
        glPushMatrix();
        // Revert scaling done by external renderer in advance
        float scaleRevert;
        if (!vislib::math::IsEqual(rencr3d->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
            scaleRevert = 2.0f/rencr3d->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        } else {
            scaleRevert = 1.0f;
        }
        scaleRevert = 1.0f/scaleRevert;
        glScalef(scaleRevert, scaleRevert, scaleRevert);
        glScalef(2,2,2); // QUICK FIX/HACK for MUX-Renderer
        //*rencr3d = *cr3d;
        //rencr3d->SetOutputBuffer(&this->opaqueFBO); // TODO: Handle incoming buffers!
        // SFB-DEMO
        rencr3d->SetTime( cr3d->Time());
        (*rencr3d)();
        glPopMatrix();
    }
    // stop rendering to the FBO for protein rendering
    this->opaqueFBO.Disable();
    // re-enable the output buffer
    //cr3d->EnableOutputBuffer();
    
    glScalef( 1.0f/scale, 1.0f/scale, 1.0f/scale);

    // =============== Refresh all parameters ===============
    this->ParameterRefresh(cr3d);
    
    /*
    unsigned int cpCnt;
    for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
        glClipPlane(GL_CLIP_PLANE0+cpCnt, this->volClipPlane[cpCnt].PeekComponents());
    }
    */

    // =============== Volume Rendering ===============
    bool retval = false;

    // try to start volume rendering using volume data
    if (volume) {
        retval = this->RenderVolumeData(cr3d, volume);
    } else if (vti) {
        retval = this->RenderVolumeData(cr3d, vti);
    }
    
    glPopMatrix();

    // unlock the current frame
    if (volume) {
        volume->Unlock();
    }
    if (vti) {
        vti->Unlock();
    }
    
    CHECK_FOR_OGL_ERROR();
    return retval;
}

 
/*
 * Volume rendering using volume data.
 */
bool LayeredIsosurfaceRenderer::RenderVolumeData(view::CallRender3D *call, core::moldyn::VolumeDataCall *volume) {
    glEnable (GL_DEPTH_TEST);
    glEnable (GL_VERTEX_PROGRAM_POINT_SIZE);

    // test for volume data
    if (volume->FrameCount() == 0)
        return false;

    glPushMatrix();
    
    vislib::math::Cuboid<float> bbox = volume->AccessBoundingBoxes().ObjectSpaceBBox();

    // translate scene for volume ray casting
    if (!vislib::math::IsEqual(bbox.LongestEdge(), 0.0f)) { 
        scale = 2.0f / bbox.LongestEdge();
    } else {
        scale = 1.0f;
    }
    vislib::math::Vector<float, 3> trans;
    trans.Set( bbox.Left(), bbox.Bottom(), bbox.Back());
    trans *= scale;
    glTranslatef(trans.GetX(), trans.GetY(), trans.GetZ());

    // ------------------------------------------------------------
    // --- Volume Rendering                                     ---
    // --- update & render the volume                           ---
    // ------------------------------------------------------------
    if (static_cast<int>(volume->FrameID()) != this->currentFrameId) {
        this->currentFrameId = static_cast<int>(volume->FrameID());
        this->UpdateVolumeTexture(volume);
        CHECK_FOR_OGL_ERROR();
    }

    // reenable second renderer
    this->opaqueFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();
    
    /*
    unsigned int cpCnt;
    if (this->volClipPlaneFlag) {
        for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
            glEnable(GL_CLIP_PLANE0+cpCnt);
        }
    }
    */
        
    if( this->doVolumeRenderingParam.Param<param::BoolParam>()->Value()) {
        this->RenderVolume(volume->BoundingBox());
        CHECK_FOR_OGL_ERROR();
    }

    
    /*
    if (this->volClipPlaneFlag) {
        for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
            glDisable(GL_CLIP_PLANE0+cpCnt);
        }
    }
    */

    glDisable (GL_VERTEX_PROGRAM_POINT_SIZE);

    glDisable (GL_DEPTH_TEST);
    
    glPopMatrix();

    return true;
}

/*
 * Volume rendering using volume data.
 */
bool LayeredIsosurfaceRenderer::RenderVolumeData(view::CallRender3D *call, protein_calls::VTIDataCall *volume) {
    glEnable (GL_DEPTH_TEST);
    glEnable (GL_VERTEX_PROGRAM_POINT_SIZE);

    // test for volume data
    if (volume->FrameCount() == 0)
        return false;

    glPushMatrix();

    vislib::math::Cuboid<float> bbox = volume->AccessBoundingBoxes().ObjectSpaceBBox();

    // translate scene for volume ray casting
    if (!vislib::math::IsEqual(bbox.LongestEdge(), 0.0f)) { 
        scale = 2.0f / bbox.LongestEdge();
    } else {
        scale = 1.0f;
    }
    vislib::math::Vector<float, 3> trans;
    trans.Set( bbox.Left(), bbox.Bottom(), bbox.Back());
    trans *= scale;
    glTranslatef(trans.GetX(), trans.GetY(), trans.GetZ());

    // ------------------------------------------------------------
    // --- Volume Rendering                                     ---
    // --- update & render the volume                           ---
    // ------------------------------------------------------------
    if ((static_cast<int>(volume->FrameID()) != this->currentFrameId)||(this->lastHash != volume->DataHash())) {
        this->currentFrameId = static_cast<int>(volume->FrameID());
        this->UpdateVolumeTexture(volume);
        CHECK_FOR_OGL_ERROR();
        this->lastHash = volume->DataHash();
    }
    
    // reenable second renderer
    this->opaqueFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();
    
    /*
    unsigned int cpCnt;
    if (this->volClipPlaneFlag) {
        for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
            glEnable(GL_CLIP_PLANE0+cpCnt);
        }
    }
    */

    if( this->doVolumeRenderingParam.Param<param::BoolParam>()->Value()) {
        this->RenderVolume(bbox);
        CHECK_FOR_OGL_ERROR();
    }

    /*
    if (this->volClipPlaneFlag) {
        for (cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt) {
            glDisable(GL_CLIP_PLANE0+cpCnt);
        }
    }
    */

    glDisable (GL_VERTEX_PROGRAM_POINT_SIZE);
    
    glDisable (GL_DEPTH_TEST);
    
    glPopMatrix();

    return true;
}

/*
 * refresh parameters
 */
void LayeredIsosurfaceRenderer::ParameterRefresh(view::CallRender3D *call) {
    
    // volume parameters
    if (this->volIsoValue0Param.IsDirty()) {
        this->isoValue0 = this->volIsoValue0Param.Param<param::FloatParam>()->Value();
        this->volIsoValue0Param.ResetDirty();
    }
    if (this->volIsoValue1Param.IsDirty()) {
        this->isoValue1 = this->volIsoValue1Param.Param<param::FloatParam>()->Value();
        this->volIsoValue1Param.ResetDirty();
    }
    if (this->volIsoValue2Param.IsDirty()) {
        this->isoValue2 = this->volIsoValue2Param.Param<param::FloatParam>()->Value();
        this->volIsoValue2Param.ResetDirty();
    }
    if (this->volIsoOpacityParam.IsDirty()) {
        this->volIsoOpacity = this->volIsoOpacityParam.Param<param::FloatParam>()->Value();
        this->volIsoOpacityParam.ResetDirty();
    }
    if (this->volClipPlaneFlagParam.IsDirty()) {
        this->volClipPlaneFlag = this->volClipPlaneFlagParam.Param<param::BoolParam>()->Value();
        this->volClipPlaneFlagParam.ResetDirty();
    }
    
    // get the clip plane 0
    view::CallClipPlane *ccp = this->clipPlane0Slot.CallAs<view::CallClipPlane>();
    float clipDat[4];
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
    }
    // set clip plane 0 normal and distance to current clip plane
    this->volClipPlane[0].Set(clipDat[0], clipDat[1], clipDat[2], clipDat[3]);
    
    // get the clip plane 1
    ccp = this->clipPlane1Slot.CallAs<view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
    }
    // set clip plane 1 normal and distance to current clip plane
    this->volClipPlane[1].Set(clipDat[0], clipDat[1], clipDat[2], clipDat[3]);
    
    // get the clip plane 2
    ccp = this->clipPlane2Slot.CallAs<view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
    }
    // set clip plane 2 normal and distance to current clip plane
    this->volClipPlane[2].Set(clipDat[0], clipDat[1], clipDat[2], clipDat[3]);
    
}

/*
 * Create a volume containing the voxel map
 */
void LayeredIsosurfaceRenderer::UpdateVolumeTexture(const core::moldyn::VolumeDataCall *volume) {
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
    //GLint mode = GL_CLAMP_TO_EDGE;
    GLint mode = GL_REPEAT;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
    glBindTexture(GL_TEXTURE_3D, 0);
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
 * Create a volume containing the voxel map
 */
void LayeredIsosurfaceRenderer::UpdateVolumeTexture(const protein_calls::VTIDataCall *volume) {
    // generate volume, if necessary
    if (!glIsTexture(this->volumeTex)) {
        glGenTextures(1, &this->volumeTex);
    }
    // get data pointer
    const float *densityData = (const float*)(volume->GetPointDataByIdx(0, 0));
    vislib::math::Vector<float, 3> gridSize = volume->GetGridsize();
    // set voxel map to volume texture
    glBindTexture(GL_TEXTURE_3D, this->volumeTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, 
        (GLsizei)gridSize.X(), 
        (GLsizei)gridSize.Y(), 
        (GLsizei)gridSize.Z(), 0, GL_LUMINANCE, GL_FLOAT, 
        densityData);
    GLint param = GL_LINEAR;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);
    //GLint mode = GL_CLAMP_TO_EDGE;
    GLint mode = GL_REPEAT;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
    glBindTexture(GL_TEXTURE_3D, 0);
    CHECK_FOR_OGL_ERROR();

    // generate FBO, if necessary
    if (!glIsFramebufferEXT(this->volFBO)) {
        glGenFramebuffersEXT(1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }
    CHECK_FOR_OGL_ERROR();
    
    
    // generate vector field texture, if necessary
    if (!glIsTexture(this->vectorfieldTex)) {
        glGenTextures(1, &this->vectorfieldTex);
    }
    // get data pointer
    const float *vectorData = (const float*)(volume->GetPointDataByIdx(1, 0));
    // set voxel map to volume texture
    glBindTexture(GL_TEXTURE_3D, this->vectorfieldTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F_ARB,
        (GLsizei)gridSize.X(), 
        (GLsizei)gridSize.Y(), 
        (GLsizei)gridSize.Z(), 0, GL_RGB, GL_FLOAT, 
        vectorData);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
    glBindTexture(GL_TEXTURE_3D, 0);
    CHECK_FOR_OGL_ERROR();

    // generate FBO, if necessary
    if (!glIsFramebufferEXT(this->volFBO)) {
        glGenFramebuffersEXT(1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }
    CHECK_FOR_OGL_ERROR();
    
    // scale[i] = 1/extent[i] --- extent = size of the bbox
    this->volScale[0] = 1.0f / (volume->GetBoundingBoxes().ObjectSpaceBBox().Width() * this->scale);
    this->volScale[1] = 1.0f / (volume->GetBoundingBoxes().ObjectSpaceBBox().Height() * this->scale);
    this->volScale[2] = 1.0f / (volume->GetBoundingBoxes().ObjectSpaceBBox().Depth() * this->scale);
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];

    // set volume size
    this->volumeSize = vislib::math::Max<unsigned int>((unsigned int)gridSize.Z(),
        vislib::math::Max<unsigned int>((unsigned int)gridSize.Y(), (unsigned int)gridSize.X()));
}

/*
 * draw the volume
 */
void LayeredIsosurfaceRenderer::RenderVolume(vislib::math::Cuboid<float> boundingbox) {
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
    //glUniform1f(_app->shader->paramsCvolume.stepSize, stepWidth);
    glUniform1f(this->volumeShader.ParameterLocation("stepSize"), stepWidth);

    //glUniform1f(_app->shader->paramsCvolume.alphaCorrection, _app->volStepSize/512.0f);
    // TODO: what is the correct value for volStepSize??
    glUniform1f(this->volumeShader.ParameterLocation("alphaCorrection"), this->volumeSize/256.0f);
    glUniform1i(this->volumeShader.ParameterLocation("numIterations"), 255);
    glUniform2f(this->volumeShader.ParameterLocation("screenResInv"), 1.0f/ float(this->width), 1.0f/ float(this->height));

    // bind depth texture
    glUniform1i(this->volumeShader.ParameterLocation("volumeSampler"), 0);
    glUniform1i(this->volumeShader.ParameterLocation("transferRGBASampler"), 1);
    glUniform1i(this->volumeShader.ParameterLocation("rayStartSampler"), 2);
    glUniform1i(this->volumeShader.ParameterLocation("rayLengthSampler"), 3);
    glUniform1i(this->volumeShader.ParameterLocation("vectorSampler"), 4);
    glUniform1i(this->volumeShader.ParameterLocation("randNoiseTex"), 5);
    glUniform1f(this->volumeShader.ParameterLocation("licDirScl"), this->volLicDirSclParam.Param<param::FloatParam>()->Value());
    glUniform1i(this->volumeShader.ParameterLocation("licLen"), this->volLicLenParam.Param<param::IntParam>()->Value());
    glUniform1f(this->volumeShader.ParameterLocation("licTCScl"), this->volLicTCSclParam.Param<param::FloatParam>()->Value());
    glUniform1f(this->volumeShader.ParameterLocation("licContrast"), this->volLicContrastStretchingParam.Param<param::FloatParam>()->Value());
    glUniform1f(this->volumeShader.ParameterLocation("licBright"), this->volLicBrightParam.Param<param::FloatParam>()->Value());
    
    glUniform3f(this->volumeShader.ParameterLocation("isoValues"), this->isoValue0, this->isoValue1, this->isoValue2);
    glUniform1f(this->volumeShader.ParameterLocation("isoOpacity"), this->volIsoOpacity);

    if( this->volClipPlaneFlag ) {
        for( unsigned int i = 0; i < vislib::math::Min( 3U, static_cast<unsigned int>(this->volClipPlane.Count())); i++ ) {
            float cp[4] = { 
                static_cast<float>(this->volClipPlane[i].X()), 
                static_cast<float>(this->volClipPlane[i].Y()),  
                static_cast<float>(this->volClipPlane[i].Z()),  
                static_cast<float>(this->volClipPlane[i].W()) };
            vislib::StringA cpPN;
            cpPN.Format( "clipPlane%i", i);
            glUniform4fv(this->volumeShader.ParameterLocation(cpPN.PeekBuffer()), 1, cp);
        }
    } else {
        for( unsigned int i = 0; i < 3U; i++ ) {
            vislib::StringA cpPN;
            cpPN.Format( "clipPlane%i", i);
            glUniform4f(this->volumeShader.ParameterLocation(cpPN.PeekBuffer()), 0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    // set bbox
    glUniform3f(this->volumeShader.ParameterLocation("osbboxdim"), boundingbox.GetSize().Width(), boundingbox.GetSize().Height(), boundingbox.GetSize().Depth());

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
    
#if 0
    // vector field texture
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_3D, this->vectorfieldTex);
    CHECK_FOR_OGL_ERROR();
    
    // 3D noise texture
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_3D, this->randNoiseTex);
    CHECK_FOR_OGL_ERROR();
#endif

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
 *  protein::CrystalStructureVolumeRenderer::initLIC
 */
void LayeredIsosurfaceRenderer::InitLIC(unsigned int licRandBuffSize) {
    using namespace vislib::sys;
    
    // Init random number generator
    srand((unsigned)time(0));

    // Create randbuffer
    unsigned int buffSize = licRandBuffSize * licRandBuffSize * licRandBuffSize;
    this->licRandBuff.AssertCapacity( buffSize);
    for(unsigned int i = 0; i < buffSize; i++) {
        float randVal = (float)rand()/float(RAND_MAX);
        this->licRandBuff.Add(randVal);
    }

    // Setup random noise texture
    glEnable(GL_TEXTURE_3D);
    if(glIsTexture(this->randNoiseTex))
        glDeleteTextures(1, &this->randNoiseTex);
    glGenTextures(1, &this->randNoiseTex);
    glBindTexture(GL_TEXTURE_3D, this->randNoiseTex);

    glTexImage3D(GL_TEXTURE_3D,
        0,
        GL_ALPHA,
        licRandBuffSize,
        licRandBuffSize,
        licRandBuffSize,
        0,
        GL_ALPHA,
        GL_FLOAT,
        this->licRandBuff.PeekElements());

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);

    // Check for opengl error
    CHECK_FOR_OGL_ERROR();
}

/*
 * write the parameters of the ray to the textures
 */
void LayeredIsosurfaceRenderer::RayParamTextures(vislib::math::Cuboid<float> boundingbox) {

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
    float r = (static_cast<float>(this->width) / static_cast<float>(this->height))*u;

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

    //enableClipPlanesVolume();

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
    // TODO reenable second renderer
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

    //disableClipPlanes();
    
    // DEBUG check texture values
    /*
    float *texdata = new float[this->width*this->height];
    float max = 0.0f;
    memset(texdata, 0, sizeof(float)*(this->width*this->height));
    glBindTexture(GL_TEXTURE_2D, this->volRayLengthTex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_ALPHA, GL_FLOAT, texdata);
    glBindTexture(GL_TEXTURE_2D, 0);
    for (unsigned int z = 1; z <= this->width*this->height; ++z) {
        std::cout << texdata[z-1] << " ";
        max = max < texdata[z-1] ? texdata[z-1] : max;
        if (z%this->width == 0)
            std::cout << std::endl;
    }
    delete[] texdata;
    */
}

/*
 * Draw the bounding box.
 */
void LayeredIsosurfaceRenderer::DrawBoundingBox(vislib::math::Cuboid<float> boundingbox) {

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
void LayeredIsosurfaceRenderer::drawClippedPolygon(vislib::math::Cuboid<float> boundingbox) {
    if (!this->volClipPlaneFlag)
        return;

    vislib::math::Vector<float, 3> position(boundingbox.GetSize().PeekDimension());
    position *= this->scale;

    // check for each clip plane
    /*
    float vcpd;
    for (int i = 0; i < static_cast<int>(this->volClipPlane.Count()); ++i) {
        slices.setupSingleSlice(this->volClipPlane[i].PeekComponents(), position.PeekComponents());
        float d = 0.0f;
        vcpd = static_cast<float>(this->volClipPlane[i].PeekComponents()[3]);
        glBegin(GL_TRIANGLE_FAN);
        slices.drawSingleSlice(-(-d + vcpd - 0.0001f));
        glEnd();
    }
    */
}

