/*
 * DirectVolumeRenderer.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1
#define STOP_SEGMENTATION

#include "DirectVolumeRenderer.h"
#include "CoreInstance.h"
#include "param/EnumParam.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "param/FloatParam.h"
#include "param/Vector3fParam.h"
#include "param/StringParam.h"
#include "utility/ShaderSourceFactory.h"
#include "view/AbstractCallRender.h"
#include "vislib/assert.h"
#include "vislib/glverify.h"
#include "vislib/Point.h"
#include "vislib/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/ShaderSource.h"
#include "vislib/AbstractOpenGLShader.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <glh/glh_genext.h>
#include <math.h>
#include <time.h>

using namespace megamol::core;

/*
 * DirectVolumeRenderer::DirectVolumeRenderer (CTOR)
 */
moldyn::DirectVolumeRenderer::DirectVolumeRenderer ( void ) : Renderer3DModule (),
        volDataCallerSlot ( "getData", "Connects the volume rendering with data storage" ),
        //protRendererCallerSlot ( "renderProtein", "Connects the volume rendering with a protein renderer" ),
        volIsoValueParam( "volIsoValue", "Isovalue for isosurface rendering"),
        volIsoOpacityParam( "volIsoOpacity", "Opacity of isosurface"),
        volClipPlaneFlagParam( "volClipPlane", "Enable volume clipping"),
        volClipPlane0NormParam( "clipPlane0Norm", "Volume clipping plane 0 normal"),
        volClipPlane0DistParam( "clipPlane0Dist", "Volume clipping plane 0 distance"),
        volClipPlaneOpacityParam( "clipPlaneOpacity", "Volume clipping plane opacity"),
        volumeTex( 0), currentFrameId(-1), volFBO( 0), width( 0), height( 0), volRayTexWidth( 0), 
        volRayTexHeight( 0), volRayStartTex( 0), volRayLengthTex( 0), volRayDistTex( 0),
        renderIsometric( true), meanDensityValue( 0.0f), isoValue( 0.5f), 
        volIsoOpacity( 0.4f), volClipPlaneFlag( false), volClipPlaneOpacity( 0.4f)
{
    // set caller slot for different data calls
    this->volDataCallerSlot.SetCompatibleCall<VolumeDataCallDescription>();
    this->MakeSlotAvailable ( &this->volDataCallerSlot );

    // set renderer caller slot
    // TODO
    //this->protRendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    //this->MakeSlotAvailable( &this->protRendererCallerSlot);

    // --- set up parameters for isovalues ---
    this->volIsoValueParam.SetParameter( new param::FloatParam( this->isoValue) );
    this->MakeSlotAvailable( &this->volIsoValueParam );

    // --- set up parameter for isosurface opacity ---
    this->volIsoOpacityParam.SetParameter( new param::FloatParam( this->volIsoOpacity, 0.0f, 1.0f ) );
    this->MakeSlotAvailable( &this->volIsoOpacityParam );

    // set default clipping plane
    this->volClipPlane.Clear();
    this->volClipPlane.Add( vislib::math::Vector<double, 4>( 0.0, 1.0, 0.0, 0.0));

    // --- set up parameter for volume clipping ---
    this->volClipPlaneFlagParam.SetParameter( new param::BoolParam( this->volClipPlaneFlag));
    this->MakeSlotAvailable( &this->volClipPlaneFlagParam );

    // --- set up parameter for volume clipping plane normal ---
    vislib::math::Vector<float, 3> cp0n(
        static_cast<float>(this->volClipPlane[0].PeekComponents()[0]), 
        static_cast<float>(this->volClipPlane[0].PeekComponents()[1]), 
        static_cast<float>(this->volClipPlane[0].PeekComponents()[2]));
    this->volClipPlane0NormParam.SetParameter( new param::Vector3fParam( cp0n) );
    this->MakeSlotAvailable( &this->volClipPlane0NormParam );

    // --- set up parameter for volume clipping plane distance ---
    float d = static_cast<float>(this->volClipPlane[0].PeekComponents()[3]);
    this->volClipPlane0DistParam.SetParameter( new param::FloatParam( d, 0.0f, 1.0f) );
    this->MakeSlotAvailable( &this->volClipPlane0DistParam );

    // --- set up parameter for clipping plane opacity ---
    this->volClipPlaneOpacityParam.SetParameter( new param::FloatParam( this->volClipPlaneOpacity, 0.0f, 1.0f ) );
    this->MakeSlotAvailable( &this->volClipPlaneOpacityParam );
    
}


/*
 * DirectVolumeRenderer::~DirectVolumeRenderer (DTOR)
 */
moldyn::DirectVolumeRenderer::~DirectVolumeRenderer ( void ) {
    this->Release ();
}


/*
 * DirectVolumeRenderer::release
 */
void moldyn::DirectVolumeRenderer::release ( void ) {

}


/*
 * DirectVolumeRenderer::create
 */
bool moldyn::DirectVolumeRenderer::create ( void ) {
    if( !glh_init_extensions( "GL_VERSION_2_0 GL_EXT_framebuffer_object GL_ARB_texture_float GL_EXT_gpu_shader4 GL_EXT_bindable_uniform") )
        return false;
    if( !glh_init_extensions( "GL_ARB_vertex_program" ) )
        return false;
    if( !vislib::graphics::gl::GLSLShader::InitialiseExtensions() )
        return false;
    if( !vislib::graphics::gl::FramebufferObject::InitialiseExtensions() )
        return false;

    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LEQUAL );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glEnable( GL_VERTEX_PROGRAM_TWO_SIDE );
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    // Load ray start shader
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::rayStartVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray start shader", this->ClassName() );
        return false;
    }
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::rayStartFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for ray start shader", this->ClassName() );
        return false;
    }
    try {
        if ( !this->volRayStartShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch ( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create ray start shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // Load ray start eye shader
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::rayStartEyeVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray start eye shader", this->ClassName() );
        return false;
    }
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::rayStartEyeFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for ray start eye shader", this->ClassName() );
        return false;
    }
    try {
        if ( !this->volRayStartEyeShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch ( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create ray start eye shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // Load ray length shader (uses same vertex shader as ray start shader)
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::rayStartVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray length shader", this->ClassName() );
        return false;
    }
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::rayLengthFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for ray length shader", this->ClassName() );
        return false;
    }
    try {
        if ( !this->volRayLengthShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch ( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create ray length shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // Load volume rendering shader
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::volumeVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for volume rendering shader", this->ClassName() );
        return false;
    }
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volumerenderer::std::volumeFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for volume rendering shader", this->ClassName() );
        return false;
    }
    try {
        if ( !this->volumeShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch ( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create volume rendering shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * ProteinRenderer::GetCapabilities
 */
bool moldyn::DirectVolumeRenderer::GetCapabilities( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities( view::CallRender3D::CAP_RENDER | 
        view::CallRender3D::CAP_LIGHTING |
        view::CallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * ProteinRenderer::GetExtents
 */
bool moldyn::DirectVolumeRenderer::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    VolumeDataCall *volume = this->volDataCallerSlot.CallAs<VolumeDataCall>();

    float scale, xoff, yoff, zoff;
    vislib::math::Cuboid<float> boundingBox;
    vislib::math::Point<float, 3> bbc;

    // try to call the volume data
    if( !(*volume)(VolumeDataCall::CallForGetExtent) ) return false;
    // get bounding box
    boundingBox = volume->BoundingBox();

    bbc = boundingBox.CalcCenter();
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    if( !vislib::math::IsEqual( boundingBox.LongestEdge(), 0.0f) ) { 
        scale = 2.0f / boundingBox.LongestEdge();
    } else {
        scale = 1.0f;
    }

    BoundingBoxes &bbox = cr3d->AccessBoundingBoxes();
    bbox.SetObjectSpaceBBox( boundingBox);
    bbox.SetWorldSpaceBBox(
        ( boundingBox.Left() + xoff) * scale,
        ( boundingBox.Bottom() + yoff) * scale,
        ( boundingBox.Back() + zoff) * scale,
        ( boundingBox.Right() + xoff) * scale,
        ( boundingBox.Top() + yoff) * scale,
        ( boundingBox.Front() + zoff) * scale);
    bbox.SetObjectSpaceClipBox( bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox( bbox.WorldSpaceBBox());

    // get the pointer to CallRender3D (protein renderer)
    // TODO
    //view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
    //vislib::math::Point<float, 3> protrenbbc;
    //if( protrencr3d ) {
    //    (*protrencr3d)(1); // GetExtents
    //    BoundingBoxes &protrenbb = protrencr3d->AccessBoundingBoxes();
    //    this->protrenScale =  protrenbb.ObjectSpaceBBox().Width() / boundingBox.Width();
    //    //this->protrenTranslate = ( protrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
    //    if( mol ) {
    //        this->protrenTranslate.Set( xoff, yoff, zoff);
    //        this->protrenTranslate *= scale;
    //    } else {
    //        this->protrenTranslate = ( protrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
    //    }
    //}

    return true;
}


/*
 * DirectVolumeRenderer::Render
 */
bool moldyn::DirectVolumeRenderer::Render( Call& call ) {
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>( &call );
    if( !cr3d ) return false;

    // get pointer to VolumeDataCall
    VolumeDataCall *volume = this->volDataCallerSlot.CallAs<VolumeDataCall>();
    
    // set frame ID and call data
    if( volume ) {
        volume->SetFrameID(static_cast<int>( cr3d->Time()));
        if( !(*volume)(VolumeDataCall::CallForGetData) ) {
            return false;
        }
    } else {
        return false;
    }

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    // =============== Query Camera View Dimensions ===============
    if( static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height ) {
        this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
    }

    // create the fbo, if necessary
    if( !this->opaqueFBO.IsValid() ) {
        this->opaqueFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }
    // resize the fbo, if necessary
    if( this->opaqueFBO.GetWidth() != this->width || this->opaqueFBO.GetHeight() != this->height ) {
        this->opaqueFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }

    // =============== Protein Rendering ===============
    // disable the output buffer
    cr3d->DisableOutputBuffer();
    // start rendering to the FBO for protein rendering
    this->opaqueFBO.Enable();
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // TODO
    //if( protrencr3d ) {
    //    // setup and call protein renderer
    //    glPushMatrix();
    //    glTranslatef( this->protrenTranslate.X(), this->protrenTranslate.Y(), this->protrenTranslate.Z());
    //    //glScalef( this->protrenScale, this->protrenScale, this->protrenScale);
    //    *protrencr3d = *cr3d;
    //    protrencr3d->SetOutputBuffer( &this->opaqueFBO); // TODO: Handle incoming buffers!
    //    (*protrencr3d)();
    //    glPopMatrix();
    //}
    // stop rendering to the FBO for protein rendering
    this->opaqueFBO.Disable();
    // re-enable the output buffer
    cr3d->EnableOutputBuffer();

    // =============== Refresh all parameters ===============
    this->ParameterRefresh( cr3d);
    
    unsigned int cpCnt;
    for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
        glClipPlane( GL_CLIP_PLANE0+cpCnt, this->volClipPlane[cpCnt].PeekComponents());
    }

    // =============== Volume Rendering ===============
    bool retval = false;

    // try to start volume rendering using volume data
    if( volume ) {
        retval = this->RenderVolumeData( cr3d, volume);
    }
    
    // unlock the current frame
    if( volume ) {
        volume->Unlock();
    }
    
    return retval;
}


/*
 * Volume rendering using volume data.
 */
bool moldyn::DirectVolumeRenderer::RenderVolumeData( view::CallRender3D *call, VolumeDataCall *volume) {
    glEnable ( GL_DEPTH_TEST );
    glEnable ( GL_VERTEX_PROGRAM_POINT_SIZE );

    // test for volume data
    if( volume->FrameCount() == 0 )
        return false;

    glPushMatrix();

    // translate scene for volume ray casting
    this->scale = 2.0f / vislib::math::Max( vislib::math::Max( 
        volume->BoundingBox().Width(),volume->BoundingBox().Height() ),
        volume->BoundingBox().Depth() );
    vislib::math::Vector<float, 3> trans( volume->BoundingBox().GetSize().PeekDimension() );
    trans *= -this->scale*0.5f;
    glTranslatef( trans.GetX(), trans.GetY(), trans.GetZ() );

    // ------------------------------------------------------------
    // --- Volume Rendering                                     ---
    // --- update & render the volume                           ---
    // ------------------------------------------------------------
    if( static_cast<int>(volume->FrameID()) != this->currentFrameId ) {
        this->currentFrameId = static_cast<int>(volume->FrameID());
        this->UpdateVolumeTexture( volume);
        CHECK_FOR_OGL_ERROR();
    }

    // reenable second renderer
    //this->opaqueFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();
    
    unsigned int cpCnt;
    if( this->volClipPlaneFlag ) {
        for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
            glEnable( GL_CLIP_PLANE0+cpCnt );
        }
    }

    this->RenderVolume( volume->BoundingBox());
    CHECK_FOR_OGL_ERROR();
    
    if( this->volClipPlaneFlag ) {
        for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
            glDisable( GL_CLIP_PLANE0+cpCnt );
        }
    }

    glDisable ( GL_VERTEX_PROGRAM_POINT_SIZE );

    glDisable ( GL_DEPTH_TEST );
    
    glPopMatrix();

    return true;
}

/*
 * refresh parameters
 */
void moldyn::DirectVolumeRenderer::ParameterRefresh( view::CallRender3D *call) {
    
    // volume parameters
    if( this->volIsoValueParam.IsDirty() ) {
        this->isoValue = this->volIsoValueParam.Param<param::FloatParam>()->Value();
        this->volIsoValueParam.ResetDirty();
    }
    if( this->volIsoOpacityParam.IsDirty() ) {
        this->volIsoOpacity = this->volIsoOpacityParam.Param<param::FloatParam>()->Value();
        this->volIsoOpacityParam.ResetDirty();
    }
    if( this->volClipPlaneFlagParam.IsDirty() ) {
        this->volClipPlaneFlag = this->volClipPlaneFlagParam.Param<param::BoolParam>()->Value();
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
    if( this->volClipPlane0NormParam.IsDirty() ) {
        // overwrite clip plane normal
        cp0n = this->volClipPlane0NormParam.Param<param::Vector3fParam>()->Value();
        // normalize clip plane normal, if necessary and set normalized clip plane normal to parameter
        if( !vislib::math::IsEqual<float>( cp0n.Length(), 1.0f) ) {
            cp0n.Normalise();
            this->volClipPlane0NormParam.Param<param::Vector3fParam>()->SetValue( cp0n);
        }
        this->volClipPlane0NormParam.ResetDirty();
    }
    // compute maximum extent
    vislib::math::Cuboid<float> bbox( call->AccessBoundingBoxes().WorldSpaceBBox());
    vislib::math::Vector<float, 3> tmpVec;
    float d, maxD, minD;
    // 1
    tmpVec.Set( bbox.GetLeftBottomBack().X(), bbox.GetLeftBottomBack().Y(), bbox.GetLeftBottomBack().Z());
    maxD = minD = cp0n.Dot( tmpVec);
    // 2
    tmpVec.Set( bbox.GetRightBottomBack().X(), bbox.GetRightBottomBack().Y(), bbox.GetRightBottomBack().Z());
    d = cp0n.Dot( tmpVec);
    if( minD > d ) minD = d;
    if( maxD < d ) maxD = d;
    // 3
    tmpVec.Set( bbox.GetLeftBottomFront().X(), bbox.GetLeftBottomFront().Y(), bbox.GetLeftBottomFront().Z());
    d = cp0n.Dot( tmpVec);
    if( minD > d ) minD = d;
    if( maxD < d ) maxD = d;
    // 4
    tmpVec.Set( bbox.GetRightBottomFront().X(), bbox.GetRightBottomFront().Y(), bbox.GetRightBottomFront().Z());
    d = cp0n.Dot( tmpVec);
    if( minD > d ) minD = d;
    if( maxD < d ) maxD = d;
    // 5
    tmpVec.Set( bbox.GetLeftTopBack().X(), bbox.GetLeftTopBack().Y(), bbox.GetLeftTopBack().Z());
    d = cp0n.Dot( tmpVec);
    if( minD > d ) minD = d;
    if( maxD < d ) maxD = d;
    // 6
    tmpVec.Set( bbox.GetRightTopBack().X(), bbox.GetRightTopBack().Y(), bbox.GetRightTopBack().Z());
    d = cp0n.Dot( tmpVec);
    if( minD > d ) minD = d;
    if( maxD < d ) maxD = d;
    // 7
    tmpVec.Set( bbox.GetLeftTopFront().X(), bbox.GetLeftTopFront().Y(), bbox.GetLeftTopFront().Z());
    d = cp0n.Dot( tmpVec);
    if( minD > d ) minD = d;
    if( maxD < d ) maxD = d;
    // 8
    tmpVec.Set( bbox.GetRightTopFront().X(), bbox.GetRightTopFront().Y(), bbox.GetRightTopFront().Z());
    d = cp0n.Dot( tmpVec);
    if( minD > d ) minD = d;
    if( maxD < d ) maxD = d;
    // check clip plane distance
    if( this->volClipPlane0DistParam.IsDirty() ) {
        cp0d = this->volClipPlane0DistParam.Param<param::FloatParam>()->Value();
        cp0d = minD + ( maxD - minD) * cp0d;
        this->volClipPlane0DistParam.ResetDirty();
    }    

    // set clip plane normal and distance to current clip plane
    this->volClipPlane[0].Set( cp0n.X(), cp0n.Y(), cp0n.Z(), cp0d);

    // check clip plane opacity parameter
    if( this->volClipPlaneOpacityParam.IsDirty() ) {
        this->volClipPlaneOpacity = this->volClipPlaneOpacityParam.Param<param::FloatParam>()->Value();
        this->volClipPlaneOpacityParam.ResetDirty();
    }

}

/*
 * Create a volume containing the voxel map
 */
void moldyn::DirectVolumeRenderer::UpdateVolumeTexture( const VolumeDataCall *volume) {
    // generate volume, if necessary
    if( !glIsTexture( this->volumeTex) ) {
        glGenTextures( 1, &this->volumeTex);
    }
    // set voxel map to volume texture
    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glTexImage3D( GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, 
        volume->VolumeDimension().GetWidth(), 
        volume->VolumeDimension().GetHeight(), 
        volume->VolumeDimension().GetDepth(), 0, GL_LUMINANCE, GL_FLOAT, 
        volume->VoxelMap() );
    GLint param = GL_LINEAR;
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);
    //GLint mode = GL_CLAMP_TO_EDGE;
    GLint mode = GL_REPEAT;
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
    glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
    glBindTexture( GL_TEXTURE_3D, 0);
    CHECK_FOR_OGL_ERROR();

    // generate FBO, if necessary
    if( !glIsFramebufferEXT( this->volFBO ) ) {
        glGenFramebuffersEXT( 1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }
    CHECK_FOR_OGL_ERROR();
    
    // scale[i] = 1/extent[i] --- extent = size of the bbox
    this->volScale[0] = 1.0f / ( volume->BoundingBox().Width() * this->scale);
    this->volScale[1] = 1.0f / ( volume->BoundingBox().Height() * this->scale);
    this->volScale[2] = 1.0f / ( volume->BoundingBox().Depth() * this->scale);
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];

    // set volume size
    this->volumeSize = vislib::math::Max<unsigned int>( volume->VolumeDimension().Depth(),
        vislib::math::Max<unsigned int>( volume->VolumeDimension().Height(), volume->VolumeDimension().Width()));
}

/*
 * draw the volume
 */
void moldyn::DirectVolumeRenderer::RenderVolume( vislib::math::Cuboid<float> boundingbox) {
    const float stepWidth = 1.0f/ ( 2.0f * float( this->volumeSize));
    glDisable( GL_BLEND);

    GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

    this->RayParamTextures( boundingbox);
    CHECK_FOR_OGL_ERROR();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFBO);

    this->volumeShader.Enable();

    glEnable( GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUniform4fv( this->volumeShader.ParameterLocation( "scaleVol"), 1, this->volScale);
    glUniform4fv( this->volumeShader.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
    //glUniform1f(_app->shader->paramsCvolume.stepSize, stepWidth);
    glUniform1f( this->volumeShader.ParameterLocation( "stepSize"), stepWidth);

    //glUniform1f(_app->shader->paramsCvolume.alphaCorrection, _app->volStepSize/512.0f);
    // TODO: what is the correct value for volStepSize??
    glUniform1f( this->volumeShader.ParameterLocation( "alphaCorrection"), this->volumeSize/256.0f);
    glUniform1i( this->volumeShader.ParameterLocation( "numIterations"), 255);
    glUniform2f( this->volumeShader.ParameterLocation( "screenResInv"), 1.0f/ float(this->width), 1.0f/ float(this->height));

    // bind depth texture
    glUniform1i( this->volumeShader.ParameterLocation( "volumeSampler"), 0);
    glUniform1i( this->volumeShader.ParameterLocation( "transferRGBASampler"), 1);
    glUniform1i( this->volumeShader.ParameterLocation( "rayStartSampler"), 2);
    glUniform1i( this->volumeShader.ParameterLocation( "rayLengthSampler"), 3);

    glUniform1f( this->volumeShader.ParameterLocation( "isoValue"), this->isoValue);
    glUniform1f( this->volumeShader.ParameterLocation( "isoOpacity"), this->volIsoOpacity);
    glUniform1f( this->volumeShader.ParameterLocation( "clipPlaneOpacity"), this->volClipPlaneOpacity);

    // transfer function
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_1D, 0);
    // ray start positions
    glActiveTexture( GL_TEXTURE2);
    glBindTexture( GL_TEXTURE_2D, this->volRayStartTex);
    // ray direction and length
    glActiveTexture( GL_TEXTURE3);
    glBindTexture( GL_TEXTURE_2D, this->volRayLengthTex);

    // volume texture
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    CHECK_FOR_OGL_ERROR();

    // draw a screen-filling quad
    glRectf(-1.0f, -1.0f, 1.0f, 1.0f);
    CHECK_FOR_OGL_ERROR();

    this->volumeShader.Disable();

    glEnable( GL_DEPTH_TEST);
    glDepthMask( GL_TRUE);
    glDisable( GL_BLEND);
    CHECK_FOR_OGL_ERROR();

    // restore depth buffer
    //glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->opaqueFBO.GetDepthTextureID(), 0);
    CHECK_FOR_OGL_ERROR();
}

/*
 * write the parameters of the ray to the textures
 */
void moldyn::DirectVolumeRenderer::RayParamTextures( vislib::math::Cuboid<float> boundingbox) {

    GLint param = GL_NEAREST;
    GLint mode = GL_CLAMP_TO_EDGE;

    // generate / resize ray start texture for volume ray casting
    if( !glIsTexture( this->volRayStartTex) ) {
        glGenTextures( 1, &this->volRayStartTex);
        glBindTexture( GL_TEXTURE_2D, this->volRayStartTex);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    } else if( this->width != this->volRayTexWidth || this->height != this->volRayTexHeight ) {
        glBindTexture( GL_TEXTURE_2D, this->volRayStartTex);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    }
    // generate / resize ray length texture for volume ray casting
    if( !glIsTexture( this->volRayLengthTex) ) {
        glGenTextures( 1, &this->volRayLengthTex);
        glBindTexture( GL_TEXTURE_2D, this->volRayLengthTex);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    } else if( this->width != this->volRayTexWidth || this->height != this->volRayTexHeight ) {
        glBindTexture( GL_TEXTURE_2D, this->volRayLengthTex);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    }
    // generate / resize ray distance texture for volume ray casting
    if( !glIsTexture( this->volRayDistTex) ) {
        glGenTextures( 1, &this->volRayDistTex);
        glBindTexture( GL_TEXTURE_2D, this->volRayDistTex);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    } else if( this->width != this->volRayTexWidth || this->height != this->volRayTexHeight ) {
        glBindTexture( GL_TEXTURE_2D, this->volRayDistTex);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, this->width, this->height, 0, GL_RGBA, GL_FLOAT, 0);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, param);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
    }
    CHECK_FOR_OGL_ERROR();
    glBindTexture( GL_TEXTURE_2D, 0);
    // set vol ray dimensions
    this->volRayTexWidth = this->width;
    this->volRayTexHeight = this->height;

    GLuint db[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->volFBO);
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
    glDrawBuffers( 2, db);

    // draw near clip plane
    glDisable( GL_DEPTH_TEST);
    glDepthMask( GL_FALSE);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);

    glColor4f(0.0f, 0.0f, 0.0f, 0.0f);

    // the shader transforms camera coords back to object space
    this->volRayStartEyeShader.Enable();

    float u = this->cameraInfo->NearClip() * tan( this->cameraInfo->ApertureAngle() * float(vislib::math::PI_DOUBLE) / 360.0f);
    float r = ( this->width / this->height)*u;

    glBegin(GL_QUADS);
        //glVertex3f(-r, -u, -_nearClip);
        glVertex3f(-r, -u, -this->cameraInfo->NearClip());
        glVertex3f( r, -u, -this->cameraInfo->NearClip());
        glVertex3f( r,  u, -this->cameraInfo->NearClip());
        glVertex3f(-r,  u, -this->cameraInfo->NearClip());
    glEnd();
    CHECK_FOR_OGL_ERROR();

    this->volRayStartEyeShader.Disable();

    glDrawBuffers( 1, db);

    this->volRayStartShader.Enable();

    // ------------ !useSphere && iso -------------
    vislib::math::Vector<float, 3> trans( boundingbox.GetSize().PeekDimension() );
    trans *= this->scale*0.5f;
    if( this->renderIsometric ) {
        glUniform3f( this->volRayStartShader.ParameterLocation( "translate"), 
            0.0f, 0.0f, 0.0f);
    } else {
        glUniform3fv( this->volRayStartShader.ParameterLocation( "translate"), 
            1, trans.PeekComponents() );
    }

    glDepthMask( GL_TRUE);
    glEnable( GL_DEPTH_TEST);

    glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
    glColor4f( 0.0f, 0.0f, 0.0f, 1.0f);

    glEnable( GL_CULL_FACE);

    // draw nearest backfaces
    glCullFace( GL_FRONT);

    //enableClipPlanesVolume();

    // draw bBox
    this->DrawBoundingBox( boundingbox);

    // draw nearest frontfaces
    glCullFace( GL_BACK);
    glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    CHECK_FOR_OGL_ERROR();
    
    // draw bBox
    this->DrawBoundingBox( boundingbox);

    this->volRayStartShader.Disable();

    // --------------------------------
    // -------- ray length ------------
    // --------------------------------
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->volFBO);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, this->volRayLengthTex, 0);
    CHECK_FOR_OGL_ERROR();

    // get clear color
    float clearCol[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, clearCol);
    glClearColor( 0, 0, 0, 0);
    glClearDepth( 0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearDepth( 1.0f);
    glDrawBuffers( 2, db);
    glClearColor( clearCol[0], clearCol[1], clearCol[2], clearCol[3]);

    this->volRayLengthShader.Enable();

    glUniform1i( this->volRayLengthShader.ParameterLocation( "sourceTex"), 0);
    glUniform1i( this->volRayLengthShader.ParameterLocation( "depthTex"), 1);
    glUniform2f( this->volRayLengthShader.ParameterLocation( "screenResInv"),
        1.0f / float(this->width), 1.0f / float(this->height));
    glUniform2f( this->volRayLengthShader.ParameterLocation( "zNearFar"),
        this->cameraInfo->NearClip(), this->cameraInfo->FarClip() );

    if( this->renderIsometric ) {
        glUniform3f( this->volRayLengthShader.ParameterLocation( "translate"), 
            0.0f, 0.0f, 0.0f);
    } else {
        glUniform3fv( this->volRayLengthShader.ParameterLocation( "translate"), 
            1, trans.PeekComponents() );
    }
    glUniform1f( this->volRayLengthShader.ParameterLocation( "scale"),
        this->scale);

    glActiveTexture( GL_TEXTURE1);
    // TODO reenable second renderer
    this->opaqueFBO.BindDepthTexture();
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, this->volRayStartTex);

    // draw farthest backfaces
    glCullFace( GL_FRONT);
    glDepthFunc( GL_GREATER);

    // draw bBox
    this->DrawBoundingBox( boundingbox);

    this->volRayLengthShader.Disable();

    glDrawBuffers( 1, db);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, 0, 0);

    glDepthFunc( GL_LESS);
    glCullFace( GL_BACK);
    glDisable( GL_CULL_FACE);

    //disableClipPlanes();
    
    // DEBUG check texture values
    /*
    float *texdata = new float[this->width*this->height];
    float max = 0.0f;
    memset( texdata, 0, sizeof(float)*(this->width*this->height));
    glBindTexture( GL_TEXTURE_2D, this->volRayLengthTex);
    glGetTexImage( GL_TEXTURE_2D, 0, GL_ALPHA, GL_FLOAT, texdata);
    glBindTexture( GL_TEXTURE_2D, 0);
    for( unsigned int z = 1; z <= this->width*this->height; ++z ) {
        std::cout << texdata[z-1] << " ";
        max = max < texdata[z-1] ? texdata[z-1] : max;
        if( z%this->width == 0 )
            std::cout << std::endl;
    }
    delete[] texdata;
    */
}

/*
 * Draw the bounding box.
 */
void moldyn::DirectVolumeRenderer::DrawBoundingBox( vislib::math::Cuboid<float> boundingbox) {

    vislib::math::Vector<float, 3> position( boundingbox.GetSize().PeekDimension() );
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
    if( this->volClipPlaneFlag )
        this->drawClippedPolygon( boundingbox);
}

/*
 * draw the clipped polygon for correct clip plane rendering
 */
void moldyn::DirectVolumeRenderer::drawClippedPolygon( vislib::math::Cuboid<float> boundingbox) {
    if( !this->volClipPlaneFlag )
        return;

    vislib::math::Vector<float, 3> position( boundingbox.GetSize().PeekDimension() );
    position *= this->scale;

    // check for each clip plane
    float vcpd;
    for( int i = 0; i < this->volClipPlane.Count(); ++i ) {
        slices.setupSingleSlice( this->volClipPlane[i].PeekComponents(), position.PeekComponents());
        float d = 0.0f;
        vcpd = static_cast<float>(this->volClipPlane[i].PeekComponents()[3]);
        glBegin(GL_TRIANGLE_FAN);
        slices.drawSingleSlice(-(-d + vcpd - 0.0001f));
        glEnd();
    }
}

