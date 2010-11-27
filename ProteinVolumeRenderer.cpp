/*
 * ProteinVolumeRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "ProteinVolumeRenderer.h"
#include "CoreInstance.h"
#include "Color.h"
#include "param/EnumParam.h"
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "param/Vector3fParam.h"
#include "utility/ShaderSourceFactory.h"
#include "view/AbstractCallRender.h"
#include "vislib/assert.h"
#include "vislib/glverify.h"
#include "vislib/File.h"
#include "vislib/String.h"
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
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * protein::ProteinVolumeRenderer::ProteinVolumeRenderer (CTOR)
 */
protein::ProteinVolumeRenderer::ProteinVolumeRenderer ( void ) : Renderer3DModule (),
		protDataCallerSlot ( "getData", "Connects the volume rendering with data storage" ),
		callFrameCalleeSlot ( "callFrame", "Connects the volume rendering with frame call from RMS renderer" ),
        protRendererCallerSlot ( "renderProtein", "Connects the volume rendering with a protein renderer" ),
		coloringModeParam ( "coloringMode", "Coloring Mode" ),
		volIsoValue1Param( "volIsoValue1", "First isovalue for isosurface rendering"),
		volIsoValue2Param( "volIsoValue2", "Second isovalue for isosurface rendering"),
		volFilterRadiusParam( "volFilterRadius", "Filter Radius for volume generation"),
		volDensityScaleParam( "volDensityScale", "Density scale factor for volume generation"),
		volIsoOpacityParam( "volIsoOpacity", "Opacity of isosurface"),
        volClipPlaneFlagParam( "volClipPlane", "Enable volume clipping"),
        volClipPlane0NormParam( "clipPlane0Norm", "Volume clipping plane 0 normal"),
        volClipPlane0DistParam( "clipPlane0Dist", "Volume clipping plane 0 distance"),
        volClipPlaneOpacityParam( "clipPlaneOpacity", "Volume clipping plane opacity"),
        currentFrameId ( 0 ), atomCount( 0 ), volumeTex( 0), volumeSize( 128), volFBO( 0),
        volFilterRadius( 1.75f), volDensityScale( 1.0f),
        width( 0), height( 0), volRayTexWidth( 0), volRayTexHeight( 0),
        volRayStartTex( 0), volRayLengthTex( 0), volRayDistTex( 0),
		renderIsometric( true), meanDensityValue( 0.0f), isoValue1( 0.5f), isoValue2(-0.5f),
        volIsoOpacity( 0.4f), volClipPlaneFlag( false), volClipPlaneOpacity( 0.4f)
{
	this->protDataCallerSlot.SetCompatibleCall<CallProteinDataDescription>();
	this->protDataCallerSlot.SetCompatibleCall<CallVolumeDataDescription>();
	this->protDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable ( &this->protDataCallerSlot );

	protein::CallFrameDescription dfd;
	this->callFrameCalleeSlot.SetCallback ( dfd.ClassName(), "CallFrame", &ProteinVolumeRenderer::ProcessFrameRequest );
	this->MakeSlotAvailable ( &this->callFrameCalleeSlot );

    this->protRendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable( &this->protRendererCallerSlot);

	// --- set the coloring mode ---

	this->SetColoringMode ( Color::ELEMENT );
	//this->SetColoringMode(AMINOACID);
	//this->SetColoringMode(STRUCTURE);
	//this->SetColoringMode(VALUE);
	//this->SetColoringMode(CHAIN_ID);
	//this->SetColoringMode(RAINBOW);
	param::EnumParam *cm = new param::EnumParam ( int ( this->currentColoringMode ) );

	cm->SetTypePair ( Color::ELEMENT, "Element" );
	cm->SetTypePair ( Color::AMINOACID, "AminoAcid" );
	cm->SetTypePair ( Color::STRUCTURE, "SecondaryStructure" );
	cm->SetTypePair(Color::VALUE, "Value");
	cm->SetTypePair ( Color::CHAIN_ID, "ChainID" );
	cm->SetTypePair(Color::RAINBOW, "Rainbow");
	cm->SetTypePair ( Color::CHARGE, "Charge" );

	this->coloringModeParam << cm;

	// --- set up parameters for isovalues ---
    this->volIsoValue1Param.SetParameter( new param::FloatParam( this->isoValue1) );
    this->volIsoValue2Param.SetParameter( new param::FloatParam( this->isoValue2) );
	// --- set up parameter for volume filter radius ---
	this->volFilterRadiusParam.SetParameter( new param::FloatParam( this->volFilterRadius, 0.0f ) );
	// --- set up parameter for volume density scale ---
	this->volDensityScaleParam.SetParameter( new param::FloatParam( this->volDensityScale, 0.0f ) );
	// --- set up parameter for isosurface opacity ---
	this->volIsoOpacityParam.SetParameter( new param::FloatParam( this->volIsoOpacity, 0.0f, 1.0f ) );

    // set default clipping plane
    this->volClipPlane.Clear();
    this->volClipPlane.Add( vislib::math::Vector<double, 4>( 0.0, 1.0, 0.0, 0.2));

    // --- set up parameter for volume clipping ---
    this->volClipPlaneFlagParam.SetParameter( new param::BoolParam( this->volClipPlaneFlag));
    // --- set up parameter for volume clipping plane normal ---
    vislib::math::Vector<float, 3> cp0n(
        this->volClipPlane[0].PeekComponents()[0], 
        this->volClipPlane[0].PeekComponents()[1], 
        this->volClipPlane[0].PeekComponents()[2]);
    this->volClipPlane0NormParam.SetParameter( new param::Vector3fParam( cp0n) );
    // --- set up parameter for volume clipping plane distance ---
    float d = this->volClipPlane[0].PeekComponents()[3];
    this->volClipPlane0DistParam.SetParameter( new param::FloatParam( d) );
	// --- set up parameter for clipping plane opacity ---
    this->volClipPlaneOpacityParam.SetParameter( new param::FloatParam( this->volClipPlaneOpacity, 0.0f, 1.0f ) );

	this->MakeSlotAvailable( &this->coloringModeParam );
	this->MakeSlotAvailable( &this->volIsoValue1Param );
	this->MakeSlotAvailable( &this->volIsoValue2Param );
	this->MakeSlotAvailable( &this->volFilterRadiusParam );
	this->MakeSlotAvailable( &this->volDensityScaleParam );
	this->MakeSlotAvailable( &this->volIsoOpacityParam );
    this->MakeSlotAvailable( &this->volClipPlaneFlagParam );
    this->MakeSlotAvailable( &this->volClipPlane0NormParam );
    this->MakeSlotAvailable( &this->volClipPlane0DistParam );
    this->MakeSlotAvailable( &this->volClipPlaneOpacityParam );

	// fill amino acid color table
	Color::FillAminoAcidColorTable( this->aminoAcidColorTable);
	// fill rainbow color table
	Color::MakeRainbowColorTable( 100, this->rainbowColors);

	this->renderRMSData = false;
	this->frameLabel = NULL;
}


/*
 * protein::ProteinVolumeRenderer::~ProteinVolumeRenderer (DTOR)
 */
protein::ProteinVolumeRenderer::~ProteinVolumeRenderer ( void ) {
	delete this->frameLabel;
	this->Release ();
}


/*
 * protein::ProteinVolumeRenderer::release
 */
void protein::ProteinVolumeRenderer::release ( void ) {

}


/*
 * protein::ProteinVolumeRenderer::create
 */
bool protein::ProteinVolumeRenderer::create ( void ) {
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
	glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB );
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE);

	using namespace vislib::sys;
	using namespace vislib::graphics::gl;

	ShaderSource vertSrc;
	ShaderSource fragSrc;

	// Load sphere shader
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::sphereVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere shader", this->ClassName() );
		return false;
	}
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::sphereFragment", fragSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere shader", this->ClassName() );
		return false;
	}
	try {
		if ( !this->sphereShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
			throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
		}
	} catch ( vislib::Exception e ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
		return false;
	}

	// Load clipped sphere shader
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::sphereClipPlaneVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for clipped sphere shader", this->ClassName() );
		return false;
	}
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::sphereClipPlaneFragment", fragSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for clipped sphere shader", this->ClassName() );
		return false;
	}
	try {
		if ( !this->clippedSphereShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
			throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
		}
	} catch ( vislib::Exception e ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create clipped sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
		return false;
	}

	// Load cylinder shader
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::cylinderVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for cylinder shader", this->ClassName() );
		return false;
	}
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "protein::std::cylinderFragment", fragSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for cylinder shader", this->ClassName() );
		return false;
	}
	try {
		if ( !this->cylinderShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
			throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
		}
	} catch ( vislib::Exception e ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create cylinder shader: %s\n", this->ClassName(), e.GetMsgA() );
		return false;
	}

	// Load volume texture generation shader
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::updateVolumeVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for volume texture update shader", this->ClassName() );
		return false;
	}
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::updateVolumeFragment", fragSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for volume texture update shader", this->ClassName() );
		return false;
	}
	try {
        if ( !this->updateVolumeShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
			throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
		}
	} catch ( vislib::Exception e ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create volume texture update shader: %s\n", this->ClassName(), e.GetMsgA() );
		return false;
	}

	// Load ray start shader
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::rayStartVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray start shader", this->ClassName() );
		return false;
	}
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::rayStartFragment", fragSrc ) ) {
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
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::rayStartEyeVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray start eye shader", this->ClassName() );
		return false;
	}
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::rayStartEyeFragment", fragSrc ) ) {
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
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::rayStartVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for ray length shader", this->ClassName() );
		return false;
	}
    if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::rayLengthFragment", fragSrc ) ) {
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
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::volumeVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for volume rendering shader", this->ClassName() );
		return false;
	}
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::volumeFragment", fragSrc ) ) {
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

	// Load dual isosurface rendering shader
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::volumeDualIsoVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for dual isosurface rendering shader", this->ClassName() );
		return false;
	}
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( "volume::std::volumeDualIsoFragment", fragSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for two isosurface rendering shader", this->ClassName() );
		return false;
	}
	try {
        if ( !this->dualIsosurfaceShader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
			throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
		}
	} catch ( vislib::Exception e ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create dual isosurface rendering shader: %s\n", this->ClassName(), e.GetMsgA() );
		return false;
	}

	return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::ProteinRenderer::GetCapabilities
 */
bool protein::ProteinVolumeRenderer::GetCapabilities( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities( view::CallRender3D::CAP_RENDER | 
		view::CallRender3D::CAP_LIGHTING |
        view::CallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * protein::ProteinRenderer::GetExtents
 */
bool protein::ProteinVolumeRenderer::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    CallProteinData *protein = this->protDataCallerSlot.CallAs<CallProteinData>();
    CallVolumeData *volume = this->protDataCallerSlot.CallAs<CallVolumeData>();
	MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();

    float scale, xoff, yoff, zoff;
    vislib::math::Cuboid<float> boundingBox;
    vislib::math::Point<float, 3> bbc;

    // try to get the bounding box from the active data call
    if( protein ) {
    // decide to use already loaded frame request from CallFrame or 'normal' rendering
    if( this->callFrameCalleeSlot.GetStatus() == AbstractSlot::STATUS_CONNECTED) {
        if( !this->renderRMSData ) return false;
    } else {
        if( !(*protein)() ) return false;
    }
        // get bounding box
        boundingBox = protein->BoundingBox();
    } else if( volume ) {
        // try to call the volume data
        if( !(*volume)() ) return false;
        // get bounding box
        boundingBox = volume->BoundingBox();
	} else if( mol ) {
		// try to call the molecular data
		if (!(*mol)(1)) return false;
		// get the bounding box
		boundingBox = mol->AccessBoundingBoxes().ObjectSpaceBBox();
		// set the frame count
		cr3d->SetTimeFramesCount( mol->FrameCount());
	} else {
        return false;
    }

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
    view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
    vislib::math::Point<float, 3> protrenbbc;
    if( protrencr3d ) {
        (*protrencr3d)(1); // GetExtents
        BoundingBoxes &protrenbb = protrencr3d->AccessBoundingBoxes();
        this->protrenScale =  protrenbb.ObjectSpaceBBox().Width() / boundingBox.Width();
        this->protrenTranslate = ( protrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
    }

    return true;
}


/*
 * protein::ProteinVolumeRenderer::Render
 */
bool protein::ProteinVolumeRenderer::Render( Call& call )
{
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>( &call );
    if( !cr3d ) return false;
    // get the pointer to CallRender3D (protein renderer)
    view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
	// get pointer to CallProteinData
	CallProteinData *protein = this->protDataCallerSlot.CallAs<CallProteinData>();
	// get pointer to CallVolumeData
    CallVolumeData *volume = this->protDataCallerSlot.CallAs<CallVolumeData>();
	// get pointer to MolecularDataCall
	MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();

	// get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    // =============== Query Camera View Dimensions ===============
    if( static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height ) {
        this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
    }

    // create the fbo, if necessary
    if( !this->proteinFBO.IsValid() ) {
        this->proteinFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }
    // resize the fbo, if necessary
    if( this->proteinFBO.GetWidth() != this->width || this->proteinFBO.GetHeight() != this->height ) {
        this->proteinFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }

    // =============== Protein Rendering ===============
    // disable the output buffer
    cr3d->DisableOutputBuffer();
    // start rendering to the FBO for protein rendering
    this->proteinFBO.Enable();
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    if( protrencr3d ) {
        // setup and call protein renderer
        glPushMatrix();
        glTranslatef( this->protrenTranslate.X(), this->protrenTranslate.Y(), this->protrenTranslate.Z());
        glScalef( this->protrenScale, this->protrenScale, this->protrenScale);
        *protrencr3d = *cr3d;
        protrencr3d->SetOutputBuffer( &this->proteinFBO); // TODO: Handle incoming buffers!
        (*protrencr3d)();
        glPopMatrix();
	}
    // stop rendering to the FBO for protein rendering
    this->proteinFBO.Disable();
    // re-enable the output buffer
    cr3d->EnableOutputBuffer();

    // =============== Refresh all parameters ===============
    this->ParameterRefresh();
	// make the atom color table if necessary
	Color::MakeColorTable( protein, 
      this->currentColoringMode,
      this->protAtomColorTable,
      this->aminoAcidColorTable,
      this->rainbowColors,
      true);

    unsigned int cpCnt;
    for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
        glClipPlane( GL_CLIP_PLANE0+cpCnt, this->volClipPlane[cpCnt].PeekComponents());
    }

    // =============== Volume Rendering ===============
    // try to start volume rendering using protein data
    if( protein ) {
        return this->RenderProteinData( cr3d, protein);
    }
    // try to start volume rendering using volume data
    if( volume ) {
        return this->RenderVolumeData( cr3d, volume);
    }
    // try to start volume rendering using protein data
    if( mol ) {
        return this->RenderMolecularData( cr3d, mol);
    }

    return false;
}


/*
 * Volume rendering using protein data.
 */
bool protein::ProteinVolumeRenderer::RenderProteinData( view::CallRender3D *call, CallProteinData *protein) {
    // get the current frame id
	if( this->currentFrameId != protein->GetCurrentFrameId() ) {
		this->currentFrameId = protein->GetCurrentFrameId();
	}

	// decide to use already loaded frame request from CallFrame or 'normal' rendering
	if( this->callFrameCalleeSlot.GetStatus() == AbstractSlot::STATUS_CONNECTED ) {
		if( !this->renderRMSData )
			return false;
	} else {
		if( !(*protein)() )
			return false;
	}

    // check last atom count with current atom count
    if( this->atomCount != protein->ProteinAtomCount() ) {
        this->atomCount = protein->ProteinAtomCount();
	}

	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_LIGHTING );
	glEnable ( GL_VERTEX_PROGRAM_POINT_SIZE );

	glPushMatrix();

    // translate scene for volume ray casting
	this->scale = 2.0f / vislib::math::Max ( vislib::math::Max ( protein->BoundingBox().Width(),
	                                   protein->BoundingBox().Height() ), protein->BoundingBox().Depth() );
    vislib::math::Vector<float, 3> trans( protein->BoundingBox().GetSize().PeekDimension() );
    trans *= -this->scale*0.5f;
    glTranslatef( trans.GetX(), trans.GetY(), trans.GetZ() );

	// ------------------------------------------------------------
	// --- Volume Rendering                                     ---
	// --- update & render the volume                           ---
	// ------------------------------------------------------------
    this->UpdateVolumeTexture( protein);
    CHECK_FOR_OGL_ERROR();

    this->proteinFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();

    unsigned int cpCnt;
    if( this->volClipPlaneFlag )
        for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
            glEnable( GL_CLIP_PLANE0+cpCnt );
    }

	this->RenderVolume( protein->BoundingBox());
    CHECK_FOR_OGL_ERROR();

    if( this->volClipPlaneFlag )
        for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
            glDisable( GL_CLIP_PLANE0+cpCnt );
    }

	glDisable ( GL_VERTEX_PROGRAM_POINT_SIZE );

	glDisable ( GL_DEPTH_TEST );
    
	glPopMatrix();

	// render label if RMS is used
	if ( this->renderRMSData )
		this->DrawLabel( protein->GetRequestedRMSFrame() );

    return true;
	}



/*
 * Volume rendering using molecular data.
 */
bool protein::ProteinVolumeRenderer::RenderMolecularData( view::CallRender3D *call, MolecularDataCall *mol) {

	// decide to use already loaded frame request from CallFrame or 'normal' rendering
	if( this->callFrameCalleeSlot.GetStatus() == AbstractSlot::STATUS_CONNECTED ) {
		if( !this->renderRMSData )
			return false;
	} else {
		if( !(*mol)() )
			return false;
	}

    // check last atom count with current atom count
	if( this->atomCount != mol->AtomCount() ) {
        this->atomCount = mol->AtomCount();
	}

	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_LIGHTING );
	glEnable ( GL_VERTEX_PROGRAM_POINT_SIZE );

	glPushMatrix();

    // translate scene for volume ray casting
	if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
		this->scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	} else {
		this->scale = 1.0f;
	}
    vislib::math::Vector<float, 3> trans( 
		mol->AccessBoundingBoxes().ObjectSpaceBBox().GetSize().PeekDimension() );
    trans *= -this->scale*0.5f;
    glTranslatef( trans.GetX(), trans.GetY(), trans.GetZ() );

		// ------------------------------------------------------------
	// --- Volume Rendering                                     ---
	// --- update & render the volume                           ---
		// ------------------------------------------------------------
    this->UpdateVolumeTexture( mol);
    CHECK_FOR_OGL_ERROR();

    this->proteinFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();

	this->RenderVolume( mol->AccessBoundingBoxes().ObjectSpaceBBox());
    CHECK_FOR_OGL_ERROR();
    
	glDisable ( GL_VERTEX_PROGRAM_POINT_SIZE );

	glDisable ( GL_DEPTH_TEST );
    
	glPopMatrix();
    
    return true;
	}


/*
 * Volume rendering using volume data.
 */
bool protein::ProteinVolumeRenderer::RenderVolumeData( view::CallRender3D *call, CallVolumeData *volume) {
	// try to call
	if( !(*volume)() ) return false;

	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_VERTEX_PROGRAM_POINT_SIZE );

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
    this->UpdateVolumeTexture( volume);
    CHECK_FOR_OGL_ERROR();

    this->proteinFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();

    this->RenderVolume( volume);
    CHECK_FOR_OGL_ERROR();
    
	glDisable ( GL_VERTEX_PROGRAM_POINT_SIZE );

	glDisable ( GL_DEPTH_TEST );
    
	glPopMatrix();

	return true;
}

/*
 * refresh parameters
 */
void protein::ProteinVolumeRenderer::ParameterRefresh() {
    
	// parameter refresh
	if( this->coloringModeParam.IsDirty() ) {
		this->SetColoringMode ( static_cast<Color::ColoringMode> ( int ( this->coloringModeParam.Param<param::EnumParam>()->Value() ) ) );
		this->coloringModeParam.ResetDirty();
	}
	// volume parameters
	if( this->volIsoValue1Param.IsDirty() ) {
		this->isoValue1 = this->volIsoValue1Param.Param<param::FloatParam>()->Value();
		this->volIsoValue1Param.ResetDirty();
	}
	if( this->volIsoValue2Param.IsDirty() ) {
		this->isoValue2 = this->volIsoValue2Param.Param<param::FloatParam>()->Value();
		this->volIsoValue2Param.ResetDirty();
	}
	if( this->volFilterRadiusParam.IsDirty() ) {
		this->volFilterRadius = this->volFilterRadiusParam.Param<param::FloatParam>()->Value();
		this->volFilterRadiusParam.ResetDirty();
	}
	if( this->volDensityScaleParam.IsDirty() ) {
		this->volDensityScale = this->volDensityScaleParam.Param<param::FloatParam>()->Value();
		this->volDensityScaleParam.ResetDirty();
	}
	if( this->volIsoOpacityParam.IsDirty() ) {
		this->volIsoOpacity = this->volIsoOpacityParam.Param<param::FloatParam>()->Value();
		this->volIsoOpacityParam.ResetDirty();
	}
    if( this->volClipPlaneFlagParam.IsDirty() ) {
        this->volClipPlaneFlag = this->volClipPlaneFlagParam.Param<param::BoolParam>()->Value();
        this->volClipPlaneFlagParam.ResetDirty();
    }
    vislib::math::Vector<float, 3> cp0n(
        (float)this->volClipPlane[0].PeekComponents()[0],
        (float)this->volClipPlane[0].PeekComponents()[1],
        (float)this->volClipPlane[0].PeekComponents()[2]);
    float cp0d = (float)this->volClipPlane[0].PeekComponents()[3];
    if( this->volClipPlane0NormParam.IsDirty() ) {
        cp0n = this->volClipPlane0NormParam.Param<param::Vector3fParam>()->Value();
        if( !vislib::math::IsEqual<float>( cp0n.Length(), 1.0f) ) {
            cp0n.Normalise();
            this->volClipPlane0NormParam.Param<param::Vector3fParam>()->SetValue( cp0n);
        }
        this->volClipPlane0NormParam.ResetDirty();
    }
    if( this->volClipPlane0DistParam.IsDirty() ) {
        cp0d = this->volClipPlane0DistParam.Param<param::FloatParam>()->Value();
        this->volClipPlane0DistParam.ResetDirty();
    }    
    this->volClipPlane[0].Set( cp0n.X(), cp0n.Y(), cp0n.Z(), cp0d);
	if( this->volClipPlaneOpacityParam.IsDirty() ) {
		this->volClipPlaneOpacity = this->volClipPlaneOpacityParam.Param<param::FloatParam>()->Value();
		this->volClipPlaneOpacityParam.ResetDirty();
	}
}

/*
 * protein::ProteinVolumeRenderer::ProcessFrameRequest
 */
bool protein::ProteinVolumeRenderer::ProcessFrameRequest ( Call& call )
{
	// get pointer to CallProteinData
	protein::CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();

	// ensure that NetCDFData uses 'RMS' specific frame handling
	protein->SetRMSUse ( true );

	// get pointer to frame call
	protein::CallFrame *pcf = dynamic_cast<protein::CallFrame*> ( &call );

	if ( pcf->NewRequest() )
	{
		// pipe frame request from frame call to protein call
		protein->SetRequestedRMSFrame ( pcf->GetFrameRequest() );
		if ( ! ( *protein ) () )
		{
			this->renderRMSData = false;
			return false;
		}
		this->renderRMSData = true;
	}

	return true;
}


/**
 * protein::ProteinVolumeRenderer::DrawLabel
 */
void protein::ProteinVolumeRenderer::DrawLabel( unsigned int frameID )
{
	using namespace vislib::graphics;
	char frameChar[10];

	glPushAttrib( GL_ENABLE_BIT);
	glDisable( GL_CULL_FACE);
	glDisable( GL_LIGHTING);

	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();

	glTranslatef(-1.0f, 1.0f, 1.0f );

	glColor3f( 1.0, 1.0, 1.0 );
	if( this->frameLabel == NULL ) {
		this->frameLabel = new vislib::graphics::gl::SimpleFont();
		if( !this->frameLabel->Initialise() ) {
			vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_WARN, "ProteinVolumeRenderer: Problems to initalise the Font" );
		}
	}
#ifdef _WIN32
        _itoa_s(frameID, frameChar, 10, 10);
#else  /* _WIN32 */
        vislib::StringA tmp; /* worst idea ever, but linux does not deserve anything better! */
        tmp.Format("%i", frameID);
        memcpy(frameChar, tmp.PeekBuffer(), 10);
#endif /* _WIN32 */

	this->frameLabel->DrawString( 0.0f, 0.0f, 0.1f, true, ( vislib::StringA( "Frame: ") + frameChar ).PeekBuffer() , AbstractFont::ALIGN_LEFT_TOP );

	glPopMatrix();

	glPopAttrib();
}


/*
 * Create a volume containing all protein atoms
 */
void protein::ProteinVolumeRenderer::UpdateVolumeTexture( const CallProteinData *protein) {
    // generate volume, if necessary
    if( !glIsTexture( this->volumeTex) ) {
        // from CellVis: cellVis.cpp, initGL
        glGenTextures( 1, &this->volumeTex);
        glBindTexture( GL_TEXTURE_3D, this->volumeTex);
        glTexImage3D( GL_TEXTURE_3D, 0, //GL_LUMINANCE32F_ARB,
                      GL_RGBA16F, 
                      this->volumeSize, this->volumeSize, this->volumeSize, 0,
                      //GL_LUMINANCE, GL_FLOAT, 0);
                      GL_RGBA, GL_FLOAT, 0);
        GLint param = GL_LINEAR;
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);
        GLint mode = GL_CLAMP_TO_EDGE;
        //GLint mode = GL_REPEAT;
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
        glBindTexture( GL_TEXTURE_3D, 0);
        CHECK_FOR_OGL_ERROR();
    }
    // generate FBO, if necessary
    if( !glIsFramebufferEXT( this->volFBO ) ) {
        glGenFramebuffersEXT( 1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }

    // counter variable
    unsigned int z;

    // store current frame buffer object ID
    GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

    glMatrixMode( GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // store old viewport
    GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport);
    // set viewport
    glViewport( 0, 0, this->volumeSize, this->volumeSize);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->volFBO);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

    glColor4f( 0.0, 0.0, 0.0, 1.0);
    
    float bgColor[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
    glClearColor( 0.1, 0.1, 0.1, 0.0);
    // clear 3d texture
    for( z = 0; z < this->volumeSize; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->volumeTex, 0, z);
        glClear( GL_COLOR_BUFFER_BIT);
        //glRecti(-1, -1, 1, 1);
    }
    glClearColor( bgColor[0], bgColor[1], bgColor[2], bgColor[3]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE);

    // scale[i] = 1/extent[i] --- extent = size of the bbox
    this->volScale[0] = 1.0f / ( protein->BoundingBox().Width() * this->scale);
    this->volScale[1] = 1.0f / ( protein->BoundingBox().Height() * this->scale);
    this->volScale[2] = 1.0f / ( protein->BoundingBox().Depth() * this->scale);
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];
    
    this->updateVolumeShader.Enable();
    vislib::math::Vector<float, 3> orig( protein->BoundingBox().GetLeftBottomBack().PeekCoordinates());
    orig = ( orig + this->translation) * this->scale;
    vislib::math::Vector<float, 3> nullVec( 0.0f, 0.0f, 0.0f);
    // set shader params

    glUniform1f( this->updateVolumeShader.ParameterLocation( "filterRadius"), this->volFilterRadius);
    glUniform1f( this->updateVolumeShader.ParameterLocation( "densityScale"), this->volDensityScale);
    glUniform3fv( this->updateVolumeShader.ParameterLocation( "scaleVol"), 1, this->volScale);
    glUniform3fv( this->updateVolumeShader.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
    glUniform3f( this->updateVolumeShader.ParameterLocation( "invVolRes"), 
        1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize));
    glUniform3fv( this->updateVolumeShader.ParameterLocation( "translate"), 1, orig.PeekComponents() );
	glUniform1f( this->updateVolumeShader.ParameterLocation( "volSize"), float( this->volumeSize));
    CHECK_FOR_OGL_ERROR();

	for( z = 0; z < this->volumeSize; ++z ) {
		// attach texture slice to FBO
		glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
		glUniform1f( this->updateVolumeShader.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
		// draw all atoms as points, using w for radius
		glBegin( GL_POINTS);
		for( unsigned int cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {
            glColor3fv( this->GetProteinAtomColor( cnt));
			glVertex4f( ( protein->ProteinAtomPositions()[cnt*3+0] + this->translation.GetX()) * this->scale,
				( protein->ProteinAtomPositions()[cnt*3+1] + this->translation.GetY()) * this->scale, 
				( protein->ProteinAtomPositions()[cnt*3+2] + this->translation.GetZ()) * this->scale, 
				protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius() * this->scale );
		}
		glEnd(); // GL_POINTS
	}

    this->updateVolumeShader.Disable();

    // restore viewport
    glViewport( viewport[0], viewport[1], viewport[2], viewport[3]);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFBO);

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthMask( GL_TRUE);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

/*
 * Create a volume containing all molecule atoms
 */
void protein::ProteinVolumeRenderer::UpdateVolumeTexture( MolecularDataCall *mol) {
    // generate volume, if necessary
    if( !glIsTexture( this->volumeTex) ) {
        // from CellVis: cellVis.cpp, initGL
        glGenTextures( 1, &this->volumeTex);
        glBindTexture( GL_TEXTURE_3D, this->volumeTex);
        glTexImage3D( GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB,
                      this->volumeSize, this->volumeSize, this->volumeSize, 0,
                      GL_LUMINANCE, GL_FLOAT, 0);
        GLint param = GL_LINEAR;
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param);
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param);
        GLint mode = GL_CLAMP_TO_EDGE;
        //GLint mode = GL_REPEAT;
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
        glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);
        glBindTexture( GL_TEXTURE_3D, 0);
        CHECK_FOR_OGL_ERROR();
    }
    // generate FBO, if necessary
    if( !glIsFramebufferEXT( this->volFBO ) ) {
        glGenFramebuffersEXT( 1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }

    // counter variable
    unsigned int z;

    // store current frame buffer object ID
    GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

    glMatrixMode( GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // store old viewport
    GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport);
    // set viewport
    glViewport( 0, 0, this->volumeSize, this->volumeSize);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->volFBO);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

    glColor4f( 0.0, 0.0, 0.0, 1.0);
    
    float bgColor[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
    glClearColor( 0.1, 0.1, 0.1, 0.0);
    // clear 3d texture
    for( z = 0; z < this->volumeSize; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->volumeTex, 0, z);
        glClear( GL_COLOR_BUFFER_BIT);
        //glRecti(-1, -1, 1, 1);
    }
    glClearColor( bgColor[0], bgColor[1], bgColor[2], bgColor[3]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE);

    // scale[i] = 1/extent[i] --- extent = size of the bbox
	this->volScale[0] = 1.0f / ( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width() * this->scale);
    this->volScale[1] = 1.0f / ( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height() * this->scale);
    this->volScale[2] = 1.0f / ( mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth() * this->scale);
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];
    
    this->updateVolumeShader.Enable();
    vislib::math::Vector<float, 3> orig( mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().PeekCoordinates());
    orig = ( orig + this->translation) * this->scale;
    vislib::math::Vector<float, 3> nullVec( 0.0f, 0.0f, 0.0f);

    // set shader params
    glUniform1f( this->updateVolumeShader.ParameterLocation( "filterRadius"), this->volFilterRadius);
    glUniform1f( this->updateVolumeShader.ParameterLocation( "densityScale"), this->volDensityScale);
    glUniform3fv( this->updateVolumeShader.ParameterLocation( "scaleVol"), 1, this->volScale);
    glUniform3fv( this->updateVolumeShader.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
    glUniform3f( this->updateVolumeShader.ParameterLocation( "invVolRes"), 
        1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize));
    glUniform3fv( this->updateVolumeShader.ParameterLocation( "translate"), 1, orig.PeekComponents() );
	glUniform1f( this->updateVolumeShader.ParameterLocation( "volSize"), float( this->volumeSize));
    CHECK_FOR_OGL_ERROR();

	for( z = 0; z < this->volumeSize; ++z ) {
		// attach texture slice to FBO
		glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
		glUniform1f( this->updateVolumeShader.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
		// draw all atoms as points, using w for radius
		glBegin( GL_POINTS);
		for( unsigned int cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
			glVertex4f( 
				( mol->AtomPositions()[3*cnt+0] + this->translation.X()) * this->scale,
				( mol->AtomPositions()[3*cnt+1] + this->translation.Y()) * this->scale,
				( mol->AtomPositions()[3*cnt+2] + this->translation.Z()) * this->scale,
				mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius() * this->scale );
		}
		glEnd(); // GL_POINTS
	}

    this->updateVolumeShader.Disable();

    // restore viewport
    glViewport( viewport[0], viewport[1], viewport[2], viewport[3]);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFBO);

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthMask( GL_TRUE);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

/*
 * Create a volume containing the voxel map
 */
void protein::ProteinVolumeRenderer::UpdateVolumeTexture( const CallVolumeData *volume) {
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
}

/*
 * draw the volume
 */
void protein::ProteinVolumeRenderer::RenderVolume( vislib::math::Cuboid<float> boundingbox) {
    const float stepWidth = 1.0f/ ( 2.0f * float( this->volumeSize));
    glDisable( GL_BLEND);

    GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

	this->RayParamTextures( boundingbox);
    CHECK_FOR_OGL_ERROR();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFBO);

    //glUseProgram(_app->shader->volume->progId);
    this->volumeShader.Enable();

    glEnable( GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //glUniform4fv(_app->shader->paramsCvolume.scaleVol, 1, vd->scale);
    glUniform4fv( this->volumeShader.ParameterLocation( "scaleVol"), 1, this->volScale);
    //glUniform4fv(_app->shader->paramsCvolume.scaleVolInv, 1, vd->scaleInv);
    glUniform4fv( this->volumeShader.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
    //glUniform1f(_app->shader->paramsCvolume.stepSize, stepWidth);
    glUniform1f( this->volumeShader.ParameterLocation( "stepSize"), stepWidth);

    //glUniform1f(_app->shader->paramsCvolume.alphaCorrection, _app->volStepSize/512.0f);
    // TODO: what is the correct value for volStepSize??
    glUniform1f( this->volumeShader.ParameterLocation( "alphaCorrection"), this->volumeSize/256.0f);
    //glUniform1i(_app->shader->paramsCvolume.numIterations, 255);
    glUniform1i( this->volumeShader.ParameterLocation( "numIterations"), 255);
    //glUniform2f(_app->shader->paramsCvolume.screenResInv, 1.0f/_fboWidth, 1.0f/_fboHeight);
    glUniform2f( this->volumeShader.ParameterLocation( "screenResInv"), 1.0f/ float(this->width), 1.0f/ float(this->height));

    // bind depth texture
    glUniform1i( this->volumeShader.ParameterLocation( "volumeSampler"), 0);
    glUniform1i( this->volumeShader.ParameterLocation( "transferRGBASampler"), 1);
    glUniform1i( this->volumeShader.ParameterLocation( "rayStartSampler"), 2);
    glUniform1i( this->volumeShader.ParameterLocation( "rayLengthSampler"), 3);

    glUniform1f( this->volumeShader.ParameterLocation( "isoValue"), this->isoValue1);
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
    //glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->proteinFBO.GetDepthTextureID(), 0);
    CHECK_FOR_OGL_ERROR();
}

/*
 * draw the volume
 */
void protein::ProteinVolumeRenderer::RenderVolume( const CallVolumeData *volume) {
    // check average density value
    if( vislib::math::Abs<float>( volume->MeanDensity() - this->meanDensityValue) > vislib::math::FLOAT_EPSILON ) {
        this->meanDensityValue = volume->MeanDensity();
        this->isoValue1 = this->meanDensityValue;
        this->isoValue2 = -this->meanDensityValue;
        this->volIsoValue1Param.Param<param::FloatParam>()->SetValue( this->isoValue1);
        this->volIsoValue2Param.Param<param::FloatParam>()->SetValue( this->isoValue2);
    }
    // compute step width
    const float stepWidth = 1.0f / ( 2.0f * float( volume->BoundingBox().LongestEdge()));
    glDisable( GL_BLEND);
    // store current FBO, if necessary
    GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);
    // generate the ray parameter textures for volume ray casting
    this->RayParamTextures( volume);
    CHECK_FOR_OGL_ERROR();
    // disable depth test and masking
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    // use the previously stored FBO (if any)
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFBO);

    // start GPU volume ray casting
    this->dualIsosurfaceShader.Enable();

    glEnable( GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // scale[i] = 1/extent[i] --- extent = size of the bbox
    this->volScale[0] = 1.0f / ( volume->BoundingBox().Width() * this->scale);
    this->volScale[1] = 1.0f / ( volume->BoundingBox().Height() * this->scale);
    this->volScale[2] = 1.0f / ( volume->BoundingBox().Depth() * this->scale);
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];
    
    glUniform4fv( this->dualIsosurfaceShader.ParameterLocation( "scaleVol"), 1, this->volScale);
    glUniform4fv( this->dualIsosurfaceShader.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
    glUniform1f( this->dualIsosurfaceShader.ParameterLocation( "stepSize"), stepWidth);

    glUniform1f( this->dualIsosurfaceShader.ParameterLocation( "alphaCorrection"), 
        float( volume->BoundingBox().LongestEdge())/256.0f);
    glUniform1i( this->dualIsosurfaceShader.ParameterLocation( "numIterations"), 255);
    glUniform2f( this->dualIsosurfaceShader.ParameterLocation( "screenResInv"), 1.0f/ float(this->width), 1.0f/ float(this->height));

    // bind depth texture
    glUniform1i( this->dualIsosurfaceShader.ParameterLocation( "volumeSampler"), 0);
    glUniform1i( this->dualIsosurfaceShader.ParameterLocation( "transferRGBASampler"), 1);
    glUniform1i( this->dualIsosurfaceShader.ParameterLocation( "rayStartSampler"), 2);
    glUniform1i( this->dualIsosurfaceShader.ParameterLocation( "rayLengthSampler"), 3);

    glUniform2f( this->dualIsosurfaceShader.ParameterLocation( "isoValues"), this->isoValue1, this->isoValue2);
	glUniform1f( this->dualIsosurfaceShader.ParameterLocation( "isoOpacity"), this->volIsoOpacity);
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

    this->dualIsosurfaceShader.Disable();

    /*
    this->volumeShader.Enable();

    glEnable( GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // scale[i] = 1/extent[i] --- extent = size of the bbox
    this->volScale[0] = 1.0f / ( volume->BoundingBox().Width() * this->scale);
    this->volScale[1] = 1.0f / ( volume->BoundingBox().Height() * this->scale);
    this->volScale[2] = 1.0f / ( volume->BoundingBox().Depth() * this->scale);
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];
    
    glUniform4fv( this->volumeShader.ParameterLocation( "scaleVol"), 1, this->volScale);
    glUniform4fv( this->volumeShader.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
    glUniform1f( this->volumeShader.ParameterLocation( "stepSize"), stepWidth);

    glUniform1f( this->volumeShader.ParameterLocation( "alphaCorrection"), 
        float( volume->BoundingBox().LongestEdge())/256.0f);
    glUniform1i( this->volumeShader.ParameterLocation( "numIterations"), 255);
    glUniform2f( this->volumeShader.ParameterLocation( "screenResInv"), 1.0f/ float(this->width), 1.0f/ float(this->height));

    // bind depth texture
    glUniform1i( this->volumeShader.ParameterLocation( "volumeSampler"), 0);
    glUniform1i( this->volumeShader.ParameterLocation( "transferRGBASampler"), 1);
    glUniform1i( this->volumeShader.ParameterLocation( "rayStartSampler"), 2);
    glUniform1i( this->volumeShader.ParameterLocation( "rayLengthSampler"), 3);

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
    */

    glEnable( GL_DEPTH_TEST);
    glDepthMask( GL_TRUE);
    glDisable( GL_BLEND);
    CHECK_FOR_OGL_ERROR();

    // restore depth buffer
    //glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->proteinFBO.GetDepthTextureID(), 0);
    CHECK_FOR_OGL_ERROR();
}

/*
 * write the parameters of the ray to the textures
 */
void protein::ProteinVolumeRenderer::RayParamTextures( vislib::math::Cuboid<float> boundingbox) {

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

    // -------- ray start ------------
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, this->volRayStartTex, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, this->volRayDistTex, 0);
    CHECK_FOR_OGL_ERROR();

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    // draw to two rendertargets (the second does not need to be cleared)
    glDrawBuffers( 2, db);
    //CHECK_FRAMEBUFFER_STATUS();

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

    //glUseProgram(_app->shader->volRayStart->progId);
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

    //glUseProgram(_app->shader->volRayLength->progId);
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
    //glBindTexture( GL_TEXTURE_2D, _depthTexId[0]);
    this->proteinFBO.BindDepthTexture();
    glActiveTexture( GL_TEXTURE0);
    //glBindTexture( GL_TEXTURE_2D, _volRayStartTex);
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
 * write the parameters of the ray to the textures
 */
void protein::ProteinVolumeRenderer::RayParamTextures( const CallVolumeData *volume) {

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

    // generate FBO, if necessary
    if( !glIsFramebufferEXT( this->volFBO ) ) {
        glGenFramebuffersEXT( 1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }

    GLuint db[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->volFBO);

    // -------- ray start ------------
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, this->volRayStartTex, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, this->volRayDistTex, 0);
    CHECK_FOR_OGL_ERROR();

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    // draw to two rendertargets (the second does not need to be cleared)
    glDrawBuffers( 2, db);
    //CHECK_FRAMEBUFFER_STATUS();

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
    vislib::math::Vector<float, 3> trans( volume->BoundingBox().GetSize().PeekDimension() );
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

    // draw bBox
    this->DrawBoundingBox( volume->BoundingBox());

    // draw nearest frontfaces
    glCullFace( GL_BACK);
    glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    CHECK_FOR_OGL_ERROR();
    
    // draw bBox
    this->DrawBoundingBox( volume->BoundingBox());

    this->volRayStartShader.Disable();
    CHECK_FOR_OGL_ERROR();

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

    //glUseProgram(_app->shader->volRayLength->progId);
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
    this->proteinFBO.BindDepthTexture();
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, this->volRayStartTex);

    // draw farthest backfaces
    glCullFace( GL_FRONT);
    glDepthFunc( GL_GREATER);

    // draw bBox
    this->DrawBoundingBox( volume->BoundingBox());

    this->volRayLengthShader.Disable();

    glDrawBuffers( 1, db);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, 0, 0);

    glDepthFunc( GL_LESS);
    glCullFace( GL_BACK);
    glDisable( GL_CULL_FACE);
}

/*
 * Draw the bounding box.
 */
void protein::ProteinVolumeRenderer::DrawBoundingBoxTranslated( vislib::math::Cuboid<float> boundingbox) {

    vislib::math::Vector<float, 3> position;
    glBegin(GL_QUADS);
    {
        // back side
        glNormal3f(0.0f, 0.0f, -1.0f);
        glColor3f( 1, 0, 0);
        //glVertex3fv( boundingbox.GetLeftBottomBack().PeekCoordinates() );
        position = boundingbox.GetLeftBottomBack();
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetLeftTopBack().PeekCoordinates() );
        position = (boundingbox.GetLeftTopBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightTopBack().PeekCoordinates() );
        position = (boundingbox.GetRightTopBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightBottomBack().PeekCoordinates() );
        position = (boundingbox.GetRightBottomBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // front side
        glNormal3f(0.0f, 0.0f, 1.0f);
        glColor3f( 0.5, 0, 0);
        //glVertex3fv( boundingbox.GetLeftTopFront().PeekCoordinates() );
        position = (boundingbox.GetLeftTopFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetLeftBottomFront().PeekCoordinates() );
        position = (boundingbox.GetLeftBottomFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightBottomFront().PeekCoordinates() );
        position = (boundingbox.GetRightBottomFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightTopFront().PeekCoordinates() );
        position = (boundingbox.GetRightTopFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // top side
        glNormal3f(0.0f, 1.0f, 0.0f);
        glColor3f( 0, 1, 0);
        //glVertex3fv( boundingbox.GetLeftTopBack().PeekCoordinates() );
        position = (boundingbox.GetLeftTopBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetLeftTopFront().PeekCoordinates() );
        position = (boundingbox.GetLeftTopFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightTopFront().PeekCoordinates() );
        position = (boundingbox.GetRightTopFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightTopBack().PeekCoordinates() );
        position = (boundingbox.GetRightTopBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // bottom side
        glNormal3f(0.0f, -1.0f, 0.0f);
        glColor3f( 0, 0.5, 0);
        //glVertex3fv( boundingbox.GetLeftBottomFront().PeekCoordinates() );
        position = (boundingbox.GetLeftBottomFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetLeftBottomBack().PeekCoordinates() );
        position = (boundingbox.GetLeftBottomBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightBottomBack().PeekCoordinates() );
        position = (boundingbox.GetRightBottomBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightBottomFront().PeekCoordinates() );
        position = (boundingbox.GetRightBottomFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // left side
        glNormal3f(-1.0f, 0.0f, 0.0f);
        glColor3f( 0, 0, 1);
        //glVertex3fv( boundingbox.GetLeftTopFront().PeekCoordinates() );
        position = (boundingbox.GetLeftTopFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetLeftTopBack().PeekCoordinates() );
        position = (boundingbox.GetLeftTopBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetLeftBottomBack().PeekCoordinates() );
        position = (boundingbox.GetLeftBottomBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetLeftBottomFront().PeekCoordinates() );
        position = (boundingbox.GetLeftBottomFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // right side
        glNormal3f(1.0f, 0.0f, 0.0f);
        glColor3f( 0, 0, 0.5);
        //glVertex3fv( boundingbox.GetRightTopBack().PeekCoordinates() );
        position = (boundingbox.GetRightTopBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightTopFront().PeekCoordinates() );
        position = (boundingbox.GetRightTopFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightBottomFront().PeekCoordinates() );
        position = (boundingbox.GetRightBottomFront());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( boundingbox.GetRightBottomBack().PeekCoordinates() );
        position = (boundingbox.GetRightBottomBack());
        position = boundingbox.CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
    }
    glEnd();
}

/*
 * Draw the bounding box.
 */
void protein::ProteinVolumeRenderer::DrawBoundingBox( vislib::math::Cuboid<float> boundingbox) {

    //vislib::math::Vector<float, 3> position( protein->BoundingBox().GetSize().PeekDimension() );
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
void ProteinVolumeRenderer::drawClippedPolygon( vislib::math::Cuboid<float> boundingbox) {
    if( !this->volClipPlaneFlag )
        return;

    //vislib::math::Vector<float, 3> position( protein->BoundingBox().GetSize().PeekDimension() );
    vislib::math::Vector<float, 3> position( boundingbox.GetSize().PeekDimension() );
    position *= this->scale;

    // check for each clip plane
    for( int i = 0; i < this->volClipPlane.Count(); ++i ) {
        slices.setupSingleSlice( this->volClipPlane[i].PeekComponents(), position.PeekComponents());
        float d = 0.0f;
        glBegin(GL_TRIANGLE_FAN);
            slices.drawSingleSlice(-(-d + this->volClipPlane[i].PeekComponents()[3]-0.0001f));
        glEnd();
    }
}
