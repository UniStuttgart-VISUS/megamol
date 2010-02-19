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
#include "param/EnumParam.h"
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "utility/ShaderSourceFactory.h"
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


/*
 * protein::ProteinVolumeRenderer::ProteinVolumeRenderer (CTOR)
 */
protein::ProteinVolumeRenderer::ProteinVolumeRenderer ( void ) : Renderer3DModule (),
		protDataCallerSlot ( "getData", "Connects the protein rendering with protein data storage" ),
		callFrameCalleeSlot ( "callFrame", "Connects the protein rendering with frame call from RMS renderer" ),
		renderingModeParam ( "renderingMode", "Rendering Mode" ),
		coloringModeParam ( "coloringMode", "Coloring Mode" ),
		drawBackboneParam ( "drawBackbone", "Draw Backbone only" ),
		drawDisulfideBondsParam ( "drawDisulfideBonds", "Draw Disulfide Bonds" ),
		stickRadiusParam ( "stickRadius", "Stick Radius for spheres and sticks with STICK_ render modes" ),
		probeRadiusParam ( "probeRadius", "Probe Radius for SAS rendering" ),
		volIsoValueParam( "volIsoValue", "Isovalue for isosurface rendering"),
		volFilterRadiusParam( "volFilterRadius", "Filter Radius for volume generation"),
		volDensityScaleParam( "volDensityScale", "Density scale factor for volume generation"),
        currentFrameId ( 0 ), atomCount( 0 ), volumeTex( 0), volumeSize( 128), volFBO( 0),
        volFilterRadius( 1.75f), volDensityScale( 1.0f),
        width( 0), height( 0), volRayTexWidth( 0), volRayTexHeight( 0),
        volRayStartTex( 0), volRayLengthTex( 0), volRayDistTex( 0),
        renderIsometric( true), isoValue( 0.6f)

{
	this->protDataCallerSlot.SetCompatibleCall<CallProteinDataDescription>();
	this->MakeSlotAvailable ( &this->protDataCallerSlot );

	protein::CallFrameDescription dfd;
	this->callFrameCalleeSlot.SetCallback ( dfd.ClassName(), "CallFrame", &ProteinVolumeRenderer::ProcessFrameRequest );
	this->MakeSlotAvailable ( &this->callFrameCalleeSlot );

	// --- set the coloring mode ---

	this->SetColoringMode ( ELEMENT );
	//this->SetColoringMode(AMINOACID);
	//this->SetColoringMode(STRUCTURE);
	//this->SetColoringMode(VALUE);
	//this->SetColoringMode(CHAIN_ID);
	//this->SetColoringMode(RAINBOW);
	param::EnumParam *cm = new param::EnumParam ( int ( this->currentColoringMode ) );

	cm->SetTypePair ( ELEMENT, "Element" );
	cm->SetTypePair ( AMINOACID, "AminoAcid" );
	cm->SetTypePair ( STRUCTURE, "SecondaryStructure" );
	cm->SetTypePair(VALUE, "Value");
	cm->SetTypePair ( CHAIN_ID, "ChainID" );
	cm->SetTypePair(RAINBOW, "Rainbow");
	cm->SetTypePair ( CHARGE, "Charge" );

	this->coloringModeParam << cm;

	// --- set the render mode ---

	SetRenderMode( LINES);
	//SetRenderMode( STICK_POLYGON);
	//SetRenderMode( STICK_RAYCASTING);
	//SetRenderMode( BALL_AND_STICK);
	//SetRenderMode( SPACEFILLING);
	param::EnumParam *rm = new param::EnumParam( int( this->currentRenderMode));

	rm->SetTypePair( LINES, "Lines" );
	//rm->SetTypePair(STICK_POLYGON, "StickPoly gonal");
	rm->SetTypePair( STICK_RAYCASTING, "StickRaycasting" );
	rm->SetTypePair( BALL_AND_STICK, "BallAndStick" );
	rm->SetTypePair( SPACEFILLING, "SpaceFilling" );
	rm->SetTypePair( SPACEFILLING_CLIP, "ClippedSpaceFilling" );
	rm->SetTypePair( SAS, "SAS" );

	this->renderingModeParam << rm;

	// --- draw only the backbone, if 'true' ---
	this->drawBackbone = false;
	//this->drawBackboneParam.SetParameter(new view::BoolParam(this->drawBackbone));

	// --- draw disulfide bonds, if 'true' ---
	this->drawDisulfideBonds = false;
	this->drawDisulfideBondsParam.SetParameter ( new param::BoolParam ( this->drawDisulfideBonds ) );

	// --- set the radius for the stick rendering mode ---
	this->radiusStick = 0.3f;
	this->stickRadiusParam.SetParameter ( new param::FloatParam ( this->radiusStick, 0.0f ) );

	// --- set the probe radius for sas rendering mode ---
	this->probeRadius = 1.4f;
	this->probeRadiusParam.SetParameter ( new param::FloatParam ( this->probeRadius, 0.0f ) );

	// --- set up parameter for isovalue ---
	this->volIsoValueParam.SetParameter( new param::FloatParam( this->isoValue, 0.0f ) );
	// --- set up parameter for volume filter radius ---
	this->volFilterRadiusParam.SetParameter( new param::FloatParam( this->volFilterRadius, 0.0f ) );
	// --- set up parameter for volume density scale ---
	this->volDensityScaleParam.SetParameter( new param::FloatParam( this->volDensityScale, 0.0f ) );

	this->MakeSlotAvailable( &this->coloringModeParam );
	this->MakeSlotAvailable( &this->renderingModeParam );
	//this->MakeSlotAvailable(&this->drawBackboneParam);
	this->MakeSlotAvailable( &this->drawDisulfideBondsParam );
	this->MakeSlotAvailable( &this->stickRadiusParam );
	this->MakeSlotAvailable( &this->probeRadiusParam );
	this->MakeSlotAvailable( &this->volIsoValueParam );
	this->MakeSlotAvailable( &this->volFilterRadiusParam );
	this->MakeSlotAvailable( &this->volDensityScaleParam );

	// set empty display list to zero
	this->proteinDisplayListLines = 0;
	// set empty display list to zero
	this->disulfideBondsDisplayList = 0;
	// STICK_RAYCASTING render mode was not prepared yet
	this->prepareStickRaycasting = true;
	// BALL_AND_STICK render mode was not prepared yet
	this->prepareBallAndStick = true;
	// SPACEFILLING render mode was not prepared yet
	this->prepareSpacefilling = true;
	// SAS render mode was not prepared yet
	this->prepareSAS = true;

	// fill amino acid color table
	this->FillAminoAcidColorTable();
	// fill rainbow color table
	this->MakeRainbowColorTable( 100);

	// draw dots for atoms when using LINES mode
	this->drawDotsWithLine = true;

	this->renderRMSData = false;
	this->frameLabel = NULL;
}


/*
 * protein::ProteinVolumeRenderer::~ProteinVolumeRenderer (DTOR)
 */
protein::ProteinVolumeRenderer::~ProteinVolumeRenderer ( void )
{
	delete this->frameLabel;
	this->Release ();
}


/*
 * protein::ProteinVolumeRenderer::release
 */
void protein::ProteinVolumeRenderer::release ( void )
{

}


/*
 * protein::ProteinVolumeRenderer::create
 */
bool protein::ProteinVolumeRenderer::create ( void )
{
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

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * protein::ProteinRenderer::GetExtents
 */
bool protein::ProteinVolumeRenderer::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    protein::CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();
    if( protein == NULL ) return false;
    // decide to use already loaded frame request from CallFrame or 'normal' rendering
    if( this->callFrameCalleeSlot.GetStatus() == AbstractSlot::STATUS_CONNECTED) {
        if( !this->renderRMSData ) return false;
    } else {
        if( !(*protein)() ) return false;
    }

    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    scale = 2.0f / vislib::math::Max(vislib::math::Max( protein->BoundingBox().Width(),
        protein->BoundingBox().Height()), protein->BoundingBox().Depth());

    BoundingBoxes &bbox = cr3d->AccessBoundingBoxes();
    bbox.SetObjectSpaceBBox(protein->BoundingBox());
    bbox.SetWorldSpaceBBox(
        (protein->BoundingBox().Left() + xoff) * scale,
        (protein->BoundingBox().Bottom() + yoff) * scale,
        (protein->BoundingBox().Back() + zoff) * scale,
        (protein->BoundingBox().Right() + xoff) * scale,
        (protein->BoundingBox().Top() + yoff) * scale,
        (protein->BoundingBox().Front() + zoff) * scale);
    bbox.SetObjectSpaceClipBox( bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox( bbox.WorldSpaceBBox());

    return true;
}


/*
 * protein::ProteinVolumeRenderer::Render
 */
bool protein::ProteinVolumeRenderer::Render( Call& call )
{
	// get pointer to CallProteinData
	CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();

	if( protein == NULL )
		return false;

	if( this->currentFrameId != protein->GetCurrentFrameId() ) {
		this->currentFrameId = protein->GetCurrentFrameId();
		this->RecomputeAll();
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
        this->RecomputeAll();
    }

	// get camera information
	this->cameraInfo = dynamic_cast<view::CallRender3D*>( &call )->GetCameraParameters();

    // =============== Query Camera View Dimensions ===============
    if( static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height ) {
        this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
    }

	// parameter refresh
	if ( this->renderingModeParam.IsDirty() ) {
		this->SetRenderMode ( static_cast<RenderMode> ( int ( this->renderingModeParam.Param<param::EnumParam>()->Value() ) ) );
		this->renderingModeParam.ResetDirty();
	}
	if ( this->coloringModeParam.IsDirty() ) {
		this->SetColoringMode ( static_cast<ColoringMode> ( int ( this->coloringModeParam.Param<param::EnumParam>()->Value() ) ) );
		this->coloringModeParam.ResetDirty();
	}
	//if (this->drawBackboneParam.IsDirty())
	//{
	//	this->drawBackbone = this->drawBackboneParam.Param<view::BoolParam>()->Value();
	//	this->drawBackboneParam.ResetDirty();
	//}
	if ( this->drawDisulfideBondsParam.IsDirty() ) {
		this->drawDisulfideBonds = this->drawDisulfideBondsParam.Param<param::BoolParam>()->Value();
		this->drawDisulfideBondsParam.ResetDirty();
	}
	if ( this->stickRadiusParam.IsDirty() ) {
		this->SetRadiusStick ( this->stickRadiusParam.Param<param::FloatParam>()->Value() );
		this->stickRadiusParam.ResetDirty();
	}
	if ( this->probeRadiusParam.IsDirty() ) {
		this->SetRadiusProbe ( this->probeRadiusParam.Param<param::FloatParam>()->Value() );
		this->probeRadiusParam.ResetDirty();
	}
	// volume parameters
	if ( this->volIsoValueParam.IsDirty() ) {
		this->isoValue = this->volIsoValueParam.Param<param::FloatParam>()->Value();
		this->volIsoValueParam.ResetDirty();
	}
	if ( this->volFilterRadiusParam.IsDirty() ) {
		this->volFilterRadius = this->volFilterRadiusParam.Param<param::FloatParam>()->Value();
		this->volFilterRadiusParam.ResetDirty();
	}
	if ( this->volDensityScaleParam.IsDirty() ) {
		this->volDensityScale = this->volDensityScaleParam.Param<param::FloatParam>()->Value();
		this->volDensityScaleParam.ResetDirty();
	}

	// make the atom color table if necessary
	this->MakeColorTable( protein );

	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_LIGHTING );
	glEnable ( GL_VERTEX_PROGRAM_POINT_SIZE );

	glPushMatrix();

	float xoff, yoff, zoff;
	vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();

	xoff = -bbc.X();
	yoff = -bbc.Y();
	zoff = -bbc.Z();

    this->translation =  protein->BoundingBox().GetLeftBottomBack();
    this->translation *= -1.0f;

	this->scale = 2.0f / vislib::math::Max ( vislib::math::Max ( protein->BoundingBox().Width(),
	                                   protein->BoundingBox().Height() ), protein->BoundingBox().Depth() );

	//glScalef ( this->scale, this->scale, this->scale );
	//glTranslatef ( xoff, yoff, zoff );
    vislib::math::Vector<float, 3> trans( protein->BoundingBox().GetSize().PeekDimension() );
    trans *= -this->scale*0.5f;
    glTranslatef( trans.GetX(), trans.GetY(), trans.GetZ() );


    // create the fbo, if necessary
    if( !this->proteinFBO.IsValid() ) {
        this->proteinFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }
    // resize the fbo, if necessary
    if( this->proteinFBO.GetWidth() != this->width || this->proteinFBO.GetHeight() != this->height ) {
        this->proteinFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }

    // start rendering to fbo
    GL_VERIFY_EXPR( this->proteinFBO.Enable());
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	if ( this->drawDisulfideBonds )
	{
		// ---------------------------------------------------------
		// --- draw disulfide bonds                              ---
		// ---------------------------------------------------------
		this->RenderDisulfideBondsLine ( protein );
	}

	if ( currentRenderMode == LINES )
	{
		// -----------------------------------------------------------------------
		// --- LINES                                                           ---
		// --- render the sceleton of the protein using GL_POINTS and GL_LINES ---
		// -----------------------------------------------------------------------
		this->RenderLines ( protein );
	}

	if ( currentRenderMode == STICK_RAYCASTING )
	{
		// ------------------------------------------------------------
		// --- STICK                                                ---
		// --- render the protein using shaders / raycasting (glsl) ---
		// ------------------------------------------------------------
		this->RenderStickRaycasting ( protein );
	}

	if ( currentRenderMode == BALL_AND_STICK )
	{
		// ------------------------------------------------------------
		// --- BALL & STICK                                         ---
		// --- render the protein using shaders / raycasting (glsl) ---
		// ------------------------------------------------------------
		this->RenderBallAndStick ( protein );
	}

	if ( currentRenderMode == SPACEFILLING )
	{
		// ------------------------------------------------------------
		// --- SPACEFILLING                                         ---
		// --- render the protein using shaders / raycasting (glsl) ---
		// ------------------------------------------------------------
		this->RenderSpacefilling ( protein );
	}

	if ( currentRenderMode == SPACEFILLING_CLIP )
	{
		// ------------------------------------------------------------
		// --- SPACEFILLING WITH CLIPPING PLANE                     ---
		// --- render the protein using shaders / raycasting (glsl) ---
		// ------------------------------------------------------------
		vislib::math::Vector<float, 3> cpDir( -1.0f, -1.0f, 0.0f);
		vislib::math::Vector<float, 3> cpBase =	bbc;
		this->RenderClippedSpacefilling ( cpDir, cpBase, protein );
	}

	if ( currentRenderMode == SAS )
	{
		// ------------------------------------------------------------
		// --- SAS (Solvent Accessible Surface)                     ---
		// --- render the protein using shaders / raycasting (glsl) ---
		// ------------------------------------------------------------
		this->RenderSolventAccessibleSurface ( protein );
	}

	// ------------------------------------------------------------
	// --- Volume Rendering                                     ---
	// --- update & render the volume for the protein atoms     ---
	// ------------------------------------------------------------

    this->proteinFBO.Disable();
    CHECK_FOR_OGL_ERROR();

    this->proteinFBO.DrawColourTexture();
    CHECK_FOR_OGL_ERROR();

    // update the volume
    this->UpdateVolumeTexture( protein);
    CHECK_FOR_OGL_ERROR();

    this->RenderVolume( protein);
    CHECK_FOR_OGL_ERROR();
    
	glDisable ( GL_VERTEX_PROGRAM_POINT_SIZE );

	glDisable ( GL_DEPTH_TEST );
    
	glPopMatrix();

	// render label if RMS is used
	if ( this->renderRMSData )
		this->DrawLabel( protein->GetRequestedRMSFrame() );

	return true;
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


/**
 * protein::ProteinVolumeRenderer::RenderLines
 */
void protein::ProteinVolumeRenderer::RenderLines( const CallProteinData *prot )
{
	// built the display list if it was not yet created
	if ( !glIsList ( this->proteinDisplayListLines ) )
	{
		// generate a new display list
		this->proteinDisplayListLines = glGenLists ( 1 );
		// compile new display list
		glNewList ( this->proteinDisplayListLines, GL_COMPILE );

		unsigned int i;
		unsigned int currentChain, currentAminoAcid, currentConnection;
		unsigned int first, second;
		// lines can not be lighted --> turn light off
		glDisable ( GL_LIGHTING );

		protein::CallProteinData::Chain chain;
		const float *protAtomPos = prot->ProteinAtomPositions();

		glPushAttrib ( GL_ENABLE_BIT | GL_POINT_BIT | GL_LINE_BIT | GL_POLYGON_BIT );
		glEnable ( GL_BLEND );
		glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glEnable ( GL_LINE_SMOOTH );
		glEnable ( GL_LINE_WIDTH );
		glEnable ( GL_POINT_SMOOTH );
		glEnable ( GL_POINT_SIZE );
		glLineWidth ( 3.0f );
		glPointSize ( 6.0f );

		if ( this->drawDotsWithLine )
		{
			// draw atoms as points
			glBegin ( GL_POINTS );
			for ( i = 0; i < prot->ProteinAtomCount(); i++ )
			{
				glColor3ubv ( this->GetProteinAtomColor ( i ) );
				glVertex3f ( protAtomPos[i*3+0], protAtomPos[i*3+1], protAtomPos[i*3+2] );
			}
			glEnd(); // GL_POINTS
		}

		// draw connections as lines
		glBegin ( GL_LINES );
		// loop over all chains
		for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
		{
			chain = prot->ProteinChain ( currentChain );
			// loop over all amino acids in the current chain
			for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
			{
				// loop over all connections of the current amino acid
				for ( currentConnection = 0;
				        currentConnection < chain.AminoAcid() [currentAminoAcid].Connectivity().Count();
				        currentConnection++ )
				{
					first = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].First();
					first += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
					second = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].Second();
					second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
					glColor3ubv ( this->GetProteinAtomColor ( first ) );
					glVertex3f ( protAtomPos[first*3+0], protAtomPos[first*3+1], protAtomPos[first*3+2] );
					glColor3ubv ( this->GetProteinAtomColor ( second ) );
					glVertex3f ( protAtomPos[second*3+0], protAtomPos[second*3+1], protAtomPos[second*3+2] );
				}
				// try to make the connection between this amino acid and its predecessor
				// --> only possible if the current amino acid is not the first in this chain
				if ( currentAminoAcid > 0 )
				{
					first = chain.AminoAcid() [currentAminoAcid-1].CCarbIndex();
					first += chain.AminoAcid() [currentAminoAcid-1].FirstAtomIndex();
					second = chain.AminoAcid() [currentAminoAcid].NIndex();
					second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
					glColor3ubv ( this->GetProteinAtomColor ( first ) );
					glVertex3f ( protAtomPos[first*3+0], protAtomPos[first*3+1], protAtomPos[first*3+2] );
					glColor3ubv ( this->GetProteinAtomColor ( second ) );
					glVertex3f ( protAtomPos[second*3+0], protAtomPos[second*3+1], protAtomPos[second*3+2] );
				}
			}
		}
		glEnd(); // GL_LINES

		glPopAttrib();

		glEndList();
		vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO+200, "%s: Display list for LINES render mode built.", this->ClassName() );
	}
	else
	{
		//draw the display list
		glCallList ( this->proteinDisplayListLines );
	}
	// turn light on after rendering
	glEnable ( GL_LIGHTING );
	glDisable ( GL_BLEND );

}


/*
 * protein::ProteinVolumeRenderer::RenderStickRaycasting
 */
void protein::ProteinVolumeRenderer::RenderStickRaycasting (
    const CallProteinData *prot )
{
	if ( this->prepareStickRaycasting )
	{
		unsigned int i1;
		unsigned int first, second;
		unsigned int currentChain, currentAminoAcid, currentConnection;
		const unsigned char *color1;
		const unsigned char *color2;

		// -----------------------------
		// -- computation for spheres --
		// -----------------------------

		// clear vertex array for spheres
		this->vertSphereStickRay.Clear();
		// clear color array for sphere colors
		this->colorSphereStickRay.Clear();

		// store the points (will be rendered as spheres by the shader)
		for ( i1 = 0; i1 < prot->ProteinAtomCount(); i1++ )
		{
            this->vertSphereStickRay.Add( ( prot->ProteinAtomPositions()[i1*3+0] + this->translation.GetX() ) * this->scale );
            this->vertSphereStickRay.Add( ( prot->ProteinAtomPositions()[i1*3+1] + this->translation.GetY() ) * this->scale );
            this->vertSphereStickRay.Add( ( prot->ProteinAtomPositions()[i1*3+2] + this->translation.GetZ() ) * this->scale );
            this->vertSphereStickRay.Add( radiusStick * this->scale );

			color1 = this->GetProteinAtomColor ( i1 );
			this->colorSphereStickRay.Add( color1[0] );
			this->colorSphereStickRay.Add( color1[1] );
			this->colorSphereStickRay.Add( color1[2] );
		}

		// -------------------------------
		// -- computation for cylinders --
		// -------------------------------
		protein::CallProteinData::Chain chain;
		vislib::math::Quaternion<float> quatC;
		quatC.Set ( 0, 0, 0, 1 );
		vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
		vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
		float angle;
		// vertex array for cylinders
		this->vertCylinderStickRay.Clear();
		// color array for first cylinder colors
		this->color1CylinderStickRay.Clear();
		// color array for second cylinder colors
		this->color2CylinderStickRay.Clear();
		// attribute array for quaterions
		this->quatCylinderStickRay.Clear();
		// attribute array for in-parameters
		this->inParaCylStickRaycasting.Clear();

		// loop over all chains
		for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
		{
			chain = prot->ProteinChain ( currentChain );
			// loop over all amino acids in the current chain
			for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
			{
				// loop over all connections of the current amino acid
				for ( currentConnection = 0;
				        currentConnection < chain.AminoAcid() [currentAminoAcid].Connectivity().Count();
				        currentConnection++ )
				{
					first = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].First();
					first += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
					second = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].Second();
					second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();

					firstAtomPos.SetX( prot->ProteinAtomPositions()[first*3+0] );
					firstAtomPos.SetY( prot->ProteinAtomPositions()[first*3+1] );
					firstAtomPos.SetZ( prot->ProteinAtomPositions()[first*3+2] );
                    firstAtomPos = ( firstAtomPos + this->translation) * this->scale;
					color1 = this->GetProteinAtomColor ( first );

					secondAtomPos.SetX ( prot->ProteinAtomPositions()[second*3+0] );
					secondAtomPos.SetY ( prot->ProteinAtomPositions()[second*3+1] );
					secondAtomPos.SetZ ( prot->ProteinAtomPositions()[second*3+2] );
                    secondAtomPos = ( secondAtomPos + this->translation) * this->scale;
					color2 = this->GetProteinAtomColor ( second );

					// compute the quaternion for the rotation of the cylinder
					dir = secondAtomPos - firstAtomPos;
					tmpVec.Set ( 1.0f, 0.0f, 0.0f );
					angle = - tmpVec.Angle ( dir );
					ortho = tmpVec.Cross ( dir );
					ortho.Normalise();
					quatC.Set ( angle, ortho );
					// compute the absolute position 'position' of the cylinder (center point)
					position = firstAtomPos + ( dir/2.0f );

                    this->inParaCylStickRaycasting.Add( radiusStick * this->scale );
					this->inParaCylStickRaycasting.Add( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

					this->quatCylinderStickRay.Add( quatC.GetX() );
					this->quatCylinderStickRay.Add( quatC.GetY() );
					this->quatCylinderStickRay.Add( quatC.GetZ() );
					this->quatCylinderStickRay.Add( quatC.GetW() );

					this->color1CylinderStickRay.Add( float ( int ( color1[0] ) ) /255.0f );
					this->color1CylinderStickRay.Add( float ( int ( color1[1] ) ) /255.0f );
					this->color1CylinderStickRay.Add( float ( int ( color1[2] ) ) /255.0f );

					this->color2CylinderStickRay.Add( float ( int ( color2[0] ) ) /255.0f );
					this->color2CylinderStickRay.Add( float ( int ( color2[1] ) ) /255.0f );
					this->color2CylinderStickRay.Add( float ( int ( color2[2] ) ) /255.0f );

					this->vertCylinderStickRay.Add( position.GetX() );
					this->vertCylinderStickRay.Add( position.GetY() );
					this->vertCylinderStickRay.Add( position.GetZ() );
					this->vertCylinderStickRay.Add( 1.0f );
				}
				// try to make the connection between this amino acid and its predecessor
				// --> only possible if the current amino acid is not the first in this chain
				if ( currentAminoAcid > 0 )
				{
					first = chain.AminoAcid()[currentAminoAcid-1].CCarbIndex();
					first += chain.AminoAcid()[currentAminoAcid-1].FirstAtomIndex();
					second = chain.AminoAcid()[currentAminoAcid].NIndex();
					second += chain.AminoAcid()[currentAminoAcid].FirstAtomIndex();

					firstAtomPos.SetX( prot->ProteinAtomPositions()[first*3+0] );
					firstAtomPos.SetY( prot->ProteinAtomPositions()[first*3+1] );
					firstAtomPos.SetZ( prot->ProteinAtomPositions()[first*3+2] );
                    firstAtomPos = ( firstAtomPos + this->translation) * this->scale;
					color1 = this->GetProteinAtomColor ( first );

					secondAtomPos.SetX ( prot->ProteinAtomPositions() [second*3+0] );
					secondAtomPos.SetY ( prot->ProteinAtomPositions() [second*3+1] );
					secondAtomPos.SetZ ( prot->ProteinAtomPositions() [second*3+2] );
                    secondAtomPos = ( secondAtomPos + this->translation) * this->scale;
					color2 = this->GetProteinAtomColor ( second );

					// compute the quaternion for the rotation of the cylinder
					dir = secondAtomPos - firstAtomPos;
					tmpVec.Set ( 1.0f, 0.0f, 0.0f );
					angle = - tmpVec.Angle ( dir );
					ortho = tmpVec.Cross ( dir );
					ortho.Normalise();
					quatC.Set ( angle, ortho );
					// compute the absolute position 'position' of the cylinder (center point)
					position = firstAtomPos + ( dir/2.0f );

					// don't draw bonds that are too long
					if ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) > 3.5f )
						continue;
					
                    this->inParaCylStickRaycasting.Add ( radiusStick * this->scale );
					this->inParaCylStickRaycasting.Add ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

					this->quatCylinderStickRay.Add( quatC.GetX() );
					this->quatCylinderStickRay.Add( quatC.GetY() );
					this->quatCylinderStickRay.Add( quatC.GetZ() );
					this->quatCylinderStickRay.Add( quatC.GetW() );

					this->color1CylinderStickRay.Add( float ( int ( color1[0] ) ) /255.0f );
					this->color1CylinderStickRay.Add( float ( int ( color1[1] ) ) /255.0f );
					this->color1CylinderStickRay.Add( float ( int ( color1[2] ) ) /255.0f );

					this->color2CylinderStickRay.Add( float ( int ( color2[0] ) ) /255.0f );
					this->color2CylinderStickRay.Add( float ( int ( color2[1] ) ) /255.0f );
					this->color2CylinderStickRay.Add( float ( int ( color2[2] ) ) /255.0f );

					this->vertCylinderStickRay.Add( position.GetX() );
					this->vertCylinderStickRay.Add( position.GetY() );
					this->vertCylinderStickRay.Add( position.GetZ() );
					this->vertCylinderStickRay.Add( 1.0f );
				}
			}
		}

		this->prepareStickRaycasting = false;
	}

	// -----------
	// -- draw  --
	// -----------
	float viewportStuff[4] =
	{
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height()
	};
	if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	glDisable ( GL_BLEND );

	// enable sphere shader
	this->sphereShader.Enable();
	glEnableClientState ( GL_VERTEX_ARRAY );
	glEnableClientState ( GL_COLOR_ARRAY );
	// set shader variables
	glUniform4fvARB ( this->sphereShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
	// set vertex and color pointers and draw them
	glVertexPointer ( 4, GL_FLOAT, 0, this->vertSphereStickRay.PeekElements() );
	glColorPointer ( 3, GL_UNSIGNED_BYTE, 0, this->colorSphereStickRay.PeekElements() );
	glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertSphereStickRay.Count() /4 ) );
	// disable sphere shader
	this->sphereShader.Disable();

	// enable cylinder shader
	this->cylinderShader.Enable();
	// set shader variables
	glUniform4fvARB ( this->cylinderShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
	glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
	glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
	glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
	// get the attribute locations
	attribLocInParams = glGetAttribLocationARB ( this->cylinderShader, "inParams" );
	attribLocQuatC = glGetAttribLocationARB ( this->cylinderShader, "quatC" );
	attribLocColor1 = glGetAttribLocationARB ( this->cylinderShader, "color1" );
	attribLocColor2 = glGetAttribLocationARB ( this->cylinderShader, "color2" );
	// enable vertex attribute arrays for the attribute locations
	glDisableClientState ( GL_COLOR_ARRAY );
	glEnableVertexAttribArrayARB ( this->attribLocInParams );
	glEnableVertexAttribArrayARB ( this->attribLocQuatC );
	glEnableVertexAttribArrayARB ( this->attribLocColor1 );
	glEnableVertexAttribArrayARB ( this->attribLocColor2 );
	// set vertex and attribute pointers and draw them
	glVertexPointer ( 4, GL_FLOAT, 0, this->vertCylinderStickRay.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylStickRaycasting.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinderStickRay.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1CylinderStickRay.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2CylinderStickRay.PeekElements() );
	glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertCylinderStickRay.Count() /4 ) );
	// disable vertex attribute arrays for the attribute locations
	glDisableVertexAttribArrayARB ( this->attribLocInParams );
	glDisableVertexAttribArrayARB ( this->attribLocQuatC );
	glDisableVertexAttribArrayARB ( this->attribLocColor1 );
	glDisableVertexAttribArrayARB ( this->attribLocColor2 );
	glDisableClientState ( GL_VERTEX_ARRAY );
	// disable cylinder shader
	this->cylinderShader.Disable();

}


/*
 * protein::ProteinVolumeRenderer::RenderBallAndStick
 */
void protein::ProteinVolumeRenderer::RenderBallAndStick (
    const CallProteinData *prot )
{
	if ( this->prepareBallAndStick )
	{
		unsigned int i1;
		unsigned int first, second;
		unsigned int currentChain, currentAminoAcid, currentConnection;
		const unsigned char *color1;
		const unsigned char *color2;

		// -----------------------------
		// -- computation for spheres --
		// -----------------------------

		// clear vertex array for spheres
		this->vertSphereStickRay.Clear();
		// clear color array for sphere colors
		this->colorSphereStickRay.Clear();

        this->vertSphereStickRay.AssertCapacity( 4 * prot->ProteinAtomCount() );
		this->colorSphereStickRay.AssertCapacity( 3 * prot->ProteinAtomCount() );
		// store the points (will be rendered as spheres by the shader)
		for ( i1 = 0; i1 < prot->ProteinAtomCount(); i1++ )
		{
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+0] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+1] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+2] );
			this->vertSphereStickRay.Add ( radiusStick );

			color1 = this->GetProteinAtomColor ( i1 );
			this->colorSphereStickRay.Add ( color1[0] );
			this->colorSphereStickRay.Add ( color1[1] );
			this->colorSphereStickRay.Add ( color1[2] );
		}

		// -------------------------------
		// -- computation for cylinders --
		// -------------------------------
		protein::CallProteinData::Chain chain;
		vislib::math::Quaternion<float> quatC;
		quatC.Set ( 0, 0, 0, 1 );
		vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
		vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
		float angle;
		// vertex array for cylinders
		this->vertCylinderStickRay.Clear();
		// color array for first cylinder colors
		this->color1CylinderStickRay.Clear();
		// color array for second cylinder colors
		this->color2CylinderStickRay.Clear();
		// attribute array for quaterions
		this->quatCylinderStickRay.Clear();
		// attribute array for in-parameters
		this->inParaCylStickRaycasting.Clear();

		// loop over all chains
		for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
		{
			chain = prot->ProteinChain ( currentChain );
			// loop over all amino acids in the current chain
			for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
			{
				// loop over all connections of the current amino acid
				for ( currentConnection = 0;
				        currentConnection < chain.AminoAcid() [currentAminoAcid].Connectivity().Count();
				        currentConnection++ )
				{
					first = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].First();
					first += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();
					second = chain.AminoAcid() [currentAminoAcid].Connectivity() [currentConnection].Second();
					second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();

					firstAtomPos.SetX ( prot->ProteinAtomPositions() [first*3+0] );
					firstAtomPos.SetY ( prot->ProteinAtomPositions() [first*3+1] );
					firstAtomPos.SetZ ( prot->ProteinAtomPositions() [first*3+2] );
					color1 = this->GetProteinAtomColor ( first );

					secondAtomPos.SetX ( prot->ProteinAtomPositions() [second*3+0] );
					secondAtomPos.SetY ( prot->ProteinAtomPositions() [second*3+1] );
					secondAtomPos.SetZ ( prot->ProteinAtomPositions() [second*3+2] );
					color2 = this->GetProteinAtomColor ( second );

					// compute the quaternion for the rotation of the cylinder
					dir = secondAtomPos - firstAtomPos;
					tmpVec.Set ( 1.0f, 0.0f, 0.0f );
					angle = - tmpVec.Angle ( dir );
					ortho = tmpVec.Cross ( dir );
					ortho.Normalise();
					quatC.Set ( angle, ortho );
					// compute the absolute position 'position' of the cylinder (center point)
					position = firstAtomPos + ( dir/2.0f );

					this->inParaCylStickRaycasting.Add ( radiusStick/3.0f );
					this->inParaCylStickRaycasting.Add ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

					this->quatCylinderStickRay.Add ( quatC.GetX() );
					this->quatCylinderStickRay.Add ( quatC.GetY() );
					this->quatCylinderStickRay.Add ( quatC.GetZ() );
					this->quatCylinderStickRay.Add ( quatC.GetW() );

					this->color1CylinderStickRay.Add ( float ( int ( color1[0] ) ) /255.0f );
					this->color1CylinderStickRay.Add ( float ( int ( color1[1] ) ) /255.0f );
					this->color1CylinderStickRay.Add ( float ( int ( color1[2] ) ) /255.0f );

					this->color2CylinderStickRay.Add ( float ( int ( color2[0] ) ) /255.0f );
					this->color2CylinderStickRay.Add ( float ( int ( color2[1] ) ) /255.0f );
					this->color2CylinderStickRay.Add ( float ( int ( color2[2] ) ) /255.0f );

					this->vertCylinderStickRay.Add ( position.GetX() );
					this->vertCylinderStickRay.Add ( position.GetY() );
					this->vertCylinderStickRay.Add ( position.GetZ() );
					this->vertCylinderStickRay.Add ( 1.0f );
				}
				// try to make the connection between this amino acid and its predecessor
				// --> only possible if the current amino acid is not the first in this chain
				if ( currentAminoAcid > 0 )
				{
					first = chain.AminoAcid() [currentAminoAcid-1].CCarbIndex();
					first += chain.AminoAcid() [currentAminoAcid-1].FirstAtomIndex();
					second = chain.AminoAcid() [currentAminoAcid].NIndex();
					second += chain.AminoAcid() [currentAminoAcid].FirstAtomIndex();

					firstAtomPos.SetX ( prot->ProteinAtomPositions() [first*3+0] );
					firstAtomPos.SetY ( prot->ProteinAtomPositions() [first*3+1] );
					firstAtomPos.SetZ ( prot->ProteinAtomPositions() [first*3+2] );
					color1 = this->GetProteinAtomColor ( first );

					secondAtomPos.SetX ( prot->ProteinAtomPositions() [second*3+0] );
					secondAtomPos.SetY ( prot->ProteinAtomPositions() [second*3+1] );
					secondAtomPos.SetZ ( prot->ProteinAtomPositions() [second*3+2] );
					color2 = this->GetProteinAtomColor ( second );

					// compute the quaternion for the rotation of the cylinder
					dir = secondAtomPos - firstAtomPos;
					tmpVec.Set ( 1.0f, 0.0f, 0.0f );
					angle = - tmpVec.Angle ( dir );
					ortho = tmpVec.Cross ( dir );
					ortho.Normalise();
					quatC.Set ( angle, ortho );
					// compute the absolute position 'position' of the cylinder (center point)
					position = firstAtomPos + ( dir/2.0f );

					// don't draw bonds that are too long
					if ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) > 3.5f )
						continue;

					this->inParaCylStickRaycasting.Add ( radiusStick/3.0f );
					this->inParaCylStickRaycasting.Add ( fabs ( ( firstAtomPos-secondAtomPos ).Length() ) );

					this->quatCylinderStickRay.Add ( quatC.GetX() );
					this->quatCylinderStickRay.Add ( quatC.GetY() );
					this->quatCylinderStickRay.Add ( quatC.GetZ() );
					this->quatCylinderStickRay.Add ( quatC.GetW() );

					this->color1CylinderStickRay.Add ( float ( int ( color1[0] ) ) /255.0f );
					this->color1CylinderStickRay.Add ( float ( int ( color1[1] ) ) /255.0f );
					this->color1CylinderStickRay.Add ( float ( int ( color1[2] ) ) /255.0f );

					this->color2CylinderStickRay.Add ( float ( int ( color2[0] ) ) /255.0f );
					this->color2CylinderStickRay.Add ( float ( int ( color2[1] ) ) /255.0f );
					this->color2CylinderStickRay.Add ( float ( int ( color2[2] ) ) /255.0f );

					this->vertCylinderStickRay.Add ( position.GetX() );
					this->vertCylinderStickRay.Add ( position.GetY() );
					this->vertCylinderStickRay.Add ( position.GetZ() );
					this->vertCylinderStickRay.Add ( 1.0f );
				}
			}
		}
		this->prepareBallAndStick = false;
	}

	// -----------
	// -- draw  --
	// -----------
	float viewportStuff[4] =
	{
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height()
	};
	if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	glDisable ( GL_BLEND );

	// enable sphere shader
	this->sphereShader.Enable();
	glEnableClientState ( GL_VERTEX_ARRAY );
	glEnableClientState ( GL_COLOR_ARRAY );
	// set shader variables
	glUniform4fvARB ( this->sphereShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
	// set vertex and color pointers and draw them
	glVertexPointer ( 4, GL_FLOAT, 0, this->vertSphereStickRay.PeekElements() );
	glColorPointer ( 3, GL_UNSIGNED_BYTE, 0, this->colorSphereStickRay.PeekElements() );
	glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertSphereStickRay.Count() /4 ) );
	// disable sphere shader
	glDisableClientState ( GL_COLOR_ARRAY );
	this->sphereShader.Disable();

	// enable cylinder shader
	this->cylinderShader.Enable();
	// set shader variables
	glUniform4fvARB ( this->cylinderShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
	glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
	glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
	glUniform3fvARB ( this->cylinderShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
	// get the attribute locations
	attribLocInParams = glGetAttribLocationARB ( this->cylinderShader, "inParams" );
	attribLocQuatC = glGetAttribLocationARB ( this->cylinderShader, "quatC" );
	attribLocColor1 = glGetAttribLocationARB ( this->cylinderShader, "color1" );
	attribLocColor2 = glGetAttribLocationARB ( this->cylinderShader, "color2" );
	// enable vertex attribute arrays for the attribute locations
	glEnableVertexAttribArrayARB ( this->attribLocInParams );
	glEnableVertexAttribArrayARB ( this->attribLocQuatC );
	glEnableVertexAttribArrayARB ( this->attribLocColor1 );
	glEnableVertexAttribArrayARB ( this->attribLocColor2 );
	// set vertex and attribute pointers and draw them
	glVertexPointer ( 4, GL_FLOAT, 0, this->vertCylinderStickRay.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylStickRaycasting.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinderStickRay.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1CylinderStickRay.PeekElements() );
	glVertexAttribPointerARB ( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2CylinderStickRay.PeekElements() );
	glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertCylinderStickRay.Count() /4 ) );
	// disable vertex attribute arrays for the attribute locations
	glDisableVertexAttribArrayARB ( this->attribLocInParams );
	glDisableVertexAttribArrayARB ( this->attribLocQuatC );
	glDisableVertexAttribArrayARB ( this->attribLocColor1 );
	glDisableVertexAttribArrayARB ( this->attribLocColor2 );
	glDisableClientState ( GL_VERTEX_ARRAY );
	// disable cylinder shader
	this->cylinderShader.Disable();

}


/*
 * protein::ProteinVolumeRenderer::RenderSpacefilling
 */
void protein::ProteinVolumeRenderer::RenderSpacefilling (
    const CallProteinData *prot )
{
	if ( this->prepareSpacefilling )
	{
		unsigned int i1;
		const unsigned char *color1;

		// -----------------------------
		// -- computation for spheres --
		// -----------------------------

		// clear vertex array for spheres
		this->vertSphereStickRay.Clear();
		// clear color array for sphere colors
		this->colorSphereStickRay.Clear();

		// store the points (will be rendered as spheres by the shader)
		for ( i1 = 0; i1 < prot->ProteinAtomCount(); i1++ )
		{
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+0] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+1] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+2] );
			this->vertSphereStickRay.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i1].TypeIndex() ].Radius() );

			color1 = this->GetProteinAtomColor ( i1 );
			this->colorSphereStickRay.Add ( color1[0] );
			this->colorSphereStickRay.Add ( color1[1] );
			this->colorSphereStickRay.Add ( color1[2] );
		}

		this->prepareSpacefilling = false;
	}

	// -----------
	// -- draw  --
	// -----------
	float viewportStuff[4] =
	{
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height()
	};
	if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	glDisable ( GL_BLEND );

	// enable sphere shader
	this->sphereShader.Enable();
	glEnableClientState ( GL_VERTEX_ARRAY );
	glEnableClientState ( GL_COLOR_ARRAY );
	// set shader variables
	glUniform4fvARB ( this->sphereShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
	// set vertex and color pointers and draw them
	glVertexPointer ( 4, GL_FLOAT, 0, this->vertSphereStickRay.PeekElements() );
	glColorPointer ( 3, GL_UNSIGNED_BYTE, 0, this->colorSphereStickRay.PeekElements() );
	glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertSphereStickRay.Count() /4 ) );
	// disable sphere shader
	glDisableClientState ( GL_VERTEX_ARRAY );
	glDisableClientState ( GL_COLOR_ARRAY );
	this->sphereShader.Disable();
}


/*
 * protein::ProteinVolumeRenderer::RenderClippedSpacefilling
 */
void protein::ProteinVolumeRenderer::RenderClippedSpacefilling (
	const vislib::math::Vector<float, 3> cpDir, const vislib::math::Vector<float, 3> cpBase,
	const CallProteinData *prot )
{
	if ( this->prepareSpacefilling )
	{
		unsigned int i1;
		const unsigned char *color1;

		// -----------------------------
		// -- computation for spheres --
		// -----------------------------

		// clear vertex array for spheres
		this->vertSphereStickRay.Clear();
		// clear color array for sphere colors
		this->colorSphereStickRay.Clear();

		// store the points (will be rendered as spheres by the shader)
		for ( i1 = 0; i1 < prot->ProteinAtomCount(); i1++ )
		{
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+0] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+1] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+2] );
			this->vertSphereStickRay.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i1].TypeIndex() ].Radius() );

			color1 = this->GetProteinAtomColor ( i1 );
			this->colorSphereStickRay.Add ( color1[0] );
			this->colorSphereStickRay.Add ( color1[1] );
			this->colorSphereStickRay.Add ( color1[2] );
		}

		this->prepareSpacefilling = false;
	}

	// -----------
	// -- draw  --
	// -----------
	float viewportStuff[4] =
	{
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height()
	};
	if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	glDisable ( GL_BLEND );

	// enable sphere shader
	this->clippedSphereShader.Enable();
	glEnableClientState ( GL_VERTEX_ARRAY );
	glEnableClientState ( GL_COLOR_ARRAY );
	// set shader variables
	glUniform4fvARB ( this->clippedSphereShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
	glUniform3fvARB ( this->clippedSphereShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
	glUniform3fvARB ( this->clippedSphereShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
	glUniform3fvARB ( this->clippedSphereShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
	glUniform3fvARB ( this->clippedSphereShader.ParameterLocation ( "clipPlaneDir" ), 1, cpDir.PeekComponents() );
	glUniform3fvARB ( this->clippedSphereShader.ParameterLocation ( "clipPlaneBase" ), 1, cpBase.PeekComponents() );
	// set vertex and color pointers and draw them
	glVertexPointer ( 4, GL_FLOAT, 0, this->vertSphereStickRay.PeekElements() );
	glColorPointer ( 3, GL_UNSIGNED_BYTE, 0, this->colorSphereStickRay.PeekElements() );
	glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertSphereStickRay.Count() /4 ) );
	// disable sphere shader
	glDisableClientState ( GL_VERTEX_ARRAY );
	glDisableClientState ( GL_COLOR_ARRAY );
	this->clippedSphereShader.Disable();
}


/*
 * protein::ProteinVolumeRenderer::RenderSolventAccessibleSurface
 */
void protein::ProteinVolumeRenderer::RenderSolventAccessibleSurface (
    const CallProteinData *prot )
{
	if ( this->prepareSAS )
	{
		unsigned int i1;
		const unsigned char *color1;

		// -----------------------------
		// -- computation for spheres --
		// -----------------------------

		// clear vertex array for spheres
		this->vertSphereStickRay.Clear();
		// clear color array for sphere colors
		this->colorSphereStickRay.Clear();

		// store the points (will be rendered as spheres by the shader)
		for ( i1 = 0; i1 < prot->ProteinAtomCount(); i1++ )
		{
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+0] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+1] );
			this->vertSphereStickRay.Add ( prot->ProteinAtomPositions() [i1*3+2] );
			this->vertSphereStickRay.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i1].TypeIndex() ].Radius() + this->probeRadius );

			color1 = this->GetProteinAtomColor ( i1 );
			this->colorSphereStickRay.Add ( color1[0] );
			this->colorSphereStickRay.Add ( color1[1] );
			this->colorSphereStickRay.Add ( color1[2] );
		}

		this->prepareSAS = false;
	}

	// -----------
	// -- draw  --
	// -----------
	float viewportStuff[4] =
	{
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height()
	};
	if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	glDisable ( GL_BLEND );

	// enable sphere shader
	this->sphereShader.Enable();
	glEnableClientState ( GL_VERTEX_ARRAY );
	glEnableClientState ( GL_COLOR_ARRAY );
	// set shader variables
	glUniform4fvARB ( this->sphereShader.ParameterLocation ( "viewAttr" ), 1, viewportStuff );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camIn" ), 1, cameraInfo->Front().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camRight" ), 1, cameraInfo->Right().PeekComponents() );
	glUniform3fvARB ( this->sphereShader.ParameterLocation ( "camUp" ), 1, cameraInfo->Up().PeekComponents() );
	// set vertex and color pointers and draw them
	glVertexPointer ( 4, GL_FLOAT, 0, this->vertSphereStickRay.PeekElements() );
	glColorPointer ( 3, GL_UNSIGNED_BYTE, 0, this->colorSphereStickRay.PeekElements() );
	glDrawArrays ( GL_POINTS, 0, ( unsigned int ) ( this->vertSphereStickRay.Count() /4 ) );
	// disable sphere shader
	glDisableClientState ( GL_VERTEX_ARRAY );
	glDisableClientState ( GL_COLOR_ARRAY );
	this->sphereShader.Disable();
}


/*
 * protein::ProteinVolumeRenderer::RenderDisulfideBondsLine
 */
void protein::ProteinVolumeRenderer::RenderDisulfideBondsLine (
    const CallProteinData *prot )
{
	// return if there are no disulfide bonds or drawDisulfideBonds is false
	if ( prot->DisulfidBondsCount() <= 0 || !drawDisulfideBonds )
		return;
	// lines can not be lighted --> turn light off
	glDisable ( GL_LIGHTING );
	
	// built the display list if it was not yet created
	if ( !glIsList ( this->disulfideBondsDisplayList ) )
	{
		// generate a new display list
		this->disulfideBondsDisplayList = glGenLists ( 1 );
		// compile new display list
		glNewList ( this->disulfideBondsDisplayList, GL_COMPILE );
	
		unsigned int i;
		unsigned int first, second;

		const float *protAtomPos = prot->ProteinAtomPositions();

		glPushAttrib ( GL_ENABLE_BIT | GL_POINT_BIT | GL_LINE_BIT | GL_POLYGON_BIT );
		glEnable ( GL_BLEND );
		glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glEnable ( GL_LINE_SMOOTH );
		glEnable ( GL_LINE_WIDTH );
		//glEnable( GL_LINE_STIPPLE);
		//glLineStipple( 1, 0xFF00);
		glLineWidth ( 3.0f );

		// set color of disulfide bonds to yellow
		glColor3f ( 1.0f, 1.0f, 0.0f );
		// draw bonds
		glBegin ( GL_LINES );
		for ( i = 0; i < prot->DisulfidBondsCount(); i++ )
		{
			first = prot->DisulfidBonds() [i].First();
			second = prot->DisulfidBonds() [i].Second();
			glVertex3f ( protAtomPos[first*3+0], protAtomPos[first*3+1], protAtomPos[first*3+2] );
			glVertex3f ( protAtomPos[second*3+0], protAtomPos[second*3+1], protAtomPos[second*3+2] );
		}
		glEnd(); // GL_LINES

		glPopAttrib();
		
		glEndList();
		vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO+200, "%s: Display list for disulfide bonds built.", this->ClassName() );
	}
	else
	{
		//draw the display list
		glCallList ( this->disulfideBondsDisplayList );
	}
	
	// turn light on after rendering
	glEnable ( GL_LIGHTING );
	glDisable ( GL_BLEND );
}


/*
 * protein::ProteinVolumeRenderer::MakeColorTable
 */
void protein::ProteinVolumeRenderer::MakeColorTable ( const CallProteinData *prot, bool forceRecompute )
{
	unsigned int i;
	unsigned int currentChain, currentAminoAcid, currentAtom, currentSecStruct;
	unsigned int cntCha, cntRes, cntAto;
	protein::CallProteinData::Chain chain;
	vislib::math::Vector<float, 3> color;
	// if recomputation is forced: clear current color table
	if ( forceRecompute )
	{
		this->protAtomColorTable.Clear();
	}
	// reserve memory for all atoms
	this->protAtomColorTable.AssertCapacity( prot->ProteinAtomCount() );
	// only compute color table if necessary
	if ( this->protAtomColorTable.IsEmpty() )
	{
		if ( this->currentColoringMode == ELEMENT )
		{
			for ( i = 0; i < prot->ProteinAtomCount(); i++ )
			{
				this->protAtomColorTable.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i].TypeIndex() ].Colour() [0] );
				this->protAtomColorTable.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i].TypeIndex() ].Colour() [1] );
				this->protAtomColorTable.Add ( prot->AtomTypes() [prot->ProteinAtomData() [i].TypeIndex() ].Colour() [2] );
			}
		} // ... END coloring mode ELEMENT
		else if ( this->currentColoringMode == AMINOACID )
		{
			// loop over all chains
			for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
			{
				chain = prot->ProteinChain ( currentChain );
				// loop over all amino acids in the current chain
				for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
				{
					// loop over all connections of the current amino acid
					for ( currentAtom = 0;
					        currentAtom < chain.AminoAcid() [currentAminoAcid].AtomCount();
					        currentAtom++ )
					{
						i = chain.AminoAcid() [currentAminoAcid].NameIndex() +1;
						i = i % ( unsigned int ) ( this->aminoAcidColorTable.Count() );
						this->protAtomColorTable.Add (
						    this->aminoAcidColorTable[i].GetX() );
						this->protAtomColorTable.Add (
						    this->aminoAcidColorTable[i].GetY() );
						this->protAtomColorTable.Add (
						    this->aminoAcidColorTable[i].GetZ() );
					}
				}
			}
		} // ... END coloring mode AMINOACID
		else if ( this->currentColoringMode == STRUCTURE )
		{
			// loop over all chains
			for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
			{
				chain = prot->ProteinChain ( currentChain );
				// loop over all secondary structure elements in this chain
				for ( currentSecStruct = 0;
				        currentSecStruct < chain.SecondaryStructureCount();
				        currentSecStruct++ )
				{
					i = chain.SecondaryStructure() [currentSecStruct].AtomCount();
					// loop over all atoms in this secondary structure element
					for ( currentAtom = 0; currentAtom < i; currentAtom++ )
					{
						if ( chain.SecondaryStructure() [currentSecStruct].Type() ==
						        protein::CallProteinData::SecStructure::TYPE_HELIX )
						{
							this->protAtomColorTable.Add ( 255 );
							this->protAtomColorTable.Add ( 0 );
							this->protAtomColorTable.Add ( 0 );
						}
						else if ( chain.SecondaryStructure() [currentSecStruct].Type() ==
						          protein::CallProteinData::SecStructure::TYPE_SHEET )
						{
							this->protAtomColorTable.Add ( 0 );
							this->protAtomColorTable.Add ( 0 );
							this->protAtomColorTable.Add ( 255 );
						}
						else if ( chain.SecondaryStructure() [currentSecStruct].Type() ==
						          protein::CallProteinData::SecStructure::TYPE_TURN )
						{
							this->protAtomColorTable.Add ( 255 );
							this->protAtomColorTable.Add ( 255 );
							this->protAtomColorTable.Add ( 0 );
						}
						else
						{
							this->protAtomColorTable.Add ( 230 );
							this->protAtomColorTable.Add ( 230 );
							this->protAtomColorTable.Add ( 230 );
						}
					}
				}
			}
			// add missing atom colors
			if ( prot->ProteinAtomCount() > ( this->protAtomColorTable.Count() / 3 ) )
			{
				currentAtom = (unsigned int)this->protAtomColorTable.Count() / 3;
				for ( ; currentAtom < prot->ProteinAtomCount(); ++currentAtom )
				{
					this->protAtomColorTable.Add ( 200 );
					this->protAtomColorTable.Add ( 200 );
					this->protAtomColorTable.Add ( 200 );
				}
			}
		} // ... END coloring mode STRUCTURE
		else if ( this->currentColoringMode == VALUE )
		{
			vislib::math::Vector<int, 3> colMax( 255,   0,   0);
			vislib::math::Vector<int, 3> colMid( 255, 255, 255);
			vislib::math::Vector<int, 3> colMin(   0,   0, 255);
			vislib::math::Vector<int, 3> col;
			
			float min( prot->MinimumTemperatureFactor() );
			float max( prot->MaximumTemperatureFactor() );
			float mid( ( max - min)/2.0f + min );
			float val;
			
			for ( i = 0; i < prot->ProteinAtomCount(); i++ )
			{
				if( min == max )
				{
					this->protAtomColorTable.Add( colMid.GetX() );
					this->protAtomColorTable.Add( colMid.GetY() );
					this->protAtomColorTable.Add( colMid.GetZ() );
					continue;
				}
				
				val = prot->ProteinAtomData()[i].TempFactor();
				// below middle value --> blend between min and mid color
				if( val < mid )
				{
					col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( val - min );
					this->protAtomColorTable.Add( col.GetX() );
					this->protAtomColorTable.Add( col.GetY() );
					this->protAtomColorTable.Add( col.GetZ() );
				}
				// above middle value --> blend between max and mid color
				else if( val > mid )
				{
					col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( val - mid );
					this->protAtomColorTable.Add( col.GetX() );
					this->protAtomColorTable.Add( col.GetY() );
					this->protAtomColorTable.Add( col.GetZ() );
				}
				// middle value --> assign mid color
				else
				{
					this->protAtomColorTable.Add( colMid.GetX() );
					this->protAtomColorTable.Add( colMid.GetY() );
					this->protAtomColorTable.Add( colMid.GetZ() );
				}
			}
		} // ... END coloring mode VALUE
		else if ( this->currentColoringMode == CHAIN_ID )
		{
			// loop over all chains
			for ( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
			{
				chain = prot->ProteinChain ( currentChain );
				// loop over all amino acids in the current chain
				for ( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
				{
					// loop over all connections of the current amino acid
					for ( currentAtom = 0;
					        currentAtom < chain.AminoAcid() [currentAminoAcid].AtomCount();
					        currentAtom++ )
					{
						i = ( currentChain + 1 ) % ( unsigned int ) ( this->aminoAcidColorTable.Count() );
						this->protAtomColorTable.Add (
						    this->aminoAcidColorTable[i].GetX() );
						this->protAtomColorTable.Add (
						    this->aminoAcidColorTable[i].GetY() );
						this->protAtomColorTable.Add (
						    this->aminoAcidColorTable[i].GetZ() );
					}
				}
			}
		} // ... END coloring mode CHAIN_ID
		else if ( this->currentColoringMode == RAINBOW )
		{
			for( cntCha = 0; cntCha < prot->ProteinChainCount(); ++cntCha )
			{
				for( cntRes = 0; cntRes < prot->ProteinChain( cntCha).AminoAcidCount(); ++cntRes )
				{
					i = int( ( float( cntRes) / float( prot->ProteinChain( cntCha).AminoAcidCount() ) ) * float( rainbowColors.size() ) );
					color = this->rainbowColors[i];
					for( cntAto = 0;
					     cntAto < prot->ProteinChain( cntCha).AminoAcid()[cntRes].AtomCount();
					     ++cntAto )
					{
						this->protAtomColorTable.Add( int(color.GetX() * 255.0f) );
						this->protAtomColorTable.Add( int(color.GetY() * 255.0f) );
						this->protAtomColorTable.Add( int(color.GetZ() * 255.0f) );
					}
				}
			}
		} // ... END coloring mode RAINBOW
		else if ( this->currentColoringMode == CHARGE )
		{
			vislib::math::Vector<int, 3> colMax( 255,   0,   0);
			vislib::math::Vector<int, 3> colMid( 255, 255, 255);
			vislib::math::Vector<int, 3> colMin(   0,   0, 255);
			vislib::math::Vector<int, 3> col;
			
			float min( prot->MinimumCharge() );
			float max( prot->MaximumCharge() );
			float mid( ( max - min)/2.0f + min );
			float charge;
			
			for ( i = 0; i < prot->ProteinAtomCount(); i++ )
			{
				if( min == max )
				{
					this->protAtomColorTable.Add( colMid.GetX() );
					this->protAtomColorTable.Add( colMid.GetY() );
					this->protAtomColorTable.Add( colMid.GetZ() );
					continue;
				}
				
				charge = prot->ProteinAtomData()[i].Charge();
				// below middle value --> blend between min and mid color
				if( charge < mid )
				{
					col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( charge - min );
					this->protAtomColorTable.Add( col.GetX() );
					this->protAtomColorTable.Add( col.GetY() );
					this->protAtomColorTable.Add( col.GetZ() );
				}
				// above middle value --> blend between max and mid color
				else if( charge > mid )
				{
					col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( charge - mid );
					this->protAtomColorTable.Add( col.GetX() );
					this->protAtomColorTable.Add( col.GetY() );
					this->protAtomColorTable.Add( col.GetZ() );
				}
				// middle value --> assign mid color
				else
				{
					this->protAtomColorTable.Add( colMid.GetX() );
					this->protAtomColorTable.Add( colMid.GetY() );
					this->protAtomColorTable.Add( colMid.GetZ() );
				}
			}
		} // ... END coloring mode CHARGE
	}
}


/*
 * protein::ProteinVolumeRenderer::RecomputeAll
 */
void protein::ProteinVolumeRenderer::RecomputeAll()
{
	this->prepareBallAndStick = true;
	this->prepareSpacefilling = true;
	this->prepareStickRaycasting = true;
	this->prepareSAS = true;

	glDeleteLists ( this->disulfideBondsDisplayList, 1 );
	this->disulfideBondsDisplayList = 0;

	glDeleteLists ( this->proteinDisplayListLines, 1 );
	this->proteinDisplayListLines = 0;

	this->protAtomColorTable.Clear();
}


/*
 * protein::ProteinVolumeRenderer::FillaminoAcidColorTable
 */
void protein::ProteinVolumeRenderer::FillAminoAcidColorTable()
{
	this->aminoAcidColorTable.Clear();
	this->aminoAcidColorTable.SetCount ( 25 );
	this->aminoAcidColorTable[0].Set ( 128, 128, 128 );
	this->aminoAcidColorTable[1].Set ( 255, 0, 0 );
	this->aminoAcidColorTable[2].Set ( 255, 255, 0 );
	this->aminoAcidColorTable[3].Set ( 0, 255, 0 );
	this->aminoAcidColorTable[4].Set ( 0, 255, 255 );
	this->aminoAcidColorTable[5].Set ( 0, 0, 255 );
	this->aminoAcidColorTable[6].Set ( 255, 0, 255 );
	this->aminoAcidColorTable[7].Set ( 128, 0, 0 );
	this->aminoAcidColorTable[8].Set ( 128, 128, 0 );
	this->aminoAcidColorTable[9].Set ( 0, 128, 0 );
	this->aminoAcidColorTable[10].Set ( 0, 128, 128 );
	this->aminoAcidColorTable[11].Set ( 0, 0, 128 );
	this->aminoAcidColorTable[12].Set ( 128, 0, 128 );
	this->aminoAcidColorTable[13].Set ( 255, 128, 0 );
	this->aminoAcidColorTable[14].Set ( 0, 128, 255 );
	this->aminoAcidColorTable[15].Set ( 255, 128, 255 );
	this->aminoAcidColorTable[16].Set ( 128, 64, 0 );
	this->aminoAcidColorTable[17].Set ( 255, 255, 128 );
	this->aminoAcidColorTable[18].Set ( 128, 255, 128 );
	this->aminoAcidColorTable[19].Set ( 192, 255, 0 );
	this->aminoAcidColorTable[20].Set ( 128, 0, 192 );
	this->aminoAcidColorTable[21].Set ( 255, 128, 128 );
	this->aminoAcidColorTable[22].Set ( 192, 255, 192 );
	this->aminoAcidColorTable[23].Set ( 192, 192, 128 );
	this->aminoAcidColorTable[24].Set ( 255, 192, 128 );
}

/*
 * protein::ProteinVolumeRenderer::makeRainbowColorTable
 * Creates a rainbow color table with 'num' entries.
 */
void protein::ProteinVolumeRenderer::MakeRainbowColorTable( unsigned int num)
{
	unsigned int n = (num/4);
	// the color table should have a minimum size of 16
	if( n < 4 )
		n = 4;
	this->rainbowColors.clear();
	float f = 1.0f/float(n);
	vislib::math::Vector<float,3> color;
	color.Set( 1.0f, 0.0f, 0.0f);
	for( unsigned int i = 0; i < n; i++)
	{
		color.SetY( vislib::math::Min( color.GetY() + f, 1.0f));
		rainbowColors.push_back( color);
	}
	for( unsigned int i = 0; i < n; i++)
	{
		color.SetX( vislib::math::Max( color.GetX() - f, 0.0f));
		rainbowColors.push_back( color);
	}
	for( unsigned int i = 0; i < n; i++)
	{
		color.SetZ( vislib::math::Min( color.GetZ() + f, 1.0f));
		rainbowColors.push_back( color);
	}
	for( unsigned int i = 0; i < n; i++)
	{
		color.SetY( vislib::math::Max( color.GetY() - f, 0.0f));
		rainbowColors.push_back( color);
	}
}

/*
 * Create a volume containing all protein atoms
 */
void protein::ProteinVolumeRenderer::UpdateVolumeTexture( const CallProteinData *protein) {
    /*
    float *tmpdata = new float[this->volumeSize*this->volumeSize*this->volumeSize];
    for( unsigned int z = 0; z < this->volumeSize; ++z ) {
        for( unsigned int y = 0; y < this->volumeSize; ++y ) {
            for( unsigned int x = 0; x < this->volumeSize; ++x ) {
                //tmpdata[x+y*this->volumeSize+z*this->volumeSize*this->volumeSize] = float(x) / float( this->volumeSize);
                //tmpdata[x+y*this->volumeSize+z*this->volumeSize*this->volumeSize] = float(y) / float( this->volumeSize);
                tmpdata[x+y*this->volumeSize+z*this->volumeSize*this->volumeSize] = float(z) / float( this->volumeSize);
            }
        }
    }
    */
    // generate volume, if necessary
    if( !glIsTexture( this->volumeTex) ) {
        // from CellVis: cellVis.cpp, initGL
        glGenTextures( 1, &this->volumeTex);
        glBindTexture( GL_TEXTURE_3D, this->volumeTex);
        glTexImage3D( GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB,
                      this->volumeSize, this->volumeSize, this->volumeSize, 0,
                      GL_LUMINANCE, GL_FLOAT, 0);
                      //GL_LUMINANCE, GL_FLOAT, tmpdata);
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
    /*
    delete[] tmpdata;
    // scale[i] = 1/extent[i] --- extent = size of the bbox
    this->volScale[0] = 1.0f / protein->BoundingBox().Width();
    this->volScale[1] = 1.0f / protein->BoundingBox().Height();
    this->volScale[2] = 1.0f / protein->BoundingBox().Depth();
    // scaleInv = 1 / scale = extend
    this->volScaleInv[0] = 1.0f / this->volScale[0];
    this->volScaleInv[1] = 1.0f / this->volScale[1];
    this->volScaleInv[2] = 1.0f / this->volScale[2];
    return;
    // END DEBUG
    */

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
    
    // clear 3d texture
    for( z = 0; z < this->volumeSize; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->volumeTex, 0, z);
        glRecti(-1, -1, 1, 1);
    }

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
    CHECK_FOR_OGL_ERROR();

    for( z = 0; z < this->volumeSize; ++z ) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
        glUniform1f( this->updateVolumeShader.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
        // draw all atoms as points, using w for radius
        glBegin( GL_POINTS);
        for( unsigned int cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {
            glVertex4f( ( protein->ProteinAtomPositions()[cnt*3+0] + this->translation.GetX()) * this->scale,
                ( protein->ProteinAtomPositions()[cnt*3+1] + this->translation.GetY()) * this->scale, 
                ( protein->ProteinAtomPositions()[cnt*3+2] + this->translation.GetZ()) * this->scale, 
                protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius() * this->scale );
        }
        glEnd(); // GL_POINTS
    }

    /*
    vislib::math::Vector<float, 3> tmpVec;
    for( z = 0; z < this->volumeSize; ++z ) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->volumeTex, 0, z);
        glUniform1f( this->updateVolumeShader.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
        // draw all atoms as points, using w for radius
        glBegin( GL_POINTS);
        
        tmpVec = protein->BoundingBox().GetLeftBottomFront();
        tmpVec = ( tmpVec + this->translation) * this->scale;
        glVertex4f( tmpVec.GetX(), tmpVec.GetY(), tmpVec.GetZ(), 1.5f * this->scale);
        
        tmpVec = protein->BoundingBox().GetLeftTopBack();
        tmpVec = ( tmpVec + this->translation) * this->scale;
        glVertex4f( tmpVec.GetX(), tmpVec.GetY(), tmpVec.GetZ(), 1.5f * this->scale);
        
        tmpVec = protein->BoundingBox().GetRightBottomBack();
        tmpVec = ( tmpVec + this->translation) * this->scale;
        glVertex4f( tmpVec.GetX(), tmpVec.GetY(), tmpVec.GetZ(), 1.5f * this->scale);
        
        tmpVec = protein->BoundingBox().GetRightTopFront();
        tmpVec = ( tmpVec + this->translation) * this->scale;
        glVertex4f( tmpVec.GetX(), tmpVec.GetY(), tmpVec.GetZ(), 1.5f * this->scale);

        tmpVec = protein->BoundingBox().CalcCenter();
        tmpVec = ( tmpVec + this->translation) * this->scale;
        glVertex4f( tmpVec.GetX(), tmpVec.GetY(), tmpVec.GetZ(), 1.5f * this->scale);
        glEnd(); // GL_POINTS
    }
    */

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

    // DEBUG check volume values
    /*
    float *texdata = new float[this->volumeSize*this->volumeSize*this->volumeSize];
    memset( texdata, 0, sizeof(float)*(this->volumeSize*this->volumeSize*this->volumeSize));
    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glGetTexImage( GL_TEXTURE_3D, 0, GL_LUMINANCE, GL_FLOAT, texdata);
    glBindTexture( GL_TEXTURE_3D, 0);
    int slab = 1;
    for( z = 1; z <= this->volumeSize*this->volumeSize; ++z ) {
        //std::cout << int( floor( texdata[slab*(this->volumeSize*this->volumeSize)+z] + 0.5)) << " ";
        std::cout << texdata[z+slab*(this->volumeSize*this->volumeSize)] << " ";
        if( z%this->volumeSize == 0 )
            std::cout << std::endl;
    }
    delete[] texdata;
    */
}

/*
 * draw the volume
 */
void protein::ProteinVolumeRenderer::RenderVolume( const CallProteinData *protein) {
    const float stepWidth = 1.0f/ ( 2.0f * float( this->volumeSize));
    glDisable( GL_BLEND);

    GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

    this->RayParamTextures( protein);
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

    glUniform1f( this->volumeShader.ParameterLocation( "isoValue"), this->isoValue);

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
 * write the parameters of the ray to the textures
 */
void protein::ProteinVolumeRenderer::RayParamTextures( const CallProteinData *protein) {

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
    vislib::math::Vector<float, 3> trans( protein->BoundingBox().GetSize().PeekDimension() );
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
    this->DrawBoundingBox( protein);

    // draw nearest frontfaces
    glCullFace( GL_BACK);
    glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    CHECK_FOR_OGL_ERROR();
    
    // draw bBox
    this->DrawBoundingBox( protein);

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
    this->DrawBoundingBox( protein);

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
void protein::ProteinVolumeRenderer::DrawBoundingBoxTranslated( const CallProteinData *protein) {

    vislib::math::Vector<float, 3> position;
    glBegin(GL_QUADS);
    {
        // back side
        glNormal3f(0.0f, 0.0f, -1.0f);
        glColor3f( 1, 0, 0);
        //glVertex3fv( protein->BoundingBox().GetLeftBottomBack().PeekCoordinates() );
        position = protein->BoundingBox().GetLeftBottomBack();
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetLeftTopBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftTopBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightTopBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightTopBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightBottomBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightBottomBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // front side
        glNormal3f(0.0f, 0.0f, 1.0f);
        glColor3f( 0.5, 0, 0);
        //glVertex3fv( protein->BoundingBox().GetLeftTopFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftTopFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetLeftBottomFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftBottomFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightBottomFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightBottomFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightTopFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightTopFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // top side
        glNormal3f(0.0f, 1.0f, 0.0f);
        glColor3f( 0, 1, 0);
        //glVertex3fv( protein->BoundingBox().GetLeftTopBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftTopBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetLeftTopFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftTopFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightTopFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightTopFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightTopBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightTopBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // bottom side
        glNormal3f(0.0f, -1.0f, 0.0f);
        glColor3f( 0, 0.5, 0);
        //glVertex3fv( protein->BoundingBox().GetLeftBottomFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftBottomFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetLeftBottomBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftBottomBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightBottomBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightBottomBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightBottomFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightBottomFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // left side
        glNormal3f(-1.0f, 0.0f, 0.0f);
        glColor3f( 0, 0, 1);
        //glVertex3fv( protein->BoundingBox().GetLeftTopFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftTopFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetLeftTopBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftTopBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetLeftBottomBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftBottomBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetLeftBottomFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetLeftBottomFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );

        // right side
        glNormal3f(1.0f, 0.0f, 0.0f);
        glColor3f( 0, 0, 0.5);
        //glVertex3fv( protein->BoundingBox().GetRightTopBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightTopBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightTopFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightTopFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightBottomFront().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightBottomFront());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
        //glVertex3fv( protein->BoundingBox().GetRightBottomBack().PeekCoordinates() );
        position = (protein->BoundingBox().GetRightBottomBack());
        position = protein->BoundingBox().CalcCenter() + position * -1.0f;
        position *= -this->scale;
        //position = ( position + this->translation ) * this->scale;
        glVertex3fv( position.PeekComponents() );
    }
    glEnd();
}

/*
 * Draw the bounding box.
 */
void protein::ProteinVolumeRenderer::DrawBoundingBox( const CallProteinData *protein) {

    vislib::math::Vector<float, 3> position( protein->BoundingBox().GetSize().PeekDimension() );
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
}
