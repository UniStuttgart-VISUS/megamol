

/*
 * SolventVolumeRenderer.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "SolventVolumeRenderer.h"
#include "VolumeSliceCall.h"
#include "Diagram2DCall.h"
#include "CoreInstance.h"
#include "Color.h"
#include "param/EnumParam.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "param/FloatParam.h"
#include "param/Vector3fParam.h"
#include "param/StringParam.h"
#include "utility/ShaderSourceFactory.h"
#include "utility/ColourParser.h"
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
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
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
 * protein::SolventVolumeRenderer::SolventVolumeRenderer (CTOR)
 */
protein::SolventVolumeRenderer::SolventVolumeRenderer ( void ) : Renderer3DModule (),
		protDataCallerSlot ( "getData", "Connects the volume rendering with data storage" ),
		callFrameCalleeSlot ( "callFrame", "Connects the volume rendering with frame call from RMS renderer" ),
	//	protRendererCallerSlot ( "renderProtein", "Connects the volume rendering with a protein renderer" ),
		dataOutSlot ( "volumeout", "Connects the volume rendering with a volume slice renderer" ),
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
		interpolParam( "posInterpolation", "Enable positional interpolation between frames" ),
		colorTableFileParam( "colorTableFilename", "The filename of the color table."),
		minGradColorParam( "minGradColor", "The color for the minimum value for gradient coloring" ),
		midGradColorParam( "midGradColor", "The color for the middle value for gradient coloring" ),
		maxGradColorParam( "maxGradColor", "The color for the maximum value for gradient coloring" ),
		solventResidues("solventResidues", ";-list of residue names which compose the solvent"),
		stickRadiusParam( "stickRadius", "The radius for stick rendering"),
		solventMolThreshold( "solventMolThreshold", "threshold of visible solvent-molecules" ),
		currentFrameId ( 0 ), atomCount( 0 ), volumeTex( 0), volumeSize( 128), volFBO( 0),
		volFilterRadius( 1.75f), volDensityScale( 1.0f),
		width( 0), height( 0), volRayTexWidth( 0), volRayTexHeight( 0),
		volRayStartTex( 0), volRayLengthTex( 0), volRayDistTex( 0),
		renderIsometric( true), meanDensityValue( 0.0f), isoValue1( 0.5f), isoValue2(-0.5f),
		volIsoOpacity( 0.4f), volClipPlaneFlag( false), volClipPlaneOpacity( 0.4f),
		forceUpdateVolumeTexture( true)
{
	// set caller slot for different data calls
	this->protDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable ( &this->protDataCallerSlot );

	// set frame callee slot
	protein::CallFrameDescription dfd;
//	this->callFrameCalleeSlot.SetCallback ( dfd.ClassName(), "CallFrame", &SolventVolumeRenderer::ProcessFrameRequest );
	this->MakeSlotAvailable ( &this->callFrameCalleeSlot );

	// set renderer caller slot
//	this->protRendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
//	this->MakeSlotAvailable( &this->protRendererCallerSlot);

	// --- set the coloring mode ---
	this->SetColoringMode ( Color::ELEMENT );
	param::EnumParam *cm = new param::EnumParam ( int ( this->currentColoringMode ) );
	MolecularDataCall *mol = new MolecularDataCall();
	unsigned int cCnt;
	Color::ColoringMode cMode;
	for( cCnt = 0; cCnt < Color::GetNumOfColoringModes( mol); ++cCnt) {
		cMode = Color::GetModeByIndex( mol, cCnt);
		cm->SetTypePair( cMode, Color::GetName( cMode).c_str());
	}
	delete mol;
	/*
	cm->SetTypePair( Color::ELEMENT, "Element" );
	cm->SetTypePair( Color::AMINOACID, "AminoAcid" );
	cm->SetTypePair( Color::STRUCTURE, "SecondaryStructure" );
	cm->SetTypePair( Color::VALUE, "Value");
	cm->SetTypePair( Color::CHAIN_ID, "ChainID" );
	cm->SetTypePair( Color::RAINBOW, "Rainbow");
	cm->SetTypePair( Color::CHARGE, "Charge" );
	*/
	this->coloringModeParam << cm;
	this->MakeSlotAvailable( &this->coloringModeParam );

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
	this->volClipPlane.Add( vislib::math::Vector<double, 4>( 0.0, 1.0, 0.0, 0.0));

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
	this->volClipPlane0DistParam.SetParameter( new param::FloatParam( d, 0.0f, 1.0f) );
	// --- set up parameter for clipping plane opacity ---
	this->volClipPlaneOpacityParam.SetParameter( new param::FloatParam( this->volClipPlaneOpacity, 0.0f, 1.0f ) );

	// en-/disable positional interpolation
	this->interpolParam.SetParameter(new param::BoolParam( true));
	this->MakeSlotAvailable( &this->interpolParam);

	// fill color table with default values and set the filename param
	vislib::StringA filename( "colors.txt");
	Color::ReadColorTableFromFile( filename, this->colorLookupTable);
	this->colorTableFileParam.SetParameter(new param::StringParam( A2T( filename)));
	this->MakeSlotAvailable( &this->colorTableFileParam);

	// the color for the minimum value (gradient coloring
	this->minGradColorParam.SetParameter(new param::StringParam( "#146496"));
	this->MakeSlotAvailable( &this->minGradColorParam);

	// the color for the middle value (gradient coloring
	this->midGradColorParam.SetParameter(new param::StringParam( "#f0f0f0"));
	this->MakeSlotAvailable( &this->midGradColorParam);

	// the color for the maximum value (gradient coloring
	this->maxGradColorParam.SetParameter(new param::StringParam( "#ae3b32"));
	this->MakeSlotAvailable( &this->maxGradColorParam);

	// ;-list of residue names which compose the solvent
	this->solventResidues.SetParameter(new param::StringParam(""));
	this->MakeSlotAvailable( &this->solventResidues);

	// fill color table with default values and set the filename param
	this->stickRadiusParam.SetParameter(new param::FloatParam( 0.3f, 0.0f));
	this->MakeSlotAvailable( &this->stickRadiusParam);

	// 
	this->solventMolThreshold.SetParameter(new param::FloatParam( 0.1f, 0.0f));
	this->MakeSlotAvailable( &this->solventMolThreshold);

	// make all slots available
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

	for(int i = 0; i < TEMPORARY_ATOM_ARRAYS_CNT; i++)
		temporaryAtomArray[i] = 0;
}


/*
 * protein::SolventVolumeRenderer::~SolventVolumeRenderer (DTOR)
 */
protein::SolventVolumeRenderer::~SolventVolumeRenderer ( void ) {
	delete this->frameLabel;
	this->Release ();

	for(int i = 0; i < TEMPORARY_ATOM_ARRAYS_CNT; i++)
		delete temporaryAtomArray[i];
}


/*
 * protein::SolventVolumeRenderer::release
 */
void protein::SolventVolumeRenderer::release ( void ) {

}


bool protein::SolventVolumeRenderer::loadShader(vislib::graphics::gl::GLSLShader& shader, const vislib::StringA& vert, const vislib::StringA& frag) {
	using namespace vislib::sys;
	using namespace vislib::graphics::gl;
	ShaderSource vertSrc;
	ShaderSource fragSrc;

	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( vert, vertSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for shader '%s'", this->ClassName(), vert.PeekBuffer() );
		return false;
	}
	if ( !this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource ( frag, fragSrc ) ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for shader '%s'", this->ClassName(), frag.PeekBuffer() );
		return false;
	}
	try {
		if ( !shader.Create ( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
			throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
		}
	} catch ( vislib::Exception e ) {
		Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA() );
		return false;
	}

	return true;
}

/*
 * protein::SolventVolumeRenderer::create
 */
bool protein::SolventVolumeRenderer::create ( void ) {
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

	// Load sphere shader
	if( !loadShader( this->sphereSolventShader, "protein::std::sphereSolventVertex", "protein::std::sphereFragment" ) )
		return false;

	// Load clipped sphere shader -> TODO: solvent version?
	if( !loadShader( this->clippedSphereShader, "protein::std::sphereClipPlaneVertex", "protein::std::sphereClipPlaneFragment" ) )
		return false;

	// Load cylinder shader
	if( !loadShader( this->cylinderSolventShader, "protein::std::cylinderSolventVertex", "protein::std::cylinderFragment" ) )
		return false;

	// Load volume texture generation shader
	if( !loadShader( this->updateVolumeShaderMoleculeVolume, "volume::std::updateVolumeVertex", "volume::std::updateSolventVolumeFragmentDensity" ) )
		return false;
	if( !loadShader( this->updateVolumeShaderSolventColor, "volume::std::updateVolumeVertex", "volume::std::updateSolventVolumeFragmentColor" ) )
		return false;

	// Load ray start shader
	if( !loadShader( this->volRayStartShader, "volume::std::rayStartVertex", "volume::std::rayStartFragment" ) )
		return false;

	// Load ray start eye shader
	if( !loadShader( this->volRayStartEyeShader, "volume::std::rayStartEyeVertex", "volume::std::rayStartEyeFragment" ) )
		return false;

	// Load ray length shader (uses same vertex shader as ray start shader)
	if( !loadShader( this->volRayLengthShader, "volume::std::rayStartVertex", "volume::std::rayLengthFragment" ) )
		return false;

	// Load volume rendering shader
	if( !loadShader( this->volumeShader, "volume::std::volumeVertex", "volume::std::volumeFragment" ) )
		return false;

	// Load dual isosurface rendering shader
	if( !loadShader( this->dualIsosurfaceShader, "volume::std::volumeDualIsoVertex", "volume::std::volumeDualIsoFragment" ) )
		return false;

	return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::ProteinRenderer::GetCapabilities
 */
bool protein::SolventVolumeRenderer::GetCapabilities( Call& call) {
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
bool protein::SolventVolumeRenderer::GetExtents( Call& call) {
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL) return false;

	MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();

	float scale, xoff, yoff, zoff;
	vislib::math::Cuboid<float> boundingBox;
	vislib::math::Point<float, 3> bbc;

	// try to get the bounding box from the active data call
	if( mol ) {
		// try to call the molecular data
		if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;
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
/*	view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
	vislib::math::Point<float, 3> protrenbbc;
	if( protrencr3d ) {
		(*protrencr3d)(1); // GetExtents
		BoundingBoxes &protrenbb = protrencr3d->AccessBoundingBoxes();
		this->protrenScale =  protrenbb.ObjectSpaceBBox().Width() / boundingBox.Width();
		//this->protrenTranslate = ( protrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
		if( mol ) {
			this->protrenTranslate.Set( xoff, yoff, zoff);
			this->protrenTranslate *= scale;
		} else {
			this->protrenTranslate = ( protrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
		}
	}*/

	if( mol ) {
		this->protrenTranslate.Set( xoff, yoff, zoff);
		this->protrenTranslate *= scale;
	}

	return true;
}


/*
 * SolventVolumeRenderer::getVolumeData
 */
bool SolventVolumeRenderer::getVolumeData( core::Call& call) {
	VolumeSliceCall *c = dynamic_cast<VolumeSliceCall*>( &call);
	if( c == NULL ) return false;

	// get the data call
	MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();

	// set the bounding box dimensions
	vislib::math::Cuboid<float> box( 0, 0, 0, 1, 1, 1);
	vislib::math::Vector<float, 3> dim;
	if( mol )
		box = mol->AccessBoundingBoxes().ObjectSpaceBBox();
	dim = vislib::math::Vector<float, 3>( box.GetSize().PeekDimension()) / box.LongestEdge();
	c->setBBoxDimensions( dim);
	// set the volume texture id
	c->setVolumeTex( this->volumeTex);
	// set the texture r coordinate
	c->setTexRCoord( this->volClipPlane0DistParam.Param<param::FloatParam>()->Value());
	// set the clip plane normal
	c->setClipPlaneNormal( this->volClipPlane0NormParam.Param<param::Vector3fParam>()->Value());
	// set the isovalue
	c->setIsovalue( this->isoValue1);

	return true;
}


/*
 * protein::SolventVolumeRenderer::Render
 */
bool protein::SolventVolumeRenderer::Render( Call& call ) {
	// cast the call to Render3D
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>( &call );
	if( !cr3d ) return false;
	// get the pointer to CallRender3D (protein renderer)
	//view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
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


	// =============== Refresh all parameters ===============
	this->ParameterRefresh( cr3d);
	
	// get the call time
	float callTime = cr3d->Time();

	// make the atom color table if necessary
	if( mol ) {
		// set frame ID and call data
		mol->SetFrameID(static_cast<int>( callTime));
		if( !(*mol)(MolecularDataCall::CallForGetData) )
			return false;
		// check if atom count is zero
		if( mol->AtomCount() == 0 ) return true;
		// get positions of the first frame
		int cnt;
		// do not use new-operator in render routines!
		float *pos0 = getTemporaryAtomArray(mol->AtomCount()*3, POS_0); //new float[mol->AtomCount() * 3];
		memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
		// check if the atom positions have to be interpolated
		bool interpolate = this->interpolParam.Param<param::BoolParam>()->Value();
		float *pos1;
		if( interpolate ) {
			// set next frame ID and get positions of the second frame
			if( ( static_cast<int>( callTime) + 1) < int(mol->FrameCount()) ) 
				mol->SetFrameID(static_cast<int>( callTime) + 1);
			else
				mol->SetFrameID(static_cast<int>( callTime));
			if (!(*mol)(MolecularDataCall::CallForGetData)) {
				//delete[] pos0;
				return false;
			}
			pos1 = getTemporaryAtomArray(mol->AtomCount()*3, POS_1);// new float[mol->AtomCount() * 3];
			memcpy( pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
			// interpolate atom positions between frames
			posInter = getTemporaryAtomArray(mol->AtomCount()*3, POS_INTER); // new float[mol->AtomCount() * 3];
			float inter = callTime - static_cast<float>(static_cast<int>( callTime));
			float threshold = vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
				vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
				mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
			for( cnt = 0; cnt < int(mol->AtomCount()); ++cnt ) {
				if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
						std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
						std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
					posInter[3*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
					posInter[3*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
					posInter[3*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
				} else if( inter < 0.5f ) {
					posInter[3*cnt+0] = pos0[3*cnt+0];
					posInter[3*cnt+1] = pos0[3*cnt+1];
					posInter[3*cnt+2] = pos0[3*cnt+2];
				} else {
					posInter[3*cnt+0] = pos1[3*cnt+0];
					posInter[3*cnt+1] = pos1[3*cnt+1];
					posInter[3*cnt+2] = pos1[3*cnt+2];
				}
			}
		} else {
			posInter = pos0;
		}

		Color::MakeColorTable( mol, 
			this->currentColoringMode,
			this->atomColorTable,
			this->colorLookupTable,
			this->rainbowColors,
			this->minGradColorParam.Param<param::StringParam>()->Value(),
			this->midGradColorParam.Param<param::StringParam>()->Value(),
			this->maxGradColorParam.Param<param::StringParam>()->Value(),
			true);
	}


	unsigned int cpCnt;
	for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
		glClipPlane( GL_CLIP_PLANE0+cpCnt, this->volClipPlane[cpCnt].PeekComponents());
	}

	// =============== Volume Rendering ===============
	bool retval = false;

	// try to start volume rendering using protein data
	if( mol ) {

#if 0
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
			//glScalef( this->protrenScale, this->protrenScale, this->protrenScale);
			*protrencr3d = *cr3d;
			protrencr3d->SetOutputBuffer( &this->proteinFBO); // TODO: Handle incoming buffers!
			(*protrencr3d)();
			glPopMatrix();
		}
		// stop rendering to the FBO for protein rendering
		this->proteinFBO.Disable();
		// re-enable the output buffer
		cr3d->EnableOutputBuffer();
#else
		// =============== Protein Rendering ===============
		// disable the output buffer
		cr3d->DisableOutputBuffer();
		// start rendering to the FBO for protein rendering
		this->proteinFBO.Enable();
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		// render molecules based on volume density (TODO!)
		glPushMatrix();
			glDisable( GL_BLEND);
			glEnable( GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
			glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
			glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
			glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

			glTranslatef( this->protrenTranslate.X(), this->protrenTranslate.Y(), this->protrenTranslate.Z());
			//glScalef( this->protrenScale, this->protrenScale, this->protrenScale);
			// compute scale factor and scale world
			float scale;
			if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) 
				scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
			else
				scale = 1.0f;
			glScalef( scale, scale, scale);
			//cr3d->SetOutputBuffer( &this->proteinFBO); // TODO: Handle incoming buffers!
			RenderStickSolvent(mol, this->posInter);
		glPopMatrix();
		// stop rendering to the FBO for protein rendering
		this->proteinFBO.Disable();
		// re-enable the output buffer
		cr3d->EnableOutputBuffer();
#endif

		// =============== Volume Rendering ===============
		retval = this->RenderMolecularData( cr3d, mol);
		// unlock the current frame
		mol->Unlock();
	}

	return retval;
}

/*
 * Render the molecular data in stick mode. Special case when using solvent rendering: only render solvent molecules near the isosurface between the solvent and the molecule.
 */
void protein::SolventVolumeRenderer::RenderStickSolvent( /*const*/ MolecularDataCall *mol, const float *atomPos) {
	// ----- prepare stick raycasting -----
	this->vertSpheres.SetCount( mol->AtomCount() * 4 );
	this->vertCylinders.SetCount( mol->ConnectionCount() * 4);
	this->quatCylinders.SetCount( mol->ConnectionCount() * 4);
	this->inParaCylinders.SetCount( mol->ConnectionCount() * 2);
	this->color1Cylinders.SetCount( mol->ConnectionCount() * 3);
	this->color2Cylinders.SetCount( mol->ConnectionCount() * 3);

	int cnt;

	// copy atom pos and radius to vertex array
#pragma omp parallel for
	for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
		this->vertSpheres[4*cnt+0] = atomPos[3*cnt+0];
		this->vertSpheres[4*cnt+1] = atomPos[3*cnt+1];
		this->vertSpheres[4*cnt+2] = atomPos[3*cnt+2];
		this->vertSpheres[4*cnt+3] =
			this->stickRadiusParam.Param<param::FloatParam>()->Value();
	}

	unsigned int idx0, idx1;
	vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
	vislib::math::Quaternion<float> quatC( 0, 0, 0, 1);
	vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
	float angle;
	// loop over all connections and compute cylinder parameters
#pragma omp parallel for private( idx0, idx1, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
	for( cnt = 0; cnt < int( mol->ConnectionCount()); ++cnt ) {
		idx0 = mol->Connection()[2*cnt];
		idx1 = mol->Connection()[2*cnt+1];

		firstAtomPos.SetX( atomPos[3*idx0+0]);
		firstAtomPos.SetY( atomPos[3*idx0+1]);
		firstAtomPos.SetZ( atomPos[3*idx0+2]);

		secondAtomPos.SetX( atomPos[3*idx1+0]);
		secondAtomPos.SetY( atomPos[3*idx1+1]);
		secondAtomPos.SetZ( atomPos[3*idx1+2]);

		// compute the quaternion for the rotation of the cylinder
		dir = secondAtomPos - firstAtomPos;
		tmpVec.Set( 1.0f, 0.0f, 0.0f);
		angle = - tmpVec.Angle( dir);
		ortho = tmpVec.Cross( dir);
		ortho.Normalise();
		quatC.Set( angle, ortho);
		// compute the absolute position 'position' of the cylinder (center point)
		position = firstAtomPos + (dir/2.0f);

		this->inParaCylinders[2*cnt] = this->stickRadiusParam.Param<param::FloatParam>()->Value();
		this->inParaCylinders[2*cnt+1] = ( firstAtomPos-secondAtomPos).Length();

		// thomasbm: hotfix for jumping molecules near bounding box
		if(this->inParaCylinders[2*cnt+1] > mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() + mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius() ) {
			this->inParaCylinders[2*cnt+1] = 0;
		}

		this->quatCylinders[4*cnt+0] = quatC.GetX();
		this->quatCylinders[4*cnt+1] = quatC.GetY();
		this->quatCylinders[4*cnt+2] = quatC.GetZ();
		this->quatCylinders[4*cnt+3] = quatC.GetW();

		this->color1Cylinders[3*cnt+0] = this->atomColorTable[3*idx0+0];
		this->color1Cylinders[3*cnt+1] = this->atomColorTable[3*idx0+1];
		this->color1Cylinders[3*cnt+2] = this->atomColorTable[3*idx0+2];

		this->color2Cylinders[3*cnt+0] = this->atomColorTable[3*idx1+0];
		this->color2Cylinders[3*cnt+1] = this->atomColorTable[3*idx1+1];
		this->color2Cylinders[3*cnt+2] = this->atomColorTable[3*idx1+2];

		this->vertCylinders[4*cnt+0] = position.X();
		this->vertCylinders[4*cnt+1] = position.Y();
		this->vertCylinders[4*cnt+2] = position.Z();
		this->vertCylinders[4*cnt+3] = 0.0f;
	}

	// ---------- actual rendering ----------

	// get viewpoint parameters for raycasting
	float viewportStuff[4] = {
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height()};
	if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
	if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	// volume texture to look up densities ...
	glActiveTexture( GL_TEXTURE0);
	glBindTexture( GL_TEXTURE_3D, this->volumeTex);
	CHECK_FOR_OGL_ERROR();

	vislib::math::Cuboid<float> bbox;
	bbox = mol->AccessBoundingBoxes().ObjectSpaceBBox();
	vislib::math::Vector<float, 3> invBBoxDimension(1/bbox.Width(), 1/bbox.Height(), 1/bbox.Depth());

	// enable sphere shader
	this->sphereSolventShader.Enable();
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	// set shader variables
	glUniform4fvARB(this->sphereSolventShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB(this->sphereSolventShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB(this->sphereSolventShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB(this->sphereSolventShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
	glUniform1iARB(this->sphereSolventShader.ParameterLocation("volumeSampler"), 0);
	glUniform3fvARB(this->sphereSolventShader.ParameterLocation("minBBox"), 1, bbox.GetOrigin().PeekCoordinates());
	glUniform3fvARB(this->sphereSolventShader.ParameterLocation("invBBoxExtend"), 1, invBBoxDimension.PeekComponents() );
	glUniform1fARB(this->sphereSolventShader.ParameterLocation("solventMolThreshold"), solventMolThreshold.Param<param::FloatParam>()->Value() );
	// set vertex and color pointers and draw them
	glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());
	glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements()); 
	glDrawArrays( GL_POINTS, 0, mol->AtomCount());
	// disable sphere shader
	this->sphereSolventShader.Disable();

	// TEST: no cylinders for now ...
#if 1
	// enable cylinder shader
	this->cylinderSolventShader.Enable();
	// set shader variables
	glUniform4fvARB( this->cylinderSolventShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB( this->cylinderSolventShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB( this->cylinderSolventShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB( this->cylinderSolventShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
	glUniform1iARB(this->cylinderSolventShader.ParameterLocation("volumeSampler"), 0);
	glUniform3fvARB(this->cylinderSolventShader.ParameterLocation("minBBox"), 1, bbox.GetOrigin().PeekCoordinates());
	glUniform3fvARB(this->cylinderSolventShader.ParameterLocation("invBBoxExtend"), 1, invBBoxDimension.PeekComponents() );
	glUniform1fARB(this->cylinderSolventShader.ParameterLocation("solventMolThreshold"), solventMolThreshold.Param<param::FloatParam>()->Value() );
	// get the attribute locations
	attribLocInParams = glGetAttribLocationARB( this->cylinderSolventShader, "inParams");
	attribLocQuatC = glGetAttribLocationARB( this->cylinderSolventShader, "quatC");
	attribLocColor1 = glGetAttribLocationARB( this->cylinderSolventShader, "color1");
	attribLocColor2 = glGetAttribLocationARB( this->cylinderSolventShader, "color2");
	// enable vertex attribute arrays for the attribute locations
	glDisableClientState( GL_COLOR_ARRAY);
	glEnableVertexAttribArrayARB( this->attribLocInParams);
	glEnableVertexAttribArrayARB( this->attribLocQuatC);
	glEnableVertexAttribArrayARB( this->attribLocColor1);
	glEnableVertexAttribArrayARB( this->attribLocColor2);
	// set vertex and attribute pointers and draw them
	glVertexPointer( 4, GL_FLOAT, 0, this->vertCylinders.PeekElements());
	glVertexAttribPointerARB( this->attribLocInParams, 2, GL_FLOAT, 0, 0, this->inParaCylinders.PeekElements());
	glVertexAttribPointerARB( this->attribLocQuatC, 4, GL_FLOAT, 0, 0, this->quatCylinders.PeekElements());
	glVertexAttribPointerARB( this->attribLocColor1, 3, GL_FLOAT, 0, 0, this->color1Cylinders.PeekElements());
	glVertexAttribPointerARB( this->attribLocColor2, 3, GL_FLOAT, 0, 0, this->color2Cylinders.PeekElements());
	glDrawArrays( GL_POINTS, 0, mol->ConnectionCount());
	// disable vertex attribute arrays for the attribute locations
	glDisableVertexAttribArrayARB( this->attribLocInParams);
	glDisableVertexAttribArrayARB( this->attribLocQuatC);
	glDisableVertexAttribArrayARB( this->attribLocColor1);
	glDisableVertexAttribArrayARB( this->attribLocColor2);
	glDisableClientState(GL_VERTEX_ARRAY);
	// disable cylinder shader
	this->cylinderSolventShader.Disable();
#endif

	glBindTexture( GL_TEXTURE_3D, 0 ); // state aufräumen
}


/*
 * Volume rendering using molecular data.
 */
bool protein::SolventVolumeRenderer::RenderMolecularData( view::CallRender3D *call, MolecularDataCall *mol) {

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
		this->forceUpdateVolumeTexture = true;
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
	// --- Volume Rendering									 ---
	// --- update & render the volume						   ---
	// ------------------------------------------------------------
	
	vislib::StringA paramSlotName( call->PeekCallerSlot()->Parent()->FullName());
	paramSlotName += "::anim::play";
	param::ParamSlot *paramSlot = dynamic_cast<param::ParamSlot*>( this->FindNamedObject( paramSlotName, true));
	if( paramSlot->Param<param::BoolParam>()->Value() || this->forceUpdateVolumeTexture ) {
		this->UpdateVolumeTexture( mol);
		CHECK_FOR_OGL_ERROR();
		this->forceUpdateVolumeTexture = false;
	}

	this->proteinFBO.DrawColourTexture();
	CHECK_FOR_OGL_ERROR();

	unsigned int cpCnt;
	if( this->volClipPlaneFlag ) {
		for( cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt ) {
			glEnable( GL_CLIP_PLANE0+cpCnt );
		}
	}

	this->RenderVolume( mol->AccessBoundingBoxes().ObjectSpaceBBox());
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
void protein::SolventVolumeRenderer::ParameterRefresh( view::CallRender3D *call) {
	
	// parameter refresh
	if( this->coloringModeParam.IsDirty() ) {
		this->SetColoringMode ( static_cast<Color::ColoringMode> ( int ( this->coloringModeParam.Param<param::EnumParam>()->Value() ) ) );
		this->coloringModeParam.ResetDirty();
		this->forceUpdateVolumeTexture = true;
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
		this->forceUpdateVolumeTexture = true;
	}
	if( this->volDensityScaleParam.IsDirty() ) {
		this->volDensityScale = this->volDensityScaleParam.Param<param::FloatParam>()->Value();
		this->volDensityScaleParam.ResetDirty();
		this->forceUpdateVolumeTexture = true;
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

	// update color table
	if( this->colorTableFileParam.IsDirty() ) {
		Color::ReadColorTableFromFile( this->colorTableFileParam.Param<param::StringParam>()->Value(), this->colorLookupTable);
		this->colorTableFileParam.ResetDirty();
		this->forceUpdateVolumeTexture = true;
	}

	// update lsit of residues that compose the solvent ...
	if (this->solventResidues.IsDirty()) {
		const vislib::TString& str = solventResidues.Param<param::StringParam>()->Value();
		vislib::Array<vislib::TString> residueTokens = vislib::StringTokeniser<vislib::TCharTraits>::Split(str, ';', true);

		// get the data call
		MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();
		if (mol && (*mol)(MolecularDataCall::CallForGetData)) {
			vislib::StringA* residueTypes = mol->ResidueTypeNames();

			for(int i = 0; i < residueTokens.Count(); i++) {
				vislib::TString token = residueTokens[i];
				token.TrimSpaces();

				for(int rIndex = 0; rIndex < mol->ResidueTypeNameCount(); rIndex++ ) {
					if (residueTypes[rIndex].Equals(token) ) {
						this->solventResidueTypeIds.Add(rIndex);
						continue;
					}
				}
			}
		}
	}
}

/**
 * protein::SolventVolumeRenderer::DrawLabel
 */
void protein::SolventVolumeRenderer::DrawLabel( unsigned int frameID )
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
			vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_WARN, "SolventVolumeRenderer: Problems to initalise the Font" );
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
 * Create a volume containing all molecule atoms
 */
void protein::SolventVolumeRenderer::UpdateVolumeTexture( MolecularDataCall *mol) {
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
	this->volScale[0] = 1.0f / ( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width() * this->scale);
	this->volScale[1] = 1.0f / ( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height() * this->scale);
	this->volScale[2] = 1.0f / ( mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth() * this->scale);
	// scaleInv = 1 / scale = extend
	this->volScaleInv[0] = 1.0f / this->volScale[0];
	this->volScaleInv[1] = 1.0f / this->volScale[1];
	this->volScaleInv[2] = 1.0f / this->volScale[2];
	


	float *updatVolumeTextureAtoms = getTemporaryAtomArray(mol->AtomCount()*4, UPDATE_VOLUME);
	float *updatVolumeTextureColors = getTemporaryAtomArray(mol->AtomCount()*3, UPDATE_COLUME_CLR);
	const float *atomColorTablePtr = this->atomColorTable.PeekElements();

/*	int atomCnt;
#pragma omp parallel for
	for( atomCnt = 0; atomCnt < mol->AtomCount(); ++atomCnt ) {
		updatVolumeTextureAtoms[atomCnt*4+0] = ( this->posInter[3*atomCnt+0] + this->translation.X()) * this->scale;
		updatVolumeTextureAtoms[atomCnt*4+1] = ( this->posInter[3*atomCnt+1] + this->translation.Y()) * this->scale;
		updatVolumeTextureAtoms[atomCnt*4+2] = ( this->posInter[3*atomCnt+2] + this->translation.Z()) * this->scale;
		updatVolumeTextureAtoms[atomCnt*4+3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomCnt]].Radius() * this->scale;
	}*/

	MolecularDataCall::Residue **residues = mol->Residues();
	/* sortierung nach atomen die das lösungsmittel bilden und der rest ... atomIdxSol  läuft rückwärts .. */
	int atomCntMol = 0, atomCntSol = 0;

	for( int residueIdx = 0; residueIdx < mol->ResidueCount(); residueIdx++ ) {
		MolecularDataCall::Residue *residue = residues[residueIdx];
		int firstAtomIndex = residue->FirstAtomIndex();
		int lastAtomIndx = residue->FirstAtomIndex() + residue->AtomCount();

		int i;
		for( i = 0 ; i < this->solventResidueTypeIds.Count(); i++ ) {
			if (solventResidueTypeIds[i] == residue->Type())
				break;
		}
		if (i < this->solventResidueTypeIds.Count()) {
			/* solvent (creates volume coloring ...) */
		//	#pragma omp parallel for
			for(int atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++) {
				atomCntSol++;
				float *atomPos = &updatVolumeTextureAtoms[(mol->AtomCount()-atomCntSol)*4];
				float *solventAtomColor = &updatVolumeTextureColors[(mol->AtomCount()-atomCntSol)*3];
				float *interPos = &this->posInter[atomIdx*3];
				const float *clr = &atomColorTablePtr[atomIdx*3];
				atomPos[0] = (interPos[0] + this->translation.X()) * this->scale;
				atomPos[1] = (interPos[1] + this->translation.Y()) * this->scale;
				atomPos[2] = (interPos[2] + this->translation.Z()) * this->scale;
				atomPos[3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius() * this->scale;
				solventAtomColor[0] = clr[0];
				solventAtomColor[1] = clr[1];
				solventAtomColor[2] = clr[2];
			}
		} else {
			/* not solvent (creates volume) */
		//	#pragma omp parallel for
			for(int atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++) {
				float *atomPos = &updatVolumeTextureAtoms[atomCntMol*4];
				float *interPos = &this->posInter[atomIdx*3];
				atomPos[0] = (interPos[0] + this->translation.X()) * this->scale;
				atomPos[1] = (interPos[1] + this->translation.Y()) * this->scale;
				atomPos[2] = (interPos[2] + this->translation.Z()) * this->scale;
				atomPos[3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius() * this->scale;
				atomCntMol++;
			}
		}
	}

	vislib::math::Vector<float, 3> orig( mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().PeekCoordinates());
	orig = ( orig + this->translation) * this->scale;
	vislib::math::Vector<float, 3> nullVec( 0.0f, 0.0f, 0.0f);

	this->updateVolumeShaderMoleculeVolume.Enable();
		// set shader params
		glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "filterRadius"), this->volFilterRadius);
		glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "densityScale"), this->volDensityScale);
		glUniform3fv( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "scaleVol"), 1, this->volScale);
		glUniform3fv( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
		glUniform3f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "invVolRes"), 
			1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize));
		glUniform3fv( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "translate"), 1, orig.PeekComponents() );
		glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "volSize"), float( this->volumeSize));
		CHECK_FOR_OGL_ERROR();

		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer( 4, GL_FLOAT, 0, updatVolumeTextureAtoms);
		for( z = 0; z < this->volumeSize; ++z ) {
			// TODO: spacial grid to speedup FBO rendering here?
			// attach texture slice to FBO
			glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
			glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
			// draw all atoms as points, using w for radius
			glDrawArrays( GL_POINTS, 0, atomCntMol);
		}
		glDisableClientState(GL_VERTEX_ARRAY);
	this->updateVolumeShaderMoleculeVolume.Disable();

	this->updateVolumeShaderSolventColor.Enable();
		// set shader params
		glUniform1f( this->updateVolumeShaderSolventColor.ParameterLocation( "filterRadius"), this->volFilterRadius /*1.7*/);
		glUniform1f( this->updateVolumeShaderSolventColor.ParameterLocation( "densityScale"), this->volDensityScale /*1.0*/);
		glUniform3fv( this->updateVolumeShaderSolventColor.ParameterLocation( "scaleVol"), 1, this->volScale);
		glUniform3fv( this->updateVolumeShaderSolventColor.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
		glUniform3f( this->updateVolumeShaderSolventColor.ParameterLocation( "invVolRes"), 
			1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize));
		glUniform3fv( this->updateVolumeShaderSolventColor.ParameterLocation( "translate"), 1, orig.PeekComponents() );
		glUniform1f( this->updateVolumeShaderSolventColor.ParameterLocation( "volSize"), float( this->volumeSize));
		CHECK_FOR_OGL_ERROR();

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glVertexPointer( 4, GL_FLOAT, 0, updatVolumeTextureAtoms+(mol->AtomCount()-atomCntSol)*4 );
		//glVertexPointer( 4, GL_FLOAT, 0, updatVolumeTextureAtoms );
		glColorPointer( 3, GL_FLOAT, 0, updatVolumeTextureColors+(mol->AtomCount()-atomCntSol)*3 );
		for( z = 0; z < this->volumeSize; ++z ) {
			// TODO: spacial grid to speedup FBO rendering here?
			// attach texture slice to FBO
			glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
			glUniform1f( this->updateVolumeShaderSolventColor.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
			// draw all atoms as points, using w for radius
			glDrawArrays( GL_POINTS, 0, atomCntSol);
			//glDrawArrays( GL_POINTS, 0, atomCntMol);
		}
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	this->updateVolumeShaderSolventColor.Disable();

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
 * draw the volume
 */
void protein::SolventVolumeRenderer::RenderVolume( vislib::math::Cuboid<float> boundingbox) {
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
 * write the parameters of the ray to the textures
 */
void protein::SolventVolumeRenderer::RayParamTextures( vislib::math::Cuboid<float> boundingbox) {

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
	float *texdata = new float[this->width*this->height]; doch kein new-operator im render routine ...
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
void protein::SolventVolumeRenderer::DrawBoundingBoxTranslated( vislib::math::Cuboid<float> boundingbox) {

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
void protein::SolventVolumeRenderer::DrawBoundingBox( vislib::math::Cuboid<float> boundingbox) {

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
void SolventVolumeRenderer::drawClippedPolygon( vislib::math::Cuboid<float> boundingbox) {
	if( !this->volClipPlaneFlag )
		return;

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

/*
 * write the current volume as a raw file
 */
void SolventVolumeRenderer::writeVolumeRAW() {
	unsigned int numVoxel = this->volumeSize * this->volumeSize * this->volumeSize;

	float *volume = new float[numVoxel];
	unsigned char *ucVol = new unsigned char[numVoxel];

	glBindTexture( GL_TEXTURE_3D, this->volumeTex);
	glGetTexImage( GL_TEXTURE_3D, 0, GL_ALPHA, GL_FLOAT, volume);
	glBindTexture( GL_TEXTURE_3D, 0);

	int cnt;
#pragma omp parallel for
	for( cnt = 0; cnt < numVoxel; ++cnt ) {
		ucVol[cnt] = int( ( volume[cnt]*10.0f) < 0 ? 0 : ( ( volume[cnt]*10.0f) > 128 ? 128 : ( volume[cnt]*10.0f)));
		//ucVol[cnt] = int( volume[cnt]*10.0f);
	}

	// write array to file
	FILE *foutRaw = fopen( "test.raw", "wb");
	if( !foutRaw ) {
		std::cout << "could not open file for writing." << std::endl;
	} else {
		fwrite( ucVol, sizeof(unsigned char), numVoxel, foutRaw );
		fclose( foutRaw);
	}

	delete[] volume;
	delete[] ucVol;
}
