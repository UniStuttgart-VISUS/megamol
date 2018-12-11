/*
 * MoleculeSESRenderer.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "MoleculeSESRenderer.h"
#include "vislib/assert.h"
#include "mmcore/CoreInstance.h"
#include "Color.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/sys/File.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include <GL/glu.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max

/*
 * MoleculeSESRenderer::MoleculeSESRenderer
 */
MoleculeSESRenderer::MoleculeSESRenderer( void ) : Renderer3DModuleDS (),
        molDataCallerSlot ( "getData", "Connects the protein SES rendering with protein data storage" ),
        bsDataCallerSlot ("getBindingSites", "Connects the molecule rendering with binding site data storage"),
        postprocessingParam( "postProcessingMode", "Enable Postprocessing Mode: "),
        rendermodeParam( "renderingMode", "Choose Render Mode: "),
	    puxelsParam( "puxels", "Enable Puxel Rendering: "),
        coloringModeParam0( "color::coloringMode0", "The first coloring mode."),
        coloringModeParam1( "color::coloringMode1", "The second coloring mode."),
        cmWeightParam( "color::colorWeighting", "The weighting of the two coloring modes."),
        silhouettecolorParam( "silhouetteColor", "Silhouette Color: "),
        sigmaParam( "SSAOsigma", "Sigma value for SSAO: " ),
        lambdaParam( "SSAOlambda", "Lambda value for SSAO: "),
        minGradColorParam( "color::minGradColor", "The color for the minimum value for gradient coloring" ),
        midGradColorParam( "color::midGradColor", "The color for the middle value for gradient coloring" ),
        maxGradColorParam( "color::maxGradColor", "The color for the maximum value for gradient coloring" ),
        debugParam( "drawRS", "Draw the Reduced Surface: "),
        drawSESParam( "drawSES", "Draw the SES: "),
        drawSASParam( "drawSAS", "Draw the SAS: "),
        fogstartParam( "fogStart", "Fog Start: "),
        molIdxListParam( "molIdxList", "The list of molecule indices for RS computation:"),
        colorTableFileParam( "color::colorTableFilename", "The filename of the color table."),
        offscreenRenderingParam( "offscreenRendering", "Toggle offscreen rendering."),
	    puxelSizeBuffer(512 << 20),
        computeSesPerMolecule(false)
{
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable ( &this->molDataCallerSlot );
    this->bsDataCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable (&this->bsDataCallerSlot);

    // set epsilon value for float-comparison
    this->epsilon = vislib::math::FLOAT_EPSILON;
    // set probe radius
    this->probeRadius = 1.4f;
    // set transparency
    this->transparency = 0.5f;

    // ----- en-/disable postprocessing -----
    this->postprocessing = NONE;
    //this->postprocessing = AMBIENT_OCCLUSION;
    //this->postprocessing = SILHOUETTE;
    //this->postprocessing = TRANSPARENCY;
    param::EnumParam *ppm = new param::EnumParam ( int ( this->postprocessing ) );
    ppm->SetTypePair( NONE, "None");
    ppm->SetTypePair( AMBIENT_OCCLUSION, "Screen Space Ambient Occlusion");
    ppm->SetTypePair( SILHOUETTE, "Silhouette");
    //ppm->SetTypePair( TRANSPARENCY, "Transparency");
    this->postprocessingParam << ppm;

    // ----- choose current render mode -----
    this->currentRendermode = GPU_RAYCASTING;
    //this->currentRendermode = POLYGONAL;
    //this->currentRendermode = POLYGONAL_GPU;
    //this->currentRendermode = GPU_RAYCASTING_INTERIOR_CLIPPING;
    //this->currentRendermode = GPU_SIMPLIFIED;
    param::EnumParam *rm = new param::EnumParam ( int ( this->currentRendermode ) );
    rm->SetTypePair( GPU_RAYCASTING, "GPU Ray Casting");
    this->rendermodeParam << rm;
	
    // ----- use Puxels param -----
    this->usePuxels = false;
    param::BoolParam *puxbpm = new param::BoolParam( this->usePuxels );
        this->puxelsParam << puxbpm;

    // ----- set the default color for the silhouette -----
    this->SetSilhouetteColor( 1.0f, 1.0f, 1.0f);
    param::IntParam *sc = new param::IntParam( this->codedSilhouetteColor, 0, 255255255 );
    this->silhouettecolorParam << sc;

    // ----- set sigma for screen space ambient occlusion (SSAO) -----
    this->sigma = 5.0f;
    param::FloatParam *ssaos = new param::FloatParam( this->sigma );
    this->sigmaParam << ssaos;

    // ----- set lambda for screen space ambient occlusion (SSAO) -----
    this->lambda = 10.0f;
    param::FloatParam *ssaol = new param::FloatParam( this->lambda );
    this->lambdaParam << ssaol;

    // ----- set start value for fogging -----
    this->fogStart = 0.5f;
    param::FloatParam *fs = new param::FloatParam( this->fogStart, 0.0f );
    this->fogstartParam << fs;
    
    // coloring modes
    this->currentColoringMode0 = Color::CHAIN;
    this->currentColoringMode1 = Color::ELEMENT;
    param::EnumParam *cm0 = new param::EnumParam ( int ( this->currentColoringMode0) );
    param::EnumParam *cm1 = new param::EnumParam ( int ( this->currentColoringMode1) );
    MolecularDataCall *mol = new MolecularDataCall();
    BindingSiteCall *bs = new BindingSiteCall();
    unsigned int cCnt;
    Color::ColoringMode cMode;
    for( cCnt = 0; cCnt < Color::GetNumOfColoringModes( mol, bs); ++cCnt) {
        cMode = Color::GetModeByIndex( mol, bs, cCnt);
        cm0->SetTypePair( cMode, Color::GetName( cMode).c_str());
        cm1->SetTypePair( cMode, Color::GetName( cMode).c_str());
    }
    delete mol;
    delete bs;
    this->coloringModeParam0 << cm0;
    this->coloringModeParam1 << cm1;
    this->MakeSlotAvailable( &this->coloringModeParam0);
    this->MakeSlotAvailable( &this->coloringModeParam1);

    // Color weighting parameter
    this->cmWeightParam.SetParameter(new param::FloatParam(0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->cmWeightParam);

    // the color for the minimum value (gradient coloring
    this->minGradColorParam.SetParameter(new param::StringParam( "#146496"));
    this->MakeSlotAvailable( &this->minGradColorParam);

    // the color for the middle value (gradient coloring
    this->midGradColorParam.SetParameter(new param::StringParam( "#f0f0f0"));
    this->MakeSlotAvailable( &this->midGradColorParam);

    // the color for the maximum value (gradient coloring
    this->maxGradColorParam.SetParameter(new param::StringParam( "#ae3b32"));
    this->MakeSlotAvailable( &this->maxGradColorParam);

    // ----- draw RS param -----
    this->drawRS = false;
    param::BoolParam *bpm = new param::BoolParam( this->drawRS );
        this->debugParam << bpm;

    // ----- draw SES param -----
    this->drawSES = true;
    param::BoolParam *sespm = new param::BoolParam( this->drawSES );
    this->drawSESParam << sespm;

    // ----- draw SAS param -----
    this->drawSAS = false;
    param::BoolParam *saspm = new param::BoolParam( this->drawSAS );
    this->drawSASParam << saspm;

    // ----- ofsfcreen rendering param -----
    this->offscreenRendering = false;
    param::BoolParam *orpm = new param::BoolParam( this->offscreenRendering );
    this->offscreenRenderingParam << orpm;

    // ----- molecular indices list param -----
    this->molIdxList.Add( "0");
    this->molIdxListParam.SetParameter(new param::StringParam( "0"));
    this->MakeSlotAvailable( &this->molIdxListParam);

    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    Color::ReadColorTableFromFile( filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(new param::StringParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);

    // fill rainbow color table
    Color::MakeRainbowColorTable( 100, this->rainbowColors);

    // set the FBOs and textures for post processing
    this->colorFBO = 0;
    this->blendFBO = 0;
    this->horizontalFilterFBO = 0;
    this->verticalFilterFBO = 0;
    this->texture0 = 0;
    this->depthTex0 = 0;
    this->hFilter = 0;
    this->vFilter = 0;
    // width and height of the screen
    this->width = 0;
    this->height = 0;

    // clear singularity texture
    singularityTexture.clear();
    // set singTexData to 0
    this->singTexData = 0;

    this->preComputationDone = false;

    // export parameters
    this->MakeSlotAvailable( &this->rendermodeParam );
    this->MakeSlotAvailable( &this->postprocessingParam );
#ifdef WITH_PUXELS
	this->MakeSlotAvailable( &this->puxelsParam );
#endif
    this->MakeSlotAvailable( &this->silhouettecolorParam );
    this->MakeSlotAvailable( &this->sigmaParam );
    this->MakeSlotAvailable( &this->lambdaParam );
    this->MakeSlotAvailable( &this->fogstartParam );
    this->MakeSlotAvailable( &this->debugParam );
    this->MakeSlotAvailable( &this->drawSESParam );
    this->MakeSlotAvailable( &this->drawSASParam );
    this->MakeSlotAvailable( &this->offscreenRenderingParam );

}


/*
 * MoleculeSESRenderer::~MoleculeSESRenderer
 */
MoleculeSESRenderer::~MoleculeSESRenderer(void) {
    if( colorFBO ) {
        glDeleteFramebuffersEXT( 1, &colorFBO);
        glDeleteFramebuffersEXT( 1, &blendFBO);
        glDeleteFramebuffersEXT( 1, &horizontalFilterFBO);
        glDeleteFramebuffersEXT( 1, &verticalFilterFBO);
        glDeleteTextures( 1, &texture0);
        glDeleteTextures( 1, &depthTex0);
        glDeleteTextures( 1, &texture1);
        glDeleteTextures( 1, &depthTex1);
        glDeleteTextures( 1, &hFilter);
        glDeleteTextures( 1, &vFilter);
    }
    // delete singularity texture
    for( unsigned int i = 0; i < singularityTexture.size(); ++i )
        glDeleteTextures( 1, &singularityTexture[i]);
    // release
    this->cylinderShader.Release();
    this->sphereShader.Release();
    this->sphereClipInteriorShader.Release();
    this->sphericalTriangleShader.Release();
    this->torusShader.Release();
    this->lightShader.Release();
    this->hfilterShader.Release();
    this->vfilterShader.Release();
    this->silhouetteShader.Release();
    this->transparencyShader.Release();
	
	this->puxelClearShader.Release();
	this->puxelOrderShader.Release();
	this->puxelDrawShader.Release();

    this->Release();
}


/*
 * protein::MoleculeSESRenderer::release
 */
void MoleculeSESRenderer::release( void ) {

}


/*
 * MoleculeSESRenderer::create
 */
bool MoleculeSESRenderer::create( void ) {
    if(!ogl_IsVersionGEQ(2,0) || !areExtsAvailable("GL_EXT_framebuffer_object GL_ARB_texture_float") )
        return false;

#ifdef WITH_PUXELS
    allowPuxels = ogl_IsVersionGEQ(4,3);
#else
    allowPuxels = false;
#endif

    if ( !vislib::graphics::gl::GLSLShader::InitialiseExtensions() )
        return false;

    //glEnable( GL_NORMALIZE);
    glEnable( GL_DEPTH_TEST);
    glDepthFunc( GL_LEQUAL);
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable( GL_VERTEX_PROGRAM_TWO_SIDE);

    float spec[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, spec);
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.0f);


    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

	ShaderSource compSrc;
    ShaderSource vertSrc;
    ShaderSource geomSrc;
    ShaderSource fragSrc;

    CoreInstance *ci = this->GetCoreInstance();
    if( !ci ) return false;

	if(allowPuxels)
	{
		/////////////////////////////////
		// load the puxel clear shader //
		/////////////////////////////////

		if( !ci->ShaderSourceFactory().MakeShaderSource( "puxels::clear", compSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load compute shader source for puxel clear shader", this->ClassName() );
			return false;
		}
		try
		{
			if( !this->puxelClearShader.Compile(compSrc.Code(), compSrc.Count()))
			{
				throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
			}
		}
		catch( vislib::Exception e )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create clear shader: %s\n", this->ClassName(), e.GetMsgA() );
			return false;
		}

		//////////////////////////
		// puxel reorder shader //
		//////////////////////////
		if( !ci->ShaderSourceFactory().MakeShaderSource( "puxels::order", compSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load compute shader source for puxel order shader", this->ClassName() );
			return false;
		}

		compSrc.Insert(3, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::sespayload"));

		try
		{
			if( !this->puxelOrderShader.Compile(compSrc.Code(), compSrc.Count()))
			{
				throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
			}
		}
		catch( vislib::Exception e )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create order shader: %s\n", this->ClassName(), e.GetMsgA() );
			return false;
		}
		
		//////////////////////////////////
		// puxel reduced surface shader //
		//////////////////////////////////
	    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::puxelsReducedSurfaceRenderVertex", vertSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for puxel render reduced surface shader", this->ClassName() );
			return false;
		}
		if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::puxelsReducedSurfaceRenderFragment", fragSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for puxel render reduced surface shader", this->ClassName() );
			return false;
		}
		fragSrc.Insert(3, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::sespayload"));
		try
		{
			if( !this->puxelRenderReducedSurfaceShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
			{
				throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
			}
		}
		catch( vislib::Exception e )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create puxel render reduced surface shader: %s\n", this->ClassName(), e.GetMsgA() );
			return false;
		}

		///////////////////////
		// puxel draw shader //
		///////////////////////
	    if( !ci->ShaderSourceFactory().MakeShaderSource( "puxels::pass_120", vertSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for puxel draw shader", this->ClassName() );
			return false;
		}
		if( !ci->ShaderSourceFactory().MakeShaderSource( "puxels::blend", fragSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for puxel draw shader", this->ClassName() );
			return false;
		}
		fragSrc.Insert(3, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::sespayload"));
		try
		{
			if( !this->puxelDrawShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
			{
				throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
			}
		}
		catch( vislib::Exception e )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create puxel draw shader: %s\n", this->ClassName(), e.GetMsgA() );
			return false;
		}
	}
    ////////////////////////////////////////////////////
    // load the shader source for the sphere renderer //
    ////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphereVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphereFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere shader", this->ClassName() );
        return false;
    }

	// when we want to use puxels, we have to replace the version snippet and insert puxel code
	if(allowPuxels)
	{
		/*
		#ifdef PUXELS
			puxels_store(gl_FragColor, gl_FragDepth);
		#endif
		*/
		fragSrc.Replace(0, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::version"));
		fragSrc.Insert(1, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::puxeluniform"));
		fragSrc.Insert(2, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::header"));
		fragSrc.Insert(3, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::sespayload"));
		fragSrc.Insert(4, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::data"));
		fragSrc.Insert(5, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::store"));
	}

    try
    {
        if( !this->sphereShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    // Sphere shader for offscreen rendering
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphereFragmentOR", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->sphereShaderOR.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create offscreen rendering sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ///////////////////////////////////////////////////
    // load the shader source for the torus renderer //
    ///////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::torusVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for torus shader", this->ClassName() );
        return false;
    }
	if(allowPuxels)
	{
		if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::torusFragmentPuxels", fragSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for torus shader", this->ClassName() );
			return false;
		}
	}
	else
	{
		if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::torusFragment", fragSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for torus shader", this->ClassName() );
			return false;
		}
	}
	// when we want to use puxels, we have to replace the version snippet and insert puxel code
	if(allowPuxels)
	{
		/*
		#ifdef PUXELS
			puxels_store(gl_FragColor, gl_FragDepth);
		#endif
		*/
		fragSrc.Replace(0, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::version"));
		fragSrc.Insert(1, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::puxeluniform"));
		fragSrc.Insert(2, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::header"));
		fragSrc.Insert(3, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::sespayload"));
		fragSrc.Insert(4, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::data"));
		fragSrc.Insert(5, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::store"));
	}

    try
    {
        if( !this->torusShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create torus shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    // Tirus shader for offscreen rendering
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::torusFragmentOR", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for torus shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->torusShaderOR.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create offscreenrendering torus shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ////////////////////////////////////////////////////////////////
    // load the shader source for the spherical triangle renderer //
    ////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphericaltriangleVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for spherical triangle shader", this->ClassName() );
        return false;
    }
	if(allowPuxels)
	{
		if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphericaltriangleFragmentPuxels", fragSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for spherical triangle shader", this->ClassName() );
			return false;
	    }
	}
	else
	{
		if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphericaltriangleFragment", fragSrc ) )
		{
			Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for spherical triangle shader", this->ClassName() );
		   return false;
		}
	}
 	// when we want to use puxels, we have to replace the version snippet and insert puxel code
	if(allowPuxels)
	{
		/*
		#ifdef PUXELS
			puxels_store(gl_FragColor, gl_FragDepth);
		#endif
		*/
		fragSrc.Replace(0, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::version"));
		fragSrc.Insert(1, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::puxeluniform"));
		fragSrc.Insert(2, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::header"));
		fragSrc.Insert(3, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::sespayload"));
		fragSrc.Insert(4, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::data"));
		fragSrc.Insert(5, ci->ShaderSourceFactory().MakeShaderSnippet("puxels::store"));
	}
	
	try
    {
        if( !this->sphericalTriangleShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create spherical triangle shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    // Spherical triangle shader for offscreenrendering
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphericaltriangleFragmentOR", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for spherical triangle shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->sphericalTriangleShaderOR.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create spherical triangle shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    //////////////////////////////////////////////////////////////////////////
    // load the shader source for the sphere renderer with clipped interior //
    //////////////////////////////////////////////////////////////////////////
    /*
fragSrc.Append( vertSrc.Append(new ShaderSource::VersionSnippet(120)));
    // vertex shader
    vertSrc.Append( this->shaderSnippetFactory().Create("commondefines",
            "protein", Utility::ShaderSnippetFactory::GLSL_COMMON_SHADER));
    vertSrc.Append( this->shaderSnippetFactory().Create("sphereSES_clipinterior",
        "protein", Utility::ShaderSnippetFactory::GLSL_VERTEX_SHADER));
    // fragment shader
    fragSrc.Append( this->shaderSnippetFactory().Create("commondefines",
            "protein", Utility::ShaderSnippetFactory::GLSL_COMMON_SHADER));
    fragSrc.Append( this->shaderSnippetFactory().Create("lightdirectional",
            "protein", Utility::ShaderSnippetFactory::GLSL_COMMON_SHADER));
    fragSrc.Append( this->shaderSnippetFactory().Create("sphereSES_clipinterior",
        "protein", Utility::ShaderSnippetFactory::GLSL_FRAGMENT_SHADER));
    // create the shader
    this->sphereClipInteriorShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());
    //clear fragment source
    fragSrc.Clear();
    //clear vertex source
    vertSrc.Clear();
    */

    //////////////////////////////////////////////////////
    // load the shader files for the per pixel lighting //
    //////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::perpixellightVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for per pixel lighting shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::perpixellightFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for per pixel lighting shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->lightShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create per pixel lighting shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    /////////////////////////////////////////////////////////////////
    // load the shader files for horizontal 1D gaussian filtering  //
    /////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::hfilterVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for horizontal 1D gaussian filter shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::hfilterFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for horizontal 1D gaussian filter shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->hfilterShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create horizontal 1D gaussian filter shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ///////////////////////////////////////////////////////////////
    // load the shader files for vertical 1D gaussian filtering  //
    ///////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::vfilterVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for vertical 1D gaussian filter shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::vfilterFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for vertical 1D gaussian filter shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->vfilterShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create vertical 1D gaussian filter shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    //////////////////////////////////////////////////////
    // load the shader files for silhouette drawing     //
    //////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::silhouetteVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for silhouette drawing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::silhouetteFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for silhouette drawing shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->silhouetteShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create vertical 1D gaussian filter shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    //////////////////////////////////////////////////////
    // load the shader files for transparency           //
    //////////////////////////////////////////////////////
    /*
    // version 120
    fragSrc.Append(vertSrc.Append(new ShaderSource::VersionSnippet(120)));
    // vertex shader
    vertSrc.Append( this->shaderSnippetFactory().Create("transparency",
        "protein", Utility::ShaderSnippetFactory::GLSL_VERTEX_SHADER));
    // fragment shader
    fragSrc.Append( this->shaderSnippetFactory().Create("transparency",
        "protein", Utility::ShaderSnippetFactory::GLSL_FRAGMENT_SHADER));
    // create the shader
    this->transparencyShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());
    //clear fragment source
    fragSrc.Clear();
    //clear vertex source
    vertSrc.Clear();
    */

    //////////////////////////////////////////////////////
    // load the shader source for the cylinder renderer //
    //////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::cylinderVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for cylinder shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::cylinderFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for cylinder shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->cylinderShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create cylinder shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    return true;
}

/*
 * MoleculeSESRenderer::GetExtents
 */
bool MoleculeSESRenderer::GetExtents(Call& call) {

    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    if (!(*mol)(1)) return false;

    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld( scale);
    cr3d->SetTimeFramesCount( mol->FrameCount());

    /*
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    protein::CallProteinData *protein = this->molDataCallerSlot.CallAs<protein::CallProteinData>();
    if (protein == NULL) return false;
    if (!(*protein)()) return false;

    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    scale = 2.0f / vislib::math::Max(vislib::math::Max(protein->BoundingBox().Width(),
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
    bbox.SetObjectSpaceClipBox(bbox.ObjectSpaceBBox());
    bbox.SetWorldSpaceClipBox(bbox.WorldSpaceBBox());
    */

    return true;
}


/*
 * MoleculeSESRenderer::Render
 */
bool MoleculeSESRenderer::Render( Call& call ) {
    // temporary variables
    unsigned int cntRS = 0;

    // cast the call to Render3D
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to CallProteinData
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    // if something went wrong --> return
    if( !mol) return false;

    // execute the call
    mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
    
    // get pointer to BindingSiteCall
    BindingSiteCall *bs = this->bsDataCallerSlot.CallAs<BindingSiteCall>();
    if( bs ) {
        (*bs)(BindingSiteCall::CallForGetData);
    }

        // ==================== check parameters ====================
    this->UpdateParameters( mol, bs);

        // ==================== Precomputations ====================

    if( this->currentRendermode == GPU_RAYCASTING ) {

        /*
        // init the reduced surfaces
        if( this->reducedSurfaceAllFrames.empty() ) {
            time_t t = clock();
            // create the reduced surfaces
            unsigned int chainIds;
            this->reducedSurfaceAllFrames.resize( mol->FrameCount());
            // compute RS for all frames
            for( unsigned int cntFrames = 0; cntFrames < mol->FrameCount(); ++cntFrames ) {
                    // execute the call
                mol->SetFrameID(static_cast<int>( cntFrames));
                if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
                // compute RS
                for( chainIds = 0; chainIds < this->molIdxList.Count(); ++chainIds ) {
                    this->reducedSurfaceAllFrames[cntFrames].push_back(
                        new ReducedSurface( atoi( this->molIdxList[chainIds]), mol, this->probeRadius) );
                        }
            }
            int framecounter, molcounter;

            for( framecounter = 0; framecounter < this->reducedSurfaceAllFrames.size(); ++framecounter ) {
                // compute RS
                for( unsigned int molcounter = 0; molcounter < this->reducedSurfaceAllFrames[framecounter].size(); ++molcounter ) {
                    this->reducedSurfaceAllFrames[framecounter][molcounter]->ComputeReducedSurfaceMolecule();
                        }
            }
                        std::cout << "RS for all frames computed in: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
                }
        unsigned int currentFrame = static_cast<unsigned int>( callTime);
        this->reducedSurface.resize( this->reducedSurfaceAllFrames[cntRS].size());
        for( cntRS = 0; cntRS < this->reducedSurfaceAllFrames[currentFrame].size(); ++cntRS ) {
            this->reducedSurface[cntRS] = this->reducedSurfaceAllFrames[currentFrame][cntRS];
        }
        this->ComputeRaycastingArrays();
        */

        // ----------------------------------------------------------------------------
        
        // init the reduced surfaces
        if( this->reducedSurface.empty() ) {
            time_t t = clock();
            // create the reduced surface
            unsigned int chainIds;
            if( !this->computeSesPerMolecule ) {
                this->reducedSurface.push_back( new ReducedSurface(mol, this->probeRadius));
                this->reducedSurface.back()->ComputeReducedSurface();
            } else {
                // if no molecule indices are given, compute the SES for all molecules
                if( this->molIdxList.IsEmpty()) {
                    for( chainIds = 0; chainIds < mol->MoleculeCount(); ++chainIds ) {
                        this->reducedSurface.push_back(
                            new ReducedSurface(chainIds, mol, this->probeRadius) );
                        this->reducedSurface.back()->ComputeReducedSurface();
                    }
                } else {
                    // else compute the SES for all selected molecules
                    for( chainIds = 0; chainIds < this->molIdxList.Count(); ++chainIds ) {
                        this->reducedSurface.push_back(
                            new ReducedSurface( atoi( this->molIdxList[chainIds]), mol, this->probeRadius) );
                        this->reducedSurface.back()->ComputeReducedSurface();
                    }
                }
            }
            vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO,
                "%s: RS computed in: %f s\n", this->ClassName(), 
                ( double( clock() - t) / double( CLOCKS_PER_SEC)));
        }
        // update the data / the RS
        for( cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS ) {
            if( this->reducedSurface[cntRS]->UpdateData( 1.0f, 5.0f) ) {
                this->ComputeRaycastingArrays( cntRS);
            }
        }
        /*
        // init the reduced surfaces
        if( this->reducedSurface.empty() ) {
            time_t t = clock();
            // create the reduced surface
            unsigned int chainIds;
            for( chainIds = 0; chainIds < this->molIdxList.Count(); ++chainIds ) {
                this->reducedSurface.push_back(
                    new ReducedSurface( atoi( this->molIdxList[chainIds]), mol, this->probeRadius) );
                this->reducedSurface.back()->ComputeReducedSurfaceMolecule();
            }
            std::cout << "RS computed in: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
        }
        // update the data / the RS
        for( cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS ) {
            if( this->reducedSurface[cntRS]->UpdateData( 1.0f, 5.0f) ) {
                this->ComputeRaycastingArrays( cntRS);
            }
        }
        */
    }

    if( !this->preComputationDone ) {
        // compute the color table
        Color::MakeColorTable(mol,
            this->currentColoringMode0,
            this->currentColoringMode1,
            this->cmWeightParam.Param<param::FloatParam>()->Value(),       // weight for the first cm
            1.0f - this->cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
            this->atomColorTable, this->colorLookupTable,
            this->rainbowColors,
            this->minGradColorParam.Param<param::StringParam>()->Value(),
            this->midGradColorParam.Param<param::StringParam>()->Value(),
            this->maxGradColorParam.Param<param::StringParam>()->Value(),
            true, bs);
        // compute the data needed for the current render mode
        if( this->currentRendermode == GPU_RAYCASTING )
            this->ComputeRaycastingArrays();
        // set the precomputation of the data as done
        this->preComputationDone = true;
    }

	bool virtualViewportChanged = false;
	if( static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetHeight()) != this->height )
    {
        this->width = static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetHeight());
		virtualViewportChanged = true;
    }

    if( this->postprocessing != NONE && virtualViewportChanged )
        this->CreateFBO();
	
	if( this->allowPuxels && virtualViewportChanged )
        puxelsCreateBuffers();

    // ==================== Scale & Translate ====================

    glPushMatrix();

    /*
        float scale, xoff, yoff, zoff;
        vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();

        xoff = -bbc.X();
        yoff = -bbc.Y();
        zoff = -bbc.Z();

        scale = 2.0f / vislib::math::Max ( vislib::math::Max ( protein->BoundingBox().Width(),
                                           protein->BoundingBox().Height() ), protein->BoundingBox().Depth() );

        glScalef ( scale, scale, scale );
        glTranslatef ( xoff, yoff, zoff );
    */

    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);


    // ==================== Start actual rendering ====================

    glDisable( GL_BLEND);
    //glEnable( GL_NORMALIZE);
    glEnable( GL_DEPTH_TEST);
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable( GL_VERTEX_PROGRAM_TWO_SIDE);

    if( this->postprocessing == TRANSPARENCY ) {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->blendFBO);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if( this->drawRS )
            this->RenderDebugStuff( mol);
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
    } else {
        if( this->drawRS ) {
            this->RenderDebugStuff( mol);
            // DEMO
            glPopMatrix();
            return true;
        }
    }

	// clear puxels buffer
	if(this->usePuxels)
		this->puxelsClear();

    // start rendering to frame buffer object
    if( this->postprocessing != NONE ) {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->colorFBO);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    // render the SES
    if( this->currentRendermode == GPU_RAYCASTING ) {
        this->RenderSESGpuRaycasting( mol);
    }

	if(this->usePuxels)
	{
		this->puxelRenderReducedSurface();
		this->puxelsReorder();
		this->puxelsDraw();
	}

    //////////////////////////////////
    // apply postprocessing effects //
    //////////////////////////////////
    if( this->postprocessing != NONE ) {
        // stop rendering to frame buffer object
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

        if( this->postprocessing == AMBIENT_OCCLUSION )
            this->PostprocessingSSAO();
        else if( this->postprocessing == SILHOUETTE )
            this->PostprocessingSilhouette();
        else if( this->postprocessing == TRANSPARENCY )
            this->PostprocessingTransparency( 0.5f);
    }

    glPopMatrix();

    // unlock the current frame
    mol->Unlock();

    return true;
}


/*
 * update parameters
 */
void MoleculeSESRenderer::UpdateParameters( const MolecularDataCall *mol, const BindingSiteCall *bs) {
    // variables
    bool recomputeColors = false;
    // ==================== check parameters ====================
    if ( this->postprocessingParam.IsDirty() ) {
        this->postprocessing = static_cast<PostprocessingMode>(  this->postprocessingParam.Param<param::EnumParam>()->Value() );
        this->postprocessingParam.ResetDirty();
    }
    if( this->rendermodeParam.IsDirty() ) {
        this->currentRendermode = static_cast<RenderMode>( this->rendermodeParam.Param<param::EnumParam>()->Value() );
        this->rendermodeParam.ResetDirty();
        this->preComputationDone = false;
    }
	if( this->puxelsParam.IsDirty() ) {
		this->usePuxels = this->puxelsParam.Param<param::BoolParam>()->Value();
		this->puxelsParam.ResetDirty();
		if(!this->allowPuxels)
			this->puxelsParam.Param<param::BoolParam>()->SetValue(false, false);
    }
    if( this->coloringModeParam0.IsDirty() || this->coloringModeParam1.IsDirty() || this->cmWeightParam.IsDirty()) {
        this->currentColoringMode0 = static_cast<Color::ColoringMode>(  this->coloringModeParam0.Param<param::EnumParam>()->Value() );
        this->currentColoringMode1 = static_cast<Color::ColoringMode>(  this->coloringModeParam1.Param<param::EnumParam>()->Value() );

        Color::MakeColorTable( mol,
            this->currentColoringMode0,
            this->currentColoringMode1,
            this->cmWeightParam.Param<param::FloatParam>()->Value(),       // weight for the first cm
            1.0f - this->cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
            this->atomColorTable, this->colorLookupTable,
            this->rainbowColors,
            this->minGradColorParam.Param<param::StringParam>()->Value(),
            this->midGradColorParam.Param<param::StringParam>()->Value(),
            this->maxGradColorParam.Param<param::StringParam>()->Value(),
            true, bs);

        this->preComputationDone = false;
        this->coloringModeParam0.ResetDirty();
        this->coloringModeParam1.ResetDirty();
        this->cmWeightParam.ResetDirty();
    }
    if( this->silhouettecolorParam.IsDirty() ) {
        this->SetSilhouetteColor( this->DecodeColor( this->silhouettecolorParam.Param<param::IntParam>()->Value() ) );
        this->silhouettecolorParam.ResetDirty();
    }
    if( this->sigmaParam.IsDirty() ) {
        this->sigma = this->sigmaParam.Param<param::FloatParam>()->Value();
        this->sigmaParam.ResetDirty();
    }
    if( this->lambdaParam.IsDirty() ) {
        this->lambda = this->lambdaParam.Param<param::FloatParam>()->Value();
        this->lambdaParam.ResetDirty();
    }
    if( this->fogstartParam.IsDirty() ) {
        this->fogStart = this->fogstartParam.Param<param::FloatParam>()->Value();
        this->fogstartParam.ResetDirty();
    }
    if( this->debugParam.IsDirty() ) {
        this->drawRS = this->debugParam.Param<param::BoolParam>()->Value();
        this->debugParam.ResetDirty();
    }
    if( this->drawSESParam.IsDirty() ) {
        this->drawSES = this->drawSESParam.Param<param::BoolParam>()->Value();
        this->drawSESParam.ResetDirty();
    }
    if( this->drawSASParam.IsDirty() ) {
        this->drawSAS = this->drawSASParam.Param<param::BoolParam>()->Value();
        this->drawSASParam.ResetDirty();
        this->preComputationDone = false;
    }
    if( this->offscreenRenderingParam.IsDirty() ) {
        this->offscreenRendering =
            this->offscreenRenderingParam.Param<param::BoolParam>()->Value();
        this->offscreenRenderingParam.ResetDirty();

    }
    if( this->molIdxListParam.IsDirty() ) {
        vislib::StringA tmpStr( this->molIdxListParam.Param<param::StringParam>()->Value());
        this->molIdxList = vislib::StringTokeniser<vislib::CharTraitsA>::Split( tmpStr, ';', true);
        this->molIdxListParam.ResetDirty();
    }
    // color table param
    if( this->colorTableFileParam.IsDirty() ) {
        Color::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::StringParam>()->Value(),
            this->colorLookupTable);
        this->colorTableFileParam.ResetDirty();
        recomputeColors = true;
    }

    if( recomputeColors ) {
        this->preComputationDone = false;
    }
    
#ifndef WITH_PUXELS
    this->usePuxels = false;
#endif
}

/*
 * postprocessing: use screen space ambient occlusion
 */
void MoleculeSESRenderer::PostprocessingSSAO() {
    // START draw overlay
    glBindTexture( GL_TEXTURE_2D, this->depthTex0);
    // --> this seems to be unnecessary since no mipmap but the original resolution is used
    //glGenerateMipmapEXT( GL_TEXTURE_2D);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glPushAttrib( GL_LIGHTING_BIT);
    glDisable( GL_LIGHTING);

    // ----- START gaussian filtering + SSAO -----
    // apply horizontal filter
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->horizontalFilterFBO);

    this->hfilterShader.Enable();

    glUniform1fARB( this->hfilterShader.ParameterLocation( "sigma"), this->sigma);
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    glBegin(GL_QUADS);
    glVertex2f( 0.0f, 0.0f);
    glVertex2f( 1.0f, 0.0f);
    glVertex2f( 1.0f, 1.0f);
    glVertex2f( 0.0f, 1.0f);
    glEnd();

    this->hfilterShader.Disable();

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

    // apply vertical filter to horizontally filtered image and compute colors
    glBindTexture( GL_TEXTURE_2D, this->hFilter);
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_2D, this->texture0);

    this->vfilterShader.Enable();

    glUniform1iARB( this->vfilterShader.ParameterLocation( "tex"), 0);
    glUniform1iARB( this->vfilterShader.ParameterLocation( "colorTex"), 1);
    glUniform1fARB( this->vfilterShader.ParameterLocation( "sigma"), this->sigma);
    glUniform1fARB( this->vfilterShader.ParameterLocation( "lambda"), this->lambda);
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    glBegin(GL_QUADS);
    glVertex2f( 0.0f, 0.0f);
    glVertex2f( 1.0f, 0.0f);
    glVertex2f( 1.0f, 1.0f);
    glVertex2f( 0.0f, 1.0f);
    glEnd();

    this->vfilterShader.Disable();
    // ----- END gaussian filtering + SSAO -----

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindTexture( GL_TEXTURE_2D, 0);
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, 0);
    // END draw overlay
}


/*
 * postprocessing: use silhouette shader
 */
void MoleculeSESRenderer::PostprocessingSilhouette() {
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glPushAttrib( GL_LIGHTING_BIT);
    glDisable( GL_LIGHTING);

    // ----- START -----
    glBindTexture( GL_TEXTURE_2D, this->depthTex0);
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_2D, this->texture0);

    this->silhouetteShader.Enable();

    glUniform1iARB( this->silhouetteShader.ParameterLocation( "tex"), 0);
    glUniform1iARB( this->silhouetteShader.ParameterLocation( "colorTex"), 1);
    glUniform1fARB( this->silhouetteShader.ParameterLocation( "difference"), 0.025f);
    glColor4f( this->silhouetteColor.GetX(), this->silhouetteColor.GetY(),
            this->silhouetteColor.GetZ(),  1.0f);
    glBegin(GL_QUADS);
    glVertex2f( 0.0f, 0.0f);
    glVertex2f( 1.0f, 0.0f);
    glVertex2f( 1.0f, 1.0f);
    glVertex2f( 0.0f, 1.0f);
    glEnd();

    this->silhouetteShader.Disable();
    // ----- END -----

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindTexture( GL_TEXTURE_2D, 0);
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, 0);
    // END draw overlay
}


/*
 * postprocessing: transparency (blend two images)
 */
void MoleculeSESRenderer::PostprocessingTransparency( float transparency) {
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glPushAttrib( GL_LIGHTING_BIT);
    glDisable( GL_LIGHTING);

    // ----- START -----
    glBindTexture( GL_TEXTURE_2D, this->depthTex0);
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_2D, this->texture0);
    glActiveTexture( GL_TEXTURE2);
    glBindTexture( GL_TEXTURE_2D, this->depthTex1);
    glActiveTexture( GL_TEXTURE3);
    glBindTexture( GL_TEXTURE_2D, this->texture1);

    this->transparencyShader.Enable();

    glUniform1iARB( this->transparencyShader.ParameterLocation( "depthTex0"), 0);
    glUniform1iARB( this->transparencyShader.ParameterLocation( "colorTex0"), 1);
    glUniform1iARB( this->transparencyShader.ParameterLocation( "depthTex1"), 2);
    glUniform1iARB( this->transparencyShader.ParameterLocation( "colorTex1"), 3);
    glUniform1fARB( this->transparencyShader.ParameterLocation( "transparency"), transparency);
    glBegin(GL_QUADS);
    glVertex2f( 0.0f, 0.0f);
    glVertex2f( 1.0f, 0.0f);
    glVertex2f( 1.0f, 1.0f);
    glVertex2f( 0.0f, 1.0f);
    glEnd();

    this->transparencyShader.Disable();
    // ----- END -----

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindTexture( GL_TEXTURE_2D, 0);
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, 0);
    // END draw overlay
}


/*
 * Create the fbo and texture needed for offscreen rendering
 */
void MoleculeSESRenderer::CreateFBO() {
    if( colorFBO ) {
        glDeleteFramebuffersEXT( 1, &colorFBO);
        glDeleteFramebuffersEXT( 1, &blendFBO);
        glDeleteFramebuffersEXT( 1, &horizontalFilterFBO);
        glDeleteFramebuffersEXT( 1, &verticalFilterFBO);
        glDeleteTextures( 1, &texture0);
        glDeleteTextures( 1, &depthTex0);
        glDeleteTextures( 1, &texture1);
        glDeleteTextures( 1, &depthTex1);
        glDeleteTextures( 1, &hFilter);
        glDeleteTextures( 1, &vFilter);
    }
    glGenFramebuffersEXT( 1, &colorFBO);
    glGenFramebuffersEXT( 1, &blendFBO);
    glGenFramebuffersEXT( 1, &horizontalFilterFBO);
    glGenFramebuffersEXT( 1, &verticalFilterFBO);
    glGenTextures( 1, &texture0);
    glGenTextures( 1, &depthTex0);
    glGenTextures( 1, &texture1);
    glGenTextures( 1, &depthTex1);
    glGenTextures( 1, &hFilter);
    glGenTextures( 1, &vFilter);

    // color and depth FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->colorFBO);
    // init texture0 (color)
    glBindTexture( GL_TEXTURE_2D, texture0);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture( GL_TEXTURE_2D, 0);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture0, 0);
    // init depth texture
    glBindTexture( GL_TEXTURE_2D, depthTex0);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, this->width, this->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->depthTex0, 0);
    glBindTexture( GL_TEXTURE_2D, 0);

    // color and depth FBO for blending (transparency)
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->blendFBO);
    // init texture1 (color)
    glBindTexture( GL_TEXTURE_2D, texture1);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture( GL_TEXTURE_2D, 0);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture1, 0);
    // init depth texture
    glBindTexture( GL_TEXTURE_2D, depthTex1);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, this->width, this->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->depthTex1, 0);
    glBindTexture( GL_TEXTURE_2D, 0);

    //horizontal filter FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->horizontalFilterFBO);
    glBindTexture( GL_TEXTURE_2D, this->hFilter);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, hFilter, 0);
    glBindTexture( GL_TEXTURE_2D, 0);

    //vertical filter FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->verticalFilterFBO);
    glBindTexture( GL_TEXTURE_2D, this->vFilter);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16_EXT, this->width, this->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, vFilter, 0);
    glBindTexture( GL_TEXTURE_2D, 0);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}


/*
 * Render the molecular surface using GPU raycasting
 */
void MoleculeSESRenderer::RenderSESGpuRaycasting(
    const MolecularDataCall *mol) {
    // TODO: attribute locations nicht jedes mal neu abfragen!

	bool virtualViewportChanged = false;
	if( static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetHeight()) != this->height )
    {
        this->width = static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(this->cameraInfo->VirtualViewSize().GetHeight());
		virtualViewportChanged = true;
    }

	if( this->allowPuxels && virtualViewportChanged )
        puxelsCreateBuffers();

    // set viewport
    float viewportStuff[4] = {
            this->cameraInfo->TileRect().Left(),
            this->cameraInfo->TileRect().Bottom(),
            this->cameraInfo->TileRect().Width(),
            this->cameraInfo->TileRect().Height() };
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // get clear color (i.e. background color) for fogging
    float *clearColor = new float[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, clearColor);
    vislib::math::Vector<float, 3> fogCol( clearColor[0], clearColor[1], clearColor[2]);

    unsigned int cntRS;

    for( cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS )
    {
        //////////////////////////////////
        // ray cast the tori on the GPU //
        //////////////////////////////////
        GLuint attribInParams;
        GLuint attribQuatC;
        GLuint attribInSphere;
        GLuint attribInColors;
        GLuint attribInCuttingPlane;

        if( this->drawSES ) {
            // enable torus shader
            if(offscreenRendering) {
                this->torusShaderOR.Enable();
                // set shader variables
                glUniform4fvARB( this->torusShaderOR.ParameterLocation( "viewAttr"), 1, viewportStuff);
                glUniform3fvARB( this->torusShaderOR.ParameterLocation( "camIn"), 1, this->cameraInfo->Front().PeekComponents());
                glUniform3fvARB( this->torusShaderOR.ParameterLocation( "camRight"), 1, this->cameraInfo->Right().PeekComponents());
                glUniform3fvARB( this->torusShaderOR.ParameterLocation( "camUp"), 1, this->cameraInfo->Up().PeekComponents());
                glUniform3fARB( this->torusShaderOR.ParameterLocation( "zValues"), fogStart, this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
                glUniform3fARB( this->torusShaderOR.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
                glUniform1fARB( this->torusShaderOR.ParameterLocation( "alpha"), this->transparency);
                // get attribute locations
                attribInParams = glGetAttribLocationARB( this->torusShaderOR, "inParams");
                attribQuatC = glGetAttribLocationARB( this->torusShaderOR, "quatC");
                attribInSphere = glGetAttribLocationARB( this->torusShaderOR, "inSphere");
                attribInColors = glGetAttribLocationARB( this->torusShaderOR, "inColors");
                attribInCuttingPlane = glGetAttribLocationARB( this->torusShaderOR, "inCuttingPlane");
            } else {
                this->torusShader.Enable();
                // set shader variables

				// puxels
#ifdef WITH_PUXELS
				glUniform1ui( this->torusShader.ParameterLocation("width"),        this->cameraInfo->TileRect().Width());
				glUniform1ui( this->torusShader.ParameterLocation("height"),       this->cameraInfo->TileRect().Height());
				glUniform1ui( this->torusShader.ParameterLocation("puxels_use"), this->usePuxels ? 1 : 0);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, puxelsBufferHeader);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, puxelsBufferData);
				glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 5, puxelsAtomicBufferNextId);
#endif

                glUniform4fvARB( this->torusShader.ParameterLocation( "viewAttr"), 1, viewportStuff);
                glUniform3fvARB( this->torusShader.ParameterLocation( "camIn"), 1, this->cameraInfo->Front().PeekComponents());
                glUniform3fvARB( this->torusShader.ParameterLocation( "camRight"), 1, this->cameraInfo->Right().PeekComponents());
                glUniform3fvARB( this->torusShader.ParameterLocation( "camUp"), 1, this->cameraInfo->Up().PeekComponents());
                glUniform3fARB( this->torusShader.ParameterLocation( "zValues"), fogStart, this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
                glUniform3fARB( this->torusShader.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
                glUniform1fARB( this->torusShader.ParameterLocation( "alpha"), this->transparency);
                // get attribute locations
                attribInParams = glGetAttribLocationARB( this->torusShader, "inParams");
                attribQuatC = glGetAttribLocationARB( this->torusShader, "quatC");
                attribInSphere = glGetAttribLocationARB( this->torusShader, "inSphere");
                attribInColors = glGetAttribLocationARB( this->torusShader, "inColors");
                attribInCuttingPlane = glGetAttribLocationARB( this->torusShader, "inCuttingPlane");
            }

            // set color to orange
            glColor3f( 1.0f, 0.75f, 0.0f);
            glEnableClientState( GL_VERTEX_ARRAY);
            // enable vertex attribute arrays for the attribute locations
            glEnableVertexAttribArrayARB( attribInParams);
            glEnableVertexAttribArrayARB( attribQuatC);
            glEnableVertexAttribArrayARB( attribInSphere);
            glEnableVertexAttribArrayARB( attribInColors);
            glEnableVertexAttribArrayARB( attribInCuttingPlane);
            // set vertex and attribute pointers and draw them
            glVertexAttribPointerARB( attribInParams, 3, GL_FLOAT, 0, 0, this->torusInParamArray[cntRS].PeekElements());
            glVertexAttribPointerARB( attribQuatC, 4, GL_FLOAT, 0, 0, this->torusQuatCArray[cntRS].PeekElements());
            glVertexAttribPointerARB( attribInSphere, 4, GL_FLOAT, 0, 0, this->torusInSphereArray[cntRS].PeekElements());
            glVertexAttribPointerARB( attribInColors, 4, GL_FLOAT, 0, 0, this->torusColors[cntRS].PeekElements());
            glVertexAttribPointerARB( attribInCuttingPlane, 3, GL_FLOAT, 0, 0, this->torusInCuttingPlaneArray[cntRS].PeekElements());
            glVertexPointer( 3, GL_FLOAT, 0, this->torusVertexArray[cntRS].PeekElements());
            glDrawArrays( GL_POINTS, 0, ((unsigned int)this->torusVertexArray[cntRS].Count())/3);
            // disable vertex attribute arrays for the attribute locations
            glDisableVertexAttribArrayARB( attribInParams);
            glDisableVertexAttribArrayARB( attribQuatC);
            glDisableVertexAttribArrayARB( attribInSphere);
            glDisableVertexAttribArrayARB( attribInColors);
            glDisableVertexAttribArrayARB( attribInCuttingPlane);
            glDisableClientState(GL_VERTEX_ARRAY);
            // enable torus shader

			// puxels
#ifdef WITH_PUXELS
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
#endif

            if(offscreenRendering) {
                this->torusShaderOR.Disable();
            }
            else {
                this->torusShader.Disable();
            }

            /////////////////////////////////////////////////
            // ray cast the spherical triangles on the GPU //
            /////////////////////////////////////////////////
            GLuint attribVec1;
            GLuint attribVec2;
            GLuint attribVec3;
            GLuint attribTexCoord1;
            GLuint attribTexCoord2;
            GLuint attribTexCoord3;
            GLuint attribColors;

            // bind texture
            glBindTexture( GL_TEXTURE_2D, singularityTexture[cntRS]);
            // enable spherical triangle shader
            if(offscreenRendering) {
                this->sphericalTriangleShaderOR.Enable();
                // set shader variables
                glUniform4fvARB( this->sphericalTriangleShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
                glUniform3fvARB( this->sphericalTriangleShaderOR.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
                glUniform3fvARB( this->sphericalTriangleShaderOR.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
                glUniform3fvARB( this->sphericalTriangleShaderOR.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
                glUniform3fARB( this->sphericalTriangleShaderOR.ParameterLocation( "zValues"), fogStart, this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
                glUniform3fARB( this->sphericalTriangleShaderOR.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
                glUniform2fARB( this->sphericalTriangleShaderOR.ParameterLocation( "texOffset"),
                                                         1.0f/(float)this->singTexWidth[cntRS], 1.0f/(float)this->singTexHeight[cntRS] );
                glUniform1fARB( this->sphericalTriangleShaderOR.ParameterLocation( "alpha"), this->transparency);
                // get attribute locations
                attribVec1 = glGetAttribLocationARB( this->sphericalTriangleShaderOR, "attribVec1");
                attribVec2 = glGetAttribLocationARB( this->sphericalTriangleShaderOR, "attribVec2");
                attribVec3 = glGetAttribLocationARB( this->sphericalTriangleShaderOR, "attribVec3");
                attribTexCoord1 = glGetAttribLocationARB( this->sphericalTriangleShaderOR, "attribTexCoord1");
                attribTexCoord2 = glGetAttribLocationARB( this->sphericalTriangleShaderOR, "attribTexCoord2");
                attribTexCoord3 = glGetAttribLocationARB( this->sphericalTriangleShaderOR, "attribTexCoord3");
                attribColors = glGetAttribLocationARB( this->sphericalTriangleShaderOR, "attribColors");
            }
            else {
                this->sphericalTriangleShader.Enable();
                // set shader variables
#ifdef WITH_PUXELS
				// puxels
				glUniform1ui( this->sphericalTriangleShader.ParameterLocation("width"),        this->cameraInfo->TileRect().Width());
				glUniform1ui( this->sphericalTriangleShader.ParameterLocation("height"),       this->cameraInfo->TileRect().Height());
				glUniform1ui( this->sphericalTriangleShader.ParameterLocation("puxels_use"), this->usePuxels ? 1 : 0);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, puxelsBufferHeader);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, puxelsBufferData);
                glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 5, puxelsAtomicBufferNextId);
#endif

                glUniform4fvARB( this->sphericalTriangleShader.ParameterLocation("viewAttr"), 1, viewportStuff);
                glUniform3fvARB( this->sphericalTriangleShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
                glUniform3fvARB( this->sphericalTriangleShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
                glUniform3fvARB( this->sphericalTriangleShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
                glUniform3fARB( this->sphericalTriangleShader.ParameterLocation( "zValues"), fogStart, this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
                glUniform3fARB( this->sphericalTriangleShader.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
                glUniform2fARB( this->sphericalTriangleShader.ParameterLocation( "texOffset"),
                                                         1.0f/(float)this->singTexWidth[cntRS], 1.0f/(float)this->singTexHeight[cntRS] );
                glUniform1fARB( this->sphericalTriangleShader.ParameterLocation( "alpha"), this->transparency);
                // get attribute locations
                attribVec1 = glGetAttribLocationARB( this->sphericalTriangleShader, "attribVec1");
                attribVec2 = glGetAttribLocationARB( this->sphericalTriangleShader, "attribVec2");
                attribVec3 = glGetAttribLocationARB( this->sphericalTriangleShader, "attribVec3");
                attribTexCoord1 = glGetAttribLocationARB( this->sphericalTriangleShader, "attribTexCoord1");
                attribTexCoord2 = glGetAttribLocationARB( this->sphericalTriangleShader, "attribTexCoord2");
                attribTexCoord3 = glGetAttribLocationARB( this->sphericalTriangleShader, "attribTexCoord3");
                attribColors = glGetAttribLocationARB( this->sphericalTriangleShader, "attribColors");
            }

            // set color to turquoise
            glColor3f( 0.0f, 0.75f, 1.0f);
            glEnableClientState( GL_VERTEX_ARRAY);
            // enable vertex attribute arrays for the attribute locations
            glEnableVertexAttribArrayARB( attribVec1);
            glEnableVertexAttribArrayARB( attribVec2);
            glEnableVertexAttribArrayARB( attribVec3);
            glEnableVertexAttribArrayARB( attribTexCoord1);
            glEnableVertexAttribArrayARB( attribTexCoord2);
            glEnableVertexAttribArrayARB( attribTexCoord3);
            glEnableVertexAttribArrayARB( attribColors);
            // set vertex and attribute pointers and draw them
            glVertexAttribPointerARB( attribVec1, 4, GL_FLOAT, 0, 0, this->sphericTriaVec1[cntRS].PeekElements());
            glVertexAttribPointerARB( attribVec2, 4, GL_FLOAT, 0, 0, this->sphericTriaVec2[cntRS].PeekElements());
            glVertexAttribPointerARB( attribVec3, 4, GL_FLOAT, 0, 0, this->sphericTriaVec3[cntRS].PeekElements());
            glVertexAttribPointerARB( attribTexCoord1, 3, GL_FLOAT, 0, 0, this->sphericTriaTexCoord1[cntRS].PeekElements());
            glVertexAttribPointerARB( attribTexCoord2, 3, GL_FLOAT, 0, 0, this->sphericTriaTexCoord2[cntRS].PeekElements());
            glVertexAttribPointerARB( attribTexCoord3, 3, GL_FLOAT, 0, 0, this->sphericTriaTexCoord3[cntRS].PeekElements());
            glVertexAttribPointerARB( attribColors, 3, GL_FLOAT, 0, 0, this->sphericTriaColors[cntRS].PeekElements());
            glVertexPointer( 4, GL_FLOAT, 0, this->sphericTriaVertexArray[cntRS].PeekElements());
            glDrawArrays( GL_POINTS, 0, ((unsigned int)this->sphericTriaVertexArray[cntRS].Count())/4);
            // disable vertex attribute arrays for the attribute locations
            glDisableVertexAttribArrayARB( attribVec1);
            glDisableVertexAttribArrayARB( attribVec2);
            glDisableVertexAttribArrayARB( attribVec3);
            glDisableVertexAttribArrayARB( attribTexCoord1);
            glDisableVertexAttribArrayARB( attribTexCoord2);
            glDisableVertexAttribArrayARB( attribTexCoord3);
            glDisableVertexAttribArrayARB( attribColors);
            glDisableClientState(GL_VERTEX_ARRAY);

			// puxels
#ifdef WITH_PUXELS
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
#endif

            // disable spherical triangle shader
            if(offscreenRendering) {
                this->sphericalTriangleShaderOR.Disable();
            }
            else {
                this->sphericalTriangleShader.Disable();
            }
            // unbind texture
            glBindTexture( GL_TEXTURE_2D, 0);
        }

        /////////////////////////////////////
        // ray cast the spheres on the GPU //
        /////////////////////////////////////
        // enable sphere shader
        if( this->currentRendermode == GPU_RAYCASTING ) {
            if(offscreenRendering) {
                this->sphereShaderOR.Enable();
                // set shader variables
                glUniform4fvARB( this->sphereShaderOR.ParameterLocation("viewAttr"), 1, viewportStuff);
                glUniform3fvARB( this->sphereShaderOR.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
                glUniform3fvARB( this->sphereShaderOR.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
                glUniform3fvARB( this->sphereShaderOR.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
                glUniform3fARB( this->sphereShaderOR.ParameterLocation( "zValues"), fogStart, this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
                glUniform3fARB( this->sphereShaderOR.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
                glUniform1fARB( this->sphereShaderOR.ParameterLocation( "alpha"), this->transparency);
            }
            else {
                this->sphereShader.Enable();
                // set shader variables

#ifdef WITH_PUXELS
				// puxels
				glUniform1ui( this->sphereShader.ParameterLocation("width"),        this->cameraInfo->TileRect().Width());
				glUniform1ui( this->sphereShader.ParameterLocation("height"),       this->cameraInfo->TileRect().Height());
				glUniform1ui( this->sphereShader.ParameterLocation("puxels_use"), this->usePuxels ? 1 : 0);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, puxelsBufferHeader);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, puxelsBufferData);
				glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 5, puxelsAtomicBufferNextId);
#endif

                glUniform4fvARB( this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
                glUniform3fvARB( this->sphereShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
                glUniform3fvARB( this->sphereShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
                glUniform3fvARB( this->sphereShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
                glUniform3fARB( this->sphereShader.ParameterLocation( "zValues"), fogStart, this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
                glUniform3fARB( this->sphereShader.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
                glUniform1fARB( this->sphereShader.ParameterLocation( "alpha"), this->transparency);
            }
        }
        else { // GPU_RAYCASTING_INTERIOR_CLIPPING
            this->sphereClipInteriorShader.Enable();
        }
        glEnableClientState( GL_VERTEX_ARRAY);
        glEnableClientState( GL_COLOR_ARRAY);
        // set vertex and color pointers and draw them
        glColorPointer( 3, GL_FLOAT, 0, this->sphereColors[cntRS].PeekElements());
        glVertexPointer( 4, GL_FLOAT, 0, this->sphereVertexArray[cntRS].PeekElements());
        glDrawArrays( GL_POINTS, 0, ((unsigned int)this->sphereVertexArray[cntRS].Count())/4);
        // disable sphere shader
        glDisableClientState( GL_COLOR_ARRAY);
        glDisableClientState( GL_VERTEX_ARRAY);
        
#ifdef WITH_PUXELS
		// puxels
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
#endif

        // disable sphere shader
        if( this->currentRendermode == GPU_RAYCASTING ) {
            if(offscreenRendering) {
                this->sphereShaderOR.Disable();
            }
            else {
                this->sphereShader.Disable();
            }
        }
        else { // GPU_RAYCASTING_INTERIOR_CLIPPING
            this->sphereClipInteriorShader.Disable();
        }
#ifdef WITH_PUXELS
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
		glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 5, 0);
#endif
    }

    // delete pointers
    delete[] clearColor;
}


/*
 * Render debug stuff
 */
void MoleculeSESRenderer::RenderDebugStuff(
    const MolecularDataCall *mol) {
    // --> USAGE: UNCOMMENT THE NEEDED PARTS

    // temporary variables
    unsigned int max1, max2;
    max1 = max2 = 0;
    vislib::math::Vector<float, 3> v1, v2, v3, n1;
    v1.Set( 0, 0, 0);
    v2 = v3 = n1 = v1;

    //////////////////////////////////////////////////////////////////////////
    // Render the atom positions as GL_POINTS
    //////////////////////////////////////////////////////////////////////////
    /*
    glDisable( GL_LIGHTING);
    glEnable( GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable( GL_POINT_SIZE);
    glEnable( GL_POINT_SMOOTH);
    glPointSize( 5.0f);
    glBegin( GL_POINTS);
    glColor3f( 1.0f, 0.0f, 0.0f);
    if( this->currentRendermode == GPU_SIMPLIFIED )
        max1 = this->simpleRS.size();
    else
        max1 = this->reducedSurface.size();
    for( unsigned int cntRS = 0; cntRS < max1; ++cntRS)
    {
        if( this->currentRendermode == GPU_SIMPLIFIED )
            max2 = this->simpleRS[cntRS]->GetRSVertexCount();
        else
            max2 = this->reducedSurface[cntRS]->GetRSVertexCount();
        for( unsigned int i = 0; i < max2; ++i )
        {
            if( this->currentRendermode == GPU_SIMPLIFIED )
                v1 = this->simpleRS[cntRS]->GetRSVertex( i)->GetPosition();
            else
                v1 = this->reducedSurface[cntRS]->GetRSVertex( i)->GetPosition();
            glVertex3fv( v1.PeekComponents());
        }
    }
    glEnd(); // GL_POINTS
    glDisable( GL_POINT_SMOOTH);
    glDisable( GL_POINT_SIZE);
    glDisable( GL_BLEND);
    glEnable( GL_LIGHTING);
    */

    //////////////////////////////////////////////////////////////////////////
    // Render the probe positions
    //////////////////////////////////////////////////////////////////////////
    /*
    glDisable( GL_LIGHTING);
    glEnable( GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable( GL_POINT_SIZE);
    glEnable( GL_POINT_SMOOTH);
    glPointSize( 5.0f);
    for( unsigned int i = 0; i < this->rsFace.size(); ++i )
    {
        glColor3f( 1.0f, 0.0f, 0.0f);
        glBegin( GL_POINTS);
            glVertex3fv( this->rsFace[i]->GetProbeCenter().PeekComponents());
        glEnd(); // GL_POINTS
    }
    glDisable( GL_POINT_SMOOTH);
    glDisable( GL_POINT_SIZE);
    glDisable( GL_BLEND);
    glEnable( GL_LIGHTING);
    */

    /*
    //////////////////////////////////////////////////////////////////////////
    // Render the probes that cut edges
    //////////////////////////////////////////////////////////////////////////
    // set viewport
    float viewportStuff[4] =
    {
        this->cameraInfo->TileRect().Left(),
        this->cameraInfo->TileRect().Bottom(),
        this->cameraInfo->TileRect().Width(),
        this->cameraInfo->TileRect().Height()
    };
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];
    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
    glColor3f( 0.8f, 0.8f, 0.8f);
    unsigned int i, j;
    glBegin( GL_POINTS);
    for( i = 0; i < this->rsEdge.size(); ++i )
    {
        for( j = 0; j < this->rsEdge[i]->cuttingProbes.size(); ++j )
        {
            glVertex4f( this->rsEdge[i]->cuttingProbes[j]->GetProbeCenter().GetX(),
                this->rsEdge[i]->cuttingProbes[j]->GetProbeCenter().GetY(),
                this->rsEdge[i]->cuttingProbes[j]->GetProbeCenter().GetZ(),
                this->probeRadius);
        }
    }
    glEnd(); // GL_POINTS
    sphereShader.Disable();
    */

    //////////////////////////////////////////////////////////////////////////
    // Draw reduced surface
    //////////////////////////////////////////////////////////////////////////
    this->RenderAtomsGPU( mol, 0.2f);
    vislib::math::Quaternion<float> quatC;
    quatC.Set( 0, 0, 0, 1);
    vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
    vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
    float angle;
    // set viewport
    float viewportStuff[4] =
    {
        this->cameraInfo->TileRect().Left(),
        this->cameraInfo->TileRect().Bottom(),
        this->cameraInfo->TileRect().Width(),
        this->cameraInfo->TileRect().Height()
    };
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];
    // enable cylinder shader
    this->cylinderShader.Enable();
    // set shader variables
    glUniform4fvARB(this->cylinderShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->cylinderShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());
    // get the attribute locations
    GLint attribLocInParams = glGetAttribLocation( this->cylinderShader, "inParams");
    GLint attribLocQuatC = glGetAttribLocation( this->cylinderShader, "quatC");
    GLint attribLocColor1 = glGetAttribLocation( this->cylinderShader, "color1");
    GLint attribLocColor2 = glGetAttribLocation( this->cylinderShader, "color2");
    glBegin( GL_POINTS);
    max1 = (unsigned int)this->reducedSurface.size();
    for( unsigned int cntRS = 0; cntRS < max1; ++cntRS) {
        max2 = this->reducedSurface[cntRS]->GetRSEdgeCount();
        for( unsigned int j = 0; j < max2; ++j ) {
            firstAtomPos = this->reducedSurface[cntRS]->GetRSEdge( j)->GetVertex1()->GetPosition();
            secondAtomPos = this->reducedSurface[cntRS]->GetRSEdge( j)->GetVertex2()->GetPosition();

            // compute the quaternion for the rotation of the cylinder
            dir = secondAtomPos - firstAtomPos;
            tmpVec.Set( 1.0f, 0.0f, 0.0f);
            angle = - tmpVec.Angle( dir);
            ortho = tmpVec.Cross( dir);
            ortho.Normalise();
            quatC.Set( angle, ortho);
            // compute the absolute position 'position' of the cylinder (center point)
            position = firstAtomPos + ( dir/2.0f);

            // draw vertex and attributes
            glVertexAttrib2f( attribLocInParams, 0.12f, (firstAtomPos-secondAtomPos).Length() );
            glVertexAttrib4fv( attribLocQuatC, quatC.PeekComponents());
            glVertexAttrib3f( attribLocColor1, 1.0f, 0.5f, 0.0f);
            glVertexAttrib3f( attribLocColor2, 1.0f, 0.5f, 0.0f);
            glVertex4f( position.GetX(), position.GetY(), position.GetZ(), 1.0f);
        }
    }
    glEnd(); // GL_POINTS
    // disable cylinder shader
    this->cylinderShader.Disable();

    glEnable( GL_COLOR_MATERIAL);
    glPolygonMode( GL_FRONT_AND_BACK, GL_TRIANGLES );
    glDisable( GL_CULL_FACE );
    this->lightShader.Enable();
    unsigned int i;
    for( unsigned int cntRS = 0; cntRS < max1; ++cntRS)
    {
        max2 = this->reducedSurface[cntRS]->GetRSFaceCount();
        for( i = 0; i < max2; ++i )
        {
            n1 = this->reducedSurface[cntRS]->GetRSFace( i)->GetFaceNormal();
            v1 = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex1()->GetPosition();
            v2 = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex2()->GetPosition();
            v3 = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex3()->GetPosition();

            glBegin( GL_TRIANGLES );
                glNormal3fv( n1.PeekComponents());
                glColor3f( 1.0f, 0.8f, 0.0f);
                glVertex3fv( v1.PeekComponents());
                //glColor3f( 0.0f, 0.7f, 0.7f);
                glVertex3fv( v2.PeekComponents());
                //glColor3f( 0.7f, 0.0f, 0.7f);
                glVertex3fv( v3.PeekComponents());
            glEnd(); //GL_TRIANGLES
        }
    }
    this->lightShader.Disable();
    glDisable( GL_COLOR_MATERIAL);

    /*
    //////////////////////////////////////////////////////////////////////////
    // Draw double edges as thick lines
    //////////////////////////////////////////////////////////////////////////
    glDisable( GL_LIGHTING);
    glLineWidth( 5.0f);
    glEnable( GL_LINE_WIDTH);

    glPushAttrib( GL_POLYGON_BIT);
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE);
    unsigned int i;
    for( i = 0; i < this->doubleEdgeList.size(); i=i+2 )
    {
        glColor3f( 1.0f, 1.0f, 0.0f);
        glBegin( GL_LINES );
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetVertex1()->GetIndex()].PeekComponents());
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetVertex2()->GetIndex()].PeekComponents());
        glEnd(); //GL_LINES
        glBegin( GL_TRIANGLES);
        // draw the coincident face
        if( this->doubleEdgeList[i]->GetFace1() == this->doubleEdgeList[i+1]->GetFace1() ||
                this->doubleEdgeList[i]->GetFace1() == this->doubleEdgeList[i+1]->GetFace2() )
        {
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex1()->GetIndex()].PeekComponents() );
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex2()->GetIndex()].PeekComponents() );
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex3()->GetIndex()].PeekComponents() );
            //draw the inner and outer faces
            if( fabs( this->doubleEdgeList[i]->GetRotationAngle() ) < fabs( this->doubleEdgeList[i+1]->GetRotationAngle() ) )
            {
                glColor3f( 0.0f, 1.0f, 0.0f);
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex1()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex2()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex3()->GetIndex()].PeekComponents() );
                glColor3f( 1.0f, 0.0f, 0.0f);
                if( this->doubleEdgeList[i]->GetFace1() == this->doubleEdgeList[i+1]->GetFace1() )
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex3()->GetIndex()].PeekComponents() );
                }
                else
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex3()->GetIndex()].PeekComponents() );
                }
            }
            else
            {
                glColor3f( 1.0f, 0.0f, 0.0f);
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex1()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex2()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex3()->GetIndex()].PeekComponents() );
                glColor3f( 0.0f, 1.0f, 0.0f);
                if( this->doubleEdgeList[i]->GetFace1() == this->doubleEdgeList[i+1]->GetFace1() )
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex3()->GetIndex()].PeekComponents() );
                }
                else
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex3()->GetIndex()].PeekComponents() );
                }
            }
        }
        else
        {
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex1()->GetIndex()].PeekComponents() );
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex2()->GetIndex()].PeekComponents() );
            glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace2()->GetVertex3()->GetIndex()].PeekComponents() );
            //draw the inner and outer faces
            if( fabs( this->doubleEdgeList[i]->GetRotationAngle() ) < fabs( this->doubleEdgeList[i+1]->GetRotationAngle() ) )
            {
                glColor3f( 0.0f, 1.0f, 0.0f);
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex1()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex2()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex3()->GetIndex()].PeekComponents() );
                glColor3f( 1.0f, 0.0f, 0.0f);
                if( this->doubleEdgeList[i]->GetFace2() == this->doubleEdgeList[i+1]->GetFace1() )
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex3()->GetIndex()].PeekComponents() );
                }
                else
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex3()->GetIndex()].PeekComponents() );
                }
            }
            else
            {
                glColor3f( 1.0f, 0.0f, 0.0f);
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex1()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex2()->GetIndex()].PeekComponents() );
                glVertex3fv( this->atomPos[this->doubleEdgeList[i]->GetFace1()->GetVertex3()->GetIndex()].PeekComponents() );
                glColor3f( 0.0f, 1.0f, 0.0f);
                if( this->doubleEdgeList[i]->GetFace2() == this->doubleEdgeList[i+1]->GetFace1() )
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace2()->GetVertex3()->GetIndex()].PeekComponents() );
                }
                else
                {
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex1()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex2()->GetIndex()].PeekComponents() );
                    glVertex3fv( this->atomPos[this->doubleEdgeList[i+1]->GetFace1()->GetVertex3()->GetIndex()].PeekComponents() );
                }
            }
        }
        glEnd(); // GL_TRIANGLES
    }
    glPopAttrib();

    glDisable( GL_LINE_WIDTH);
    glEnable( GL_LIGHTING);
    */

    /*
    //////////////////////////////////////////////////////////////////////////
    // Draw tetrahedra defining the convace spherical triangles
    //////////////////////////////////////////////////////////////////////////
    glDisable( GL_LIGHTING);
    glLineWidth( 3.0f);
    glEnable( GL_LINE_WIDTH);
    for( i = 0; i < this->rsFace.size(); ++i )
    {
        glColor3f( 1.0f, 1.0f, 0.0f);
        glBegin( GL_LINES );
            glVertex3fv( this->atomPos[this->rsFace[i]->GetVertex1()->GetIndex()].PeekComponents());
            glVertex3fv( this->rsFace[i]->GetProbeCenter().PeekComponents());
            glVertex3fv( this->atomPos[this->rsFace[i]->GetVertex2()->GetIndex()].PeekComponents());
            glVertex3fv( this->rsFace[i]->GetProbeCenter().PeekComponents());
            glVertex3fv( this->atomPos[this->rsFace[i]->GetVertex3()->GetIndex()].PeekComponents());
            glVertex3fv( this->rsFace[i]->GetProbeCenter().PeekComponents());
        glEnd(); //GL_LINES
    }
    glDisable( GL_LINE_WIDTH);
    glEnable( GL_LIGHTING);
    */

    /*
    for( i = 0; i < this->rsFace.size(); ++i )
    {
        this->RenderProbe( this->rsFace[i]->GetProbeCenter(), info);
    }
    */

    /*
    //////////////////////////////////////////////////////////////////////////
    // Draw edges
    //////////////////////////////////////////////////////////////////////////
    glPushAttrib( GL_LINE_BIT);
    glLineWidth( 3.0f);
    glEnable( GL_LINE_WIDTH);
    glDisable( GL_LIGHTING);
    glBegin( GL_LINES);
    for( i = 0; i < this->rsEdge.size(); ++i )
    {
        if( this->rsEdge[i]->GetFace2() < 0 ||
            ( *this->rsEdge[i]->GetFace1() == *this->rsEdge[i]->GetFace2() ) )
        {
            glColor3f( 1.0f, 0.0f, 1.0f);
            //std::cout << "Edge: " << i << " (" <<
            //      this->rsEdge[i]->GetFace1() << ")--(" << this->rsEdge[i]->GetFace2() << ") " <<
            //      this->rsEdge[i].GetRotationAngle() << std::endl;
        }
        else
        {
            glColor3f( 0.0f, 1.0f, 0.0f);
        }
        glVertex3fv( this->atomPos[this->rsEdge[i]->GetVertex1()->GetIndex()].PeekComponents());
        glVertex3fv( this->atomPos[this->rsEdge[i]->GetVertex2()->GetIndex()].PeekComponents());
    }
    glEnd(); // GL_LINES
    glEnable( GL_LIGHTING);
    glDisable( GL_LINE_WIDTH);
    glPopAttrib();
    */

    /*
    //////////////////////////////////////////////////////////////////////////
    // Draw lines between probes
    //////////////////////////////////////////////////////////////////////////
    glPushAttrib( GL_LINE_BIT);
    glEnable( GL_LINE_WIDTH);
    glLineWidth( 5.0f);
    glDisable( GL_LIGHTING);
    glBegin( GL_LINES);
    for( i = 0; i < this->rsEdge.size(); ++i )
    {
        glColor3f( 1.0f, 1.0f, 0.0f);
        glVertex3fv( this->rsFace[this->rsEdge[i]->GetFace1()]->GetProbeCenter().PeekComponents());
        glVertex3fv( this->rsFace[this->rsEdge[i]->GetFace2()]->GetProbeCenter().PeekComponents());
    }
    glEnd(); // GL_LINES
    glEnable( GL_LIGHTING);
    glDisable( GL_LINE_WIDTH);
    glPopAttrib();
    */
}


/*
 * Compute the vertex and attribute arrays for the raycasting shaders
 * (spheres, spherical triangles & tori)
 */
void MoleculeSESRenderer::ComputeRaycastingArrays() {
    //time_t t = clock();

    unsigned int cntRS;
    unsigned int i;

    // resize lists of vertex, attribute and color arrays
    this->sphericTriaVertexArray.resize( this->reducedSurface.size());
    this->sphericTriaVec1.resize( this->reducedSurface.size());
    this->sphericTriaVec2.resize( this->reducedSurface.size());
    this->sphericTriaVec3.resize( this->reducedSurface.size());
    this->sphericTriaTexCoord1.resize( this->reducedSurface.size());
    this->sphericTriaTexCoord2.resize( this->reducedSurface.size());
    this->sphericTriaTexCoord3.resize( this->reducedSurface.size());
    this->sphericTriaColors.resize( this->reducedSurface.size());
    this->torusVertexArray.resize( this->reducedSurface.size());
    this->torusInParamArray.resize( this->reducedSurface.size());
    this->torusQuatCArray.resize( this->reducedSurface.size());
    this->torusInSphereArray.resize( this->reducedSurface.size());
    this->torusColors.resize( this->reducedSurface.size());
    this->torusInCuttingPlaneArray.resize( this->reducedSurface.size());
    this->sphereVertexArray.resize( this->reducedSurface.size());
    this->sphereColors.resize( this->reducedSurface.size());


    // compute singulatity textures
    this->CreateSingularityTextures();

    for( cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS )
    {
        ///////////////////////////////////////////////////////////////////////
        // compute arrays for ray casting the spherical triangles on the GPU //
        ///////////////////////////////////////////////////////////////////////
        vislib::math::Vector<float, 3> tmpVec;
        vislib::math::Vector<float, 3> tmpDualProbe( 1.0f, 1.0f, 1.0f);
        float dualProbeRad = 0.0f;

        this->sphericTriaVertexArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaVec1[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaVec2[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaVec3[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 4);
        this->sphericTriaTexCoord1[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 3);
        this->sphericTriaTexCoord2[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 3);
        this->sphericTriaTexCoord3[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 3);
        this->sphericTriaColors[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSFaceCount() * 3);

        // loop over all RS-faces
        for( i = 0; i < this->reducedSurface[cntRS]->GetRSFaceCount(); ++i )
        {
            // if the face has a dual face --> store the probe of this face
            if( this->reducedSurface[cntRS]->GetRSFace( i)->GetDualFace() != NULL )
            {
                tmpDualProbe = this->reducedSurface[cntRS]->GetRSFace( i)->GetDualFace()->GetProbeCenter();
                dualProbeRad = this->probeRadius;
            }
            // first RS-vertex
            tmpVec = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex1()->GetPosition() - this->reducedSurface[cntRS]->GetRSFace( i)->GetProbeCenter();
            this->sphericTriaVec1[cntRS][i*4+0] = tmpVec.GetX();
            this->sphericTriaVec1[cntRS][i*4+1] = tmpVec.GetY();
            this->sphericTriaVec1[cntRS][i*4+2] = tmpVec.GetZ();
            this->sphericTriaVec1[cntRS][i*4+3] = 1.0f;
            // second RS-vertex
            tmpVec = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex2()->GetPosition() - this->reducedSurface[cntRS]->GetRSFace( i)->GetProbeCenter();
            this->sphericTriaVec2[cntRS][i*4+0] = tmpVec.GetX();
            this->sphericTriaVec2[cntRS][i*4+1] = tmpVec.GetY();
            this->sphericTriaVec2[cntRS][i*4+2] = tmpVec.GetZ();
            this->sphericTriaVec2[cntRS][i*4+3] = 1.0f;
            // third RS-vertex
            tmpVec = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex3()->GetPosition() - this->reducedSurface[cntRS]->GetRSFace( i)->GetProbeCenter();
            this->sphericTriaVec3[cntRS][i*4+0] = tmpVec.GetX();
            this->sphericTriaVec3[cntRS][i*4+1] = tmpVec.GetY();
            this->sphericTriaVec3[cntRS][i*4+2] = tmpVec.GetZ();
            this->sphericTriaVec3[cntRS][i*4+3] = dualProbeRad*dualProbeRad;
            // store number of cutting probes and texture coordinates for each edge
            this->sphericTriaTexCoord1[cntRS][i*3+0] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge1()->cuttingProbes.size();
            this->sphericTriaTexCoord1[cntRS][i*3+1] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge1()->GetTexCoordX();
            this->sphericTriaTexCoord1[cntRS][i*3+2] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge1()->GetTexCoordY();
            this->sphericTriaTexCoord2[cntRS][i*3+0] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge2()->cuttingProbes.size();
            this->sphericTriaTexCoord2[cntRS][i*3+1] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge2()->GetTexCoordX();
            this->sphericTriaTexCoord2[cntRS][i*3+2] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge2()->GetTexCoordY();
            this->sphericTriaTexCoord3[cntRS][i*3+0] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge3()->cuttingProbes.size();
            this->sphericTriaTexCoord3[cntRS][i*3+1] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge3()->GetTexCoordX();
            this->sphericTriaTexCoord3[cntRS][i*3+2] = (float)this->reducedSurface[cntRS]->GetRSFace( i)->GetEdge3()->GetTexCoordY();
            // colors
            this->sphericTriaColors[cntRS][i*3+0] = CodeColor( &this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex1()->GetIndex()*3]);
            this->sphericTriaColors[cntRS][i*3+1] = CodeColor( &this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex2()->GetIndex()*3]);
            this->sphericTriaColors[cntRS][i*3+2] = CodeColor( &this->atomColorTable[this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex3()->GetIndex()*3]);
            // sphere center
            this->sphericTriaVertexArray[cntRS][i*4+0] = this->reducedSurface[cntRS]->GetRSFace( i)->GetProbeCenter().GetX();
            this->sphericTriaVertexArray[cntRS][i*4+1] = this->reducedSurface[cntRS]->GetRSFace( i)->GetProbeCenter().GetY();
            this->sphericTriaVertexArray[cntRS][i*4+2] = this->reducedSurface[cntRS]->GetRSFace( i)->GetProbeCenter().GetZ();
            this->sphericTriaVertexArray[cntRS][i*4+3] = this->GetProbeRadius();
        }

        ////////////////////////////////////////////////////////
        // compute arrays for ray casting the tori on the GPU //
        ////////////////////////////////////////////////////////
        vislib::math::Quaternion<float> quatC;
        vislib::math::Vector<float, 3> zAxis, torusAxis, rotAxis, P, X1, X2, C, planeNormal;
        zAxis.Set( 0.0f, 0.0f, 1.0f);
        float distance, d;
        vislib::math::Vector<float, 3> tmpDir1, tmpDir2, tmpDir3, cutPlaneNorm;

        this->torusVertexArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSEdgeCount() * 3);
        this->torusInParamArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSEdgeCount() * 3);
        this->torusQuatCArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSEdgeCount() * 4);
        this->torusInSphereArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSEdgeCount() * 4);
        this->torusColors[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSEdgeCount() * 4);
        this->torusInCuttingPlaneArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSEdgeCount() * 3);

        // loop over all RS-edges
        for( i = 0; i < this->reducedSurface[cntRS]->GetRSEdgeCount(); ++i )
        {
            // get the rotation axis of the torus
            torusAxis = this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter();
            torusAxis.Normalise();
            // get the axis for rotating the torus rotations axis on the z-axis
            rotAxis = torusAxis.Cross( zAxis);
            rotAxis.Normalise();
            // compute quaternion
            quatC.Set( torusAxis.Angle( zAxis), rotAxis);
            // compute the tangential point X2 of the spheres
            P = this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter() + rotAxis *
                this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusRadius();

            X1 = P - this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetPosition();
            X1.Normalise();
            X1 *= this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetRadius();
            X2 = P - this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetPosition();
            X2.Normalise();
            X2 *= this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetRadius();
            d = ( X1 + this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter()).Dot( torusAxis);

            C = this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetPosition() -
                    this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetPosition();
            C = ( ( P - this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetPosition()).Length() /
                ( ( P - this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetPosition()).Length() +
                ( P - this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetPosition()).Length())) * C;
            distance = ( X2 - C).Length();
            C = ( C + this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetPosition()) -
                this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter();

            // compute normal of the cutting plane
            tmpDir1 = this->reducedSurface[cntRS]->GetRSEdge( i)->GetFace1()->GetProbeCenter();
            tmpDir2 = this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetPosition() - tmpDir1;
            tmpDir2.Normalise();
            tmpDir2 *= this->probeRadius;
            tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter();
            tmpDir3 = this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - tmpDir1;
            tmpDir3.Normalise();
            tmpDir3 *= this->probeRadius;
            tmpDir3 = tmpDir3 + tmpDir1 - this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter();
            // tmpDir2 and tmpDir3 now store the position of the intersection points for face 1
            cutPlaneNorm = tmpDir2 - tmpDir3;
            // cutPlaneNorm now stores the vector between the two intersection points for face 1
            tmpDir1 = this->reducedSurface[cntRS]->GetRSEdge( i)->GetFace2()->GetProbeCenter();
            tmpDir2 = this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - tmpDir1;
            tmpDir2.Normalise();
            tmpDir2 *= this->probeRadius;
            tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter();
            // tmpDir2 now stores the position of the intersection point 1 for face 2
            tmpDir2 = tmpDir2 - tmpDir3;
            // tmpDir2 and tmpDir3 now span the plane containing the four intersection points
            cutPlaneNorm = cutPlaneNorm.Cross( tmpDir2);
            cutPlaneNorm = torusAxis.Cross( cutPlaneNorm);
            cutPlaneNorm.Normalise();

            // attributes
            this->torusInParamArray[cntRS][i*3+0] = this->probeRadius;
            this->torusInParamArray[cntRS][i*3+1] = this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusRadius();
            this->torusInParamArray[cntRS][i*3+2] = this->reducedSurface[cntRS]->GetRSEdge( i)->GetRotationAngle();
            this->torusQuatCArray[cntRS][i*4+0] = quatC.GetX();
            this->torusQuatCArray[cntRS][i*4+1] = quatC.GetY();
            this->torusQuatCArray[cntRS][i*4+2] = quatC.GetZ();
            this->torusQuatCArray[cntRS][i*4+3] = quatC.GetW();
            this->torusInSphereArray[cntRS][i*4+0] = C.GetX();
            this->torusInSphereArray[cntRS][i*4+1] = C.GetY();
            this->torusInSphereArray[cntRS][i*4+2] = C.GetZ();
            this->torusInSphereArray[cntRS][i*4+3] = distance;
            // colors
            this->torusColors[cntRS][i*4+0] = CodeColor( &this->atomColorTable[this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex1()->GetIndex()*3]);
            this->torusColors[cntRS][i*4+1] = CodeColor( &this->atomColorTable[this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetIndex()*3]);
            this->torusColors[cntRS][i*4+2] = d;
            //this->torusColors[cntRS][i*4+3] = ( X2 - X1).Length();
            this->torusColors[cntRS][i*4+3] = ( X2 + this->reducedSurface[cntRS]->GetRSEdge( i)->GetVertex2()->GetPosition() - this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter()).Dot( torusAxis) - d;
            // cutting plane
            this->torusInCuttingPlaneArray[cntRS][i*3+0] = cutPlaneNorm.GetX();
            this->torusInCuttingPlaneArray[cntRS][i*3+1] = cutPlaneNorm.GetY();
            this->torusInCuttingPlaneArray[cntRS][i*3+2] = cutPlaneNorm.GetZ();
            // torus center
            this->torusVertexArray[cntRS][i*3+0] = this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter().GetX();
            this->torusVertexArray[cntRS][i*3+1] = this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter().GetY();
            this->torusVertexArray[cntRS][i*3+2] = this->reducedSurface[cntRS]->GetRSEdge( i)->GetTorusCenter().GetZ();
        }

        ///////////////////////////////////////////////////////////
        // compute arrays for ray casting the spheres on the GPU //
        ///////////////////////////////////////////////////////////
        /*
        this->sphereVertexArray[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSVertexCount() * 4);
        this->sphereColors[cntRS].SetCount( this->reducedSurface[cntRS]->GetRSVertexCount() * 3);
        */
        this->sphereVertexArray[cntRS].AssertCapacity(
                        this->reducedSurface[cntRS]->GetRSVertexCount() * 4);
        this->sphereVertexArray[cntRS].Clear();
        this->sphereColors[cntRS].AssertCapacity(
                        this->reducedSurface[cntRS]->GetRSVertexCount() * 3);
        this->sphereColors[cntRS].Clear();

        // loop over all RS-vertices (i.e. all protein atoms)
        for( i = 0; i < this->reducedSurface[cntRS]->GetRSVertexCount(); ++i )
        {
            // add only surface atoms (i.e. with not buried RS-vertices)
            if( this->reducedSurface[cntRS]->GetRSVertex( i)->IsBuried() )
                continue;
            // set vertex color
            this->sphereColors[cntRS].Append( this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex( i)->GetIndex()*3+0]);
            this->sphereColors[cntRS].Append( this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex( i)->GetIndex()*3+1]);
            this->sphereColors[cntRS].Append( this->atomColorTable[this->reducedSurface[cntRS]->GetRSVertex( i)->GetIndex()*3+2]);
            // set vertex position
            this->sphereVertexArray[cntRS].Append(
                this->reducedSurface[cntRS]->GetRSVertex( i)->GetPosition().GetX());
            this->sphereVertexArray[cntRS].Append(
                this->reducedSurface[cntRS]->GetRSVertex( i)->GetPosition().GetY());
            this->sphereVertexArray[cntRS].Append(
                this->reducedSurface[cntRS]->GetRSVertex( i)->GetPosition().GetZ());
            if( this->drawSAS ){
                this->sphereVertexArray[cntRS].Append(
                    this->reducedSurface[cntRS]->GetRSVertex( i)->GetRadius() + this->probeRadius);
            } else {
                this->sphereVertexArray[cntRS].Append(
                    this->reducedSurface[cntRS]->GetRSVertex( i)->GetRadius());
            }
        }
    }
    // print the time of the computation
    //std::cout << "computation of arrays for GPU ray casting finished:" << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
}


/*
 * Compute the vertex and attribute arrays for the raycasting shaders
 * (spheres, spherical triangles & tori)
 */
void MoleculeSESRenderer::ComputeRaycastingArrays( unsigned int idxRS) {
    // do nothing if the given index is out of bounds
    if( idxRS > this->reducedSurface.size() )
        return;

    // check if all arrays have the correct size
    if( this->sphericTriaVertexArray.size() != this->reducedSurface.size() ||
        this->sphericTriaVec1.size() != this->reducedSurface.size() ||
        this->sphericTriaVec2.size() != this->reducedSurface.size() ||
        this->sphericTriaVec3.size() != this->reducedSurface.size() ||
        this->sphericTriaTexCoord1.size() != this->reducedSurface.size() ||
        this->sphericTriaTexCoord2.size() != this->reducedSurface.size() ||
        this->sphericTriaTexCoord3.size() != this->reducedSurface.size() ||
        this->sphericTriaColors.size() != this->reducedSurface.size() ||
        this->torusVertexArray.size() != this->reducedSurface.size() ||
        this->torusInParamArray.size() != this->reducedSurface.size() ||
        this->torusQuatCArray.size() != this->reducedSurface.size() ||
        this->torusInSphereArray.size() != this->reducedSurface.size() ||
        this->torusColors.size() != this->reducedSurface.size() ||
        this->torusInCuttingPlaneArray.size() != this->reducedSurface.size() ||
        this->sphereVertexArray.size() != this->reducedSurface.size() ||
        this->sphereColors.size() != this->reducedSurface.size() )
    {
        // recompute everything if one of the arrays has the wrong size
        //ComputeRaycastingArrays();
        this->preComputationDone = false;
        return;
    }

    unsigned int i;

    // compute singulatity textures
    this->CreateSingularityTexture( idxRS);

    ///////////////////////////////////////////////////////////////////////
    // compute arrays for ray casting the spherical triangles on the GPU //
    ///////////////////////////////////////////////////////////////////////
    vislib::math::Vector<float, 3> tmpVec;
    vislib::math::Vector<float, 3> tmpDualProbe( 1.0f, 1.0f, 1.0f);
    float dualProbeRad = 0.0f;

    this->sphericTriaVertexArray[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaVec1[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaVec2[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaVec3[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 4);
    this->sphericTriaTexCoord1[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 3);
    this->sphericTriaTexCoord2[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 3);
    this->sphericTriaTexCoord3[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 3);
    this->sphericTriaColors[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSFaceCount() * 3);

    // loop over all RS-faces
    for( i = 0; i < this->reducedSurface[idxRS]->GetRSFaceCount(); ++i )
    {
        // if the face has a dual face --> store the probe of this face
        if( this->reducedSurface[idxRS]->GetRSFace( i)->GetDualFace() != NULL )
        {
            tmpDualProbe = this->reducedSurface[idxRS]->GetRSFace( i)->GetDualFace()->GetProbeCenter();
            dualProbeRad = this->probeRadius;
        }
        // first RS-vertex
        tmpVec = this->reducedSurface[idxRS]->GetRSFace( i)->GetVertex1()->GetPosition() - this->reducedSurface[idxRS]->GetRSFace( i)->GetProbeCenter();
        this->sphericTriaVec1[idxRS][i*4+0] = tmpVec.GetX();
        this->sphericTriaVec1[idxRS][i*4+1] = tmpVec.GetY();
        this->sphericTriaVec1[idxRS][i*4+2] = tmpVec.GetZ();
        this->sphericTriaVec1[idxRS][i*4+3] = 1.0f;
        // second RS-vertex
        tmpVec = this->reducedSurface[idxRS]->GetRSFace( i)->GetVertex2()->GetPosition() - this->reducedSurface[idxRS]->GetRSFace( i)->GetProbeCenter();
        this->sphericTriaVec2[idxRS][i*4+0] = tmpVec.GetX();
        this->sphericTriaVec2[idxRS][i*4+1] = tmpVec.GetY();
        this->sphericTriaVec2[idxRS][i*4+2] = tmpVec.GetZ();
        this->sphericTriaVec2[idxRS][i*4+3] = 1.0f;
        // third RS-vertex
        tmpVec = this->reducedSurface[idxRS]->GetRSFace( i)->GetVertex3()->GetPosition() - this->reducedSurface[idxRS]->GetRSFace( i)->GetProbeCenter();
        this->sphericTriaVec3[idxRS][i*4+0] = tmpVec.GetX();
        this->sphericTriaVec3[idxRS][i*4+1] = tmpVec.GetY();
        this->sphericTriaVec3[idxRS][i*4+2] = tmpVec.GetZ();
        this->sphericTriaVec3[idxRS][i*4+3] = dualProbeRad*dualProbeRad;
        // store number of cutting probes and texture coordinates for each edge
        this->sphericTriaTexCoord1[idxRS][i*3+0] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge1()->cuttingProbes.size();
        this->sphericTriaTexCoord1[idxRS][i*3+1] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge1()->GetTexCoordX();
        this->sphericTriaTexCoord1[idxRS][i*3+2] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge1()->GetTexCoordY();
        this->sphericTriaTexCoord2[idxRS][i*3+0] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge2()->cuttingProbes.size();
        this->sphericTriaTexCoord2[idxRS][i*3+1] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge2()->GetTexCoordX();
        this->sphericTriaTexCoord2[idxRS][i*3+2] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge2()->GetTexCoordY();
        this->sphericTriaTexCoord3[idxRS][i*3+0] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge3()->cuttingProbes.size();
        this->sphericTriaTexCoord3[idxRS][i*3+1] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge3()->GetTexCoordX();
        this->sphericTriaTexCoord3[idxRS][i*3+2] = (float)this->reducedSurface[idxRS]->GetRSFace( i)->GetEdge3()->GetTexCoordY();
        // colors
        this->sphericTriaColors[idxRS][i*3+0] = CodeColor( &this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace( i)->GetVertex1()->GetIndex()*3]);
        this->sphericTriaColors[idxRS][i*3+1] = CodeColor( &this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace( i)->GetVertex2()->GetIndex()*3]);
        this->sphericTriaColors[idxRS][i*3+2] = CodeColor( &this->atomColorTable[this->reducedSurface[idxRS]->GetRSFace( i)->GetVertex3()->GetIndex()*3]);
        // sphere center
        this->sphericTriaVertexArray[idxRS][i*4+0] = this->reducedSurface[idxRS]->GetRSFace( i)->GetProbeCenter().GetX();
        this->sphericTriaVertexArray[idxRS][i*4+1] = this->reducedSurface[idxRS]->GetRSFace( i)->GetProbeCenter().GetY();
        this->sphericTriaVertexArray[idxRS][i*4+2] = this->reducedSurface[idxRS]->GetRSFace( i)->GetProbeCenter().GetZ();
        this->sphericTriaVertexArray[idxRS][i*4+3] = this->GetProbeRadius();
    }

    ////////////////////////////////////////////////////////
    // compute arrays for ray casting the tori on the GPU //
    ////////////////////////////////////////////////////////
    vislib::math::Quaternion<float> quatC;
    vislib::math::Vector<float, 3> zAxis, torusAxis, rotAxis, P, X1, X2, C, planeNormal;
    zAxis.Set( 0.0f, 0.0f, 1.0f);
    float distance, d;
    vislib::math::Vector<float, 3> tmpDir1, tmpDir2, tmpDir3, cutPlaneNorm;

    this->torusVertexArray[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSEdgeCount() * 3);
    this->torusInParamArray[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSEdgeCount() * 3);
    this->torusQuatCArray[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSEdgeCount() * 4);
    this->torusInSphereArray[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSEdgeCount() * 4);
    this->torusColors[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSEdgeCount() * 4);
    this->torusInCuttingPlaneArray[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSEdgeCount() * 3);

    // loop over all RS-edges
    for( i = 0; i < this->reducedSurface[idxRS]->GetRSEdgeCount(); ++i )
    {
        // get the rotation axis of the torus
        torusAxis = this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter();
        torusAxis.Normalise();
        // get the axis for rotating the torus rotations axis on the z-axis
        rotAxis = torusAxis.Cross( zAxis);
        rotAxis.Normalise();
        // compute quaternion
        quatC.Set( torusAxis.Angle( zAxis), rotAxis);
        // compute the tangential point X2 of the spheres
        P = this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter() + rotAxis *
                this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusRadius();

        X1 = P - this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetPosition();
        X1.Normalise();
        X1 *= this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetRadius();
        X2 = P - this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetPosition();
        X2.Normalise();
        X2 *= this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetRadius();
        d = ( X1 + this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter()).Dot( torusAxis);

        C = this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetPosition() -
                this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetPosition();
        C = ( ( P - this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetPosition()).Length() /
                ( ( P - this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetPosition()).Length() +
                ( P - this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetPosition()).Length())) * C;
        distance = ( X2 - C).Length();
        C = ( C + this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetPosition()) -
                this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter();

        // compute normal of the cutting plane
        tmpDir1 = this->reducedSurface[idxRS]->GetRSEdge( i)->GetFace1()->GetProbeCenter();
        tmpDir2 = this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetPosition() - tmpDir1;
        tmpDir2.Normalise();
        tmpDir2 *= this->probeRadius;
        tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter();
        tmpDir3 = this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - tmpDir1;
        tmpDir3.Normalise();
        tmpDir3 *= this->probeRadius;
        tmpDir3 = tmpDir3 + tmpDir1 - this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter();
        // tmpDir2 and tmpDir3 now store the position of the intersection points for face 1
        cutPlaneNorm = tmpDir2 - tmpDir3;
        // cutPlaneNorm now stores the vector between the two intersection points for face 1
        tmpDir1 = this->reducedSurface[idxRS]->GetRSEdge( i)->GetFace2()->GetProbeCenter();
        tmpDir2 = this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetPosition() - tmpDir1;
        tmpDir2.Normalise();
        tmpDir2 *= this->probeRadius;
        tmpDir2 = tmpDir2 + tmpDir1 - this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter();
        // tmpDir2 now stores the position of the intersection point 1 for face 2
        tmpDir2 = tmpDir2 - tmpDir3;
        // tmpDir2 and tmpDir3 now span the plane containing the four intersection points
        cutPlaneNorm = cutPlaneNorm.Cross( tmpDir2);
        cutPlaneNorm = torusAxis.Cross( cutPlaneNorm);
        cutPlaneNorm.Normalise();

        // attributes
        this->torusInParamArray[idxRS][i*3+0] = this->probeRadius;
        this->torusInParamArray[idxRS][i*3+1] = this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusRadius();
        this->torusInParamArray[idxRS][i*3+2] = this->reducedSurface[idxRS]->GetRSEdge( i)->GetRotationAngle();
        this->torusQuatCArray[idxRS][i*4+0] = quatC.GetX();
        this->torusQuatCArray[idxRS][i*4+1] = quatC.GetY();
        this->torusQuatCArray[idxRS][i*4+2] = quatC.GetZ();
        this->torusQuatCArray[idxRS][i*4+3] = quatC.GetW();
        this->torusInSphereArray[idxRS][i*4+0] = C.GetX();
        this->torusInSphereArray[idxRS][i*4+1] = C.GetY();
        this->torusInSphereArray[idxRS][i*4+2] = C.GetZ();
        this->torusInSphereArray[idxRS][i*4+3] = distance;
        // colors
        this->torusColors[idxRS][i*4+0] = CodeColor( &this->atomColorTable[this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex1()->GetIndex()*3]);
        this->torusColors[idxRS][i*4+1] = CodeColor( &this->atomColorTable[this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetIndex()*3]);
        this->torusColors[idxRS][i*4+2] = d;
        this->torusColors[idxRS][i*4+3] = ( X2 + this->reducedSurface[idxRS]->GetRSEdge( i)->GetVertex2()->GetPosition() - this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter()).Dot( torusAxis) - d;
        // cutting plane
        this->torusInCuttingPlaneArray[idxRS][i*3+0] = cutPlaneNorm.GetX();
        this->torusInCuttingPlaneArray[idxRS][i*3+1] = cutPlaneNorm.GetY();
        this->torusInCuttingPlaneArray[idxRS][i*3+2] = cutPlaneNorm.GetZ();
        // torus center
        this->torusVertexArray[idxRS][i*3+0] = this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter().GetX();
        this->torusVertexArray[idxRS][i*3+1] = this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter().GetY();
        this->torusVertexArray[idxRS][i*3+2] = this->reducedSurface[idxRS]->GetRSEdge( i)->GetTorusCenter().GetZ();
    }

    ///////////////////////////////////////////////////////////
    // compute arrays for ray casting the spheres on the GPU //
    ///////////////////////////////////////////////////////////
    /*
    this->sphereVertexArray[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSVertexCount() * 4);
    this->sphereColors[idxRS].SetCount( this->reducedSurface[idxRS]->GetRSVertexCount() * 3);
    */
    this->sphereVertexArray[idxRS].AssertCapacity(
                    this->reducedSurface[idxRS]->GetRSVertexCount() * 4);
    this->sphereVertexArray[idxRS].Clear();
    this->sphereColors[idxRS].AssertCapacity(
                    this->reducedSurface[idxRS]->GetRSVertexCount() * 3);
    this->sphereColors[idxRS].Clear();

    // loop over all RS-vertices (i.e. all protein atoms)
    for( i = 0; i < this->reducedSurface[idxRS]->GetRSVertexCount(); ++i )
    {
        // add only surface atoms (i.e. with not buried RS-vertices)
        if( this->reducedSurface[idxRS]->GetRSVertex( i)->IsBuried() )
                continue;
        // set vertex color
        this->sphereColors[idxRS].Append( this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex( i)->GetIndex()*3+0]);
        this->sphereColors[idxRS].Append( this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex( i)->GetIndex()*3+1]);
        this->sphereColors[idxRS].Append( this->atomColorTable[this->reducedSurface[idxRS]->GetRSVertex( i)->GetIndex()*3+2]);
        // set vertex position
        this->sphereVertexArray[idxRS].Append(
                this->reducedSurface[idxRS]->GetRSVertex( i)->GetPosition().GetX());
        this->sphereVertexArray[idxRS].Append(
                this->reducedSurface[idxRS]->GetRSVertex( i)->GetPosition().GetY());
        this->sphereVertexArray[idxRS].Append(
                this->reducedSurface[idxRS]->GetRSVertex( i)->GetPosition().GetZ());
        if( this->drawSAS ){
                this->sphereVertexArray[idxRS].Append(
            this->reducedSurface[idxRS]->GetRSVertex( i)->GetRadius() + this->probeRadius);
        } else {
            this->sphereVertexArray[idxRS].Append(
                this->reducedSurface[idxRS]->GetRSVertex( i)->GetRadius());
        }
        /*
        // set vertex color
        this->sphereColors[idxRS][i*3+0] = this->atomColor[this->reducedSurface[idxRS]->GetRSVertex( i)->GetIndex()*3+0];
        this->sphereColors[idxRS][i*3+1] = this->atomColor[this->reducedSurface[idxRS]->GetRSVertex( i)->GetIndex()*3+1];
        this->sphereColors[idxRS][i*3+2] = this->atomColor[this->reducedSurface[idxRS]->GetRSVertex( i)->GetIndex()*3+2];
        // set vertex position
        this->sphereVertexArray[idxRS][i*4+0] =
            this->reducedSurface[idxRS]->GetRSVertex( i)->GetPosition().GetX();
        this->sphereVertexArray[idxRS][i*4+1] =
            this->reducedSurface[idxRS]->GetRSVertex( i)->GetPosition().GetY();
        this->sphereVertexArray[idxRS][i*4+2] =
            this->reducedSurface[idxRS]->GetRSVertex( i)->GetPosition().GetZ();
        this->sphereVertexArray[idxRS][i*4+3] =
            this->reducedSurface[idxRS]->GetRSVertex( i)->GetRadius();
        */
    }
}


/*
 * code a rgb-color into one float
 */
float MoleculeSESRenderer::CodeColor( const float *col) const {
    return float(
          (int)( col[0] * 255.0f)*1000000   // red
        + (int)( col[1] * 255.0f)*1000      // green
        + (int)( col[2] * 255.0f) );        // blue
}
 /*
float MoleculeSESRenderer::CodeColor( const vislib::math::Vector<float, 3> &col) const {
    return float(
          (int)( col.GetX() * 255.0f)*1000000   // red
        + (int)( col.GetY() * 255.0f)*1000      // green
        + (int)( col.GetZ() * 255.0f) );        // blue
}*/


/*
 * decode a coded color to the original rgb-color
 */
vislib::math::Vector<float, 3> MoleculeSESRenderer::DecodeColor( int codedColor) const {
        int col = codedColor;
        vislib::math::Vector<float, 3> color;
        float red, green;
        if( col >= 1000000 )
            red = floor( (float)col / 1000000.0f);
        else
                red = 0.0;
        col = col - int( red * 1000000.0f);
        if( col > 1000)
            green = floor( (float)col / 1000.0f);
        else
            green = 0.0;
        col = col - int( green * 1000.0f);
        //color.Set( red / 255.0f, green / 255.0f, float(col) / 255.0f);
        color.Set( std::min( 1.0f, std::max( 0.0f, red / 255.0f ) ),
                   std::min( 1.0f, std::max( 0.0f, green / 255.0f ) ),
                   std::min( 1.0f, std::max( 0.0f, col / 255.0f ) ) );
        return color;
}


/*
 * Creates the texture for singularity handling.
 */
void MoleculeSESRenderer::CreateSingularityTextures() {
    // time_t t = clock();
    unsigned int cnt1, cnt2, cntRS;

    // delete old singularity textures
    for( cnt1 = 0; cnt1 < this->singularityTexture.size(); ++cnt1 )
    {
        glDeleteTextures( 1, &singularityTexture[cnt1]);
    }
    // check if the singularity texture has the right size
    if( this->reducedSurface.size() != this->singularityTexture.size() )
    {
        // store old singularity texture size
        unsigned int singTexSizeOld = (unsigned int)this->singularityTexture.size();
        // resize singularity texture to fit the number of reduced surfaces
        this->singularityTexture.resize( this->reducedSurface.size());
        // generate a new texture for each new singularity texture
        for( cnt1 = singTexSizeOld; cnt1 < singularityTexture.size(); ++cnt1 )
        {
            glGenTextures( 1, &singularityTexture[cnt1]);
        }
    }
    // resize singularity texture dimension arrays
    this->singTexWidth.resize( this->reducedSurface.size());
    this->singTexHeight.resize( this->reducedSurface.size());

    // get maximum texture size
    GLint texSize;
    glGetIntegerv( GL_MAX_TEXTURE_SIZE, &texSize);

    // TODO: compute proper maximum number of cutting probes
    unsigned int numProbes = 16;

    for( cntRS = 0; cntRS < this->reducedSurface.size(); ++cntRS )
    {
        // set width and height of texture
        if( (unsigned int)texSize < this->reducedSurface[cntRS]->GetCutRSEdgesCount() )
        {
            this->singTexHeight[cntRS] = texSize;
            this->singTexWidth[cntRS] = numProbes * (int)ceil( double(
                this->reducedSurface[cntRS]->GetCutRSEdgesCount()) / (double)texSize);
        }
        else
        {
            this->singTexHeight[cntRS] = this->reducedSurface[cntRS]->GetCutRSEdgesCount();
            this->singTexWidth[cntRS] = numProbes;
        }
        // generate float-array for texture with the appropriate dimension
        if( this->singTexData )
            delete[] this->singTexData;
        this->singTexData = new float[this->singTexWidth[cntRS]*this->singTexHeight[cntRS]*3];
        // write probes to singularity texture
        unsigned int coordX = 0;
        unsigned int coordY = 0;
        unsigned int counter = 0;
        for( cnt1 = 0; cnt1 < this->reducedSurface[cntRS]->GetRSEdgeCount(); ++cnt1 )
        {
            if( this->reducedSurface[cntRS]->GetRSEdge( cnt1)->cuttingProbes.empty() )
            {
                this->reducedSurface[cntRS]->GetRSEdge( cnt1)->SetTexCoord( 0, 0);
            }
            else
            {
                // set texture coordinates
                this->reducedSurface[cntRS]->GetRSEdge( cnt1)->SetTexCoord( coordX, coordY);
                // compute texture coordinates for next entry
                coordY++;
                if( coordY == this->singTexHeight[cntRS] )
                {
                    coordY = 0;
                    coordX = coordX + numProbes;
                }
                // write probes to texture
                for( cnt2 = 0; cnt2 < numProbes; ++cnt2 )
                {
                    if( cnt2 < this->reducedSurface[cntRS]->GetRSEdge( cnt1)->cuttingProbes.size() )
                    {
                        singTexData[counter] =
                            this->reducedSurface[cntRS]->GetRSEdge( cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetX();
                        counter++;
                        singTexData[counter] =
                            this->reducedSurface[cntRS]->GetRSEdge( cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetY();
                        counter++;
                        singTexData[counter] =
                            this->reducedSurface[cntRS]->GetRSEdge( cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetZ();
                        counter++;
                    }
                    else
                    {
                        singTexData[counter] = 0.0f;
                        counter++;
                        singTexData[counter] = 0.0f;
                        counter++;
                        singTexData[counter] = 0.0f;
                        counter++;
                    }
                }
            }
        }
        // texture generation
        glBindTexture( GL_TEXTURE_2D, singularityTexture[cntRS]);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, this->singTexWidth[cntRS], this->singTexHeight[cntRS], 0, GL_RGB, GL_FLOAT, this->singTexData);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture( GL_TEXTURE_2D, 0);
    }
    //std::cout << "Create texture: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
}


/*
 * Creates the texture for singularity handling.
 */
void MoleculeSESRenderer::CreateSingularityTexture( unsigned int idxRS) {
    // do nothing if the index is out of bounds
    if( idxRS > this->reducedSurface.size() )
        return;

    // check if all arrays have the appropriate size
    if( this->singularityTexture.size() != this->reducedSurface.size() ||
        this->singTexWidth.size() != this->reducedSurface.size() ||
        this->singTexHeight.size() != this->reducedSurface.size() )
    {
        // create all singularity textures
        CreateSingularityTextures();
        return;
    }

    unsigned int cnt1, cnt2;

    // delete old singularity texture
    glDeleteTextures( 1, &singularityTexture[idxRS]);

    // get maximum texture size
    GLint texSize;
    glGetIntegerv( GL_MAX_TEXTURE_SIZE, &texSize);

    // TODO: compute proper maximum number of cutting probes
    unsigned int numProbes = 16;

    // set width and height of texture
    if( (unsigned int)texSize < this->reducedSurface[idxRS]->GetCutRSEdgesCount() )
    {
        this->singTexHeight[idxRS] = texSize;
        this->singTexWidth[idxRS] = numProbes * (int)ceil( double(
            this->reducedSurface[idxRS]->GetCutRSEdgesCount()) / (double)texSize);
    }
    else
    {
        this->singTexHeight[idxRS] = this->reducedSurface[idxRS]->GetCutRSEdgesCount();
        this->singTexWidth[idxRS] = numProbes;
    }
    // generate float-array for texture with the appropriate dimension
    if( this->singTexData )
        delete[] this->singTexData;
    this->singTexData = new float[this->singTexWidth[idxRS]*this->singTexHeight[idxRS]*3];
    // write probes to singularity texture
    unsigned int coordX = 0;
    unsigned int coordY = 0;
    unsigned int counter = 0;
    for( cnt1 = 0; cnt1 < this->reducedSurface[idxRS]->GetRSEdgeCount(); ++cnt1 )
    {
        if( this->reducedSurface[idxRS]->GetRSEdge( cnt1)->cuttingProbes.empty() )
        {
            this->reducedSurface[idxRS]->GetRSEdge( cnt1)->SetTexCoord( 0, 0);
        }
        else
        {
            // set texture coordinates
            this->reducedSurface[idxRS]->GetRSEdge( cnt1)->SetTexCoord( coordX, coordY);
            // compute texture coordinates for next entry
            coordY++;
            if( coordY == this->singTexHeight[idxRS] )
            {
                coordY = 0;
                coordX = coordX + numProbes;
            }
            // write probes to texture
            for( cnt2 = 0; cnt2 < numProbes; ++cnt2 )
            {
                if( cnt2 < this->reducedSurface[idxRS]->GetRSEdge( cnt1)->cuttingProbes.size() )
                {
                    singTexData[counter] =
                        this->reducedSurface[idxRS]->GetRSEdge( cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetX();
                    counter++;
                    singTexData[counter] =
                        this->reducedSurface[idxRS]->GetRSEdge( cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetY();
                    counter++;
                    singTexData[counter] =
                        this->reducedSurface[idxRS]->GetRSEdge( cnt1)->cuttingProbes[cnt2]->GetProbeCenter().GetZ();
                    counter++;
                }
                else
                {
                    singTexData[counter] = 0.0f;
                    counter++;
                    singTexData[counter] = 0.0f;
                    counter++;
                    singTexData[counter] = 0.0f;
                    counter++;
                }
            }
        }
    }
    // texture generation
    glBindTexture( GL_TEXTURE_2D, singularityTexture[idxRS]);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, this->singTexWidth[idxRS], this->singTexHeight[idxRS], 0, GL_RGB, GL_FLOAT, this->singTexData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);

}


/*
 * Render all atoms
 */
void MoleculeSESRenderer::RenderAtomsGPU( const MolecularDataCall *mol, const float scale) {
    unsigned int cnt, cntRS, max1, max2;

    // set viewport
    float viewportStuff[4] =
    {
        this->cameraInfo->TileRect().Left(),
        this->cameraInfo->TileRect().Bottom(),
        this->cameraInfo->TileRect().Width(),
        this->cameraInfo->TileRect().Height()
    };
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());

    glBegin( GL_POINTS);

    glColor3f( 1.0f, 0.0f, 0.0f);
    max1 = (unsigned int)this->reducedSurface.size();
    for( cntRS = 0; cntRS < max1; ++cntRS )
    {
        max2 = this->reducedSurface[cntRS]->GetRSVertexCount();
        // loop over all protein atoms
        for( cnt = 0; cnt < max2; ++cnt )
        {
            if( this->reducedSurface[cntRS]->GetRSVertex( cnt)->IsBuried() )
                continue;
            //glColor3ubv( protein->AtomTypes()[protein->ProteinAtomData()[this->reducedSurface[cntRS]->GetRSVertex( cnt)->GetIndex()].TypeIndex()].Colour());
            glColor3f( 1.0f, 0.0f, 0.0f);
            glVertex4f(
                this->reducedSurface[cntRS]->GetRSVertex( cnt)->GetPosition().GetX(),
                this->reducedSurface[cntRS]->GetRSVertex( cnt)->GetPosition().GetY(),
                this->reducedSurface[cntRS]->GetRSVertex( cnt)->GetPosition().GetZ(),
                this->reducedSurface[cntRS]->GetRSVertex( cnt)->GetRadius()*scale );
        }
    }

    glEnd(); // GL_POINTS

    // disable sphere shader
    this->sphereShader.Disable();
}


/*
 * Renders the probe at postion 'm'
 */
void MoleculeSESRenderer::RenderProbe( const vislib::math::Vector<float, 3> m) {
    GLUquadricObj *sphere = gluNewQuadric();
    gluQuadricNormals( sphere, GL_SMOOTH);

    glEnable( GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE);

    glPushMatrix();
    glTranslatef( m.GetX(), m.GetY(), m.GetZ());
    glColor4f( 1.0f, 1.0f, 1.0f, 0.6f);
    gluSphere( sphere, probeRadius, 16, 8);
    glPopMatrix();

    glDisable( GL_BLEND);

}


/*
 * Renders the probe at postion 'm'
 */
void MoleculeSESRenderer::RenderProbeGPU( const vislib::math::Vector<float, 3> m) {
    // set viewport
    float viewportStuff[4] =
    {
        this->cameraInfo->TileRect().Left(),
        this->cameraInfo->TileRect().Bottom(),
        this->cameraInfo->TileRect().Width(),
        this->cameraInfo->TileRect().Height()
    };
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, this->cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, this->cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, this->cameraInfo->Up().PeekComponents());

    glBegin( GL_POINTS);
            glColor3f( 1.0f, 1.0f, 1.0f );
            glVertex4f( m.GetX(), m.GetY(), m.GetZ(), probeRadius );
    glEnd();

    // disable sphere shader
    this->sphereShader.Disable();
}


/*
 * MoleculeSESRenderer::deinitialise
 */
void MoleculeSESRenderer::deinitialise(void) {
    if( colorFBO ) {
        glDeleteFramebuffersEXT( 1, &colorFBO);
        glDeleteFramebuffersEXT( 1, &blendFBO);
        glDeleteFramebuffersEXT( 1, &horizontalFilterFBO);
        glDeleteFramebuffersEXT( 1, &verticalFilterFBO);
        glDeleteTextures( 1, &texture0);
        glDeleteTextures( 1, &depthTex0);
        glDeleteTextures( 1, &texture1);
        glDeleteTextures( 1, &depthTex1);
        glDeleteTextures( 1, &hFilter);
        glDeleteTextures( 1, &vFilter);
    }
    // delete singularity texture
    for( unsigned int i = 0; i < singularityTexture.size(); ++i )
        glDeleteTextures( 1, &singularityTexture[i]);
    // release shaders
    this->cylinderShader.Release();
    this->sphereShader.Release();
    this->sphereClipInteriorShader.Release();
    this->sphericalTriangleShader.Release();
    this->torusShader.Release();
    this->lightShader.Release();
    this->hfilterShader.Release();
    this->vfilterShader.Release();
    this->silhouetteShader.Release();
    this->transparencyShader.Release();
	
#ifdef WITH_PUXELS
	this->puxelClearShader.Release();
	this->puxelOrderShader.Release();
	this->puxelDrawShader.Release();
#endif
}


/*
 * returns the color of the atom 'idx' for the current coloring mode
 */
vislib::math::Vector<float, 3> MoleculeSESRenderer::GetProteinAtomColor( unsigned int idx) {
    if( idx < this->atomColorTable.Count()/3 )
        //return this->atomColorTable[idx];
        return vislib::math::Vector<float, 3>( this->atomColorTable[idx*3+0],
                                               this->atomColorTable[idx*3+1],
                                               this->atomColorTable[idx*3+0]);
    else
        return vislib::math::Vector<float, 3>( 0.5f, 0.5f, 0.5f);
}

/**
 * (Re)initializes the buffers needed for Puxel rendering.
 */
void MoleculeSESRenderer::puxelsCreateBuffers()
{
	if(puxelsAtomicBufferNextId)
		glDeleteBuffers(1, &puxelsAtomicBufferNextId);
	// atomic counter buffer
	glGenBuffers(1, &puxelsAtomicBufferNextId);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, puxelsAtomicBufferNextId);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), NULL, GL_STREAM_COPY);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

	if(puxelsAtomicBufferNextId)
		glDeleteBuffers(1, &puxelsBufferHeader);
	// puxels header
	glGenBuffers(1, &puxelsBufferHeader);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, puxelsBufferHeader);
	glBufferData(GL_SHADER_STORAGE_BUFFER, this->width * this->height * sizeof(unsigned int), NULL, GL_STREAM_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	if(puxelsAtomicBufferNextId)
		glDeleteBuffers(1, &puxelsBufferData);
	// puxels data
	glGenBuffers(1, &puxelsBufferData);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, puxelsBufferData);
	glBufferData(GL_SHADER_STORAGE_BUFFER, puxelSizeBuffer, NULL, GL_STREAM_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

/**
 * Calls the puxelClearShader shader and resets all values of 
 * puxelsBufferHeaderand puxelsAtomicBufferNextId to zero.
 */
void MoleculeSESRenderer::puxelsClear()
{
	puxelClearShader.Enable();
	glUniform1ui(this->puxelClearShader.ParameterLocation("width"), (GLuint)this->cameraInfo->TileRect().Width());
	glUniform1ui(this->puxelClearShader.ParameterLocation("height"), (GLuint)this->cameraInfo->TileRect().Height());
	
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, puxelsBufferHeader);

	puxelClearShader.Dispatch((unsigned int)this->cameraInfo->TileRect().Width()/16, (unsigned int)this->cameraInfo->TileRect().Height()/16, 1);
	
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	puxelClearShader.Disable();
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

	unsigned int null(0);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, puxelsAtomicBufferNextId);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), &null, GL_DYNAMIC_COPY);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

/**
 * Calls the puxelOrderShader shader and orders each puxel tube
 * according to the depth of the fragments.
 */
void MoleculeSESRenderer::puxelsReorder()
{
	puxelOrderShader.Enable();
	glUniform1ui(this->puxelOrderShader.ParameterLocation("width"), (GLuint)this->cameraInfo->TileRect().Width());
	glUniform1ui(this->puxelOrderShader.ParameterLocation("height"), (GLuint)this->cameraInfo->TileRect().Height());
	
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, puxelsBufferHeader);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, puxelsBufferData);

	puxelOrderShader.Dispatch(this->width/16, this->height/16, 1);
	
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	puxelOrderShader.Disable();
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
}

/**
 * Renders a reduced surface of the molecule for later discarding internal fragments.
 */
void MoleculeSESRenderer::puxelRenderReducedSurface()
{
	return;
    // temporary variables
    unsigned int max1, max2;
    max1 = max2 = 0;
    vislib::math::Vector<float, 3> v1, v2, v3, n1;
    v1.Set( 0, 0, 0);
    v2 = v3 = n1 = v1;

    glEnable( GL_COLOR_MATERIAL);
    glPolygonMode( GL_FRONT_AND_BACK, GL_TRIANGLES );
    glDisable( GL_CULL_FACE );
	this->puxelRenderReducedSurfaceShader.Enable();
	glUniform1ui(this->puxelRenderReducedSurfaceShader.ParameterLocation("width"), this->width);
	glUniform1ui(this->puxelRenderReducedSurfaceShader.ParameterLocation("height"), this->height);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, puxelsBufferHeader);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, puxelsBufferData);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 5, puxelsAtomicBufferNextId);
    unsigned int i;
    for( unsigned int cntRS = 0; cntRS < (unsigned int)this->reducedSurface.size(); ++cntRS)
    {
        max2 = this->reducedSurface[cntRS]->GetRSFaceCount();
        for( i = 0; i < max2; ++i )
        {
            n1 = this->reducedSurface[cntRS]->GetRSFace( i)->GetFaceNormal();
            v1 = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex1()->GetPosition();
            v2 = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex2()->GetPosition();
            v3 = this->reducedSurface[cntRS]->GetRSFace( i)->GetVertex3()->GetPosition();

            glBegin( GL_TRIANGLES );
                glNormal3fv( n1.PeekComponents());
                glColor4f( 1.0f, 0.8f, 0.0f, 0.2f);
                glVertex3fv( v1.PeekComponents());
                //glColor3f( 0.0f, 0.7f, 0.7f);
                glVertex3fv( v2.PeekComponents());
                //glColor3f( 0.7f, 0.0f, 0.7f);
                glVertex3fv( v3.PeekComponents());
            glEnd(); //GL_TRIANGLES
        }
    }
    this->puxelRenderReducedSurfaceShader.Disable();
    glDisable( GL_COLOR_MATERIAL);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 5, 0);
}

/**
 * Calls the puxelBlend shader and displays the contents of the puxel buffer
 * on the current Framebuffer or screen
 */
void MoleculeSESRenderer::puxelsDraw()
{
	glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);


	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	puxelDrawShader.Enable();
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, puxelsBufferHeader);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, puxelsBufferData);
	
	glUniform1ui(this->puxelDrawShader.ParameterLocation("width"), this->width);
	glUniform1ui(this->puxelDrawShader.ParameterLocation("height"), this->height);

	// fullscreenquad
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    glBegin(GL_QUADS);
    glVertex2f( 0.0f, 0.0f);
    glVertex2f( 1.0f, 0.0f);
    glVertex2f( 1.0f, 1.0f);
    glVertex2f( 0.0f, 1.0f);
    glEnd();

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	
	puxelDrawShader.Disable();
	
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

