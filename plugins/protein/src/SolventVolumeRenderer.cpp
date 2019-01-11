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
#include "mmcore/CoreInstance.h"
#include "Color.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/AbstractCallRender.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/glverify.h"
#include "vislib/sys/File.h"
#include "vislib/String.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include <math.h>
#include <time.h>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;


#define USE_VERTEX_SKIP_SHADER

/*
 * protein::SolventVolumeRenderer::SolventVolumeRenderer (CTOR)
 */
protein::SolventVolumeRenderer::SolventVolumeRenderer ( void ) : Renderer3DModule (),
        protDataCallerSlot ( "getData", "Connects the volume rendering with data storage" ),
        protRendererCallerSlot ( "renderMolecule", "Connects the volume rendering with a molecule renderer" ),
        dataOutSlot ( "volumeout", "Connects the volume rendering with a volume slice renderer" ),
        coloringModeSolventParam ( "coloringModeSolvent", "solvent coloring mode" ),
        coloringModePolymerParam ( "coloringModePolymer", "polymer coloring node" ),
        coloringModeVolSurfParam ( "coloringModeVolSurf", "volume surface coloring mode"),
        colorFilterRadiusParam( "colorFilterRadius", "color filter radius"),
        colorIntensityScaleParam( "colorIntensityScale", "color intensity scale"),
        volIsoValue1Param( "volIsoValue1", "First isovalue for isosurface rendering"),
        //volIsoValue2Param( "volIsoValue2", "Second isovalue for isosurface rendering"),
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
        //solventResidues("solventResidues", ";-list of residue names which compose the solvent"),
        stickRadiusParam( "stickRadius", "The radius for stick rendering"),
        atomRadiusFactorParam("atomRadiusFactor", "factor for atom sphere radius"),
        atomSpaceFillingParam("atomSpaceFilling", "render atoms space filling using Van-der-Waals radii"),
        solventMolThreshold( "solventMolThreshold", "threshold of visible solvent-molecules" ),
        accumulateColors("accumulateColors", "accumulate color distribution on the volume surface over time"),
        accumulateVolume("accumulateVolume", "accumulate volume density over time"),
        accumulateFactor("accumulateFactor", "accumulate factor for color/volume accumulation ..."),
        countSolMolParam("countSolventMol", "compute number of solvent molecules (by residue type) in hydration shell"),
        currentFrameId ( 0 ), atomCount( 0 ), volumeTex( 0), volumeSize( 128), volFBO( 0),
        volFilterRadius( 1.75f), volDensityScale( 1.0f),
        width( 0), height( 0), volRayTexWidth( 0), volRayTexHeight( 0),
        volRayStartTex( 0), volRayLengthTex( 0), volRayDistTex( 0),
        renderIsometric( true), meanDensityValue( 0.0f), isoValue1( 0.5f), /*isoValue2(-0.5f),*/
        volIsoOpacity( 0.4f), volClipPlaneFlag( false), volClipPlaneOpacity( 0.4f),
        forceUpdateVolumeTexture( true), lastUpdateVolumeTextureTime(0), forceUpdateColoringMode(true),
        molVisVbo( 0)
{
    // set caller slot for different data calls
    this->protDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable ( &this->protDataCallerSlot );
    
    // set renderer caller slot
    this->protRendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable( &this->protRendererCallerSlot);

    // --- set the coloring mode ---
    param::EnumParam *polymerCMEnum = new param::EnumParam ( int ( Color::ELEMENT ) );
    MolecularDataCall *mol = new MolecularDataCall();
    Color::ColoringMode cMode;
    unsigned int numClrModes = Color::GetNumOfColoringModes( mol);
    for(unsigned int cCnt = 0; cCnt < numClrModes; ++cCnt) {
        cMode = Color::GetModeByIndex( mol, cCnt);
        polymerCMEnum->SetTypePair( cMode, Color::GetName( cMode).c_str());
    }
    delete mol;

    param::EnumParam *solventCMEnum = new param::EnumParam ( *polymerCMEnum );

    this->coloringModePolymerParam << polymerCMEnum;
    this->coloringModeSolventParam << solventCMEnum;

    /*
    cm->SetTypePair( Color::ELEMENT, "Element" );
    cm->SetTypePair( Color::AMINOACID, "AminoAcid" );
    cm->SetTypePair( Color::STRUCTURE, "SecondaryStructure" );
    cm->SetTypePair( Color::VALUE, "Value");
    cm->SetTypePair( Color::CHAIN_ID, "ChainID" );
    cm->SetTypePair( Color::RAINBOW, "Rainbow");
    cm->SetTypePair( Color::CHARGE, "Charge" );
    */


    this->MakeSlotAvailable( &this->coloringModePolymerParam );
    this->MakeSlotAvailable( &this->coloringModeSolventParam );

    param::EnumParam *isoSurfCMEnum = new param::EnumParam (VOLCM_SolventConcentration);
    isoSurfCMEnum->SetTypePair(VOLCM_SolventConcentration, "SolventConcentration");
    isoSurfCMEnum->SetTypePair(VOLCM_HydrogenBonds, "HydrogenBonds");
    isoSurfCMEnum->SetTypePair(VOlCM_HydrogenBondStats, "HydrogenBondStats");
    this->coloringModeVolSurfParam << isoSurfCMEnum;
    this->MakeSlotAvailable( &this->coloringModeVolSurfParam );

    this->colorFilterRadiusParam.SetParameter( new param::FloatParam(this->volFilterRadius, 0.0f));
    this->MakeSlotAvailable( &this->colorFilterRadiusParam );
    this->colorIntensityScaleParam.SetParameter( new param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable( &this->colorIntensityScaleParam );

    // --- set up parameters for isovalues ---
    this->volIsoValue1Param.SetParameter( new param::FloatParam( this->isoValue1) );
    //this->volIsoValue2Param.SetParameter( new param::FloatParam( this->isoValue2) );
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
        (float)this->volClipPlane[0].PeekComponents()[0], 
        (float)this->volClipPlane[0].PeekComponents()[1], 
        (float)this->volClipPlane[0].PeekComponents()[2]);
    this->volClipPlane0NormParam.SetParameter( new param::Vector3fParam( cp0n) );
    // --- set up parameter for volume clipping plane distance ---
    float d = (float)this->volClipPlane[0].PeekComponents()[3];
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
//    this->solventResidues.SetParameter(new param::StringParam(""));
//    this->MakeSlotAvailable( &this->solventResidues);

    // fill color table with default values and set the filename param
    this->stickRadiusParam.SetParameter(new param::FloatParam( 0.3f, 0.0f));
    this->MakeSlotAvailable( &this->stickRadiusParam);

    this->atomRadiusFactorParam.SetParameter(new param::FloatParam( 1.0f, 0.0f));
    this->MakeSlotAvailable( &this->atomRadiusFactorParam);


    this->atomSpaceFillingParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable( &this->atomSpaceFillingParam);

    // 
    this->solventMolThreshold.SetParameter(new param::FloatParam( 0.01f, 0.0f));
    this->MakeSlotAvailable( &this->solventMolThreshold);

    this->accumulateColors.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable( &this->accumulateColors);

    this->accumulateVolume.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable( &this->accumulateVolume);

    this->accumulateFactor.SetParameter(new param::FloatParam(0.2f, 0.0f));
    this->MakeSlotAvailable( &this->accumulateFactor);

    this->countSolMolParam.SetParameter(new param::BoolParam( false));
    this->MakeSlotAvailable( &this->countSolMolParam);

    // make all slots available
    this->MakeSlotAvailable( &this->volIsoValue1Param );
    //this->MakeSlotAvailable( &this->volIsoValue2Param );
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

}


/*
 * protein::SolventVolumeRenderer::~SolventVolumeRenderer (DTOR)
 */
protein::SolventVolumeRenderer::~SolventVolumeRenderer ( void ) {
    this->Release ();
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
    if( !areExtsAvailable( "GL_EXT_framebuffer_object GL_ARB_texture_float GL_EXT_gpu_shader4 GL_EXT_bindable_uniform") )
        return false;
    if (!isExtAvailable( "GL_ARB_vertex_program")  || !ogl_IsVersionGEQ(2,0))
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
    if( !loadShader( this->sphereShader, "protein::std::sphereVertex", "protein::std::sphereFragment" ) )
        return false;

    // Load sphere shader
    if( !loadShader( this->sphereSolventShader, "protein::std::sphereSolventVertex", "protein::std::sphereFragment" ) )
        return false;

    // Load clipped sphere shader -> TODO: solvent version?
    if( !loadShader( this->clippedSphereShader, "protein::std::sphereClipPlaneVertex", "protein::std::sphereClipPlaneFragment" ) )
        return false;

    // Load cylinder shader
    if( !loadShader( this->cylinderSolventShader, "protein::std::cylinderSolventVertex", "protein::std::cylinderFragment" ) )
        return false;

    // Load shader for hydrogen bonds
//    if( !loadShader( this->hbondLineSolventShader, "protein::std::hbondLineSolventVertex", "protein::std::hbondLineSolventFragment" ) )
//        return false;

    // Load volume texture generation shader
    if( !loadShader( this->updateVolumeShaderMoleculeVolume, "volume::std::updateVolumeVertex", "volume::std::updateSolventVolumeFragmentDensity" ) )
        return false;

#ifdef USE_VERTEX_SKIP_SHADER
    // this version of the volume vertex shader saves a lot of fragment processing power in UpdateVolumeTexture()
#pragma message ( "WARNING: since the user can controll density and coloing filter radius separately, this optimization might skip too many vertices!" )
    vislib::StringA updateVolumeVertex("volume::std::updateVolumeSkipDensityVertex");
#else
    vislib::StringA updateVolumeVertex("volume::std::updateVolumeVertex");
#endif
    if( !loadShader( this->updateVolumeShaderSolventColor, updateVolumeVertex, "volume::std::updateSolventVolumeFragmentColor" ) )
        return false;
    if( !loadShader( this->updateVolumeShaderHBondColor, updateVolumeVertex, "volume::std::updateSolventVolumeFragmentHBondClr" ) )
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

    // Load visible solvent molecule shader
    if( !loadShader( this->visMolShader, "protein::std::visibleSolventMoleculeVertex", "protein::std::visibleSolventMoleculeFragment" ) )
        return false;

    // Load solvent type counting shader
    if( !loadShader( this->solTypeCountShader, "protein::std::solventTypeCountVertex", "protein::std::visibleSolventMoleculeFragment" ) )
        return false;

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/


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
/*    view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
    vislib::math::Point<float, 3> protrenbbc;
    if( protrencr3d ) {
        (*protrencr3d)(core::view::AbstractCallRender::FnGetExtents); // GetExtents
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


void protein::SolventVolumeRenderer::ColorAtom(float *atomColor, MolecularDataCall *mol, int colorMode, int atomIdx, int residueIdx ) {
    switch(colorMode) {
    default:
    case Color::ELEMENT: {
#if 1
        const unsigned char *c = mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Colour();
        atomColor[0] = c[0] / 255.0f;
        atomColor[1] = c[1] / 255.0f;
        atomColor[2] = c[2] / 255.0f;
#else
        vislib::math::Vector<float, 3>& c = this->colorLookupTable[mol->AtomTypeIndices()[atomIdx]%this->colorLookupTable.Count()];
        atomColor[0] = c.X();
        atomColor[1] = c.Y();
        atomColor[2] = c.Z();
#endif
    } break;
    case Color::RESIDUE: {
        int resTypeIdx = mol->Residues()[residueIdx]->Type();
        vislib::math::Vector<float, 3>& c = this->colorLookupTable[resTypeIdx%this->colorLookupTable.Count()];
        atomColor[0] = c.X();
        atomColor[1] = c.Y();
        atomColor[2] = c.Z();
    } break;
    case Color::STRUCTURE: {
        ; // MolecularDataCall::SecStructure::TYPE_HELIX, TYPE_COIL etc
    } break;
    case Color::BFACTOR: {
        ;
    } break;
    case Color::CHARGE: {
        ;
    } break;
    case Color::OCCUPANCY:
        break;
    case Color::CHAIN: {
        const MolecularDataCall::Residue *residue = mol->Residues()[residueIdx];
        int moleculeIdx = residue->MoleculeIndex();
        int chainIdx = mol->Molecules()[moleculeIdx].ChainIndex();
        vislib::math::Vector<float, 3>& c = this->colorLookupTable[chainIdx%this->colorLookupTable.Count()];
        atomColor[0] = c.X();
        atomColor[1] = c.Y();
        atomColor[2] = c.Z();
    } break;
    case Color::MOLECULE: {
        const MolecularDataCall::Residue *residue = mol->Residues()[residueIdx];
        int moleculeIdx = residue->MoleculeIndex();
        vislib::math::Vector<float, 3>& c = this->colorLookupTable[moleculeIdx%this->colorLookupTable.Count()];
        atomColor[0] = c.X();
        atomColor[1] = c.Y();
        atomColor[2] = c.Z();
    } break;
    case Color::RAINBOW:
        break;
    }
}

void protein::SolventVolumeRenderer::UpdateColorTable(MolecularDataCall *mol) {
    if (!forceUpdateColoringMode && this->atomColorTable.Count() == mol->AtomCount()*3)
        return;

    forceUpdateColoringMode = false;

/*    Color::MakeColorTable( mol,
        this->currentColoringMode,
        this->atomColorTable,
        this->colorLookupTable,
        this->rainbowColors,
        this->minGradColorParam.Param<param::StringParam>()->Value(),
        this->midGradColorParam.Param<param::StringParam>()->Value(),
        this->maxGradColorParam.Param<param::StringParam>()->Value(),
        true);*/

    if (this->atomColorTable.Count() < mol->AtomCount()*3)
        this->atomColorTable.SetCount(mol->AtomCount()*3);

    float *atomColorTablePtr = &this->atomColorTable[0];

    int solventColorMode = this->coloringModeSolventParam.Param<param::EnumParam>()->Value();
    int polymerColorMode = this->coloringModePolymerParam.Param<param::EnumParam>()->Value();

    for( unsigned int residueIdx = 0; residueIdx < mol->ResidueCount(); residueIdx++ ) {
        const MolecularDataCall::Residue *residue = mol->Residues()[residueIdx];
        int firstAtomIndex = residue->FirstAtomIndex();
        int lastAtomIndx = residue->FirstAtomIndex() + residue->AtomCount();

        bool isSolvent = mol->IsSolvent(residue);

        for(int atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++) {
            float *atomColor = &atomColorTablePtr[atomIdx*3];
            if (isSolvent)
                ColorAtom(atomColor, mol, solventColorMode, atomIdx, residueIdx );
            else {
                Color::ColoringMode currentColoringMode0 = static_cast<Color::ColoringMode>(int(this->coloringModePolymerParam.Param<param::EnumParam>()->Value()));
                //ColorAtom(atomColor, mol, polymerColorMode, atomIdx, residueIdx );
                Color::MakeColorTable( mol,
                    currentColoringMode0,
                    this->atomColorTable, this->colorLookupTable, this->rainbowColors,
                    this->minGradColorParam.Param<param::StringParam>()->Value(),
                    this->midGradColorParam.Param<param::StringParam>()->Value(),
                    this->maxGradColorParam.Param<param::StringParam>()->Value(),
                    true);
            }
        }
    }
}


/*
 * protein::SolventVolumeRenderer::Render
 */
bool protein::SolventVolumeRenderer::Render( Call& call ) {
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>( &call );
    if( !cr3d ) return false;

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();

    if (!mol)
        return false;

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
    this->ParameterRefresh( cr3d, mol);
    
    // get the call time
    float callTime = cr3d->Time();

    // use floor/ceil here?!
    unsigned int frameID0 = static_cast<unsigned int>(callTime);
    unsigned int frameID1 = (frameID0+1) < mol->FrameCount() ? frameID0+1 : frameID0;
    float frameInterp = callTime - static_cast<float>(static_cast<int>( callTime));
    float *interAtomPosFrame0Ptr = 0;
    int *interHBondFrame0Ptr = 0;

    /* get first frame and copy it to the temporary array for now.
      we could force the datacall 'mol' to keep the locked frame data to avoid that copy.
     (setting mol->SetUnlocker(0, false) will prevent unlocking of the current frame after loading another)*/
    mol->SetFrameID(frameID0);
    if( !(*mol)(MolecularDataCall::CallForGetData) || mol->AtomCount() == 0 )
        return false;
    if (this->interpAtomPosTmpArray.Count() < mol->AtomCount()*3)
        this->interpAtomPosTmpArray.SetCount(mol->AtomCount()*3);
    interAtomPosFrame0Ptr = &this->interpAtomPosTmpArray[0];
    memcpy(interAtomPosFrame0Ptr, mol->AtomPositions(), sizeof(float)*mol->AtomCount()*3);

    if (mol->AtomHydrogenBondIndices()) {
        if (this->interpHBondTmpArray.Count() < mol->AtomCount())
            this->interpHBondTmpArray.SetCount(mol->AtomCount());
        interHBondFrame0Ptr = &this->interpHBondTmpArray[0];
        memcpy(interHBondFrame0Ptr, mol->AtomHydrogenBondIndices(), sizeof(int)*mol->AtomCount());
    }

    // force volume texture update if data has changed
    if (frameID0 != this->interpFrame0 || mol->DataHash() != this->interpDataHash0)
        this->forceUpdateVolumeTexture = true;
    this->interpFrame0 = frameID0;
    this->interpDataHash0 = mol->DataHash();


    // check if the atom positions have to be interpolated
    if (frameID0 != frameID1 && this->interpolParam.Param<param::BoolParam>()->Value()) {
        const float *interAtomPosFrame1Ptr;
        const int *interHBondFrame1Ptr;

        mol->Unlock(); // very important because we're goning to load a new frame ...
        mol->SetFrameID(frameID1);
        if( !(*mol)(MolecularDataCall::CallForGetData) )
            return false;
        interAtomPosFrame1Ptr = mol->AtomPositions();
        interHBondFrame1Ptr = mol->AtomHydrogenBondIndices();

    /*    // force volume texture update if data has changed -> will be done anyway in animation mode?!
        if (frameID1 != this->interpFrame1 || mol->DataHash() != this->interpDataHash1)
            this->forceUpdateVolumeTexture = true;
        this->interpFrame1 = frameID1;
        this->interpDataHash1 = mol->DataHash();*/

        const vislib::math::Cuboid<float>& bbox = mol->AccessBoundingBoxes().ObjectSpaceBBox();

        // wg zyklischer Randbedingung ...
        float threshold = vislib::math::Min( bbox.Width(), vislib::math::Min( bbox.Height(), bbox.Depth())) * 0.4f; // 0.75f;

        // interpolate atom positions between frames
#pragma omp parallel for
        for(int atomIdx = 0; atomIdx < (int)mol->AtomCount(); atomIdx++) {
            float localInter = frameInterp;
#if 0
            vislib::math::ShallowPoint<float,3> atomPosFrame0( interAtomPosFrame0Ptr + atomIdx*3 );
            vislib::math::ShallowPoint<float,3> atomPosFrame1( const_cast<float*>(interAtomPosFrame1Ptr) + atomIdx*3 );

            if( atomPosFrame0.Distance(atomPosFrame1) >= threshold ) {
                if( localInter < 0.5f )
                    localInter = 0;
                else
                    localInter = 1;
            }

            atomPosFrame0 = atomPosFrame0.Interpolate(atomPosFrame1, localInter);
#else
            int tmp = atomIdx * 3;
            float *p0 = interAtomPosFrame0Ptr + tmp;
            const float *p1 = interAtomPosFrame1Ptr + tmp;
            float dx = p0[0]-p1[0];
            float dy = p0[1]-p1[1];
            float dz = p0[2]-p1[2];
            if (sqrtf(dx*dx+dy*dy+dz*dz) >= threshold) {
                if( localInter < 0.5f )
                    localInter = 0;
                else
                    localInter = 1;
            }
            p0[0] = (1.0f-localInter)*p0[0] + localInter*p1[0];
            p0[1] = (1.0f-localInter)*p0[1] + localInter*p1[1];
            p0[2] = (1.0f-localInter)*p0[2] + localInter*p1[2];
#endif

            // maybe test here the distance between atom[atomIdx] and atom[interHBonds0/1[atomIdx]] ?!
            if (interHBondFrame0Ptr)
                interHBondFrame0Ptr[atomIdx] = /*inter*/localInter < 0.5 ? interHBondFrame0Ptr[atomIdx] : interHBondFrame1Ptr[atomIdx];
        }
    }

    this->atomPosInterPtr = interAtomPosFrame0Ptr;
    this->hBondInterPtr = interHBondFrame0Ptr;

    UpdateColorTable(mol);

    for(unsigned int cpCnt = 0; cpCnt < this->volClipPlane.Count(); ++cpCnt )
        glClipPlane( GL_CLIP_PLANE0+cpCnt, this->volClipPlane[cpCnt].PeekComponents());

    // =============== Volume Rendering ===============
    bool retval = false;

    // try to start volume rendering using protein data
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
    // =============== Molecule Rendering ===============

    if( this->countSolMolParam.Param<param::BoolParam>()->Value() ) {
        // disable the output buffer
        cr3d->DisableOutputBuffer();
        // find visible molecules
        //glEnable( GL_DEPTH_TEST);
        //glDepthFunc(GL_LEQUAL);
        glDisable( GL_DEPTH_TEST);
        this->FindVisibleSolventMolecules( mol);
        // re-enable the output buffer
        cr3d->EnableOutputBuffer();
    }

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
        
#if 1 // HACKHACKHACK
        // get the pointer to CallRender3D (protein renderer)
        view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
        if( protrencr3d ) {
            // setup and call protein renderer
            glPushMatrix();
            *protrencr3d = *cr3d;
            protrencr3d->SetOutputBuffer( &this->proteinFBO); // TODO: Handle incoming buffers!
            (*protrencr3d)();
            glPopMatrix();
        } else
#endif // HACKHACKHACK
            RenderMolecules(mol, this->atomPosInterPtr);

        RenderHydrogenBounds(mol, this->atomPosInterPtr); // TEST
        

    glPopMatrix();
    // stop rendering to the FBO for protein rendering
    this->proteinFBO.Disable();
    // re-enable the output buffer
    cr3d->EnableOutputBuffer();
    
#endif
        
// DEBUG
#if 0
    float dist = 50.f;
    unsigned int ai = 0;
    gnf.SetPointData( mol->AtomPositions(), mol->AtomCount(), mol->AccessBoundingBoxes().ObjectSpaceBBox(), dist);
    vislib::Array<unsigned int> na;
    gnf.FindNeighboursInRange( &mol->AtomPositions()[ai*3], dist, na);
    
    // ==================== Scale & Translate ====================
    glPushMatrix();
    // compute scale factor and scale world
    scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);    
    vislib::math::Vector<float, 3> trans( 
        mol->AccessBoundingBoxes().ObjectSpaceBBox().GetSize().PeekDimension() );
    //trans *= -this->scale*0.5f;
    trans *= -0.5f;
    glTranslatef( trans.GetX(), trans.GetY(), trans.GetZ());

    // ==================== Start actual rendering ====================
    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable( GL_VERTEX_PROGRAM_TWO_SIDE);
    
    float viewportStuff[4] = {
        cameraInfo->TileRect().Left(), cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(), cameraInfo->TileRect().Height()};
    if( viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if( viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    int cnt1, cnt2;
    
    vislib::Array<float> colTab( mol->AtomCount() * 3);
    colTab.SetCount( mol->AtomCount() * 3);
#pragma omp parallel for
    for( cnt1 = 0; cnt1 < (int)mol->AtomCount(); ++cnt1 ) {
        colTab[cnt1*3+0] = 1.0f;
        colTab[cnt1*3+1] = 0.0f;
        colTab[cnt1*3+2] = 0.0f;
    }

    cnt1 = ai;
    vislib::math::Vector<float, 3> vec1( &mol->AtomPositions()[cnt1*3]);
    for( cnt2 = 0; cnt2 < mol->AtomCount(); ++cnt2 ) {
        if( cnt2 == cnt1 ) continue;
        vislib::math::Vector<float, 3> vec2( &mol->AtomPositions()[cnt2*3]);
        if( ( vec2 - vec1).Length() < dist ) {
            colTab[cnt2*3+0] = 0.0f;
            colTab[cnt2*3+1] = 0.0f;
            colTab[cnt2*3+2] = 1.0f;
        }
    }

#pragma omp parallel for
    for( cnt1 = 0; cnt1 < na.Count(); ++cnt1 ) {
        colTab[na[cnt1]*3+0] = 1.0f;
        colTab[na[cnt1]*3+1] = 1.0f;
        colTab[na[cnt1]*3+2] = 0.0f;
    }
    colTab[ai*3+0] = 0.0f;
    colTab[ai*3+1] = 1.0f;
    colTab[ai*3+2] = 0.0f;

    this->sphereShader.Enable();
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // draw atoms
    glBegin( GL_POINTS);
    for( cnt1 = 0; cnt1 < mol->AtomCount(); ++cnt1 ) {
        if( cnt1 == ai ) {
            glColor3fv( &colTab[cnt1*3]);
            glVertex4f(
                mol->AtomPositions()[cnt1*3+0],
                mol->AtomPositions()[cnt1*3+1],
                mol->AtomPositions()[cnt1*3+2],
                mol->AtomTypes()[mol->AtomTypeIndices()[cnt1]].Radius());
        } else {    
            glColor3fv( &colTab[cnt1*3]);
            glVertex4f(
                mol->AtomPositions()[cnt1*3+0],
                mol->AtomPositions()[cnt1*3+1],
                mol->AtomPositions()[cnt1*3+2],
                mol->AtomTypes()[mol->AtomTypeIndices()[cnt1]].Radius() / 5.0f);
        }
    }
    glEnd();
    this->sphereShader.Disable();

    glPopMatrix();
#else    
    // =============== Volume Rendering ===============
    retval = this->RenderMolecularData( cr3d, mol);
#endif

    // unlock the current frame
    mol->Unlock();

    return retval;
}

/*
 * count visible solvent molecules
 */
void protein::SolventVolumeRenderer::FindVisibleSolventMolecules( MolecularDataCall *mol) {
    // TODO: write list of solvent atoms and residue indices only once, update atom positions for each frame

    // atom counter
    unsigned int atomCnt = 0;
    unsigned int molCnt = 0;
    // reserve memory
    this->solventAtomPos.AssertCapacity( mol->AtomCount() * 3);
    this->solventAtomPos.Clear();
    this->solventAtomParams.AssertCapacity( mol->AtomCount() * 2);
    this->solventAtomParams.Clear();
    // get all atoms of all solvent molecules
    // loop over all chains
    for( unsigned int chCnt = 0; chCnt < mol->ChainCount(); chCnt++ ) {
        // check chain type (do nothing if not solvent
        if( mol->Chains()[chCnt].Type() != MolecularDataCall::Chain::SOLVENT )
            continue;
        // molecule indices of current chain
        unsigned int firstMolIdx = mol->Chains()[chCnt].FirstMoleculeIndex();
        unsigned int lastMolIdx = firstMolIdx + mol->Chains()[chCnt].MoleculeCount();
        for( unsigned int mCnt = firstMolIdx; mCnt < lastMolIdx; mCnt++ ) {
            // count all solvent molecules
            molCnt++;
            // residue indices of current molecule
            unsigned int firstResIdx = mol->Molecules()[mCnt].FirstResidueIndex();
            unsigned int lastResIdx = firstResIdx + mol->Molecules()[mCnt].ResidueCount();            
            for( unsigned int rCnt = firstResIdx; rCnt < lastResIdx; rCnt++ ) {
                // atom indices of current residue
                unsigned int firstAtomIdx = mol->Residues()[rCnt]->FirstAtomIndex();
                unsigned int lastAtomIdx = firstAtomIdx + mol->Residues()[rCnt]->AtomCount();
                // search for solvent residue type index
                unsigned int solResIdx = 0;
                for( unsigned int i = 0; i < mol->AtomSolventResidueCount(); i++ ) {
                    if( mol->SolventResidueIndices()[i] == mol->Residues()[rCnt]->Type() ) {
                        solResIdx = i;
                        break;
                    }
                }
                for( unsigned int aCnt = firstAtomIdx; aCnt < lastAtomIdx; aCnt++ ) {
                    // set solvent residue type index
                    this->solventAtomParams.Add( static_cast<float>(solResIdx));
                    this->solventAtomParams.Add( static_cast<float>(mCnt));
                    // set the atom position
                    this->solventAtomPos.Add( mol->AtomPositions()[aCnt*3]);
                    this->solventAtomPos.Add( mol->AtomPositions()[aCnt*3+1]);
                    this->solventAtomPos.Add( mol->AtomPositions()[aCnt*3+2]);
                    // count new atom
                    atomCnt++;
                }
            }
        }
    }

    // compute the dimensions of the FBO
    this->molVisFboWidth = molCnt / 8192 + 1;
    this->molVisFboHeight = vislib::math::Min( molCnt, 8192U);
    //this->molVisFboWidth = mol->AtomSolventResidueCount();
    //this->molVisFboHeight = 1;
    // create fbo
    if( this->molVisFbo.IsValid() ) {
        if( this->molVisFbo.GetHeight() != this->molVisFboHeight || this->molVisFbo.GetWidth() != this->molVisFboWidth ) {
            this->molVisFbo.Create( this->molVisFboWidth, this->molVisFboHeight, GL_RGB32F, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
        }
    } else {
        this->molVisFbo.Create( this->molVisFboWidth, this->molVisFboHeight, GL_RGB32F, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }

    // create VBO
    if( !glIsBuffer( this->molVisVbo) )
        glGenBuffers( 1, &this->molVisVbo);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, molVisVbo);
    glBufferData( GL_PIXEL_PACK_BUFFER_EXT, this->molVisFboWidth*this->molVisFboHeight*3*sizeof(float), NULL, GL_DYNAMIC_DRAW_ARB );
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0);

    // --- start rendering ---
    vislib::math::Cuboid<float> bbox = mol->AccessBoundingBoxes().ObjectSpaceBBox();
    vislib::math::Vector<float, 3> invBBoxDimension( 1.0f/bbox.Width(), 1.0f/bbox.Height(), 1.0f/bbox.Depth());

    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    //glOrtho( 0.0, mol->AtomSolventResidueCount(), 0.0, 1.0, 0.0, 1.0);
    glOrtho( 0.0, molVisFboWidth, 0.0, molVisFboHeight, 0.0, 1.0);
    glDisable( GL_LIGHTING);

    // start rendering to FBO
    this->molVisFbo.Enable();
    // get and set clear color and clear
    float clearCol[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, clearCol);
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // volume texture to look up densities ...
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_3D, this->volumeTex);

    //glBlendFunc( GL_ONE, GL_ONE);
    //glEnable( GL_BLEND);

    // render all atoms as spheres and write the Id to the red color channel
    this->visMolShader.Enable();
    // set uniform parameters
    glUniform1i( this->visMolShader.ParameterLocation("volumeSampler"), 0);
    glUniform3fv( this->visMolShader.ParameterLocation("minBBox"), 1, bbox.GetOrigin().PeekCoordinates());
    glUniform3fv( this->visMolShader.ParameterLocation("invBBoxExtend"), 1, invBBoxDimension.PeekComponents() );
    glUniform1f( this->visMolShader.ParameterLocation("solventMolThreshold"), solventMolThreshold.Param<param::FloatParam>()->Value() );
    GLint inparams = glGetAttribLocation( this->visMolShader, "inParams");
    glEnableClientState( GL_VERTEX_ARRAY);
    glEnableVertexAttribArray( inparams);
    glVertexAttribPointer( inparams, 2, GL_FLOAT, GL_FALSE, 0, this->solventAtomParams.PeekElements());
    glVertexPointer( 3, GL_FLOAT, 0, this->solventAtomPos.PeekElements());
    glDrawArrays( GL_POINTS, 0, atomCnt);
    glDisableVertexAttribArray( inparams);
    glDisableClientState( GL_VERTEX_ARRAY);
    this->visMolShader.Disable();

    //glDisable( GL_BLEND);

    glBindTexture( GL_TEXTURE_3D, 0);

    // read FBO to VBO
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, this->molVisVbo);
    glReadPixels( 0, 0, this->molVisFboWidth, this->molVisFboHeight, GL_RGB, GL_FLOAT, 0);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0 );
    // stop rendering to FBO
    this->molVisFbo.Disable();

    // reset viewport to normal view
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    
    //// DEBUG ...
    //float *t = new float[this->molVisFboWidth * this->molVisFboHeight * 3];
    //glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, this->molVisVbo);
    //float *vboPtr = static_cast<float*>(glMapBuffer( GL_PIXEL_PACK_BUFFER_EXT, GL_READ_ONLY));
    //memcpy( t, vboPtr, this->molVisFboWidth*this->molVisFboHeight*3*sizeof(float));
    //glUnmapBuffer( GL_PIXEL_PACK_BUFFER_EXT);
    //glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0);
    ////this->molVisFbo.GetColourTexture( t, 0, GL_RGB, GL_FLOAT);
    //for( unsigned int i = 0; i < this->molVisFboWidth; i++ ) {
    //    for( unsigned int j = 0; j < this->molVisFboHeight; j++ ) {
    //        if( t[( i * 8192 + j) * 3 + 0] > 0 ) {
    //            std::cout << "value " << i << "," << j << " = " << t[( i * 8192 + j) * 3 + 0] << " " << t[( i * 8192 + j) * 3 + 1] << " " << t[( i * 8192 + j) * 3 + 2] << std::endl;
    //        }
    //    }
    //}
    //// ... DEBUG

    // ----- render the results of the last step (stored in a vertex buffer object) ------

    // create fbo
    if( this->solTypeCountFbo.IsValid() ) {
        if( this->solTypeCountFbo.GetWidth() != mol->AtomSolventResidueCount() ) {
            this->solTypeCountFbo.Create( mol->AtomSolventResidueCount(), 1, GL_RGB32F, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
        }
    } else {
        this->solTypeCountFbo.Create( mol->AtomSolventResidueCount(), 1, GL_RGB32F, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
    }

    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glOrtho( 0.0, this->solTypeCountFbo.GetWidth(), 0.0, 1.0, 0.0, 1.0);
    glDisable( GL_LIGHTING);
    // start rendering to FBO
    this->solTypeCountFbo.Enable();
    // set clear color and clear
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // TODO
    glBlendFunc( GL_ONE, GL_ONE);
    glEnable( GL_BLEND);
    this->solTypeCountShader.Enable();

    // TODO: fix buggy rendering
    // count molecules of the same solvent type
    glEnableClientState( GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, this->molVisVbo);
    glVertexPointer( 3, GL_FLOAT, 0, 0);
    glDrawArrays( GL_POINTS, 0, this->molVisFboWidth * this->molVisFboHeight);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    glDisableClientState( GL_VERTEX_ARRAY);
    
    //glBegin( GL_POINTS);
    //glVertex3f( 1.0, 0.0, 0.0);
    //glVertex3f( 1.0, 0.0, 0.0);
    //glVertex3f( 1.0, 1.0, 0.0);
    //glVertex3f( 1.0, 1.0, 0.0);
    //glVertex3f( 1.0, 1.0, 0.0);
    //glEnd(); // GL_POINTS

    this->solTypeCountShader.Disable();
    glDisable( GL_BLEND);

    // stop rendering to FBO
    this->solTypeCountFbo.Disable();
    // reset viewport to normal view
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    float *counter = new float[mol->AtomSolventResidueCount()];
    this->solTypeCountFbo.GetColourTexture( counter, 0, GL_RED, GL_FLOAT);
    vislib::StringA str, tmpStr;
    for( unsigned int i = 0; i < mol->AtomSolventResidueCount(); i++ ) {
        tmpStr.Format( "Solvent Molecule Type %i, count: %i; ", i, int( counter[i]));
        str += tmpStr;
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO, "Solvent Molecule Type %i, count: %i", i, int( counter[i]));
    }
/*    vislib::graphics::gl::SimpleFont f;
    if( f.Initialise() ) {
        glColor3f( 0.5f, 0.5f, 0.5f);    
        f.DrawString( 0.0, 0.0, 10.0f, true, str.PeekBuffer());
    }*/

    delete[] counter;

    // reset clear color
    glClearColor( clearCol[0], clearCol[1], clearCol[2], clearCol[3]);
}

/*
 * boese dreckig hingerotzt ...
 */
void protein::SolventVolumeRenderer::RenderHydrogenBounds(MolecularDataCall *mol, const float *atomPos) {
    const int *hydrogenConnections = this->hBondInterPtr;

    if (!hydrogenConnections)
        return;

    /* render hydrogen bounds using sticks ... */

    // stick radius of hbounds is significant smaller than regular atom bindings ...
    float stickRadius = this->stickRadiusParam.Param<param::FloatParam>()->Value() * 0.4f; // * 0.2f;

    vislib::math::Quaternion<float> quatC( 0, 0, 0, 1);
    vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
    float angle;
    // loop over all connections and compute cylinder parameters
    unsigned int curCount = 0;
    // parallelisierung geht hier nicht?! weil es von den dynamischen daten abhaengt an welchem index ein zylinder sitzt?!
    // (variable 'cnt' haengt von 'hydrogenConnections' ab ...)
//#pragma omp parallel for private( idx0, idx1, connection, firstAtomPos, secondAtomPos, quatC, tmpVec, ortho, dir, position, angle)
    for (unsigned int atomIdx = 0; atomIdx < mol->AtomCount(); atomIdx++) {
        int connection = hydrogenConnections[atomIdx];
        if (connection == -1)
            continue;

        if (this->vertCylinders.Count() <= curCount*4) {
            this->vertCylinders.SetCount( (curCount + 100)*4 );
            this->quatCylinders.SetCount( (curCount + 100)*4 );
            this->inParaCylinders.SetCount( (curCount + 100)*2 );
            this->color1Cylinders.SetCount( (curCount + 100)*3 );
            this->color2Cylinders.SetCount( (curCount + 100)*3 );
        }

        unsigned int idx0 = atomIdx;
        unsigned int idx1 = (unsigned)connection;

        vislib::math::Vector<float, 3> firstAtomPos(atomPos+3*idx0), secondAtomPos(atomPos+3*idx1);

        // compute the quaternion for the rotation of the cylinder
        dir = secondAtomPos - firstAtomPos;
        tmpVec.Set( 1.0f, 0.0f, 0.0f);
        angle = - tmpVec.Angle( dir);
        ortho = tmpVec.Cross( dir);
        ortho.Normalise();
        quatC.Set( angle, ortho);
        // compute the absolute position 'position' of the cylinder (center point)
        position = firstAtomPos + (dir/2.0f);

        this->inParaCylinders[2*curCount] = stickRadius;
        this->inParaCylinders[2*curCount+1] = ( firstAtomPos-secondAtomPos).Length();

        // thomasbm: hotfix for jumping molecules near bounding box
        if(this->inParaCylinders[2*curCount+1] > mol->AtomHydrogenBondDistance() * 1.5f
                /*mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() + mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius()*/ ) {
            this->inParaCylinders[2*curCount+1] = 0;
        }

        this->quatCylinders[4*curCount+0] = quatC.GetX();
        this->quatCylinders[4*curCount+1] = quatC.GetY();
        this->quatCylinders[4*curCount+2] = quatC.GetZ();
        this->quatCylinders[4*curCount+3] = quatC.GetW();

        // red at the oxygen/acceptor-part end of the hydrogen bound
        this->color1Cylinders[3*curCount+0] = 1; // this->atomColorTable[3*idx0+0];
        this->color1Cylinders[3*curCount+1] = 0; // this->atomColorTable[3*idx0+1];
        this->color1Cylinders[3*curCount+2] = 0; // this->atomColorTable[3*idx0+2];

        this->color2Cylinders[3*curCount+0] = 1; // this->atomColorTable[3*idx1+0];
        this->color2Cylinders[3*curCount+1] = 1; // this->atomColorTable[3*idx1+1];
        this->color2Cylinders[3*curCount+2] = 0; // this->atomColorTable[3*idx1+2];

        this->vertCylinders[4*curCount+0] = position.X();
        this->vertCylinders[4*curCount+1] = position.Y();
        this->vertCylinders[4*curCount+2] = position.Z();
        this->vertCylinders[4*curCount+3] = 0.0f;

        curCount++;
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

    vislib::math::Cuboid<float> bbox = mol->AccessBoundingBoxes().ObjectSpaceBBox();
    vislib::math::Vector<float, 3> invBBoxDimension(1/bbox.Width(), 1/bbox.Height(), 1/bbox.Depth());

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

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
    glDrawArrays( GL_POINTS, 0, curCount);
    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArrayARB( this->attribLocInParams);
    glDisableVertexAttribArrayARB( this->attribLocQuatC);
    glDisableVertexAttribArrayARB( this->attribLocColor1);
    glDisableVertexAttribArrayARB( this->attribLocColor2);
    glDisableClientState(GL_VERTEX_ARRAY);
    // disable cylinder shader
    this->cylinderSolventShader.Disable();

    glBindTexture( GL_TEXTURE_3D, 0 ); // state aufraeumen
}

/*
 * Render the molecular data in stick mode. Special case when using solvent rendering: only render solvent molecules near the isosurface between the solvent and the molecule.
 */
void protein::SolventVolumeRenderer::RenderMolecules(/*const*/ MolecularDataCall *mol, const float *atomPos) {
    // ----- prepare stick raycasting -----
    if (this->vertSpheres.Count() <= mol->AtomCount() * 4) {
        this->vertSpheres.SetCount( mol->AtomCount() * 4 );
        this->vertCylinders.SetCount( mol->ConnectionCount() * 4);
        this->quatCylinders.SetCount( mol->ConnectionCount() * 4);
        this->inParaCylinders.SetCount( mol->ConnectionCount() * 2);
        this->color1Cylinders.SetCount( mol->ConnectionCount() * 3);
        this->color2Cylinders.SetCount( mol->ConnectionCount() * 3);
    }

    float stickRadius = this->stickRadiusParam.Param<param::FloatParam>()->Value();
    float atomRadiusFactor = this->atomRadiusFactorParam.Param<param::FloatParam>()->Value();
    bool spaceFilling = this->atomSpaceFillingParam.Param<param::BoolParam>()->Value();

    // copy atom pos and radius to vertex array
    if (this->coloringModeVolSurfParam.Param<param::EnumParam>()->Value() == VOlCM_HydrogenBondStats &&
            mol->AtomHydrogenBondStatistics()) {
        // render atom spheres size according to their hydrogen bond statistics ...
        float factor = 1.0f / (float)mol->FrameCount();
        const unsigned int *hbStatistics = mol->AtomHydrogenBondStatistics();
        unsigned int numSolventResidues = mol->AtomSolventResidueCount();

        #pragma omp parallel for
        for (int atomIdx = 0; atomIdx < (int)mol->AtomCount(); atomIdx++) {
            this->vertSpheres[4*atomIdx+0] = atomPos[3*atomIdx+0];
            this->vertSpheres[4*atomIdx+1] = atomPos[3*atomIdx+1];
            this->vertSpheres[4*atomIdx+2] = atomPos[3*atomIdx+2];
            int atomStatistics = 0;
            for(unsigned int j = 0; j < numSolventResidues; j++) {
                atomStatistics += hbStatistics[numSolventResidues*atomIdx+j];
            }
            float radius = (spaceFilling ? mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius() : stickRadius)*atomRadiusFactor;
            this->vertSpheres[4*atomIdx+3] = radius + radius*(
                    (float)(atomStatistics /*hbStatistics[cnt]*/)*factor);
        }
    } else {
        // render spheres in normal mode (sphere-radius is the same as stick-radius)
        #pragma omp parallel for
        for (int atomIdx = 0; atomIdx < (int)mol->AtomCount(); atomIdx++) {
            this->vertSpheres[4*atomIdx+0] = atomPos[3*atomIdx+0];
            this->vertSpheres[4*atomIdx+1] = atomPos[3*atomIdx+1];
            this->vertSpheres[4*atomIdx+2] = atomPos[3*atomIdx+2];
            this->vertSpheres[4*atomIdx+3] = (spaceFilling ? mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius() : stickRadius)*atomRadiusFactor;
        }
    }

    /*if (!spaceFilling)*/ {
        //vislib::math::Vector<float, 3> firstAtomPos, secondAtomPos;
        vislib::math::Quaternion<float> quatC( 0, 0, 0, 1);
        vislib::math::Vector<float,3> tmpVec, ortho, dir, position;
        float angle;
        // loop over all connections and compute cylinder parameters
    #pragma omp parallel for private(quatC, tmpVec, ortho, dir, position, angle)
        for (int atomIdx = 0; atomIdx < (int)mol->ConnectionCount(); atomIdx++) {
            unsigned int idx0 = mol->Connection()[2*atomIdx];
            unsigned int idx1 = mol->Connection()[2*atomIdx+1];

            vislib::math::Vector<float, 3> firstAtomPos(atomPos+3*idx0), secondAtomPos(atomPos+3*idx1);

            // compute the quaternion for the rotation of the cylinder
            dir = secondAtomPos - firstAtomPos;
            tmpVec.Set( 1.0f, 0.0f, 0.0f);
            angle = - tmpVec.Angle( dir);
            ortho = tmpVec.Cross( dir);
            ortho.Normalise();
            quatC.Set( angle, ortho);
            // compute the absolute position 'position' of the cylinder (center point)
            position = firstAtomPos + (dir/2.0f);

            this->inParaCylinders[2*atomIdx] = stickRadius;
            this->inParaCylinders[2*atomIdx+1] = ( firstAtomPos-secondAtomPos).Length();

            // thomasbm: hotfix for jumping molecules near bounding box
            if(this->inParaCylinders[2*atomIdx+1] >
                    mol->AtomTypes()[mol->AtomTypeIndices()[idx0]].Radius() + mol->AtomTypes()[mol->AtomTypeIndices()[idx1]].Radius() ) {
                this->inParaCylinders[2*atomIdx+1] = 0;
            }

            this->quatCylinders[4*atomIdx+0] = quatC.GetX();
            this->quatCylinders[4*atomIdx+1] = quatC.GetY();
            this->quatCylinders[4*atomIdx+2] = quatC.GetZ();
            this->quatCylinders[4*atomIdx+3] = quatC.GetW();

            this->color1Cylinders[3*atomIdx+0] = this->atomColorTable[3*idx0+0];
            this->color1Cylinders[3*atomIdx+1] = this->atomColorTable[3*idx0+1];
            this->color1Cylinders[3*atomIdx+2] = this->atomColorTable[3*idx0+2];

            this->color2Cylinders[3*atomIdx+0] = this->atomColorTable[3*idx1+0];
            this->color2Cylinders[3*atomIdx+1] = this->atomColorTable[3*idx1+1];
            this->color2Cylinders[3*atomIdx+2] = this->atomColorTable[3*idx1+2];

            this->vertCylinders[4*atomIdx+0] = position.X();
            this->vertCylinders[4*atomIdx+1] = position.Y();
            this->vertCylinders[4*atomIdx+2] = position.Z();
            this->vertCylinders[4*atomIdx+3] = 0.0f;
        }
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

    vislib::math::Cuboid<float> bbox = mol->AccessBoundingBoxes().ObjectSpaceBBox();
    vislib::math::Vector<float, 3> invBBoxDimension(1/bbox.Width(), 1/bbox.Height(), 1/bbox.Depth());

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // enable sphere shader
    this->sphereSolventShader.Enable();
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


    // enable cylinder shader
    /*if (!spaceFilling)*/ {
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
    }

    glBindTexture( GL_TEXTURE_3D, 0 ); // state aufraeumen
}


/*
 * Volume rendering using molecular data.
 */
bool protein::SolventVolumeRenderer::RenderMolecularData( view::CallRender3D *call, MolecularDataCall *mol) {

    // decide to use already loaded frame request from CallFrame or 'normal' rendering
    if( !(*mol)() )
        return false;

    // check last atom count with current atom count
    if( this->atomCount != mol->AtomCount() ) {
        this->atomCount = mol->AtomCount();
        this->forceUpdateVolumeTexture = true;
        this->forceUpdateColoringMode = true;
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
    
    vislib::StringA paramSlotName( call->PeekCallerSlot()->Parent()->FullName());
    paramSlotName += "::anim::play";
    param::ParamSlot *playSlot = dynamic_cast<param::ParamSlot*>( this->FindNamedObject( paramSlotName, true).get());

    if( playSlot->Param<param::BoolParam>()->Value() || this->forceUpdateVolumeTexture || fabs(call->Time() - this->lastUpdateVolumeTextureTime) > 0.01) {
        this->lastUpdateVolumeTextureTime = call->Time();
        this->UpdateVolumeTexture( mol);
        CHECK_FOR_OGL_ERROR();
        this->forceUpdateVolumeTexture = false;
    }

    //glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable( GL_BLEND);
    this->proteinFBO.DrawColourTexture();
    //glDisable( GL_BLEND);
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
void protein::SolventVolumeRenderer::ParameterRefresh( view::CallRender3D *call, MolecularDataCall *mol) {
    
    // parameter refresh
    if( this->coloringModeSolventParam.IsDirty() || this->coloringModePolymerParam.IsDirty() || this->coloringModeVolSurfParam.IsDirty() ) {
        this->coloringModeSolventParam.ResetDirty();
        this->coloringModePolymerParam.ResetDirty();
        this->coloringModeVolSurfParam.ResetDirty();
        /* hydrogen statistics map surface by residue color -> only residue coloring makes sense for solvent here */
        if (this->coloringModeVolSurfParam.Param<param::EnumParam>()->Value() == VOlCM_HydrogenBondStats)
            this->coloringModeSolventParam.Param<param::EnumParam>()->SetValue(Color::RESIDUE);
        this->forceUpdateVolumeTexture = true;
        this->forceUpdateColoringMode = true;
    }
    if (this->colorIntensityScaleParam.IsDirty() || this->colorFilterRadiusParam.IsDirty()) {
        this->colorIntensityScaleParam.ResetDirty();
        this->colorFilterRadiusParam.ResetDirty();
        this->forceUpdateVolumeTexture = true;
    }
    if (this->accumulateColors.IsDirty()) {
        this->accumulateColors.ResetDirty();
        this->forceUpdateVolumeTexture = true;
    }
    if (this->accumulateVolume.IsDirty()) {
        this->accumulateVolume.ResetDirty();
        this->forceUpdateVolumeTexture = true;
    }
    // volume parameters
    if( this->volIsoValue1Param.IsDirty() ) {
        this->isoValue1 = this->volIsoValue1Param.Param<param::FloatParam>()->Value();
        this->volIsoValue1Param.ResetDirty();
    }
/*    if( this->volIsoValue2Param.IsDirty() ) {
        this->isoValue2 = this->volIsoValue2Param.Param<param::FloatParam>()->Value();
        this->volIsoValue2Param.ResetDirty();
    }*/
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
        this->forceUpdateColoringMode = true;
    }

/*    // update lsit of residues that compose the solvent ...
    if (this->solventResidues.IsDirty()) {
        this->solventResidues.ResetDirty();
        const vislib::TString& str = solventResidues.Param<param::StringParam>()->Value();
        vislib::Array<vislib::TString> residueTokens = vislib::StringTokeniser<vislib::TCharTraits>::Split(str, ';', true);

        if (mol && (*mol)(MolecularDataCall::CallForGetData)) {
            const vislib::StringA* residueTypes = mol->ResidueTypeNames();

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
        this->forceUpdateVolumeTexture = true;
    }*/
}


#if 0
void protein::SolventVolumeRenderer::CreateSpatialProbabilitiesTexture( MolecularDataCall *mol) {
    // generate volume, if necessary
    if( !glIsTexture( this->spatialProbVolumeTex) ) {
        // from CellVis: cellVis.cpp, initGL
        glGenTextures( 1, &this->spatialProbVolumeTex);
        glBindTexture( GL_TEXTURE_3D, this->spatialProbVolumeTex);
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
    for(int z = 0; z < this->volumeSize; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->spatialProbVolumeTex, 0, z);
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
    

    if (update_vol.Count() < mol->AtomCount()*4) update_vol.SetCount(mol->AtomCount()*4);
    float *updatVolumeTextureAtoms = &update_vol[0];

    if (update_clr.Count() < mol->AtomCount()*3) update_clr.SetCount(mol->AtomCount()*3);
    float *updatVolumeTextureColors = &update_clr[0];

    /* make local variables to avoid function alls in for-loops (compiler won't inline in debug mode...) */
    int atomCount = mol->AtomCount();
    const float *atomColorTablePtr = this->atomColorTable.PeekElements();
    const unsigned int *atomTypeIndices = mol->AtomTypeIndices();
    const MolecularDataCall::AtomType *atomTypes = mol->AtomTypes();

    memset(updatVolumeTextureAtoms, 0, sizeof(float)*atomCount*4);
    memset(updatVolumeTextureColors, 0, sizeof(float)*atomCount*3);

    vislib::math::Vector<float, 3> orig( mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().PeekCoordinates());
    orig = ( orig + this->translation) * this->scale;
    vislib::math::Vector<float, 3> nullVec( 0.0f, 0.0f, 0.0f);

    this->createSpatialProbabilityVolume.Enable();
        // set shader params
        glUniform1f( this->createSpatialProbabilityVolume.ParameterLocation( "filterRadius"), this->volFilterRadius);
        glUniform1f( this->createSpatialProbabilityVolume.ParameterLocation( "densityScale"), this->volDensityScale);
        glUniform3fv( this->createSpatialProbabilityVolume.ParameterLocation( "scaleVol"), 1, this->volScale);
        glUniform3fv( this->createSpatialProbabilityVolume.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
        glUniform3f( this->createSpatialProbabilityVolume.ParameterLocation( "invVolRes"), 
            1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize));
        glUniform3fv( this->createSpatialProbabilityVolume.ParameterLocation( "translate"), 1, orig.PeekComponents() );
        glUniform1f( this->createSpatialProbabilityVolume.ParameterLocation( "volSize"), float( this->volumeSize));
        CHECK_FOR_OGL_ERROR();
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        /* sortierung nach atomen die das loesungsmittel bilden und der rest ... atomIdxSol  laeuuft rueckwaerts .. */
        int numFrames = mol->FrameCount();
        float normalizeFactor = 1.0f / numFrames;
        for(int frameId = 0; frameId < numFrames; frameId++) {
            //int atomCntDensity = 0, atomCntColor = 0;
            const MolecularDataCall::Residue **residues = mol->Residues();

#if 1
            #pragma omp parallel for
            for(int atomCnt = 0; atomCnt < atomCount; ++atomCnt ) {
                updatVolumeTextureAtoms[atomCnt*4+0] += ( this->atomPosInter[3*atomCnt+0] + this->translation.X()) * this->scale * normalizeFactor;
                updatVolumeTextureAtoms[atomCnt*4+1] += ( this->atomPosInter[3*atomCnt+1] + this->translation.Y()) * this->scale * normalizeFactor;
                updatVolumeTextureAtoms[atomCnt*4+2] += ( this->atomPosInter[3*atomCnt+2] + this->translation.Z()) * this->scale * normalizeFactor;
                updatVolumeTextureAtoms[atomCnt*4+3] += atomTypes[atomTypeIndices[atomCnt]].Radius() * this->scale * normalizeFactor;
                updatVolumeTextureColors[atomCnt*3+0] += atomColorTablePtr[atomCnt*3+0] * normalizeFactor;
                updatVolumeTextureColors[atomCnt*3+1] += atomColorTablePtr[atomCnt*3+1] * normalizeFactor;
                updatVolumeTextureColors[atomCnt*3+2] += atomColorTablePtr[atomCnt*3+2] * normalizeFactor;
            }
#else
            for( int residueIdx = 0; residueIdx < mol->ResidueCount(); residueIdx++ ) {
                const MolecularDataCall::Residue *residue = residues[residueIdx];
                int firstAtomIndex = residue->FirstAtomIndex();
                int lastAtomIndx = residue->FirstAtomIndex() + residue->AtomCount();

                int i;
                for( i = 0 ; i < this->solventResidueTypeIds.Count(); i++ ) {
                    if (solventResidueTypeIds[i] == residue->Type())
                        break;
                }
                if (i < this->solventResidueTypeIds.Count()) {
                    /* solvent (creates volume coloring ...) */
                //    #pragma omp parallel for
                    for(int atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++) {
                        updatVolumeTextureAtoms[]...
                        updatVolumeTextureColors[]+= atomColorTablePtr...
                    }
                    //atomCntColor += ?
                } else {
                    /* not solvent (creates volume) */
                }
            }
#endif

            glVertexPointer( 4, GL_FLOAT, 0, updatVolumeTextureAtoms+?);
            glColorPointer( 3, GL_FLOAT, 0, /*updatVolumeTextureColors*/ atomColorTablePtr);
            for(int z = 0; z < this->volumeSize; ++z ) {
                // TODO: spacial grid to speedup FBO rendering here?
                // attach texture slice to FBO
                glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->spatialProbVolumeTex, 0, z);
                glUniform1f( this->createSpatialProbabilityVolume.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
                // draw all atoms as points, using w for radius
                glDrawArrays( GL_POINTS, 0, ?);
            }
        }

        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    this->createSpatialProbabilityVolume.Disable();

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
#endif

/*
 * Create a volume containing all molecule atoms
 */
void protein::SolventVolumeRenderer::UpdateVolumeTexture( MolecularDataCall *mol) {
    bool firstVolumeUpdate = false;

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
        firstVolumeUpdate = true; // clear volume on first update in any case
    }
    // generate FBO, if necessary
    if( !glIsFramebufferEXT( this->volFBO ) ) {
        glGenFramebuffersEXT( 1, &this->volFBO);
        CHECK_FOR_OGL_ERROR();
    }

    int coloringModeVolSurf = this->coloringModeVolSurfParam.Param<param::EnumParam>()->Value();

    // coloring mode
    bool coloringByHydroBonds = coloringModeVolSurf == VOLCM_HydrogenBonds && hBondInterPtr;
    bool coloringByHydroBondStats = coloringModeVolSurf== VOlCM_HydrogenBondStats && mol->AtomHydrogenBondStatistics();

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

    glColor4f( 0.0f, 0.0f, 0.0f, 1.0f);

    float bgColor[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
    glClearColor( 0.1f, 0.1f, 0.1f, 0.0f);
    //glClearColor( 0.5, 0.5, 0.5, 0.0);
    // clear 3d texture

    bool accumulateColors = this->accumulateColors.Param<param::BoolParam>()->Value();
    bool accumulateVolume = this->accumulateVolume.Param<param::BoolParam>()->Value();

    for(unsigned int z = 0; z < this->volumeSize; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->volumeTex, 0, z);
        int colorMask[4] = {1,1,1,1};
        if (!firstVolumeUpdate) {
            if (accumulateColors)
                colorMask[0] = colorMask[1] = colorMask[2] = 0;
            if (accumulateVolume)
                colorMask[3] = 0;
        }
        glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        glClear( GL_COLOR_BUFFER_BIT);
        glColorMask(1,1,1,1);
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

    if (update_vol.Count() < mol->AtomCount()*4) update_vol.SetCount(mol->AtomCount()*4);
    float *updatVolumeTextureAtoms = &update_vol[0];

    if (update_clr.Count() < mol->AtomCount()*3) update_clr.SetCount(mol->AtomCount()*3);
    float *updatVolumeTextureColors = &update_clr[0];

    /* make local variables to avoid function alls in for-loops (compiler won't inline in debug mode...) */
    int atomCount = mol->AtomCount();
    const float *atomColorTablePtr = this->atomColorTable.PeekElements();
    const unsigned int *atomTypeIndices = mol->AtomTypeIndices();
    const MolecularDataCall::AtomType *atomTypes = mol->AtomTypes();

/*    int atomCnt;
#pragma omp parallel for
    for( atomCnt = 0; atomCnt < mol->AtomCount(); ++atomCnt ) {
        updatVolumeTextureAtoms[atomCnt*4+0] = ( this->atomPosInter[3*atomCnt+0] + this->translation.X()) * this->scale;
        updatVolumeTextureAtoms[atomCnt*4+1] = ( this->atomPosInter[3*atomCnt+1] + this->translation.Y()) * this->scale;
        updatVolumeTextureAtoms[atomCnt*4+2] = ( this->atomPosInter[3*atomCnt+2] + this->translation.Z()) * this->scale;
        updatVolumeTextureAtoms[atomCnt*4+3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomCnt]].Radius() * this->scale;
    }*/

    const MolecularDataCall::Residue **residues = mol->Residues();
    /* sortierung nach atomen die das loesungsmittel bilden und der rest ... atomIdxSol  laeuuft rueckwaerts .. */
    int atomCntDensity = 0, atomCntColor = 0;

    //float colorScale = this->colorFilterRadiusParam.Param<param::FloatParam>()->Value();

    for (unsigned int residueIdx = 0; residueIdx < mol->ResidueCount(); residueIdx++) {
        const MolecularDataCall::Residue *residue = residues[residueIdx];
        int firstAtomIndex = residue->FirstAtomIndex();
        int lastAtomIndx = residue->FirstAtomIndex() + residue->AtomCount();

        bool isSolvent = false;
#if 1
        isSolvent = mol->IsSolvent(residue);
#else
        for(int i = 0 ; i < this->solventResidueTypeIds.Count(); i++ ) {
            if (this->solventResidueTypeIds[i] == residue->Type()) {
                isSolvent = true;
                break;
            }
        }
#endif

        // test to visualize hydrogen bounds from the polymer to the solvent and map color to the volume surface
        if (coloringByHydroBonds) {
            for(int atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++) {
                if (this->hBondInterPtr[atomIdx] == -1)
                    continue;
                int sortedVolumeIdx = atomCount - (atomCntColor+1);
                int connection = this->hBondInterPtr[atomIdx];
                float *atomPos = &updatVolumeTextureAtoms[sortedVolumeIdx*4];
                float *solventAtomColor = &updatVolumeTextureColors[sortedVolumeIdx*3];
                float *interPos = &this->atomPosInterPtr[atomIdx*3];
                float *interPos2 = &this->atomPosInterPtr[connection*3];
                // hydrogen bonds in the middle between the two connected atoms ...
                atomPos[0] = ((interPos[0]+interPos2[0])*0.5f + this->translation.X()) * this->scale;
                atomPos[1] = ((interPos[1]+interPos2[1])*0.5f + this->translation.Y()) * this->scale;
                atomPos[2] = ((interPos[2]+interPos2[2])*0.5f + this->translation.Z()) * this->scale;
                //atomPos[3] = atomTypes[atomTypeIndices[atomIdx]].Radius() * this->scale;
                atomPos[3] = mol->AtomHydrogenBondDistance() * this->scale; // * colorScale;
                //atomPos[3] = (atomTypes[atomTypeIndices[atomIdx]].Radius()
                //         +atomTypes[atomTypeIndices[connection]].Radius()) * this->scale * 0.5;
                solventAtomColor[0] = 1;
                solventAtomColor[1] = 0;
                solventAtomColor[2] = 0;
                atomCntColor++;
            }
        }
#if 1
        else if (coloringByHydroBondStats && !isSolvent) {
            // visalize hydrogen bond statistics ...
            unsigned int numSolventResidues = mol->AtomSolventResidueCount();
            float factor = 1.0f / (float)mol->FrameCount();
            const unsigned int *hbStatistics = mol->AtomHydrogenBondStatistics();
            const unsigned int *solventResidueIndices = mol->SolventResidueIndices();

            // colorLookupTable[i]; error: we don't have residue index of HB statistics ...
            for(int atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++) {
                int sortedVolumeIdx = atomCount - (atomCntColor+1);
                float *atomPos = &updatVolumeTextureAtoms[sortedVolumeIdx*4];
                float *solventAtomColor = &updatVolumeTextureColors[sortedVolumeIdx*3];
                float *interPos = &this->atomPosInterPtr[atomIdx*3];
                // hydrogen bonds in the middle between the two connected atoms ...
                atomPos[0] = (interPos[0] + this->translation.X()) * this->scale;
                atomPos[1] = (interPos[1] + this->translation.Y()) * this->scale;
                atomPos[2] = (interPos[2] + this->translation.Z()) * this->scale;
                atomPos[3] = atomTypes[atomTypeIndices[atomIdx]].Radius() * this->scale; // * colorScale;
                //atomPos[3] = mol->AtomHydrogenBondDistance() * this->scale;

            // for quick testing mix the three color channels ...
                solventAtomColor[0] = solventAtomColor[1] = solventAtomColor[2] = 0;
                for(unsigned int j = 0; j < numSolventResidues; j++) {
                    float value = (float)hbStatistics[numSolventResidues*atomIdx+j] * factor;
                    //solventAtomColor[j%3] += value;
                    int residueType = solventResidueIndices[j];
                    const vislib::math::Vector<float, 3>& residueColor = this->colorLookupTable[residueType%this->colorLookupTable.Count()];
                    solventAtomColor[0] += residueColor.X() * value;
                    solventAtomColor[1] += residueColor.Y() * value;
                    solventAtomColor[2] += residueColor.Z() * value;
                }

                atomCntColor++;
            }
        }
#endif

        /* sort atoms into different arrays depending whether they form the solvent or the polymer ... */
        if (isSolvent) {
            if (coloringModeVolSurf == VOLCM_SolventConcentration) {
                /* solvent (creates volume coloring ...) */
                int atomIdx;
                //#pragma omp parallel for // das verlangsamt alles enorm unter windows! (komisch ...)
                for(atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++) {
                    // fill the solvent atoms in reverse into the array ...
                    int sortedVolumeIdx = atomCount - (atomCntColor+(atomIdx-firstAtomIndex)+1);
                    float *atomPos = &updatVolumeTextureAtoms[sortedVolumeIdx*4];
                    float *solventAtomColor = &updatVolumeTextureColors[sortedVolumeIdx*3];
                    float *interPos = &this->atomPosInterPtr[atomIdx*3];
                    const float *clr = &atomColorTablePtr[atomIdx*3];
                    atomPos[0] = (interPos[0] + this->translation.X()) * this->scale;
                    atomPos[1] = (interPos[1] + this->translation.Y()) * this->scale;
                    atomPos[2] = (interPos[2] + this->translation.Z()) * this->scale;
                    atomPos[3] = atomTypes[atomTypeIndices[atomIdx]].Radius() * this->scale; // * colorScale;
                    solventAtomColor[0] = clr[0];
                    solventAtomColor[1] = clr[1];
                    solventAtomColor[2] = clr[2];
                }
                atomCntColor += (lastAtomIndx - firstAtomIndex);
            }
        } else {
            /* not solvent (creates volume) */
            int atomIdx;
        //    #pragma omp parallel for // geht hier nicht, bringt wohl auch nix
            for(atomIdx = firstAtomIndex; atomIdx < lastAtomIndx; atomIdx++ ) {
                int sortedVolumeIdx = atomCntDensity + (atomIdx-firstAtomIndex);
                float *atomPos = &updatVolumeTextureAtoms[sortedVolumeIdx*4];
                float *interPos = &this->atomPosInterPtr[atomIdx*3];
                atomPos[0] = (interPos[0] + this->translation.X()) * this->scale;
                atomPos[1] = (interPos[1] + this->translation.Y()) * this->scale;
                atomPos[2] = (interPos[2] + this->translation.Z()) * this->scale;
                atomPos[3] = atomTypes[atomTypeIndices[atomIdx]].Radius() * this->scale;          
#if 1 //HACKHACKHACK
                float *atomCol = &updatVolumeTextureColors[atomIdx*3];
                atomCol[0] = this->atomColorTable[3*atomIdx];
                atomCol[1] = this->atomColorTable[3*atomIdx+1];
                atomCol[2] = this->atomColorTable[3*atomIdx+2];
#endif //HACKHACKHACK
            }
            atomCntDensity += (lastAtomIndx - firstAtomIndex);
        }
    }

    vislib::math::Vector<float, 3> orig( mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().PeekCoordinates());
    orig = ( orig + this->translation) * this->scale;
    vislib::math::Vector<float, 3> nullVec( 0.0f, 0.0f, 0.0f);

//    vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO, "rendering %d polymer atoms", atomCntDensity );

    float accumulateFactor = accumulateVolume ? this->accumulateFactor.Param<param::FloatParam>()->Value() : 1.0f;
    this->updateVolumeShaderMoleculeVolume.Enable();
        // set shader params
        glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "filterRadius"), this->volFilterRadius);
        glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "densityScale"), this->volDensityScale * accumulateFactor);
        glUniform3fv( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "scaleVol"), 1, this->volScale);
        glUniform3fv( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
        glUniform3f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "invVolRes"), 
            1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize));
        glUniform3fv( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "translate"), 1, orig.PeekComponents() );
        glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "volSize"), float( this->volumeSize));
        CHECK_FOR_OGL_ERROR();

        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer( 4, GL_FLOAT, 0, updatVolumeTextureAtoms);
#if 1 //HACKHACKHACK
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer( 3, GL_FLOAT, 0, updatVolumeTextureColors);
#endif // HACKHACKHACK
        for(unsigned int z = 0; z < this->volumeSize; ++z ) {
            // TODO: spacial grid to speedup FBO rendering here?
            // attach texture slice to FBO
            glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
            glUniform1f( this->updateVolumeShaderMoleculeVolume.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
            // draw all atoms as points, using w for radius
            glDrawArrays( GL_POINTS, 0, atomCntDensity);
        }
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    this->updateVolumeShaderMoleculeVolume.Disable();

#ifdef USE_VERTEX_SKIP_SHADER
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
#endif

    accumulateFactor = accumulateColors ? this->accumulateFactor.Param<param::FloatParam>()->Value() : 1.0f;
//    vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_INFO, "rendering %d solvent atoms", atomCntColor );
    vislib::graphics::gl::GLSLShader& volumeColorShader =  coloringByHydroBonds ? this->updateVolumeShaderHBondColor : this->updateVolumeShaderSolventColor;
    volumeColorShader.Enable();
        // set shader params
        glUniform1f( volumeColorShader.ParameterLocation( "filterRadius"), this->colorFilterRadiusParam.Param<param::FloatParam>()->Value()/*this->volFilterRadius*/);
        glUniform1f( volumeColorShader.ParameterLocation( "densityScale"), this->colorIntensityScaleParam.Param<param::FloatParam>()->Value() * accumulateFactor/*this->volDensityScale*/);
        glUniform3fv( volumeColorShader.ParameterLocation( "scaleVol"), 1, this->volScale);
        glUniform3fv( volumeColorShader.ParameterLocation( "scaleVolInv"), 1, this->volScaleInv);
        glUniform3f( volumeColorShader.ParameterLocation( "invVolRes"),
            1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize), 1.0f/ float(this->volumeSize));
        glUniform3fv( volumeColorShader.ParameterLocation( "translate"), 1, orig.PeekComponents() );
        glUniform1f( volumeColorShader.ParameterLocation( "volSize"), float( this->volumeSize));
#ifdef USE_VERTEX_SKIP_SHADER
        glUniform1i( volumeColorShader.ParameterLocation( "volumeSampler"), 0);
#endif
        CHECK_FOR_OGL_ERROR();

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glVertexPointer( 4, GL_FLOAT, 0, updatVolumeTextureAtoms+(mol->AtomCount()-atomCntColor)*4 );
        glColorPointer( 3, GL_FLOAT, 0, updatVolumeTextureColors+(mol->AtomCount()-atomCntColor)*3 );
        for(unsigned int z = 0; z < this->volumeSize; ++z ) {
            // TODO: spacial grid to speedup FBO rendering here?
            // attach texture slice to FBO
            glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
            glUniform1f( volumeColorShader.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
            // draw all atoms as points, using w for radius
            glDrawArrays( GL_POINTS, 0, atomCntColor);
            //glDrawArrays( GL_POINTS, 0, atomCntDensity);
        }
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    volumeColorShader.Disable();

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
    for(unsigned int i = 0; i < this->volClipPlane.Count(); ++i ) {
        slices.setupSingleSlice( this->volClipPlane[i].PeekComponents(), position.PeekComponents());
        float d = 0.0f;
        glBegin(GL_TRIANGLE_FAN);
            slices.drawSingleSlice(-(-d + (float)this->volClipPlane[i].PeekComponents()[3] - 0.0001f));
        glEnd();
    }
}
