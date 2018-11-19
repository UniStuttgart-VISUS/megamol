/*
 * ProteinVolumeRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1
#define STOP_SEGMENTATION

#include "ProteinVolumeRenderer.h"
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
#include "mmcore/param/ButtonParam.h"
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
#include <algorithm>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max

/*
 * ProteinVolumeRenderer::ProteinVolumeRenderer (CTOR)
 */
ProteinVolumeRenderer::ProteinVolumeRenderer ( void ) : Renderer3DModule (),
        protDataCallerSlot ( "getData", "Connects the volume rendering with data storage" ),
        protRendererCallerSlot ( "renderProtein", "Connects the volume rendering with a protein renderer" ),
        dataOutSlot ( "volumeout", "Connects the volume rendering with a volume slice renderer" ),
        diagramDataOutSlot ( "segmentationout", "Connects the volume rendering with a 2D diagram renderer" ),
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
        initialSegmentationSizeParam( "initSegSize", "The maximum initial segmentation size"),
        stopSegmentationParam( "stopSegmentation", "Stop segmentation and replay when segmentation threshold is reached"), 
        segmentationDeltaParam( "segmentationDelta", "The maximum allowed difference of the segmented voxels between"),
        writeRawFileParam( "writeRawFile", "Write the volume to a RAW file."),
        currentFrameId ( 0 ), atomCount( 0 ), volumeTex( 0), volumeSize( 128), volFBO( 0),
        volFilterRadius( 1.75f), volDensityScale( 1.0f),
        width( 0), height( 0), volRayTexWidth( 0), volRayTexHeight( 0),
        volRayStartTex( 0), volRayLengthTex( 0), volRayDistTex( 0),
        renderIsometric( true), meanDensityValue( 0.0f), isoValue1( 0.5f), isoValue2(-0.5f),
        volIsoOpacity( 0.4f), volClipPlaneFlag( false), volClipPlaneOpacity( 0.4f),
        mousePos( 0, 0, 0), clickedPos( 0, 0, 0), startVolSeg( false), segmentationTime( 0.0f),
        oldVoxelCount( 0), stopSegmentation( true), fixedNumberOfVoxels( false),
        drawMarker( true), forceUpdateVolumeTexture( true)
{
    // set caller slot for different data calls
    this->protDataCallerSlot.SetCompatibleCall<CallProteinVolumeDataDescription>();
    this->protDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable ( &this->protDataCallerSlot );

    // set renderer caller slot
    this->protRendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable( &this->protRendererCallerSlot);

    // set data out callee slot
    VolumeSliceCallDescription vscd;
    this->dataOutSlot.SetCallback( vscd.ClassName(), 
        VolumeSliceCall::FunctionName( VolumeSliceCall::CallForGetData), 
        &ProteinVolumeRenderer::getVolumeData);
    this->MakeSlotAvailable( &this->dataOutSlot);

    // set data out callee slot
    Diagram2DCallDescription d2dcd;
    this->diagramDataOutSlot.SetCallback( d2dcd.ClassName(), 
        Diagram2DCall::FunctionName( Diagram2DCall::CallForGetData), 
        &ProteinVolumeRenderer::getSegmentationData);
    this->MakeSlotAvailable( &this->diagramDataOutSlot);

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
        static_cast<float>(this->volClipPlane[0].PeekComponents()[0]), 
        static_cast<float>(this->volClipPlane[0].PeekComponents()[1]), 
        static_cast<float>(this->volClipPlane[0].PeekComponents()[2]));
    this->volClipPlane0NormParam.SetParameter( new param::Vector3fParam( cp0n) );
    // --- set up parameter for volume clipping plane distance ---
    float d = static_cast<float>(this->volClipPlane[0].PeekComponents()[3]);
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

    // the stop segmentation parameter
    this->initialSegmentationSizeParam.SetParameter(new param::IntParam( 2000, 0, int(this->volumeSize*this->volumeSize*this->volumeSize)));
    this->MakeSlotAvailable( &this->initialSegmentationSizeParam);

    // the stop segmentation parameter
    this->stopSegmentationParam.SetParameter(new param::BoolParam( true));
    this->MakeSlotAvailable( &this->stopSegmentationParam);
    
    // the segmentation delta parameter
    this->segmentationDeltaParam.SetParameter(new param::FloatParam( 2.0f, 0.0f, 10.0f));
    this->MakeSlotAvailable( &this->segmentationDeltaParam);

    // write volume to raw file
    this->writeRawFileParam.SetParameter(new param::ButtonParam());
    this->writeRawFileParam.SetUpdateCallback(&ProteinVolumeRenderer::writeVolumeRAWFloat);
    this->MakeSlotAvailable( &this->writeRawFileParam);

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

    // empty segmented voxel list
    this->segmentedVoxels.clear();

}


/*
 * ProteinVolumeRenderer::~ProteinVolumeRenderer (DTOR)
 */
ProteinVolumeRenderer::~ProteinVolumeRenderer ( void ) {
    this->Release ();
}


/*
 * ProteinVolumeRenderer::release
 */
void ProteinVolumeRenderer::release ( void ) {

}


/*
 * ProteinVolumeRenderer::create
 */
bool ProteinVolumeRenderer::create ( void ) {
    if( !ogl_IsVersionGEQ(2,0) || !areExtsAvailable( "GL_EXT_framebuffer_object GL_ARB_texture_float GL_EXT_gpu_shader4 GL_EXT_bindable_uniform") )
        return false;
    if( !isExtAvailable( "GL_ARB_vertex_program" ) )
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
 * ProteinRenderer::GetExtents
 */
bool ProteinVolumeRenderer::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    CallProteinVolumeData *volume = this->protDataCallerSlot.CallAs<CallProteinVolumeData>();
    MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();

    float scale, xoff, yoff, zoff;
    vislib::math::Cuboid<float> boundingBox;
    vislib::math::Point<float, 3> bbc;

    // try to get the bounding box from the active data call
    if( volume ) {
        // try to call the volume data
        if( !(*volume)() ) return false;
        // get bounding box
        boundingBox = volume->BoundingBox();
    } else if( mol ) {
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
    view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
    vislib::math::Point<float, 3> protrenbbc;
    if( protrencr3d ) {
        (*protrencr3d)(core::view::AbstractCallRender::FnGetExtents);
        BoundingBoxes &protrenbb = protrencr3d->AccessBoundingBoxes();
        this->protrenScale =  protrenbb.ObjectSpaceBBox().Width() / boundingBox.Width();
        //this->protrenTranslate = ( protrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
        if( mol ) {
            this->protrenTranslate.Set( xoff, yoff, zoff);
            this->protrenTranslate *= scale;
        } else {
        this->protrenTranslate = ( protrenbb.ObjectSpaceBBox().CalcCenter() - bbc) * scale;
    }
    }
    return true;
}


/*
 * ProteinVolumeRenderer::getVolumeData
 */
bool ProteinVolumeRenderer::getVolumeData( core::Call& call) {
    VolumeSliceCall *c = dynamic_cast<VolumeSliceCall*>( &call);
    if( c == NULL ) return false;

    // get the data call
    CallProteinVolumeData *volume = this->protDataCallerSlot.CallAs<CallProteinVolumeData>();
    MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();
    // set the bounding box dimensions
    vislib::math::Cuboid<float> box( 0, 0, 0, 1, 1, 1);
    vislib::math::Vector<float, 3> dim;
    if( volume )
        box = volume->BoundingBox();
    else if( mol )
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

    // get the mouse pos
    this->mousePos = c->getMousePos();
    // get the clicked mouse pos
    if( this->clickedPos != c->getClickPos() ) {
        this->clickedPos = c->getClickPos();
        this->startVolSeg = true;
    }

    return true;
}

/*
 * ProteinVolumeRenderer::getSegmentationData
 */
bool ProteinVolumeRenderer::getSegmentationData( core::Call& call) {
    Diagram2DCall *c = dynamic_cast<Diagram2DCall*>( &call);
    if( c == NULL ) return false;

    // get the data call
    MolecularDataCall *mol = this->protDataCallerSlot.CallAs<MolecularDataCall>();

    // set the range
    vislib::math::Vector<float, 2> range;
    if( mol )
#ifdef PLOT_PERCENTAGE
        range.Set( mol->FrameCount(), 1.0f);
#else
        range.Set( static_cast<float>(mol->FrameCount()), 
            static_cast<float>(this->volumeSize * this->volumeSize));
#endif
    else
        return false;
    c->SetRange( range.X(), range.Y());

    // set the data point
#ifdef PLOT_PERCENTAGE
    float z = fabsf( float( this->oldVoxelCount) - float( this->segmentedVoxels.size()));
    float n = float( std::max( 1U, this->oldVoxelCount));
    c->SetValue( this->segmentationTime, z / n > 1.0f ? 1.0f : z / n );
#else
    c->SetValue( this->segmentationTime, float( this->segmentedVoxels.size()));
#endif

    // set the marker flag
    c->SetMarkerFlag( this->drawMarker);
    this->drawMarker = false;

    // set the last call time
    if( mol )
        c->SetCallTime( this->callTime);
    else
        return false;

    return true;
}


/*
 * ProteinVolumeRenderer::Render
 */
bool ProteinVolumeRenderer::Render( Call& call ) {
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>( &call );
    if( !cr3d ) return false;
    // get the pointer to CallRender3D (protein renderer)
    view::CallRender3D *protrencr3d = this->protRendererCallerSlot.CallAs<view::CallRender3D>();
    // get pointer to CallProteinVolumeData
    CallProteinVolumeData *volume = this->protDataCallerSlot.CallAs<CallProteinVolumeData>();
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

    // render the current mouse position
    this->RenderMousePosition( cr3d, 0.3f);

    // draw segmented voxels
    this->RenderSegmentedVoxels( cr3d);

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

    // =============== Refresh all parameters ===============
    this->ParameterRefresh( cr3d);
    this->posInter = 0;
    
    // get the call time
    this->callTime = cr3d->Time();

    // DEBUG
#ifdef STOP_SEGMENTATION
    if( !this->stopSegmentation ) {
        float z = fabsf( float( this->oldVoxelCount) - float( this->segmentedVoxels.size()));
        float n = float( std::max( 1U, this->oldVoxelCount));
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO, 
            "Total # of voxels: %5.i, old # of voxels: %5.i, difference: %6.2f", 
            this->segmentedVoxels.size(), this->oldVoxelCount, 
            ( z / n) * 100.0f );
        if( ( z / n) > ( this->segmentationDeltaParam.Param<param::FloatParam>()->Value() - 1.0f) ) {

            vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_WARN, "Difference greater than 100 per cent!");
            // get and set play param
            vislib::StringA paramSlotName( cr3d->PeekCallerSlot()->Parent()->FullName());
            paramSlotName += "::anim::play";
            param::ParamSlot *paramSlot = dynamic_cast<param::ParamSlot*>(this->FindNamedObject(paramSlotName, true).get());
            if( paramSlot && this->stopSegmentationParam.Param<param::BoolParam>()->Value() ) {
                paramSlot->Param<param::BoolParam>()->SetValue( false);
            }
            // stop the segmentation
            this->stopSegmentation = true;
            // set fixed number of voxels flag
            if( int( this->oldVoxelCount) < int( this->segmentedVoxels.size()) )
                this->fixedNumberOfVoxels = true;
            // set the flag to draw a marker
            this->drawMarker = true;
        }
        // store number of voxels
        this->oldVoxelCount = static_cast<unsigned int>(this->segmentedVoxels.size());
    }
#endif

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
        float *pos0 = new float[mol->AtomCount() * 3];
        memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
        // check if the atom positions have to be interpolated
        bool interpolate = this->interpolParam.Param<param::BoolParam>()->Value();
        float *pos1 = 0;
        if( interpolate ) {
            // set next frame ID and get positions of the second frame
            if( ( static_cast<int>( callTime) + 1) < int(mol->FrameCount()) ) 
                mol->SetFrameID(static_cast<int>( callTime) + 1);
            else
                mol->SetFrameID(static_cast<int>( callTime));
            if (!(*mol)(MolecularDataCall::CallForGetData)) {
                delete[] pos0;
                return false;
            }
            pos1 = new float[mol->AtomCount() * 3];
            memcpy( pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
            // interpolate atom positions between frames
            this->posInter = new float[mol->AtomCount() * 3];
            float inter = callTime - static_cast<float>(static_cast<int>( callTime));
            float threshold = vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
                vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
                mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
            for( cnt = 0; cnt < int(mol->AtomCount()); ++cnt ) {
                if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                        std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                        std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
                    this->posInter[3*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
                    this->posInter[3*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
                    this->posInter[3*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
                } else if( inter < 0.5f ) {
                    this->posInter[3*cnt+0] = pos0[3*cnt+0];
                    this->posInter[3*cnt+1] = pos0[3*cnt+1];
                    this->posInter[3*cnt+2] = pos0[3*cnt+2];
                } else {
                    this->posInter[3*cnt+0] = pos1[3*cnt+0];
                    this->posInter[3*cnt+1] = pos1[3*cnt+1];
                    this->posInter[3*cnt+2] = pos1[3*cnt+2];
                }
            }

            // delete temporary arrays
            if( pos0 )
                delete[] pos0;
            if( pos1 )
                delete[] pos1;
        } else {
            this->posInter = pos0;
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
    // try to start volume rendering using volume data
    if( volume ) {
        retval = this->RenderVolumeData( cr3d, volume);
    }
    // try to start volume rendering using protein data
    if( mol ) {
        retval = this->RenderMolecularData( cr3d, mol);
    }

    
    // TEST ...
    vislib::StringA paramSlotName( cr3d->PeekCallerSlot()->Parent()->FullName());
    paramSlotName += "::anim::play";
    param::ParamSlot *paramSlot = dynamic_cast<param::ParamSlot*>(this->FindNamedObject(paramSlotName, true).get());
    if( paramSlot ) {
        this->stopSegmentation = !paramSlot->Param<param::BoolParam>()->Value();
    }
    // ... TEST
    if( this->startVolSeg ) {
        this->startVolumeSegmentation( callTime);
        this->startVolSeg = false;
        this->stopSegmentation = false;
        this->oldVoxelCount = static_cast<unsigned int>(this->segmentedVoxels.size());
        this->fixedNumberOfVoxels = false;
    } else if( !this->stopSegmentation ) {
        //this->updateVolumeSegmentation( callTime);
        //this->updateVolumeSegmentation2( callTime);
        this->updateVolumeSegmentationRmsd( callTime);
    }

    // unlock the current frame
    if( mol )
        mol->Unlock();

    if( this->posInter )
        delete[] this->posInter;

    return retval;
}


/*
 * Volume rendering using molecular data.
 */
bool ProteinVolumeRenderer::RenderMolecularData( view::CallRender3D *call, MolecularDataCall *mol) {

    if( !(*mol)() )
        return false;

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
    // --- Volume Rendering                                     ---
    // --- update & render the volume                           ---
    // ------------------------------------------------------------
    
    vislib::StringA paramSlotName( call->PeekCallerSlot()->Parent()->FullName());
    paramSlotName += "::anim::play";
    param::ParamSlot *paramSlot = dynamic_cast<param::ParamSlot*>(this->FindNamedObject(paramSlotName, true).get());
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
 * Volume rendering using volume data.
 */
bool ProteinVolumeRenderer::RenderVolumeData( view::CallRender3D *call, CallProteinVolumeData *volume) {
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
 * Render the current mouse position on the clipping plane as a sphere
 */
void ProteinVolumeRenderer::RenderMousePosition( view::CallRender3D *call, float rad) {
    // do not draw if mouse position is in origin
    if( vislib::math::IsEqual( this->mousePos.Length(), 0.0f) ) return;

    glPushMatrix();
    // translate scene for volume ray casting
    if( !vislib::math::IsEqual( call->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        this->scale = 2.0f / call->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        this->scale = 1.0f;
    }
    vislib::math::Vector<float, 3> trans( 
        call->AccessBoundingBoxes().ObjectSpaceBBox().GetSize().PeekDimension() );
    trans *= -this->scale*0.5f;
    glTranslatef( trans.GetX(), trans.GetY(), trans.GetZ() );

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

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fv( this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

    // draw mouse position
    glBegin( GL_POINTS);
    glColor3f( 1, 1, 1);
    glVertex4f( 
        this->mousePos.X() * call->AccessBoundingBoxes().WorldSpaceBBox().Width(), 
        this->mousePos.Y() * call->AccessBoundingBoxes().WorldSpaceBBox().Height(), 
        this->mousePos.Z() * call->AccessBoundingBoxes().WorldSpaceBBox().Depth(),
        rad * scale );
    glEnd(); // GL_POINTS
    
    // disable sphere shader
    this->sphereShader.Disable();

    glPopMatrix();
}

/*
 * Render the segmented voxels as spheres
 */
void ProteinVolumeRenderer::RenderSegmentedVoxels( view::CallRender3D *call) {
    // do nothing if the voxel list is empty
    if( this->segmentedVoxels.empty() ) return;

    glPushMatrix();
    // translate scene for volume ray casting
    if( !vislib::math::IsEqual( call->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        this->scale = 2.0f / call->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        this->scale = 1.0f;
    }
    vislib::math::Vector<float, 3> trans( 
        call->AccessBoundingBoxes().ObjectSpaceBBox().GetSize().PeekDimension() );
    trans *= -this->scale*0.5f;
    glTranslatef( trans.GetX(), trans.GetY(), trans.GetZ() );

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

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fv( this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

    // draw mouse position
    glBegin( GL_POINTS);
    //glColor3f( 1, 1, 0);
    if( this->fixedNumberOfVoxels )
        glColor3f( 0.0f/255.0f, 240.0f/255.0f, 213.0f/255.0f);
    else
        glColor3f( 243.0f/255.0f, 215.0f/255.0f, 73.0f/255.0f);
    std::list<vislib::math::Vector<int, 3> >::iterator iter;
    int a, b, c;
    iter = this->segmentedVoxels.begin();
    unsigned int cnt = 0;
    vislib::math::Vector<int, 3> pos( 0, 0, 0);
    while( iter != this->segmentedVoxels.end() ) {
        a = iter->X();
        b = iter->Y();
        c = iter->Z();
        glVertex4f( 
            ( float( a) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Width(), 
            ( float( b) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Height(), 
            ( float( c) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Depth(),
            call->AccessBoundingBoxes().WorldSpaceBBox().LongestEdge() / float( this->volumeSize));
        pos += *iter;
        cnt++;
        iter++;
    }
    
#if 0
    pos /= cnt;
    glColor3f( 0, 1, 1);
    glVertex4f( 
            ( float( pos.X()) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Width(), 
            ( float( pos.Y()) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Height(), 
            ( float( pos.Z()) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Depth(),
            call->AccessBoundingBoxes().WorldSpaceBBox().LongestEdge() / float( this->volumeSize));

    iter = this->segmentedVoxels.begin();
    vislib::math::Vector<int, 3> pos2( *iter);
    iter++;
    while( iter != this->segmentedVoxels.end() ) {
        if( ( pos - *iter).Length() < ( pos - pos2).Length() ) {
            pos2 = *iter;
        }
        iter++;
    }
    
    glColor3f( 1, 0, 1);
    glVertex4f( 
            ( float( pos2.X()) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Width(), 
            ( float( pos2.Y()) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Height(), 
            ( float( pos2.Z()) / float( this->volumeSize)) * call->AccessBoundingBoxes().WorldSpaceBBox().Depth(),
            ( call->AccessBoundingBoxes().WorldSpaceBBox().LongestEdge() / float( this->volumeSize)) * 1.5f);
#endif
    glEnd(); // GL_POINTS
    // disable sphere shader
    this->sphereShader.Disable();

    glPopMatrix();
}


/*
 * refresh parameters
 */
void ProteinVolumeRenderer::ParameterRefresh( view::CallRender3D *call) {
    
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

}


/*
 * Create a volume containing all molecule atoms
 */
void ProteinVolumeRenderer::UpdateVolumeTexture( MolecularDataCall *mol) {
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
    glClearColor( 0.1f, 0.1f, 0.1f, 0.0f);
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

    float *atoms = new float[mol->AtomCount()*4];
    int atomCnt;
#pragma omp parallel for
    for( atomCnt = 0; atomCnt < static_cast<int>(mol->AtomCount()); ++atomCnt ) {
        atoms[atomCnt*4+0] = ( this->posInter[3*atomCnt+0] + this->translation.X()) * this->scale;
        atoms[atomCnt*4+1] = ( this->posInter[3*atomCnt+1] + this->translation.Y()) * this->scale;
        atoms[atomCnt*4+2] = ( this->posInter[3*atomCnt+2] + this->translation.Z()) * this->scale;
        atoms[atomCnt*4+3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomCnt]].Radius() * this->scale;
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer( 4, GL_FLOAT, 0, atoms);
    glColorPointer( 3, GL_FLOAT, 0, this->atomColorTable.PeekElements());
    for( z = 0; z < this->volumeSize; ++z ) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volumeTex, 0, z);
        glUniform1f( this->updateVolumeShader.ParameterLocation( "sliceDepth"), (float( z) + 0.5f) / float(this->volumeSize));
        // draw all atoms as points, using w for radius
        /*
        glBegin( GL_POINTS);
        for( unsigned int cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
            glColor3fv( this->GetAtomColor( cnt));
            glVertex4f( 
                //( mol->AtomPositions()[3*cnt+0] + this->translation.X()) * this->scale,
                //( mol->AtomPositions()[3*cnt+1] + this->translation.Y()) * this->scale,
                //( mol->AtomPositions()[3*cnt+2] + this->translation.Z()) * this->scale,
                ( this->posInter[3*cnt+0] + this->translation.X()) * this->scale,
                ( this->posInter[3*cnt+1] + this->translation.Y()) * this->scale,
                ( this->posInter[3*cnt+2] + this->translation.Z()) * this->scale,
                mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius() * this->scale );
        }
        glEnd(); // GL_POINTS
        */
        // set vertex and color pointers and draw them
        glDrawArrays( GL_POINTS, 0, mol->AtomCount());
    }
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    delete[] atoms;

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
void ProteinVolumeRenderer::UpdateVolumeTexture( const CallProteinVolumeData *volume) {
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
void ProteinVolumeRenderer::RenderVolume( vislib::math::Cuboid<float> boundingbox) {
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
void ProteinVolumeRenderer::RenderVolume( const CallProteinVolumeData *volume) {
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
void ProteinVolumeRenderer::RayParamTextures( vislib::math::Cuboid<float> boundingbox) {

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

    // save OpenGL state Clip planes
    glPushAttrib(GL_ENABLE_BIT);
    // NVIDIA sets number of clip planes to 6!
    for (int i=0; i<6; ++i) {
        glDisable(GL_CLIP_PLANE0 + i);
    }

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
    
    // restore OpenGL state Clip planes
    glPopAttrib();

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
void ProteinVolumeRenderer::RayParamTextures( const CallProteinVolumeData *volume) {

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
void ProteinVolumeRenderer::DrawBoundingBoxTranslated( vislib::math::Cuboid<float> boundingbox) {

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
void ProteinVolumeRenderer::DrawBoundingBox( vislib::math::Cuboid<float> boundingbox) {

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
    float vcpd;
	for (int i = 0; i < (int)this->volClipPlane.Count(); ++i) {
        slices.setupSingleSlice( this->volClipPlane[i].PeekComponents(), position.PeekComponents());
        float d = 0.0f;
        vcpd = static_cast<float>(this->volClipPlane[i].PeekComponents()[3]);
        glBegin(GL_TRIANGLE_FAN);
        slices.drawSingleSlice(-(-d + vcpd - 0.0001f));
        glEnd();
    }
}

/*
 * Start the segmentation of the volume
 */
void ProteinVolumeRenderer::startVolumeSegmentation( float time) {
    time_t t = clock(); // DEBUG
#if 0
    // write the volume as a raw file
    this->writeVolumeRAW();
    std::cout << "Volume written to RAW file in " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << " seconds." << std::endl;
    t = clock();
#endif

    this->segmentedVoxels.clear();
    unsigned int numVoxel = this->volumeSize * this->volumeSize * this->volumeSize;
    unsigned int z = this->volumeSize * this->volumeSize;
    unsigned int y = this->volumeSize;

    float *volume = new float[numVoxel];
    float *mask = new float[numVoxel];
    memset( mask, 0, numVoxel);

    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glGetTexImage( GL_TEXTURE_3D, 0, GL_ALPHA, GL_FLOAT, volume);
    glBindTexture( GL_TEXTURE_3D, 0);

    std::list<vislib::math::Vector<int, 3> > volPosList; 
    volPosList.push_back( this->clickedPos * float(this->volumeSize));

    vislib::math::Vector<int, 3> volPos; 
    int idx;
    int cnt = 0;

    while( !volPosList.empty() ) {
        // get and pop first element of the list
        volPos = volPosList.front();
        volPosList.pop_front();
        // process element
        idx = volPos.Z() * z + volPos.Y() * y + volPos.X();
        if( mask[idx] < 0.5f && volume[idx] < this->isoValue1 ) {
            mask[idx] = 1.0f;
            this->segmentedVoxels.push_back( volPos);
            cnt++;
            if( cnt > initialSegmentationSizeParam.Param<param::IntParam>()->Value() )
                break;
            if( volPos.X() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()+1, volPos.Y(), volPos.Z()));
            if( volPos.X() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()-1, volPos.Y(), volPos.Z()));
            if( volPos.Y() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()+1, volPos.Z()));
            if( volPos.Y() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()-1, volPos.Z()));
            if( volPos.Z() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()+1));
            if( volPos.Z() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()-1));
        }
    }

    // store time
    this->segmentationTime = time;

    std::cout << "Flood fill for " << cnt << " voxels executed in " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << " seconds." << std::endl;

    delete[] volume;
    delete[] mask;
}

/*
 * Update the segmentation of the volume
 */
void ProteinVolumeRenderer::updateVolumeSegmentation( float time) {
    time_t t = clock(); // DEBUG
    // do nothing if no previous segmentation exists
    if( this->segmentedVoxels.empty() ) return;
    int segmentSize = static_cast<int>(this->segmentedVoxels.size());

    unsigned int numVoxel = this->volumeSize * this->volumeSize * this->volumeSize;
    unsigned int z = this->volumeSize * this->volumeSize;
    unsigned int y = this->volumeSize;

    vislib::math::Vector<int, 3> volPos; 
    int idx;
    int cnt = 0;

    std::list<vislib::math::Vector<int, 3> > volPosList; 

    float *volume = new float[numVoxel];
    float *mask = new float[numVoxel];
    memset( mask, 0, numVoxel);

    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glGetTexImage( GL_TEXTURE_3D, 0, GL_ALPHA, GL_FLOAT, volume);
    glBindTexture( GL_TEXTURE_3D, 0);

    // write all current voxels to voxel map, if they are still within the cavity
    for( cnt = 0; cnt < segmentSize; ++cnt ) {
        // get and pop first element of the list
        volPos = this->segmentedVoxels.front();
        this->segmentedVoxels.pop_front();
        // process element
        idx = volPos.Z() * z + volPos.Y() * y + volPos.X();
        if( volume[idx] < this->isoValue1 ) {
            volPosList.push_back( volPos);
        }
    }

    cnt = 0;
    while( !volPosList.empty() ) {
        // get and pop first element of the list
        volPos = volPosList.front();
        volPosList.pop_front();
        // process element
        idx = volPos.Z() * z + volPos.Y() * y + volPos.X();
        if( mask[idx] < 0.5f && volume[idx] < this->isoValue1 ) {
            mask[idx] = 1.0f;
            this->segmentedVoxels.push_back( volPos);
            cnt++;
            if( cnt > 1000 ) 
                break;
            if( volPos.X() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()+1, volPos.Y(), volPos.Z()));
            if( volPos.X() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()-1, volPos.Y(), volPos.Z()));
            if( volPos.Y() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()+1, volPos.Z()));
            if( volPos.Y() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()-1, volPos.Z()));
            if( volPos.Z() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()+1));
            if( volPos.Z() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()-1));
        }
    }

    // store time
    this->segmentationTime = time;

    //std::cout << "Flood fill for " << cnt << " voxels executed in " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << " seconds." << std::endl;
    double elapsedTime = ( double( clock() - t) / double( CLOCKS_PER_SEC) );

    delete[] volume;
    delete[] mask;
}

/*
 * Update the segmentation of the volume
 */
void ProteinVolumeRenderer::updateVolumeSegmentation2( float time) {
    time_t t = clock(); // DEBUG
    // do nothing if no previous segmentation exists
    if( this->segmentedVoxels.empty() ) return;
    int segmentSize = static_cast<int>(this->segmentedVoxels.size());

    unsigned int numVoxel = this->volumeSize * this->volumeSize * this->volumeSize;
    unsigned int z = this->volumeSize * this->volumeSize;
    unsigned int y = this->volumeSize;

    vislib::math::Vector<int, 3> volPos; 
    int idx;
    int cnt = 0;

    std::list<vislib::math::Vector<int, 3> > volPosList; 

    float *volume = new float[numVoxel];
    float *mask = new float[numVoxel];
    memset( mask, 0, numVoxel);

    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glGetTexImage( GL_TEXTURE_3D, 0, GL_ALPHA, GL_FLOAT, volume);
    glBindTexture( GL_TEXTURE_3D, 0);

    // start ...
    std::list<vislib::math::Vector<int, 3> >::iterator iter;

    iter = this->segmentedVoxels.begin();
    cnt = 0;
    vislib::math::Vector<int, 3> pos( 0, 0, 0);
    while( iter != this->segmentedVoxels.end() ) {
        pos += *iter;
        cnt++;
        iter++;
    }
    pos /= cnt;
    pos += this->clickedPos * static_cast<float>(this->volumeSize);
    pos /= 2;

    // write all current voxels to voxel map, if they are still within the cavity
    for( cnt = 0; cnt < segmentSize; ++cnt ) {
        // get and pop first element of the list
        volPos = this->segmentedVoxels.front();
        this->segmentedVoxels.pop_front();
        // process element
        idx = volPos.Z() * z + volPos.Y() * y + volPos.X();
        if( volume[idx] < this->isoValue1 ) {
            volPosList.push_back( volPos);
        }
    }

    iter = volPosList.begin();
    vislib::math::Vector<int, 3> pos2( *iter);
    iter++;
    while( iter != volPosList.end() ) {
        if( ( pos - *iter).Length() < ( pos - pos2).Length() ) {
            pos2 = *iter;
        }
        iter++;
    }
    // ... end

    volPosList.clear();
    volPosList.push_back( pos2);

    cnt = 0;
    while( !volPosList.empty() ) {
        // get and pop first element of the list
        volPos = volPosList.front();
        volPosList.pop_front();
        // process element
        idx = volPos.Z() * z + volPos.Y() * y + volPos.X();
        if( mask[idx] < 0.5f && volume[idx] < this->isoValue1 ) {
            mask[idx] = 1.0f;
            this->segmentedVoxels.push_back( volPos);
            cnt++;
#ifdef STOP_SEGMENTATION
            if( cnt > int( this->segmentationDeltaParam.Param<param::FloatParam>()->Value() * float( this->oldVoxelCount)) ) 
                break;
#endif
            if( volPos.X() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()+1, volPos.Y(), volPos.Z()));
            if( volPos.X() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()-1, volPos.Y(), volPos.Z()));
            if( volPos.Y() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()+1, volPos.Z()));
            if( volPos.Y() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()-1, volPos.Z()));
            if( volPos.Z() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()+1));
            if( volPos.Z() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()-1));
        }
    }

    // store time
    this->segmentationTime = time;

    //std::cout << "Flood fill for " << cnt << " voxels executed in " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << " seconds." << std::endl;
    double elapsedTime = ( double( clock() - t) / double( CLOCKS_PER_SEC) );

    delete[] volume;
    delete[] mask;
}
/*
 * Update the segmentation of the volume
 */
void ProteinVolumeRenderer::updateVolumeSegmentationRmsd( float time) {
    time_t t = clock(); // DEBUG
    // do nothing if no previous segmentation exists
    if( this->segmentedVoxels.empty() ) return;
    int segmentSize = static_cast<int>(this->segmentedVoxels.size());

    unsigned int numVoxel = this->volumeSize * this->volumeSize * this->volumeSize;
    unsigned int z = this->volumeSize * this->volumeSize;
    unsigned int y = this->volumeSize;

    vislib::math::Vector<int, 3> volPos; 
    int idx;
    int cnt = 0;

    std::list<vislib::math::Vector<int, 3> > volPosList; 

    float *volume = new float[numVoxel];
    float *mask = new float[numVoxel];
    memset( mask, 0, numVoxel);

    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glGetTexImage( GL_TEXTURE_3D, 0, GL_ALPHA, GL_FLOAT, volume);
    glBindTexture( GL_TEXTURE_3D, 0);

    // start ...
    std::list<vislib::math::Vector<int, 3> >::iterator iter;
    vislib::math::Vector<int, 3> pos( this->clickedPos * static_cast<float>(this->volumeSize));
    vislib::math::Vector<int, 3> pos2( pos);

    // write all current voxels to voxel map, if they are still within the cavity
    for( cnt = 0; cnt < segmentSize; ++cnt ) {
        // get and pop first element of the list
        volPos = this->segmentedVoxels.front();
        this->segmentedVoxels.pop_front();
        // process element
        idx = volPos.Z() * z + volPos.Y() * y + volPos.X();
        if( volume[idx] < this->isoValue1 ) {
            volPosList.push_back( volPos);
        }
    }

    int l1, l2;
    if( !volPosList.empty() ) {
        iter = volPosList.begin();
        pos2 = *iter;
        iter++;
        while( iter != volPosList.end() ) {
            l1 = ( pos - *iter).Length();
            l2 = ( pos - pos2).Length();
            if( l1 < l2 ) {
                pos2 = *iter;
            }
            iter++;
        }
    } else {
        std::cout << "empty" << std::endl;
    }
    // ... end

    volPosList.clear();
    volPosList.push_back( pos2);

    cnt = 0;
    while( !volPosList.empty() ) {
        // get and pop first element of the list
        volPos = volPosList.front();
        volPosList.pop_front();
        // process element
        idx = volPos.Z() * z + volPos.Y() * y + volPos.X();
        if( mask[idx] < 0.5f && volume[idx] < this->isoValue1 ) {
            mask[idx] = 1.0f;
            this->segmentedVoxels.push_back( volPos);
            cnt++;
#ifdef STOP_SEGMENTATION
            if( cnt > int( this->segmentationDeltaParam.Param<param::FloatParam>()->Value() * float( this->oldVoxelCount)) && !this->fixedNumberOfVoxels ) 
                break;
            if( cnt == this->oldVoxelCount && this->fixedNumberOfVoxels )
                break;
#endif
            if( volPos.X() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()+1, volPos.Y(), volPos.Z()));
            if( volPos.X() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X()-1, volPos.Y(), volPos.Z()));
            if( volPos.Y() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()+1, volPos.Z()));
            if( volPos.Y() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y()-1, volPos.Z()));
            if( volPos.Z() < static_cast<int>(this->volumeSize-1) )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()+1));
            if( volPos.Z() > 0 )
                volPosList.push_back( vislib::math::Vector<int, 3>( volPos.X(), volPos.Y(), volPos.Z()-1));
        }
    }
    // check size of segmented region
    if( cnt < static_cast<int>(this->oldVoxelCount) )
        this->fixedNumberOfVoxels = false;

    // store time
    this->segmentationTime = time;

    std::cout << "Flood fill for " << cnt << " voxels executed in " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << " seconds." << std::endl;
    double elapsedTime = ( double( clock() - t) / double( CLOCKS_PER_SEC) );

    delete[] volume;
    delete[] mask;
}

/*
 * write the current volume as a raw file
 */
bool ProteinVolumeRenderer::writeVolumeRAWUChar(core::param::ParamSlot& slot) {
    unsigned int numVoxel = this->volumeSize * this->volumeSize * this->volumeSize;

    float *volume = new float[numVoxel];
    unsigned char *ucVol = new unsigned char[numVoxel];

    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glGetTexImage( GL_TEXTURE_3D, 0, GL_ALPHA, GL_FLOAT, volume);
    glBindTexture( GL_TEXTURE_3D, 0);

    int cnt;
#pragma omp parallel for
    for( cnt = 0; cnt < static_cast<int>(numVoxel); ++cnt ) {
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

    return true;
}

/*
 * write the current volume as a raw file
 */
bool ProteinVolumeRenderer::writeVolumeRAWFloat(core::param::ParamSlot& slot) {
    unsigned int numVoxel = this->volumeSize * this->volumeSize * this->volumeSize;

    float *volume = new float[numVoxel];

    glBindTexture( GL_TEXTURE_3D, this->volumeTex);
    glGetTexImage( GL_TEXTURE_3D, 0, GL_ALPHA, GL_FLOAT, volume);
    glBindTexture( GL_TEXTURE_3D, 0);
    
    // write array to file
    FILE *foutRaw = fopen( "test.raw", "wb");
    if( !foutRaw ) {
        std::cout << "could not open file for writing." << std::endl;
        delete[] volume;
        return false;
    } else {
        fwrite( volume, sizeof(float), numVoxel, foutRaw);
        fclose( foutRaw);
    }
    delete[] volume;
    return true;
}

    
