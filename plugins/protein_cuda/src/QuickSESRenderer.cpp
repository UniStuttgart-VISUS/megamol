/*
 * QuickSESRenderer.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "QuickSESRenderer.h"
#include "mmcore/CoreInstance.h"
#include "Color.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/StringConverter.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include <omp.h>
#include <algorithm>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_cuda;


/*
 * protein_cuda::QuickSESRenderer::QuickSESRenderer (CTOR)
 */
QuickSESRenderer::QuickSESRenderer(void) : Renderer3DModule (),
    molDataCallerSlot( "getData", "Connects the molecule rendering with molecule data storage"),
    colorTableFileParam( "color::colorTableFilename", "The filename of the color table."),
    coloringModeParam( "color::coloringMode", "The first coloring mode."),
    minGradColorParam( "color::minGradColor", "The color for the minimum value for gradient coloring" ),
    midGradColorParam( "color::midGradColor", "The color for the middle value for gradient coloring" ),
    maxGradColorParam( "color::maxGradColor", "The color for the maximum value for gradient coloring" ),
    interpolParam( "posInterpolation", "Enable positional interpolation between frames" ),
    probeRadiusParam( "QuickSES::probeRadius", "The probe radius" ),
	gridSpacingParam("QuickSES::gridspacing", "Grid spacing"),
    offscreenRenderingParam( "offscreenRendering", "Toggle offscreenRendering" ),
	pos0(NULL), pos1(NULL), posArraySize(0),
    setCUDAGLDevice(true)
{
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    Color::ReadColorTableFromFile( filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(new param::FilePathParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);

    // coloring mode #0
    this->currentColoringMode = Color::CHAIN;
    param::EnumParam *cm0 = new param::EnumParam(int(this->currentColoringMode));
    cm0->SetTypePair( Color::ELEMENT, "Element");
    cm0->SetTypePair( Color::RESIDUE, "Residue");
    cm0->SetTypePair( Color::STRUCTURE, "Structure");
    cm0->SetTypePair( Color::BFACTOR, "BFactor");
    cm0->SetTypePair( Color::CHARGE, "Charge");
    cm0->SetTypePair( Color::OCCUPANCY, "Occupancy");
    cm0->SetTypePair( Color::CHAIN, "Chain");
    cm0->SetTypePair( Color::MOLECULE, "Molecule");
    cm0->SetTypePair( Color::RAINBOW, "Rainbow");
    this->coloringModeParam << cm0;
    this->MakeSlotAvailable( &this->coloringModeParam);

    // the color for the minimum value (gradient coloring
    this->minGradColorParam.SetParameter(new param::StringParam( "#146496"));
    this->MakeSlotAvailable( &this->minGradColorParam);

    // the color for the middle value (gradient coloring
    this->midGradColorParam.SetParameter(new param::StringParam( "#f0f0f0"));
    this->MakeSlotAvailable( &this->midGradColorParam);

    // the color for the maximum value (gradient coloring
    this->maxGradColorParam.SetParameter(new param::StringParam( "#ae3b32"));
    this->MakeSlotAvailable( &this->maxGradColorParam);

    // en-/disable positional interpolation
    this->interpolParam.SetParameter(new param::BoolParam( true));
    this->MakeSlotAvailable( &this->interpolParam);

    // make the rainbow color table
    Color::MakeRainbowColorTable( 100, this->rainbowColors);
    
    this->probeRadiusParam.SetParameter( new param::FloatParam( 1.4f, 0.0f));
    this->MakeSlotAvailable( &this->probeRadiusParam);

    this->gridSpacingParam.SetParameter( new param::FloatParam( 1.0f, 0.1f));
    this->MakeSlotAvailable( &this->gridSpacingParam);
    
    // Toggle offscreen rendering
    this->offscreenRenderingParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable( &this->offscreenRenderingParam);

    cudaqses = 0;
}


/*
 * protein_cuda::QuickSESRenderer::~QuickSESRenderer (DTOR)
 */
QuickSESRenderer::~QuickSESRenderer(void)  {
    if (cudaqses) {
        CUDAQuickSES *cqs = (CUDAQuickSES *)this->cudaqses;
        delete cqs;
    }

    this->Release();
}


/*
 * protein_cuda::QuickSESRenderer::release
 */
void QuickSESRenderer::release(void) {

}


/*
 * protein_cuda::QuickSESRenderer::create
 */
bool QuickSESRenderer::create(void) {
    if( !isExtAvailable( "GL_ARB_vertex_program") || !ogl_IsVersionGEQ(2,0) )
        return false;

    if ( !vislib::graphics::gl::GLSLShader::InitialiseExtensions() )
        return false;
    
    //cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId() );
    //printf( "cudaGLSetGLDevice: %s\n", cudaGetErrorString( cudaGetLastError()));

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;
    
    //////////////////////////////////////////////////////
    // load the shader files for the per pixel lighting //
    //////////////////////////////////////////////////////
    // vertex shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::cartoon::perpixellight::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for perpixellight shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::cartoon::perpixellight::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for perpixellight shader");
        return false;
    }
    this->lightShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());

    ///////////////////////////////////////////////////////////////////////////////////////
    // load the shader files for the per pixel lighting (OFFSCREEN/PASSTHROUGH RENDERER) //
    ///////////////////////////////////////////////////////////////////////////////////////
    // vertex shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::cartoon::perpixellight::vertexOR", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for perpixellight OR shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::cartoon::perpixellight::fragmentOR", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for perpixellight OR shader");
        return false;
    }
    this->lightShaderOR.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());

    return true;
}


/*
 * protein_cuda::QuickSESRenderer::GetExtents
 */
bool QuickSESRenderer::GetExtents(Call& call) {
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;

    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld( scale);
    cr3d->SetTimeFramesCount( mol->FrameCount());

    return true;
}


/**********************************************************************
 * 'render'-function
 **********************************************************************/

/*
 * protein_cuda::QuickSESRenderer::Render
 */
bool QuickSESRenderer::Render(Call& call) {

    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    if( setCUDAGLDevice ) {
#ifdef _WIN32
        if( cr3d->IsGpuAffinity() ) {
            HGPUNV gpuId = cr3d->GpuAffinity<HGPUNV>();
            int devId;
            cudaWGLGetDevice( &devId, gpuId);
            cudaGLSetGLDevice( devId);
        } else {
            cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId());
        }
#else
        cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId());
#endif
        printf( "cudaGLSetGLDevice: %s\n", cudaGetErrorString( cudaGetLastError()));
        setCUDAGLDevice = false;
    }

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL) return false;

    int cnt;
    
    // set call time
    mol->SetCalltime(callTime);
    // set frame ID and call data
    mol->SetFrameID(static_cast<int>( callTime));

    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
    // check if atom count is zero
    if( mol->AtomCount() == 0 ) return true;
    // get positions of the first frame
	if (posArraySize < mol->AtomCount() * 3) {
		posArraySize = mol->AtomCount() * 3;
		if (pos0) {
			delete[] pos0;
			delete[] pos1;
		}
		pos0 = new float[posArraySize];
		pos1 = new float[posArraySize];
	}
    memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
    // set next frame ID and get positions of the second frame
    if( ( ( static_cast<int>( callTime) + 1) < int( mol->FrameCount()) ) &&
        this->interpolParam.Param<param::BoolParam>()->Value() )
        mol->SetFrameID(static_cast<int>( callTime) + 1);
    else
        mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        return false;
    }
    memcpy( pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));

    // interpolate atom positions between frames
    posInter.SetCount(mol->AtomCount() * 4);
    float inter = callTime - static_cast<float>(static_cast<int>( callTime));
    float threshold = vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
            posInter[4*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
            posInter[4*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
            posInter[4*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
        } else if( inter < 0.5f ) {
            posInter[4*cnt+0] = pos0[3*cnt+0];
            posInter[4*cnt+1] = pos0[3*cnt+1];
            posInter[4*cnt+2] = pos0[3*cnt+2];
        } else {
            posInter[4*cnt+0] = pos1[3*cnt+0];
            posInter[4*cnt+1] = pos1[3*cnt+1];
            posInter[4*cnt+2] = pos1[3*cnt+2];
        }
		posInter[4 * cnt + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
    }

    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

	this->UpdateParameters(mol);

    // recompute color table, if necessary
    if( this->atomColorTable.Count()/3 < mol->AtomCount() ) {
        // Use one coloring mode
        Color::MakeColorTable( mol,
          this->currentColoringMode,
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);
    }

    // ---------- render ----------

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    
    float spec[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, spec);
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.0f);
    glEnable( GL_COLOR_MATERIAL);

    // DEBUG
	glPointSize(5.0f);
    glBegin( GL_POINTS);
    for( unsigned int i = 0; i < mol->AtomCount(); i++ ) {
        glColor3fv( &atomColorTable.PeekElements()[3*i]);
        glVertex3fv( &posInter[4*i]);
    }
    glEnd(); // GL_POINTS

    // calculate surface
    if( !this->cudaqses ) {
        this->cudaqses = new CUDAQuickSES();
    }
    
    // enable per-pixel light shader
    if(!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
        // direct rendering
        this->lightShader.Enable();
    } else {
        // offscreen rendering (Render to fragment buffer)
        this->lightShaderOR.Enable();
        glUniform2fARB(this->lightShaderOR.ParameterLocation("zValues"), cameraInfo->NearClip(), cameraInfo->FarClip());
    }

	// TODO calculate and render SES
	this->calcSurf(mol, this->posInter.PeekElements());
    
    // disable per-pixel light shader
    if(!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
        this->lightShader.Disable();
    }
    else {
        this->lightShaderOR.Disable();
    }

    glPopMatrix();

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
* update parameters
*/
void QuickSESRenderer::UpdateParameters(const MolecularDataCall *mol) {
	// color table param
	if (this->colorTableFileParam.IsDirty()) {
		Color::ReadColorTableFromFile(
			this->colorTableFileParam.Param<param::FilePathParam>()->Value(),
			this->colorLookupTable);
		this->colorTableFileParam.ResetDirty();
	}
	// Recompute color table
	if (this->coloringModeParam.IsDirty()) {

		this->currentColoringMode = static_cast<Color::ColoringMode>(int(
			this->coloringModeParam.Param<param::EnumParam>()->Value()));

		// Use one coloring mode
		Color::MakeColorTable(mol,
			this->currentColoringMode,
			this->atomColorTable, this->colorLookupTable, this->rainbowColors,
			this->minGradColorParam.Param<param::StringParam>()->Value(),
			this->midGradColorParam.Param<param::StringParam>()->Value(),
			this->maxGradColorParam.Param<param::StringParam>()->Value(),
			true);

		this->coloringModeParam.ResetDirty();
	}
}


bool QuickSESRenderer::calcSurf(MolecularDataCall *mol, const float *pos) {
	// tmp variables
	//int i;
	//float mincoord[3], maxcoord[3];
	float numvoxels[3];
	// get grid spacing 
	float gridspacing = this->gridSpacingParam.Param<param::FloatParam>()->Value();
	// get bounding box
	vislib::math::Cuboid<float> bbox = mol->AccessBoundingBoxes().ObjectSpaceBBox();
	bbox.EnforcePositiveSize();

	// compute the real grid dimensions from the selected atoms
	numvoxels[0] = static_cast<float>((int)ceil(bbox.Width() / gridspacing));
	numvoxels[1] = static_cast<float>((int)ceil(bbox.Height() / gridspacing));
	numvoxels[2] = static_cast<float>((int)ceil(bbox.Depth() / gridspacing));
	
	CUDAQuickSES *cqs = (CUDAQuickSES *)this->cudaqses;

	return true;
}
