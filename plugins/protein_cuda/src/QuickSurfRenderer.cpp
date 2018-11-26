/*
 * QuickSurfRenderer.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "QuickSurfRenderer.h"
#include "mmcore/CoreInstance.h"
#include "Color.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"
#include <GL/glu.h>
#include <omp.h>
#include <algorithm>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_cuda;


/*
 * protein_cuda::QuickSurfRenderer::QuickSurfRenderer (CTOR)
 */
QuickSurfRenderer::QuickSurfRenderer(void) : Renderer3DModuleDS (),
    molDataCallerSlot( "getData", "Connects the molecule rendering with molecule data storage"),
    colorTableFileParam( "color::colorTableFilename", "The filename of the color table."),
    coloringModeParam( "color::coloringMode", "The first coloring mode."),
    minGradColorParam( "color::minGradColor", "The color for the minimum value for gradient coloring" ),
    midGradColorParam( "color::midGradColor", "The color for the middle value for gradient coloring" ),
    maxGradColorParam( "color::maxGradColor", "The color for the maximum value for gradient coloring" ),
    interpolParam( "posInterpolation", "Enable positional interpolation between frames" ),
    qualityParam( "quicksurf::quality", "Quality" ),
    radscaleParam( "quicksurf::radscale", "Radius scale" ),
    gridspacingParam( "quicksurf::gridspacing", "Grid spacing" ),
    isovalParam( "quicksurf::isoval", "Isovalue" ),
    offscreenRenderingParam( "offscreenRendering", "Toggle offscreenRendering" ),
	transparencyValueParam("alphaValue", "Alpha value of the whole surface"),
    setCUDAGLDevice(true)
{
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    Color::ReadColorTableFromFile( filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(new param::StringParam( A2T( filename)));
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
    
    this->qualityParam.SetParameter( new param::IntParam( 1, 0, 4));
    this->MakeSlotAvailable( &this->qualityParam);

    this->radscaleParam.SetParameter( new param::FloatParam( 1.0f, 0.0f));
    this->MakeSlotAvailable( &this->radscaleParam);

    this->gridspacingParam.SetParameter( new param::FloatParam( 1.0f, 0.0f));
    this->MakeSlotAvailable( &this->gridspacingParam);

    this->isovalParam.SetParameter( new param::FloatParam( 0.5f, 0.0f));
    this->MakeSlotAvailable( &this->isovalParam);

	this->transparencyValueParam.SetParameter(new param::FloatParam(1.0f, 0.0f, 1.0f));
	this->MakeSlotAvailable(&this->transparencyValueParam);
    
    // Toggle offscreen rendering
    this->offscreenRenderingParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable( &this->offscreenRenderingParam);

    volmap = NULL;
    voltexmap = NULL;
    isovalue = 0.5f;

    numvoxels[0] = 128;
    numvoxels[1] = 128;
    numvoxels[2] = 128;

    origin[0] = 0.0f;
    origin[1] = 0.0f;
    origin[2] = 0.0f;

    xaxis[0] = 1.0f;
    xaxis[1] = 0.0f;
    xaxis[2] = 0.0f;

    yaxis[0] = 0.0f;
    yaxis[1] = 1.0f;
    yaxis[2] = 0.0f;

    zaxis[0] = 0.0f;
    zaxis[1] = 0.0f;
    zaxis[2] = 1.0f;
   
    cudaqsurf = 0;

    gpuvertexarray=0;
    gpunumverts=0;
    gv=gn=gc=NULL;
    gpunumfacets=0;
    gf=NULL;

    timer = wkf_timer_create();
}


/*
 * protein_cuda::QuickSurfRenderer::~QuickSurfRenderer (DTOR)
 */
QuickSurfRenderer::~QuickSurfRenderer(void)  {
    if (cudaqsurf) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *) cudaqsurf;
        delete cqs;
    }

    if (gv) 
        free(gv);
    if (gn) 
        free(gn);
    if (gc) 
        free(gc);
    if (gf) 
        free(gf);

    if (voltexmap != NULL)
        free(voltexmap);
    voltexmap = NULL;

    wkf_timer_destroy(timer);

    this->Release();
}


/*
 * protein_cuda::QuickSurfRenderer::release
 */
void QuickSurfRenderer::release(void) {

}


/*
 * protein_cuda::QuickSurfRenderer::create
 */
bool QuickSurfRenderer::create(void) {
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
 * protein_cuda::QuickSurfRenderer::GetExtents
 */
bool QuickSurfRenderer::GetExtents(Call& call) {
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
 * 'render'-functions
 **********************************************************************/

/*
 * protein_cuda::QuickSurfRenderer::Render
 */
bool QuickSurfRenderer::Render(Call& call) {

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
    float *pos0 = new float[mol->AtomCount() * 3];
    memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
    // set next frame ID and get positions of the second frame
    if( ( ( static_cast<int>( callTime) + 1) < int( mol->FrameCount()) ) &&
        this->interpolParam.Param<param::BoolParam>()->Value() )
        mol->SetFrameID(static_cast<int>( callTime) + 1);
    else
        mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    float *pos1 = new float[mol->AtomCount() * 3];
    memcpy( pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));

    // interpolate atom positions between frames
    float *posInter = new float[mol->AtomCount() * 3];
    float inter = callTime - static_cast<float>(static_cast<int>( callTime));
    float threshold = vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
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

    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

    // ---------- update parameters ----------
    this->UpdateParameters( mol);

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

    // TODO
    /*
    glBegin( GL_POINTS);
    for( int i = 0; i < mol->AtomCount(); i++ ) {
        glColor3fv( &atomColorTable.PeekElements()[3*i]);
        glVertex3fv( &posInter[3*i]);
    }
    glEnd(); // GL_POINTS
    */

    // calculate surface
    if( !cudaqsurf ) {
        cudaqsurf = new CUDAQuickSurf();
    }

	float alpha = this->transparencyValueParam.Param<param::FloatParam>()->Value();
	bool sort = false;

	if (std::abs(alpha - 1.0f) > vislib::math::FLOAT_EPSILON) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		sort = true;
	}

    // enable per-pixel light shader
    if(!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
        // direct rendering
        this->lightShader.Enable();
		glUniform1fARB(this->lightShader.ParameterLocation("alpha"), alpha);
    } else {
        // offscreen rendering (Render to fragment buffer)
        this->lightShaderOR.Enable();
        glUniform2fARB(this->lightShaderOR.ParameterLocation("zValues"), cameraInfo->NearClip(), cameraInfo->FarClip());
		glUniform1fARB(this->lightShaderOR.ParameterLocation("alpha"), alpha);
    }

    this->calcSurf( mol, posInter, 
        this->qualityParam.Param<param::IntParam>()->Value(),
        this->radscaleParam.Param<param::FloatParam>()->Value(),
        this->gridspacingParam.Param<param::FloatParam>()->Value(),
        this->isovalParam.Param<param::FloatParam>()->Value(),
        true, sort);
    
    // disable per-pixel light shader
    if(!this->offscreenRenderingParam.Param<param::BoolParam>()->Value()) {
        this->lightShader.Disable();
    }
    else {
        this->lightShaderOR.Disable();
    }

	if (std::abs(alpha - 1.0f) > vislib::math::FLOAT_EPSILON) {
		glDisable(GL_BLEND);
	}

    glPopMatrix();


    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
 * update parameters
 */
void QuickSurfRenderer::UpdateParameters( const MolecularDataCall *mol) {
    // color table param
    if( this->colorTableFileParam.IsDirty() ) {
        Color::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::StringParam>()->Value(),
            this->colorLookupTable);
        this->colorTableFileParam.ResetDirty();
    }
    // Recompute color table
    if( this->coloringModeParam.IsDirty() ) {

        this->currentColoringMode = static_cast<Color::ColoringMode>(int(
            this->coloringModeParam.Param<param::EnumParam>()->Value()));

        // Use one coloring mode
        Color::MakeColorTable( mol,
          this->currentColoringMode,
          this->atomColorTable, this->colorLookupTable, this->rainbowColors,
          this->minGradColorParam.Param<param::StringParam>()->Value(),
          this->midGradColorParam.Param<param::StringParam>()->Value(),
          this->maxGradColorParam.Param<param::StringParam>()->Value(),
          true);

        this->coloringModeParam.ResetDirty();
    }
}


int QuickSurfRenderer::calcSurf(MolecularDataCall *mol, float *posInter,
                         int quality, float radscale, float gridspacing,
                         float isoval, bool useCol, bool sortTriangles) {
    wkf_timer_start(timer);

    // clean up any existing CPU arrays before going any further...
    if (voltexmap != NULL)
    free(voltexmap);
    voltexmap = NULL;

    // initialize class variables
    isovalue=isoval;

    // If no volumetric texture will be computed we will use the cmap
    // parameter to pass in the solid color to be applied to all vertices
    //vec_copy(solidcolor, cmap);

    // compute min/max atom radius, build list of selected atom radii,
    // and compute bounding box for the selected atoms
    float minx, miny, minz, maxx, maxy, maxz;
    float minrad, maxrad;
    int i;
    float mincoord[3], maxcoord[3];

    minx = maxx = posInter[0];
    miny = maxy = posInter[1];
    minz = maxz = posInter[2];
    minrad = maxrad = mol->AtomTypes()[mol->AtomTypeIndices()[0]].Radius();
	for (i = 0; i < (int)mol->AtomCount(); i++) {
#ifdef COMPUTE_BBOX
        int ind = i * 3;
        float tmpx = posInter[ind  ];
        float tmpy = posInter[ind+1];
        float tmpz = posInter[ind+2];

        minx = (tmpx < minx) ? tmpx : minx;
        maxx = (tmpx > maxx) ? tmpx : maxx;

        miny = (tmpy < miny) ? tmpy : miny;
        maxy = (tmpy > maxy) ? tmpy : maxy;

        minz = (tmpz < minz) ? tmpz : minz;
        maxz = (tmpz > maxz) ? tmpz : maxz;
#endif
  
        float r = mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius();
        minrad = (r < minrad) ? r : minrad;
        maxrad = (r > maxrad) ? r : maxrad;
    }

    mincoord[0] = minx;
    mincoord[1] = miny;
    mincoord[2] = minz;
    maxcoord[0] = maxx;
    maxcoord[1] = maxy;
    maxcoord[2] = maxz;

    // crude estimate of the grid padding we require to prevent the
    // resulting isosurface from being clipped
    float gridpadding = radscale * maxrad * 1.5f;
    float padrad = gridpadding;
    padrad = static_cast<float>(0.4 * sqrt(4.0/3.0*M_PI*padrad*padrad*padrad));
    gridpadding = std::max(gridpadding, padrad);

#if VERBOSE
    printf("  Padding radius: %.3f  (minrad: %.3f maxrad: %.3f)\n", 
        gridpadding, minrad, maxrad);
#endif

    mincoord[0] -= gridpadding;
    mincoord[1] -= gridpadding;
    mincoord[2] -= gridpadding;
    maxcoord[0] += gridpadding;
    maxcoord[1] += gridpadding;
    maxcoord[2] += gridpadding;

    // kroneml
    mincoord[0] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    mincoord[1] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    mincoord[2] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    maxcoord[0] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Right();
    maxcoord[1] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Top();
    maxcoord[2] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Front();

    // compute the real grid dimensions from the selected atoms
    xaxis[0] = maxcoord[0]-mincoord[0];
    yaxis[1] = maxcoord[1]-mincoord[1];
    zaxis[2] = maxcoord[2]-mincoord[2];
    numvoxels[0] = (int) ceil(xaxis[0] / gridspacing);
    numvoxels[1] = (int) ceil(yaxis[1] / gridspacing);
    numvoxels[2] = (int) ceil(zaxis[2] / gridspacing);

    // recalc the grid dimensions from rounded/padded voxel counts
    xaxis[0] = (numvoxels[0]-1) * gridspacing;
    yaxis[1] = (numvoxels[1]-1) * gridspacing;
    zaxis[2] = (numvoxels[2]-1) * gridspacing;
    maxcoord[0] = mincoord[0] + xaxis[0];
    maxcoord[1] = mincoord[1] + yaxis[1];
    maxcoord[2] = mincoord[2] + zaxis[2];

#if VERBOSE
    printf("  Final bounding box: (%.1f %.1f %.1f) -> (%.1f %.1f %.1f)\n",
        mincoord[0], mincoord[1], mincoord[2],
        maxcoord[0], maxcoord[1], maxcoord[2]);

    printf("  Grid size: (%d %d %d)\n",
        numvoxels[0], numvoxels[1], numvoxels[2]);
#endif

    //vec_copy(origin, mincoord);
    origin[0] = mincoord[0];
    origin[1] = mincoord[1];
    origin[2] = mincoord[2];

    // build compacted lists of bead coordinates, radii, and colors
    float *xyzr = NULL;
    float *colors = NULL;

    int ind = 0;
    int ind4 = 0; 
    xyzr = (float *) malloc( mol->AtomCount() * sizeof(float) * 4);
	float alphaVal = this->transparencyValueParam.Param<param::FloatParam>()->Value();
    if (useCol) {
        colors = (float *) malloc( mol->AtomCount() * sizeof(float) * 4);

        // build compacted lists of atom coordinates, radii, and colors
		for (i = 0; i < (int)mol->AtomCount(); i++) {
            const float *fp = posInter + ind;
            xyzr[ind4    ] = fp[0]-origin[0];
            xyzr[ind4 + 1] = fp[1]-origin[1];
            xyzr[ind4 + 2] = fp[2]-origin[2];
            xyzr[ind4 + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius();
 
            //const float *cp = &cmap[colidx[i] * 3];
            const float *cp = &this->atomColorTable[ind];
            colors[ind4    ] = cp[0];
            colors[ind4 + 1] = cp[1];
            colors[ind4 + 2] = cp[2];
			colors[ind4 + 3] = alphaVal;

            ind4 += 4;
            ind += 3;
        }
    } else {
        // build compacted lists of atom coordinates and radii only
		for (i = 0; i < (int)mol->AtomCount(); i++) {
            const float *fp = posInter + ind;
            xyzr[ind4    ] = fp[0]-origin[0];
            xyzr[ind4 + 1] = fp[1]-origin[1];
            xyzr[ind4 + 2] = fp[2]-origin[2];
            xyzr[ind4 + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius();
            ind4 += 4;
            ind += 3;
        }
    }

    // set gaussian window size based on user-specified quality parameter
    float gausslim = 2.0f;
    switch (quality) {
    case 3: gausslim = 4.0f; break; // max quality

    case 2: gausslim = 3.0f; break; // high quality

    case 1: gausslim = 2.5f; break; // medium quality

    case 0: 
    default: gausslim = 2.0f; // low quality
        break;
    }

    pretime = wkf_timer_timenow(timer);

    CUDAQuickSurf *cqs = (CUDAQuickSurf *) cudaqsurf;

	// Calculate cam pos using last column of inverse modelview matrix
	float3 camPos;
	GLfloat m[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> modelMatrix(&m[0]);
	modelMatrix.Invert();
	camPos.x = modelMatrix.GetAt(0, 3);
	camPos.y = modelMatrix.GetAt(1, 3);
	camPos.z = modelMatrix.GetAt(2, 3);

	cqs->copyCamPosToDevice(camPos);

    // compute both density map and floating point color texture map
    int rc = cqs->calc_surf( mol->AtomCount(), &xyzr[0],
        (useCol) ? &colors[0] : &this->atomColorTable[0],
        useCol, origin, numvoxels, maxrad,
        radscale, gridspacing, isovalue, gausslim,
        gpunumverts, gv, gn, gc, gpunumfacets, gf, sortTriangles);

    if (rc == 0) {
        gpuvertexarray = 1;
        free(xyzr);
        if (colors) free(colors);
        voltime = wkf_timer_timenow(timer);
        return 0;
    } else {
        gpuvertexarray = 0;
        free(xyzr);
        if (colors) free(colors);  
        voltime = wkf_timer_timenow(timer);
        return 1;
    }
}
