q/*
 * QuickSurfRenderer2.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#ifdef WITH_CUDA

#define _USE_MATH_DEFINES 1

#include "QuickSurfRenderer2.h"
#include "CoreInstance.h"
#include "Color.h"
#include "utility/ShaderSourceFactory.h"
#include "utility/ColourParser.h"
#include "param/StringParam.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/ShaderSource.h"
#include "vislib/AbstractOpenGLShader.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <glh/glh_extensions.h>
#include <omp.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::core::moldyn;


/*
 * protein::QuickSurfRenderer2::QuickSurfRenderer2 (CTOR)
 */
QuickSurfRenderer2::QuickSurfRenderer2(void) : Renderer3DModuleDS (),
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
    m_hPos(0)
{
    this->molDataCallerSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
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
    this->minGradColorParam.SetParameter(new param::FloatParam( 0.5f, 0.0f));
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
 * protein::QuickSurfRenderer2::~QuickSurfRenderer2 (DTOR)
 */
QuickSurfRenderer2::~QuickSurfRenderer2(void)  {
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
 * protein::QuickSurfRenderer2::release
 */
void QuickSurfRenderer2::release(void) {

}


/*
 * protein::QuickSurfRenderer2::create
 */
bool QuickSurfRenderer2::create(void) {
	if( !glh_init_extensions( "GL_ARB_vertex_program GL_VERSION_2_0") )
        return false;

    if ( !vislib::graphics::gl::GLSLShader::InitialiseExtensions() )
        return false;
    
    cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId() );
    printf( "cudaGLSetGLDevice: %s\n", cudaGetErrorString( cudaGetLastError()));

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
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::cartoon::perpixellight::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for perpixellight shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::cartoon::perpixellight::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for perpixellight shader");
        return false;
    }
    this->lightShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());

    return true;
}


/*
 * protein::QuickSurfRenderer2::GetCapabilities
 */
bool QuickSurfRenderer2::GetCapabilities(Call& call) {
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::AbstractCallRender3D::CAP_RENDER
        | view::AbstractCallRender3D::CAP_LIGHTING
        | view::AbstractCallRender3D::CAP_ANIMATION );

    return true;
}


/*
 * protein::QuickSurfRenderer2::GetExtents
 */
bool QuickSurfRenderer2::GetExtents(Call& call) {
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    MultiParticleDataCall *mol = this->molDataCallerSlot.CallAs<MultiParticleDataCall>();
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

    return true;
}


/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::QuickSurfRenderer2::Render
 */
bool QuickSurfRenderer2::Render(Call& call) {

    // cast the call to Render3D
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to MultiParticleDataCall
    MultiParticleDataCall *c2 = this->molDataCallerSlot.CallAs<MultiParticleDataCall>();
    if( c2 == NULL) return false;

    int cnt;

    // set frame ID and call data
    c2->SetFrameID(static_cast<int>( callTime));
    if (!(*c2)(0)) return false;
    
    
    // set number of particles
    numParticles = 0;
    for( unsigned int i = 0; i < c2->GetParticleListCount(); i++ ) {
        numParticles += c2->AccessParticles(i).GetCount();
    }
    
    // allocate host storage
    if( m_hPos )
        delete[] m_hPos;
    m_hPos = new float[numParticles*3];
    memset(m_hPos, 0, numParticles*3*sizeof(float));

    UINT64 particleCnt = 0;
    for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
        megamol::core::moldyn::MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
        const float *pos = static_cast<const float*>(parts.GetVertexData());
        unsigned int posStride = parts.GetVertexDataStride();
        float globRad = parts.GetGlobalRadius();
        bool useGlobRad = (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
        if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
            continue;
        }
        if (useGlobRad) {
            if (posStride < 12) posStride = 12;
        } else {
            if (posStride < 16) posStride = 16;
        }
        for (UINT64 j = 0; j < parts.GetCount(); j++,
                pos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride)) {
            m_hPos[particleCnt*3+0] = pos[0];
            m_hPos[particleCnt*3+1] = pos[1];
            m_hPos[particleCnt*3+2] = pos[2];
            particleCnt++;
        }
    }

    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

    // ---------- update parameters ----------
    this->UpdateParameters( c2);

    // recompute color table, if necessary
    if( this->atomColorTable.Count()/3 < numParticles ) {
        // Use one coloring mode
        this->atomColorTable.AssertCapacity( numParticles * 3);
        for( unsigned int i = 0; i < numParticles; i++ ) {
            this->atomColorTable.Add( 112.0f/255.0f);
            this->atomColorTable.Add( 146.0f/255.0f);
            this->atomColorTable.Add( 190.0f/255.0f);
        }
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
    for( int i = 0; i < numParticles; i++ ) {
        glColor3fv( &atomColorTable.PeekElements()[3*i]);
        glVertex3fv( &posInter[3*i]);
    }
    glEnd(); // GL_POINTS
    */

    // calculate surface
    if( !cudaqsurf ) {
        cudaqsurf = new CUDAQuickSurf();
    }
    this->lightShader.Enable();
    this->calcSurf( c2, m_hPos, 
        this->qualityParam.Param<param::IntParam>()->Value(),
        this->radscaleParam.Param<param::FloatParam>()->Value(),
        this->gridspacingParam.Param<param::FloatParam>()->Value(),
        this->isovalParam.Param<param::FloatParam>()->Value(),
        true);
    this->lightShader.Disable();

    glPopMatrix();


    // unlock the current frame
    c2->Unlock();

    return true;
}

/*
 * update parameters
 */
void QuickSurfRenderer2::UpdateParameters( const MultiParticleDataCall *mol) {

}


int QuickSurfRenderer2::calcSurf(MultiParticleDataCall *mol, float *posInter,
                         int quality, float radscale, float gridspacing,
                         float isoval, bool useCol) {
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
    //minrad = maxrad = mol->AtomTypes()[mol->AtomTypeIndices()[0]].Radius();
    minrad = maxrad = minGradColorParam.Param<param::FloatParam>()->Value();
    /*
    for ( i = 0; i < numParticles; i++) {
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
    */
    mincoord[0] = minx;
    mincoord[1] = miny;
    mincoord[2] = minz;
    maxcoord[0] = maxx;
    maxcoord[1] = maxy;
    maxcoord[2] = maxz;

    // crude estimate of the grid padding we require to prevent the
    // resulting isosurface from being clipped
    float gridpadding = radscale * maxrad * 1.5;
    float padrad = gridpadding;
    padrad = 0.4 * sqrt(4.0/3.0*M_PI*padrad*padrad*padrad);
    gridpadding = std::max(gridpadding, padrad);

#if VERBOSE
    printf("  Padding radius: %.3f  (minrad: %.3f maxrad: %.3f)\n", 
        gridpadding, minrad, maxrad);
#endif
    gridpadding *= 2.0f,

    // kroneml
    mincoord[0] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    mincoord[1] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    mincoord[2] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    maxcoord[0] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Right();
    maxcoord[1] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Top();
    maxcoord[2] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Front();
    
    mincoord[0] -= gridpadding;
    mincoord[1] -= gridpadding;
    mincoord[2] -= gridpadding;
    maxcoord[0] += gridpadding;
    maxcoord[1] += gridpadding;
    maxcoord[2] += gridpadding;

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
    xyzr = (float *) malloc( numParticles * sizeof(float) * 4);
    if (useCol) {
        colors = (float *) malloc( numParticles * sizeof(float) * 4);

        // build compacted lists of atom coordinates, radii, and colors
        for (i = 0; i < numParticles; i++) {
            const float *fp = posInter + ind;
            xyzr[ind4    ] = fp[0]-origin[0];
            xyzr[ind4 + 1] = fp[1]-origin[1];
            xyzr[ind4 + 2] = fp[2]-origin[2];
            xyzr[ind4 + 3] = minrad;
 
            //const float *cp = &cmap[colidx[i] * 3];
            const float *cp = &this->atomColorTable[ind];
            colors[ind4    ] = cp[0];
            colors[ind4 + 1] = cp[1];
            colors[ind4 + 2] = cp[2];
            colors[ind4 + 3] = 1.0f;

            ind4 += 4;
            ind += 3;
        }
    } else {
        // build compacted lists of atom coordinates and radii only
        for (i = 0; i < numParticles; i++) {
            const float *fp = posInter + ind;
            xyzr[ind4    ] = fp[0]-origin[0];
            xyzr[ind4 + 1] = fp[1]-origin[1];
            xyzr[ind4 + 2] = fp[2]-origin[2];
            xyzr[ind4 + 3] = minrad;
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

    // compute both density map and floating point color texture map
    int rc = cqs->calc_surf( numParticles, &xyzr[0],
        (useCol) ? &colors[0] : &this->atomColorTable[0],
        useCol, origin, numvoxels, maxrad,
        radscale, gridspacing, isovalue, gausslim,
        gpunumverts, gv, gn, gc, gpunumfacets, gf);

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


#endif // WITH_CUDA
