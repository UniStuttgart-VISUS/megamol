/*
 * QuickSurfRenderer2.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define WRITE_DIAGRAM_FILE

#define _USE_MATH_DEFINES 1

#include "QuickSurfRenderer2.h"
#include "MolecularSurfaceFeature.h"
#include "mmcore/CoreInstance.h"
#include "Color.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
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
#include <GL/glu.h>
#include <omp.h>
#include <cfloat>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;
using namespace megamol::core::moldyn;
using namespace megamol::protein_calls;


/*
 * protein_cuda::QuickSurfRenderer2::QuickSurfRenderer2 (CTOR)
 */
QuickSurfRenderer2::QuickSurfRenderer2(void) : Renderer3DModuleDS (),
    molDataCallerSlot( "getData", "Connects the molecule rendering with molecule data storage"),
    areaDiagramCalleeSlot ("areadiagramout", "Provides data for the area line graph"),
    colorTableFileParam( "color::colorTableFilename", "The filename of the color table."),
    interpolParam( "posInterpolation", "Enable positional interpolation between frames" ),
    qualityParam( "quicksurf::quality", "Quality" ),
    radscaleParam( "quicksurf::radscale", "Radius scale" ),
    gridspacingParam( "quicksurf::gridspacing", "Grid spacing" ),
    isovalParam( "quicksurf::isoval", "Isovalue" ),
    twoSidedLightParam( "twoSidedLight", "Turns two-sided lighting on and off" ),
    surfaceColorParam( "surfaceColor", "The color of the surface" ),
    recomputeAreaDiagramParam( "recomputeAreaDiagram", "Recompute the area diagram"),
    m_hPos(0), m_hPosSize(0), numParticles(0), currentSurfaceArea(0.0f), recomputeAreaDiagram(true),
    areaDiagramData(0), callTime(0.0f),
    setCUDAGLDevice(true),
    getClipPlaneSlot("getclipplane", "Connects to a clipping plane module")
{
    this->molDataCallerSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    this->areaDiagramCalleeSlot.SetCallback(DiagramCall::ClassName(), DiagramCall::FunctionName(DiagramCall::CallForGetData), &QuickSurfRenderer2::GetAreaDiagramData);
    this->MakeSlotAvailable(&this->areaDiagramCalleeSlot);

    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    Color::ReadColorTableFromFile( filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(new param::StringParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);

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

    this->twoSidedLightParam.SetParameter( new param::BoolParam(false));
    this->MakeSlotAvailable( &this->twoSidedLightParam);

    // the surface color
    this->surfaceColorParam.SetParameter(new param::StringParam("#7092be"));
    this->MakeSlotAvailable(&this->surfaceColorParam);

    // the recompute area diagram button
    this->recomputeAreaDiagramParam.SetParameter( new param::ButtonParam());
    this->recomputeAreaDiagramParam.SetUpdateCallback(&QuickSurfRenderer2::recomputeAreaDiagramCallback);
    this->MakeSlotAvailable(&this->recomputeAreaDiagramParam);

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

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);
}


/*
 * protein_cuda::QuickSurfRenderer2::~QuickSurfRenderer2 (DTOR)
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
 * protein_cuda::QuickSurfRenderer2::release
 */
void QuickSurfRenderer2::release(void) {

}


/*
 * protein_cuda::QuickSurfRenderer2::create
 */
bool QuickSurfRenderer2::create(void) {
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
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quicksurf::perpixellightVertexClip", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for perpixellight shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quicksurf::perpixellightFragmentClip", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for perpixellight shader");
        return false;
    }
    this->lightShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());

    return true;
}


/*
 * protein_cuda::QuickSurfRenderer2::GetExtents
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
 * protein_cuda::QuickSurfRenderer2::Render
 */
bool QuickSurfRenderer2::Render(Call& call) {

    // cast the call to Render3D
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
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

    callTime = cr3d->Time();
    if (callTime < 1.0f) callTime = 1.0f;

    // get pointer to MultiParticleDataCall
    MultiParticleDataCall *c2 = this->molDataCallerSlot.CallAs<MultiParticleDataCall>();
    if( c2 == NULL) return false;

    // set frame ID and call data
    c2->SetFrameID(static_cast<int>( callTime));
    if (!(*c2)(0)) return false;


    // set number of particles
    numParticles = 0;
    for( unsigned int i = 0; i < c2->GetParticleListCount(); i++ ) {
        numParticles += c2->AccessParticles(i).GetCount();
    }

    // Do nothing if no particles are present
    if (numParticles == 0) {
        return true;
    }

    // allocate host storage
    if (m_hPosSize < numParticles*4) {
        if (m_hPos)
            delete[] m_hPos;
        m_hPos = new float[numParticles*4];
        this->m_hPosSize = numParticles*4;
    }
    memset(m_hPos, 0, numParticles*4*sizeof(float));

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
            m_hPos[particleCnt*4+0] = pos[0];
            m_hPos[particleCnt*4+1] = pos[1];
            m_hPos[particleCnt*4+2] = pos[2];
            if (useGlobRad) {
                m_hPos[particleCnt*4+3] = globRad;
            } else {
                m_hPos[particleCnt*4+3] = pos[3];
            }

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
    if(this->atomColorTable.Count()/3 < numParticles || this->surfaceColorParam.IsDirty()) {
        float r, g, b;
        utility::ColourParser::FromString( this->surfaceColorParam.Param<param::StringParam>()->Value(), r, g, b);
        // Use one coloring mode
        this->atomColorTable.AssertCapacity( numParticles * 3);
        this->atomColorTable.Clear();
        for( unsigned int i = 0; i < numParticles; i++ ) {
            this->atomColorTable.Add( r);
            this->atomColorTable.Add( g);
            this->atomColorTable.Add( b);
        }
        this->surfaceColorParam.ResetDirty();
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

    glDisable(GL_CULL_FACE);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);

    // TODO
    /*
    glBegin( GL_POINTS);
    for( int i = 0; i < numParticles; i++ ) {
        glColor3fv( &atomColorTable.PeekElements()[3*i]);
        glVertex3fv( &posInter[3*i]);
    }
    glEnd(); // GL_POINTS
    */

    float clipDat[4];
    float clipCol[4];
    this->getClipData(clipDat, clipCol);

    // calculate surface
    if( !cudaqsurf ) {
        cudaqsurf = new CUDAQuickSurf();
    }
    this->lightShader.Enable();
    glUniform1i(this->lightShader.ParameterLocation("twoSidedLight"), this->twoSidedLightParam.Param<param::BoolParam>()->Value());
    glUniform4fv(this->lightShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform4fv(this->lightShader.ParameterLocation("clipCol"), 1, clipCol);

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


bool QuickSurfRenderer2::recomputeAreaDiagramCallback(core::param::ParamSlot& slot) {
    this->recomputeAreaDiagram = true;

#ifdef WRITE_DIAGRAM_FILE
    // get pointer to MultiParticleDataCall
    MultiParticleDataCall *parts = this->molDataCallerSlot.CallAs<MultiParticleDataCall>();
    if( parts == NULL) return false;
    // call extent
    if (!(*parts)(1)) return false;

    // check if the data has to be recomputed
    if (this->recomputeAreaDiagram) {
        FILE *areaFile = fopen( "areas.txt", "w");
        // loop over all frames
        for (unsigned int i = 1; i < parts->FrameCount(); i++) {
            parts->SetFrameID(i);
            if (!(*parts)(0)) {
                fclose( areaFile);
                return false;
            }
            // set number of particles
            numParticles = 0;
            for( unsigned int j = 0; j < parts->GetParticleListCount(); j++ ) {
                numParticles += parts->AccessParticles(j).GetCount();
            }
            // allocate host storage
            if (m_hPosSize < numParticles*3) {
                if (m_hPos)
                    delete[] m_hPos;
                m_hPos = new float[numParticles*3];
                this->m_hPosSize = numParticles*3;
            }
            memset(m_hPos, 0, numParticles*3*sizeof(float));
            UINT64 particleCnt = 0;
            for (unsigned int k = 0; k < parts->GetParticleListCount(); k++) {
                megamol::core::moldyn::MultiParticleDataCall::Particles &particles = parts->AccessParticles(k);
                const float *pos = static_cast<const float*>(particles.GetVertexData());
                unsigned int posStride = particles.GetVertexDataStride();
                bool useGlobRad = (particles.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
                if (particles.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
                    continue;
                }
                if (useGlobRad) {
                    if (posStride < 12) posStride = 12;
                } else {
                    if (posStride < 16) posStride = 16;
                }
                for (UINT64 j = 0; j < particles.GetCount(); j++,
                        pos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride)) {
                    m_hPos[particleCnt*3+0] = pos[0];
                    m_hPos[particleCnt*3+1] = pos[1];
                    m_hPos[particleCnt*3+2] = pos[2];
                    particleCnt++;
                }
            }
            // calculate the surface
            if( !cudaqsurf ) {
                cudaqsurf = new CUDAQuickSurf();
            }
            this->calcSurf (parts, m_hPos,
                this->qualityParam.Param<param::IntParam>()->Value(),
                this->radscaleParam.Param<param::FloatParam>()->Value(),
                this->gridspacingParam.Param<param::FloatParam>()->Value(),
                this->isovalParam.Param<param::FloatParam>()->Value(),
                true);
            // write surface area to file
            fprintf( areaFile, "%f\n", this->currentSurfaceArea);
        }
        fclose( areaFile);
    }
#endif

    return true;
}

#pragma warning(push)
#pragma warning(disable : 4258)
bool QuickSurfRenderer2::GetAreaDiagramData(core::Call& call) {

    DiagramCall *dc = dynamic_cast<DiagramCall*>(&call);
    if (dc == NULL) return false;

    // set current call time for sliding guide line
    if (dc->GetGuideCount() == 0) {
        dc->AddGuide(log(callTime), DiagramCall::DIAGRAM_GUIDE_VERTICAL);
    } else {
        dc->GetGuide(0)->SetPosition(log(callTime));
    }

    // get pointer to MultiParticleDataCall
    MultiParticleDataCall *parts = this->molDataCallerSlot.CallAs<MultiParticleDataCall>();
    if( parts == NULL) return false;
    // call extent
    if (!(*parts)(1)) return false;


    // set diagram data series if it was not yet set
    if (dc->GetSeriesCount() == 0) {
        this->areaDiagramData = new DiagramCall::DiagramSeries( "Surface Area",
            new MolecularSurfaceFeature(log(static_cast<float>(parts->FrameCount()))));
        dc->AddSeries( this->areaDiagramData);
    }

    // check if the data has to be recomputed
    if (this->recomputeAreaDiagram) {
        // delete old data series and create new one
        dc->DeleteSeries(this->areaDiagramData);
        delete this->areaDiagramData;
        this->areaDiagramData = new DiagramCall::DiagramSeries( "Surface Area",
            new MolecularSurfaceFeature(log(static_cast<float>(parts->FrameCount()))));
        float r, g, b;
        utility::ColourParser::FromString( this->surfaceColorParam.Param<param::StringParam>()->Value(), r, g, b);
        this->areaDiagramData->SetColor(r, g, b);
        dc->AddSeries( this->areaDiagramData);
        // loop over all frames
        for (unsigned int i = 1; i < parts->FrameCount(); i++) {
            parts->SetFrameID(i);
            if (!(*parts)(0)) return false;
            // set number of particles
            numParticles = 0;
            for( unsigned int i = 0; i < parts->GetParticleListCount(); i++ ) {
                numParticles += parts->AccessParticles(i).GetCount();
            }
            // allocate host storage
            if (m_hPosSize < numParticles*3) {
                if (m_hPos)
                    delete[] m_hPos;
                m_hPos = new float[numParticles*3];
                this->m_hPosSize = numParticles*3;
            }
            memset(m_hPos, 0, numParticles*3*sizeof(float));
            UINT64 particleCnt = 0;
            for (unsigned int i = 0; i < parts->GetParticleListCount(); i++) {
                megamol::core::moldyn::MultiParticleDataCall::Particles &particles = parts->AccessParticles(i);
                const float *pos = static_cast<const float*>(particles.GetVertexData());
                unsigned int posStride = particles.GetVertexDataStride();
                bool useGlobRad = (particles.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
                if (particles.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
                    continue;
                }
                if (useGlobRad) {
                    if (posStride < 12) posStride = 12;
                } else {
                    if (posStride < 16) posStride = 16;
                }
                for (UINT64 j = 0; j < particles.GetCount(); j++,
                        pos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride)) {
                    m_hPos[particleCnt*3+0] = pos[0];
                    m_hPos[particleCnt*3+1] = pos[1];
                    m_hPos[particleCnt*3+2] = pos[2];
                    particleCnt++;
                }
            }
            // recompute color table, if necessary
            if(this->atomColorTable.Count()/3 < numParticles || this->surfaceColorParam.IsDirty()) {
                float r, g, b;
                utility::ColourParser::FromString( this->surfaceColorParam.Param<param::StringParam>()->Value(), r, g, b);
                // Use one coloring mode
                this->atomColorTable.AssertCapacity( numParticles * 3);
                this->atomColorTable.Clear();
                for( unsigned int i = 0; i < numParticles; i++ ) {
                    this->atomColorTable.Add( r);
                    this->atomColorTable.Add( g);
                    this->atomColorTable.Add( b);
                }
                this->surfaceColorParam.ResetDirty();
            }
            // calculate the surface
            if( !cudaqsurf ) {
                cudaqsurf = new CUDAQuickSurf();
            }
            this->calcSurf (parts, m_hPos,
                this->qualityParam.Param<param::IntParam>()->Value(),
                this->radscaleParam.Param<param::FloatParam>()->Value(),
                this->gridspacingParam.Param<param::FloatParam>()->Value(),
                this->isovalParam.Param<param::FloatParam>()->Value(),
                true);
            // set surface area to diagram
            static_cast<MolecularSurfaceFeature*>(this->areaDiagramData->GetMappable())->AppendValue(
                log(static_cast<float>(i)), this->currentSurfaceArea);
        }

        this->recomputeAreaDiagram = false;
    }

    return true;
}
#pragma warning(pop)

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

#define PADDING
#define USE_BOUNDING_BOX

    // compute min/max atom radius, build list of selected atom radii,
    // and compute bounding box for the selected atoms
#ifndef USE_BOUNDING_BOX
    float minx, miny, minz, maxx, maxy, maxz;
#endif
    float minrad, maxrad;
    int i;
    float mincoord[3], maxcoord[3];

    minrad = FLT_MAX;
    maxrad = FLT_MIN;
//    // get min and max radius
//    for (unsigned int i = 0; i < mol->GetParticleListCount(); i++) {
//        megamol::core::moldyn::MultiParticleDataCall::Particles &parts = mol->AccessParticles(i);
//        for (unsigned int j = 0; j < parts.)
//        if( minrad > parts.GetGlobalRadius() ) minrad = parts.GetGlobalRadius();
//        if( maxrad < parts.GetGlobalRadius() ) maxrad = parts.GetGlobalRadius();
//    }

    for (unsigned int i = 0; i < numParticles; ++i)
    {
        if( minrad > posInter[i*4+3]) minrad = posInter[i*4+3];
        if( maxrad < posInter[i*4+3]) maxrad = posInter[i*4+3];
    }

    // crude estimate of the grid padding we require to prevent the
    // resulting isosurface from being clipped
#ifdef PADDING
    float gridpadding = radscale * maxrad * 3.0f;
    float padrad = gridpadding;
    padrad = static_cast<float>(0.4f * sqrt(4.0f/3.0f*M_PI*padrad*padrad*padrad));
    gridpadding = std::max(gridpadding, padrad);
    gridpadding *= 2.0f;
#else
    float gridpadding = 0.0f;
#endif

#if VERBOSE
    printf("  Padding radius: %.3f  (minrad: %.3f maxrad: %.3f)\n",
        gridpadding, minrad, maxrad);
#endif

#ifdef USE_BOUNDING_BOX
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
#endif // VERBOSE
#else // USE_BOUNDING_BOX
    minx = maxx = posInter[0];
    miny = maxy = posInter[1];
    minz = maxz = posInter[2];
    minrad = FLT_MAX;
    maxrad = FLT_MIN;
    // get min and max radius
    for (unsigned int i = 0; i < mol->GetParticleListCount(); i++) {
        megamol::core::moldyn::MultiParticleDataCall::Particles &parts = mol->AccessParticles(i);
        if( minrad > parts.GetGlobalRadius() ) minrad = parts.GetGlobalRadius();
        if( maxrad < parts.GetGlobalRadius() ) maxrad = parts.GetGlobalRadius();
    }

    for ( i = 0; i < numParticles; i++) {
        float tmpx = posInter[i * 3    ];
        float tmpy = posInter[i * 3 + 1];
        float tmpz = posInter[i * 3 + 2];

        minx = (tmpx < minx) ? tmpx : minx;
        maxx = (tmpx > maxx) ? tmpx : maxx;

        miny = (tmpy < miny) ? tmpy : miny;
        maxy = (tmpy > maxy) ? tmpy : maxy;

        minz = (tmpz < minz) ? tmpz : minz;
        maxz = (tmpz > maxz) ? tmpz : maxz;
    }
    mincoord[0] = minx - maxrad * 0.0f;
    mincoord[1] = miny - maxrad * 0.0f;
    mincoord[2] = minz - maxrad * 0.0f;
    maxcoord[0] = maxx + maxrad * 0.0f;
    maxcoord[1] = maxy + maxrad * 0.0f;
    maxcoord[2] = maxz + maxrad * 0.0f;
#endif // USE_BOUNDING_BOX

    // update grid spacing
    float bBoxDim[3] = {maxcoord[0] - mincoord[0], maxcoord[1] - mincoord[1], maxcoord[2] - mincoord[2]};
    float gridDim[3] = {
        ceilf(bBoxDim[0] / gridspacing) + 1,
        ceilf(bBoxDim[1] / gridspacing) + 1,
        ceilf(bBoxDim[2] / gridspacing) + 1};
    float3 gridSpacing3D;
    gridSpacing3D.x = bBoxDim[0] / (gridDim[0] - 1.0f);
    gridSpacing3D.y = bBoxDim[1] / (gridDim[1] - 1.0f);
    gridSpacing3D.z = bBoxDim[2] / (gridDim[2] - 1.0f);
    numvoxels[0] = static_cast<int>(ceilf(bBoxDim[0] / gridSpacing3D.x)) + 1;
    numvoxels[1] = static_cast<int>(ceilf(bBoxDim[1] / gridSpacing3D.y)) + 1;
    numvoxels[2] = static_cast<int>(ceilf(bBoxDim[2] / gridSpacing3D.z)) + 1;

    //vec_copy(origin, mincoord);
    origin[0] = mincoord[0];
    origin[1] = mincoord[1];
    origin[2] = mincoord[2];

    // build compacted lists of bead coordinates, radii, and colors
//    float *xyzr = NULL;
    float *colors = NULL;

    int ind = 0;
    int ind4 = 0;
//    xyzr = (float *) malloc( numParticles * sizeof(float) * 4);
    if (useCol) {
        colors = (float *) malloc( numParticles * sizeof(float) * 4);

        // build compacted lists of atom coordinates, radii, and colors
        for (i = 0; i < numParticles; i++) {
//            const float *fp = posInter + ind;
//            xyzr[ind4    ] = fp[0]-origin[0];
//            xyzr[ind4 + 1] = fp[1]-origin[1];
//            xyzr[ind4 + 2] = fp[2]-origin[2];
//            xyzr[ind4 + 3] = minrad;

            //const float *cp = &cmap[colidx[i] * 3];
            const float *cp = &this->atomColorTable[ind];
            colors[ind4    ] = cp[0];
            colors[ind4 + 1] = cp[1];
            colors[ind4 + 2] = cp[2];
            colors[ind4 + 3] = 1.0f;

            ind4 += 4;
            ind += 3;
        }
    }
//    else {
//        // build compacted lists of atom coordinates and radii only
//        for (i = 0; i < numParticles; i++) {
//            const float *fp = posInter + ind;
//            xyzr[ind4    ] = fp[0]-origin[0];
//            xyzr[ind4 + 1] = fp[1]-origin[1];
//            xyzr[ind4 + 2] = fp[2]-origin[2];
//            xyzr[ind4 + 3] = minrad;
//            ind4 += 4;
//            ind += 3;
//        }
//    }

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
    cqs->useGaussKernel = true;

    // compute both density map and floating point color texture map
    int rc = cqs->calc_surf( static_cast<long>(numParticles), posInter,
        (useCol) ? &colors[0] : &this->atomColorTable[0], useCol, 
        origin, numvoxels, maxrad,
        radscale, gridSpacing3D, isovalue, gausslim,
        gpunumverts, gv, gn, gc, gpunumfacets, gf);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return -1;

    this->currentSurfaceArea = cqs->surfaceArea;

    if (rc == 0) {
        gpuvertexarray = 1;
//        free(xyzr);
        if (colors) free(colors);
        voltime = wkf_timer_timenow(timer);
        return 0;
    } else {
        gpuvertexarray = 0;
//        free(xyzr);
        if (colors) free(colors);
        voltime = wkf_timer_timenow(timer);
        return 1;
    }
}

/*
 * QuickSurfRenderer2::getClipData
 */
void QuickSurfRenderer2::getClipData(float *clipDat, float *clipCol) {
    view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        clipCol[3] = 1.0f;
    }
}
