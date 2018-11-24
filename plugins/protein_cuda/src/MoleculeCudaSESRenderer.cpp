/*
 * MoleculeCudaSESRenderer.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MoleculeCudaSESRenderer.h"

#define _USE_MATH_DEFINES 1

// define the maximum allowed number of atoms in the vicinity of an atom
#define MAX_ATOM_VICINITY 50
// define the maximum allowed number of probes in the vicinity of a probe
#define MAX_PROBE_VICINITY 32
// define the maximum dimension for the visibility fbo
#define VISIBILITY_FBO_DIM 512

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "vislib/graphics/gl/ShaderSource.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <iostream>
#include <algorithm>

#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <cuda_gl_interop.h>
#include "cuda_error_check.h"


extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_cuda;
        
/*
 * MoleculeCudaSESRenderer::MoleculeCudaSESRenderer
 */
MoleculeCudaSESRenderer::MoleculeCudaSESRenderer( void ) : Renderer3DModule(),
    protDataCallerSlot( "getData", "Connects the CUDA SES rendering with molecular data" ),
    interpolParam( "posInterpolation", "Enable positional interpolation between frames" ),
    debugParam( "debugParam", "Debugging parameter" ),
    probeRadiusParam( "probeRadiusParam", "Probe radius" ),
    proteinAtomId( 0), proteinAtomCount( 0),
    visibilityFBO( 0), visibilityColor( 0), visibilityDepth( 0),
    visibilityVertex( 0), visibilityVertexVBO( 0), visibleAtomMask( 0),
    vicinityTable( 0), posInter( 0), fogStart( 0.5f), transparency( 1.0), probeRadius( 1.4f),
    numAtomsPerVoxel( 100), voxelMap( 0), voxelMapSize( 0), vicinityTableTex( 0),
    atomPosRSGS( 0), atomColRSGS( 0),
    visibleAtomsList( 0), visibleAtomsIdList( 0), visibleAtomCount( 0),
    visibleAtomsTex( 0), visibleAtomsIdTex( 0),
    triangleFBO( 0), triangleColor0( 0), triangleColor1( 0), triangleColor2( 0), triangleNormal( 0), triangleDepth( 0),
    triangleVBO( 0), triangleVertex( 0),
    visibleTriangleFBO( 0), visibleTriangleColor( 0), visibleTriangleDepth( 0),
    visibilityTexVBO( 0), query( 0),
    adjacentTriangleFBO( 0), adjacentTriangleColor( 0), adjacentTriangleDepth( 0),
    adjacentTriangleVertex( 0), adjacentTriangleVBO( 0), adjacentTriangleTexVBO( 0),
    singTexData( 0), singTexCoords( 0), singTex( 0), probeVoxelMap( 0),
    delta( 0.01f), first( true), cudaInitalized( false),
    m_dPoint1( 0), m_dPoint2( 0), m_dPoint3( 0), m_dProbePosTable( 0),
    pointIdx( 0), cudaTexResource( 0), visTriaTestVerts( 0),
    cudaVisTriaVboResource( 0), torusVbo( 0),
    m_dProbePos( 0), m_dSortedProbePos( 0), m_dProbeNeighbors( 0), m_dProbeNeighborCount( 0),
    cudaSTriaResource( 0), m_dGridProbeHash( 0), m_dGridProbeIndex( 0)
{
    this->protDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable ( &this->protDataCallerSlot );

    // en-/disable positional interpolation
    this->interpolParam.SetParameter(new param::BoolParam( true));
    this->MakeSlotAvailable( &this->interpolParam);

    // debugging parameter
    this->debugParam.SetParameter(new param::IntParam( 0, 0));
    this->MakeSlotAvailable( &this->debugParam);
    // probe radius parameter
    this->probeRadiusParam.SetParameter(new param::FloatParam( this->probeRadius, 0.0f, 2.4f));
    this->MakeSlotAvailable( &this->probeRadiusParam);
}
/*
 * MoleculeCudaSESRenderer::~MoleculeCudaSESRenderer
 */
MoleculeCudaSESRenderer::~MoleculeCudaSESRenderer(void) {
    this->Release();
}


/*
 * MoleculeCudaSESRenderer::release
 */
void protein_cuda::MoleculeCudaSESRenderer::release( void ) {
    this->drawPointShader.Release();
    this->writeSphereIdShader.Release();
    this->sphereShader.Release();

    if( this->proteinAtomId ) delete[] this->proteinAtomId;
    if( this->visibilityVertex ) delete[] this->visibilityVertex;
    if( this->visibleAtomMask ) delete[] this->visibleAtomMask;
    if( this->vicinityTable ) delete[] this->vicinityTable;
    if( this->voxelMap ) delete[] this->voxelMap;
    if( this->atomPosRSGS ) delete[] this->atomPosRSGS;
    if( this->atomColRSGS ) delete[] this->atomColRSGS;
    if( this->visibleAtomsList ) delete[] this->visibleAtomsList;
    if( this->visibleAtomsIdList ) delete[] this->visibleAtomsIdList;
    if( this->triangleVertex ) delete[] this->triangleVertex;
    if( this->adjacentTriangleVertex ) delete[] this->adjacentTriangleVertex;
    if( this->probeVoxelMap ) delete[] this->probeVoxelMap;
    if( this->singTexData ) delete[] this->singTexData;
    if( this->singTexCoords ) delete[] this->singTexCoords;

    //cudppDestroyPlan( this->sortHandle);
    //cudppDestroyPlan( this->sortHandleProbe);
}


/*
 * MoleculeCudaSESRenderer::create
 */
bool MoleculeCudaSESRenderer::create( void ) {
    if( !areExtsAvailable("GL_EXT_framebuffer_object GL_ARB_texture_float GL_EXT_gpu_shader4 GL_EXT_geometry_shader4 GL_EXT_bindable_uniform")
        || !ogl_IsVersionGEQ(2,0))
        return false;

    if( !isExtAvailable( "GL_NV_transform_feedback") )
        return false;

    if ( !vislib::graphics::gl::GLSLShader::InitialiseExtensions() )
        return false;

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
    
    ShaderSource vertSrc;
    ShaderSource geomSrc;
    ShaderSource fragSrc;

    CoreInstance *ci = this->GetCoreInstance();
    if( !ci ) return false;
    
    ///////////////////////////////////////////////////////////////
    // load the shader source for the sphere Id writing renderer //
    ///////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::writeSphereIdVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere id writing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::writeSphereIdFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere id writing shader", this->ClassName() );
        return false;
    }
    try {
        if( !this->writeSphereIdShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create sphere Id writing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // load the shader source for point drawing renderer (fetch vertex position from texture //
    ///////////////////////////////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawPointVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for point drawing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawPointFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for point drawing shader", this->ClassName() );
        return false;
    }
    try {
        if( !this->drawPointShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create point drawing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ////////////////////////////////////////////////////
    // load the shader source for the sphere renderer //
    ////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein_cuda::ses::sphereVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein_cuda::ses::sphereFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere shader", this->ClassName() );
        return false;
    }
    try {
        if( !this->sphereShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ////////////////////////////////////////////////////////////////
    // load the shader source for the reduced surface computation //
    ////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::reducedSurface2Vertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for reduced surface computation shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::reducedSurface2Fragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for reduced surface computation shader", this->ClassName() );
        return false;
    }
    try {
        if( !this->reducedSurfaceShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create reduced surface computation shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    /////////////////////////////////////////////////////////////////
    // load the shader source for reduced surface triangle drawing //
    /////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawTriangleVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for RS triangle drawing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawTriangleFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for RS triangle drawing shader", this->ClassName() );
        return false;
    }
    try {
        fragSrc.Count();

        if( !this->drawTriangleShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create RS triangle drawing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    /////////////////////////////////////////////////////////
    // load the shader source for visible triangle drawing //
    /////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawVisibleTriangleVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for visible triangle drawing", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawVisibleTriangleGeometry", geomSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load geometry shader source for visible triangle drawing", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawVisibleTriangleFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for visible triangle drawing", this->ClassName() );
        return false;
    }
    try {
        if( !this->drawVisibleTriangleShader.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count()) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        } else {    
            // set varying active
            glActiveVaryingNV( this->drawVisibleTriangleShader, "attribVec1");
            glActiveVaryingNV( this->drawVisibleTriangleShader, "attribVec2");
            glActiveVaryingNV( this->drawVisibleTriangleShader, "attribVec3");
            // set GL_POINTS primitives as INPUT
            this->drawVisibleTriangleShader.SetProgramParameter( GL_GEOMETRY_INPUT_TYPE_EXT , GL_POINTS);
            // set GL_TRIANGLE_STRIP as OUTPUT
            this->drawVisibleTriangleShader.SetProgramParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_POINTS);
            // set maximum number of vertices to be generated by geometry shader
            this->drawVisibleTriangleShader.SetProgramParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 20);
            // link the shader
            if( !this->drawVisibleTriangleShader.Link() ) {
                throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
            }
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create visible triangle drawing geometry shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    ///////////////////////////////////////////////////
    // load the shader source for the torus renderer //
    ///////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein_cuda::ses::torusVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for torus shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein_cuda::ses::torusFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for torus shader", this->ClassName() );
        return false;
    }
    try {
        if( !this->torusShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create torus shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ////////////////////////////////////////////////////////////////
    // load the shader source for the spherical triangle renderer //
    ////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein_cuda::ses::sphericaltriangleVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for spherical triangle shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein_cuda::ses::sphericaltriangleFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for spherical triangle shader", this->ClassName() );
        return false;
    }
    try {
        if( !this->sphericalTriangleShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create spherical triangle shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    //////////////////////////////////////////////////////////
    // load the shader source for adjacent triangle finding //
    //////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentTriangleVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for adjacent triangle finding shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentTriangleFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for adjacent triangle finding shader", this->ClassName() );
        return false;
    }
    try {
        fragSrc.Count();

        if( !this->adjacentTriangleShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create adjacent triangle finding shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    //////////////////////////////////////////////////////
    // load the shader source for adjacent atom finding //
    //////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentAtomVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for adjacent atom marking shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentAtomFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for adjacent atom marking shader", this->ClassName() );
        return false;
    }
    try {
        fragSrc.Count();

        if( !this->adjacentAtomsShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create adjacent atom marking shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    /////////////////////////////////////////////////////////////////
    // load the shader source for reduced surface triangle drawing //
    /////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawCUDATriangleVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for CUDA triangle drawing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawCUDATriangleFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for CUDA triangle drawing shader", this->ClassName() );
        return false;
    }
    try {
        fragSrc.Count();

        if( !this->drawCUDATriangleShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create CUDA triangle drawing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    ////////////////////////////////////////////////////////////////
    // load the shader source for visible triangle index emission //
    ////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::visibleTriangleIdxVertex", vertSrc ) ) {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for visible triangle index emission", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::visibleTriangleIdxGeometry", geomSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load geometry shader source for visible triangle index emission", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::visibleTriangleIdxFragment", fragSrc ) ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for visible triangle index emission", this->ClassName() );
        return false;
    }
    try {
        if( !this->visibleTriangleIdxShader.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count()) ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        } else {
            // set GL_POINTS primitives as INPUT
            this->visibleTriangleIdxShader.SetProgramParameter( GL_GEOMETRY_INPUT_TYPE_EXT , GL_POINTS);
            // set GL_POINTS as OUTPUT
            this->visibleTriangleIdxShader.SetProgramParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_POINTS);
            // set maximum number of vertices to be generated by geometry shader
            this->visibleTriangleIdxShader.SetProgramParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 1);
            // link the shader
            if( !this->visibleTriangleIdxShader.Link() ) {
                throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
            }
        }
    } catch( vislib::Exception e ) {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create visible triangle index emission geometry shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    // everything went well, return success
    return true;
}


/*
 * Initialize CUDA
 */
bool MoleculeCudaSESRenderer::initCuda( MolecularDataCall *protein, uint gridDim, view::CallRender3D *cr3d) {
    // do not initialize twice!
    // TODO: make this better...
    if( this->cudaInitalized ) return true;
	// set number of atoms
	this->numAtoms = protein->AtomCount();

	// use CUDA device with highest Gflops/s
	//cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    
#ifdef _WIN32
    if( cr3d->IsGpuAffinity() ) {
        HGPUNV gpuId = cr3d->GpuAffinity<HGPUNV>();
        int devId;
        cudaWGLGetDevice( &devId, gpuId);
        cudaGLSetGLDevice( devId);
    } else {
		cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    }
#else
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId());
#endif
    printf( "cudaGLSetGLDevice: %s\n", cudaGetErrorString( cudaGetLastError()));

	// set grid dimensions
    this->gridSize.x = this->gridSize.y = this->gridSize.z = gridDim;
    this->numGridCells = this->gridSize.x * this->gridSize.y * this->gridSize.z;
    //float3 worldSize = make_float3( 2.0f, 2.0f, 2.0f);
	float3 worldSize = make_float3(
        protein->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
		protein->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
		protein->AccessBoundingBoxes().ObjectSpaceBBox().Depth() );
    this->gridSortBits = 16;    // increase this for larger grids

    // set parameters
    this->params.gridSize = this->gridSize;
    this->params.numCells = this->numGridCells;
    this->params.numBodies = this->numAtoms;
    //this->params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	this->params.worldOrigin = make_float3(
		protein->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetX(),
		protein->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetY(),
		protein->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetZ());
	this->params.cellSize = make_float3( worldSize.x / this->gridSize.x, worldSize.y / this->gridSize.y, worldSize.z / this->gridSize.z);
	this->params.probeRadius = this->probeRadius;
    this->params.maxNumNeighbors = MAX_ATOM_VICINITY;

    // allocate host storage
    //m_hPos = new float[this->numAtoms*4];
	cudaMallocHost( (void**)&m_hPos, this->numAtoms*4);
    memset(m_hPos, 0, this->numAtoms*4*sizeof(float));

	m_hNeighborCount = new uint[this->numAtoms];
	memset( m_hNeighborCount, 0, this->numAtoms*sizeof(uint));

	m_hNeighbors = new uint[this->numAtoms*MAX_ATOM_VICINITY];
	memset( m_hNeighbors, 0, this->numAtoms*MAX_ATOM_VICINITY*sizeof(uint));

    m_hParticleIndex = new uint[this->numAtoms];
	memset( m_hParticleIndex, 0, this->numAtoms*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * this->numAtoms;

	// array for atom positions
    allocateArray((void**)&m_dPos, memSize);
	// array for sorted atom positions
    allocateArray((void**)&m_dSortedPos, memSize);
	// array for the counted number of atoms
	allocateArray((void**)&m_dNeighborCount, this->numAtoms*sizeof(uint));
	// array for the neighbor atoms
	allocateArray((void**)&m_dNeighbors, this->numAtoms*MAX_ATOM_VICINITY*sizeof(uint));

    allocateArray((void**)&m_dGridParticleHash, this->numAtoms*sizeof(uint));
    allocateArray((void**)&m_dGridParticleIndex, this->numAtoms*sizeof(uint));

    allocateArray((void**)&m_dCellStart, this->numGridCells*sizeof(uint));
    allocateArray((void**)&m_dCellEnd, this->numGridCells*sizeof(uint));

    // Create the CUDPP radix sort
    //CUDPPConfiguration sortConfig;
    //sortConfig.algorithm = CUDPP_SORT_RADIX;
    //sortConfig.datatype = CUDPP_UINT;
    //sortConfig.op = CUDPP_ADD;
    //sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    //cudppPlan( &this->sortHandle, sortConfig, this->numAtoms, 1, 0);

	setParameters( &this->params);

    //cudaError e;
    //e = cudaGetLastError();

	return true;
}


/*
 * MoleculeCudaSESRenderer::GetExtents
 */
bool MoleculeCudaSESRenderer::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    MolecularDataCall *protein = this->protDataCallerSlot.CallAs<MolecularDataCall>();
    if (protein == NULL) return false;
    if (!(*protein)(MolecularDataCall::CallForGetExtent)) return false;

    float scale;
    if( !vislib::math::IsEqual( protein->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / protein->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = protein->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld( scale);
    cr3d->SetTimeFramesCount( protein->FrameCount());

    return true;
}


/*
 * MoleculeCudaSESRenderer::Render
 */
bool MoleculeCudaSESRenderer::Render( Call& call ) {

    // update probe radius
    this->probeRadius = this->probeRadiusParam.Param<param::FloatParam>()->Value();
    /*
    // change the probe size over time
    if( this->probeRadius > 2.5f )
        delta = -delta;
    else if( this->probeRadius < 0.95f )
        delta = -delta;
    this->probeRadius += delta;
    */
	this->params.probeRadius = this->probeRadius;

    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;
    
    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();
    // get the call time
    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    MolecularDataCall *protein = this->protDataCallerSlot.CallAs<MolecularDataCall>();
    // if something went wrong --> return
    if( !protein) return false;
    
    // set frame ID and call data
    protein->SetFrameID(static_cast<int>( callTime));
    if ( !(*protein)(MolecularDataCall::CallForGetData) )
        return false;
    // check if atom count is zero
    if( protein->AtomCount() == 0 ) return true;
    // get positions of the first frame
    int cnt;
    float *pos0 = new float[protein->AtomCount() * 3];
    memcpy( pos0, protein->AtomPositions(), protein->AtomCount() * 3 * sizeof( float));
    // check if the atom positions have to be interpolated
    bool interpolate = this->interpolParam.Param<param::BoolParam>()->Value();
    float *pos1 = NULL;
    if( interpolate ) {
        // set next frame ID and get positions of the second frame
        if( ( static_cast<int>( callTime) + 1) < int(protein->FrameCount()) ) 
            protein->SetFrameID(static_cast<int>( callTime) + 1);
        else
            protein->SetFrameID(static_cast<int>( callTime));
        if (!(*protein)(MolecularDataCall::CallForGetData)) {
            delete[] pos0;
            return false;
        }
        pos1 = new float[protein->AtomCount() * 3];
        memcpy( pos1, protein->AtomPositions(), protein->AtomCount() * 3 * sizeof( float));
        // interpolate atom positions between frames
        posInter = new float[protein->AtomCount() * 3];
        float inter = callTime - static_cast<float>(static_cast<int>( callTime));
        float threshold = vislib::math::Min( protein->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
            vislib::math::Min( protein->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
            protein->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
        for( cnt = 0; cnt < int(protein->AtomCount()); ++cnt ) {
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
    
	// try to initialize CUDA
	if( !this->cudaInitalized ) {
        cudaInitalized = this->initCuda( protein, 16, cr3d);
		vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO, 
			"%s: CUDA initialization: %i", this->ClassName(), cudaInitalized );
	}

    // get bounding box of the protein
    this->bBox = protein->AccessBoundingBoxes().ObjectSpaceBBox();
    
    // get clear color
    glGetFloatv( GL_COLOR_CLEAR_VALUE, this->clearCol);

    // ==================== Scale & Translate ====================
    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if( !vislib::math::IsEqual( protein->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) { 
        scale = 2.0f / protein->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef( scale, scale, scale);

    // =============== Query Camera View Dimensions ===============
    if( static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height ) {
        this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
        this->CreateVisibilityFBO( VISIBILITY_FBO_DIM);
    }

    // ========================== Render ==========================
    glDisable( GL_BLEND);

//time_t //t = clock();
    // Step 1: Find all visible atoms (using GPU-based visibility query)
	this->FindVisibleAtoms( protein); // GTX260, 1m40: 480 fps
//std::cout << "FindVisibleAtoms: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
//t = clock();

    // Step 2a: Compute the vicinity table for all visible atoms
    this->ComputeVicinityTableCUDA( protein); // GTX260, 1m40: 135 fps
//std::cout << "ComputeVicinityTableCUDA: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
//t = clock();

    // Step 2b: Compute the Reduced Surface via Fragment Shader
    this->ComputeRSCuda( protein); // GTX260, 1m40: 85 fps
//std::cout << "ComputeRSCuda: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
//t = clock();

    // Step 3a: Find all visible potential RS-faces
    this->FindVisibleTrianglesCuda( protein); // GTX260, 1m40: 50 fps
//std::cout << "FindVisibleTrianglesCuda: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
//t = clock();

    // Step 3b: Find adjacent, occluded RS-faces
    this->FindAdjacentTrianglesCUDA( protein);
//std::cout << "FindAdjacentTrianglesCUDA: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
//t = clock();

    // Step 4a: Extract geometric primitives for ray casting
	unsigned int primitiveCount = this->CreateGeometricPrimitivesCuda( protein);
//std::cout << "CreateGeometricPrimitivesCuda: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
//t = clock();

    // Step 4b: Search intersecting probes for singularity handling
    this->CreateSingularityTextureCuda( protein, primitiveCount*2);
//std::cout << "CreateSingularityTextureCuda: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
//t = clock();

    // Step 5: Render the SES using GPU ray casting
    this->RenderSESCuda( protein, primitiveCount);
//std::cout << "RenderSESCuda: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;

    // check for GL errors
    CHECK_FOR_OGL_ERROR();

    // reset clear color
    glClearColor( clearCol[0], clearCol[1], clearCol[2], clearCol[3]);

    // START draw overlay
#if 0
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_LIGHTING);
    glLoadIdentity();
    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    glEnable( GL_TEXTURE_2D);
    glBindTexture( GL_TEXTURE_2D, this->visibleTriangleColor);
    //glBindTexture( GL_TEXTURE_2D, this->singTex);
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    float y = 0.5f;
    glBegin(GL_QUADS);
    glTexCoord2f( 0, 0);
    glVertex2f( 0, 0);
    glTexCoord2f( 1, 0);
    glVertex2f( 1, 0);
    glTexCoord2f( 1, 1);
    glVertex2f( 1, y);
    glTexCoord2f( 0, 1);
    glVertex2f( 0, y);
    glEnd();
    //glBindTexture( GL_TEXTURE_2D, this->triangleColor0);
    //glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    //glBegin(GL_QUADS);
    //glTexCoord2f( 0, 0);
    //glVertex2f( 0, 0);
    //glTexCoord2f( 1, 0);
    //glVertex2f( 1, 0);
    //glTexCoord2f( 1, 1);
    //glVertex2f( 1, 0.3f);
    //glTexCoord2f( 0, 1);
    //glVertex2f( 0, 0.3f);
    //glEnd();
    //glBindTexture( GL_TEXTURE_2D, this->triangleColor1);
    //glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    //glBegin(GL_QUADS);
    //glTexCoord2f( 0, 0);
    //glVertex2f( 0, 0.35f);
    //glTexCoord2f( 1, 0);
    //glVertex2f( 1, 0.35f);
    //glTexCoord2f( 1, 1);
    //glVertex2f( 1, 0.65f);
    //glTexCoord2f( 0, 1);
    //glVertex2f( 0, 0.65f);
    //glEnd();
    //glBindTexture( GL_TEXTURE_2D, this->triangleColor2);
    //glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    //glBegin(GL_QUADS);
    //glTexCoord2f( 0, 0);
    //glVertex2f( 0, 0.7f);
    //glTexCoord2f( 1, 0);
    //glVertex2f( 1, 0.7f);
    //glTexCoord2f( 1, 1);
    //glVertex2f( 1, 1);
    //glTexCoord2f( 0, 1);
    //glVertex2f( 0, 1);
    //glEnd();
    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable( GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
#endif

    // ================= Undo Scale & Translate =================
    glPopMatrix();

    glEnable( GL_BLEND);

    delete[] pos0;
    if( interpolate ) {
        delete[] pos1;
        delete[] posInter;
    }
    posInter = 0;

    // everything went well
    return true;
}

/*
 * MoleculeCudaSESRenderer::deinitialise
 */
void MoleculeCudaSESRenderer::deinitialise(void)
{
}


/*
 * renders all atoms using GPU ray casting and write atom ID
 */
void MoleculeCudaSESRenderer::RenderAtomIdGPU( MolecularDataCall *protein) {
    const float *positions = protein->AtomPositions();
    if( posInter ) positions = posInter;
    // initialize Id array if necessary
    if( protein->AtomCount() != this->proteinAtomCount ) {
        // set correct number of protein atoms
        this->proteinAtomCount = protein->AtomCount();
        // create the FBO for atom visibility testing
        this->CreateVisibleAtomsFBO( this->proteinAtomCount);
        // delete old Id array, if available
        if( this->proteinAtomId )
            delete[] this->proteinAtomId;
        // create new Id array
        this->proteinAtomId = new float[this->proteinAtomCount*3];
        // fill the array
        for( unsigned int cnt = 0; cnt < this->proteinAtomCount; ++cnt ) {
            this->proteinAtomId[cnt*3+0] = float( cnt);
            this->proteinAtomId[cnt*3+1] = protein->AtomTypes()[protein->AtomTypeIndices()[cnt]].Radius();
        }
        // resize the lists for visible atoms and the Ids of the visible atoms
        if( this->visibleAtomsList )
            delete[] this->visibleAtomsList;
        this->visibleAtomsList = new float[this->proteinAtomCount*4];
        if( this->visibleAtomsIdList )
            delete[] this->visibleAtomsIdList;
        this->visibleAtomsIdList = new unsigned int[this->proteinAtomCount];
        this->visibleAtomCount = 0;
    }
	// -----------
	// -- draw  --
	// -----------
	float viewportStuff[4] = {
		cameraInfo->TileRect().Left(),
		cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(),
		cameraInfo->TileRect().Height()
	};
	if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];
	
    glDisable( GL_LIGHTING);
    glEnable( GL_DEPTH_TEST);

	// enable sphere shader
	this->writeSphereIdShader.Enable();

	// set shader variables
    glUniform4fv( this->writeSphereIdShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->writeSphereIdShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->writeSphereIdShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->writeSphereIdShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    glUniform3f( this->writeSphereIdShader.ParameterLocation( "zValues"), 1.0f, cameraInfo->NearClip(), cameraInfo->FarClip());

	glEnableClientState( GL_VERTEX_ARRAY);
	glEnableClientState( GL_COLOR_ARRAY);
	// set vertex and color pointers and draw them
    glColorPointer( 3, GL_FLOAT, 0, this->proteinAtomId);
    //glVertexPointer( 3, GL_FLOAT, 0, protein->AtomPositions());
    glVertexPointer( 3, GL_FLOAT, 0, positions);
    glDrawArrays( GL_POINTS, 0, protein->AtomCount());
	// disable sphere shader
	glDisableClientState( GL_COLOR_ARRAY);
	glDisableClientState( GL_VERTEX_ARRAY);

	// disable sphere shader
	this->writeSphereIdShader.Disable();
}


/*
 * create the FBO for visibility test
 */
void MoleculeCudaSESRenderer::CreateVisibilityFBO( unsigned int maxSize) {
    // compute texture dimensions
    if( this->width > this->height )
    {
        this->visibilityTexWidth = maxSize;
        this->visibilityTexHeight = (unsigned int)( float(maxSize) * ( float(this->height) / float(this->width)));
    }
    else
    {
        this->visibilityTexWidth = (unsigned int)( float(maxSize) * ( float(this->width) / float(this->height)));
        this->visibilityTexHeight = maxSize;
    }
    // delete FBO & textures, if necessary
    if( this->visibilityFBO )
        glDeleteFramebuffersEXT( 1, &this->visibilityFBO);
    if( this->visibilityColor )
        glDeleteTextures( 1, &this->visibilityColor);
    if( this->visibilityDepth )
        glDeleteTextures( 1, &this->visibilityDepth);
    // generate FBO & textures
    glGenFramebuffersEXT( 1, &visibilityFBO);
    glGenTextures( 1, &this->visibilityColor);
    glGenTextures( 1, &this->visibilityDepth);
    // color and depth FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->visibilityFBO);
    // init texture0 (color)
    glBindTexture( GL_TEXTURE_2D, this->visibilityColor);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, this->visibilityTexWidth, this->visibilityTexHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture( GL_TEXTURE_2D, 0);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->visibilityColor, 0);
    // init depth texture
    glBindTexture( GL_TEXTURE_2D, this->visibilityDepth);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, this->visibilityTexWidth, this->visibilityTexHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->visibilityDepth, 0);
    glBindTexture( GL_TEXTURE_2D, 0);

    // unbind the fbo
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    CHECK_FOR_OGL_ERROR();

    // (re)create the vertex array for drawing the visibility texture
    if( this->visibilityVertex )
        delete[] this->visibilityVertex;
    this->visibilityVertex = new float[this->visibilityTexWidth*this->visibilityTexHeight*3];
    // fill the vertices with the coordinates
    unsigned int counter = 0;
    for( unsigned int cntX = 0; cntX < this->visibilityTexWidth; ++cntX ) {
        for( unsigned int cntY = 0; cntY < this->visibilityTexHeight; ++cntY ) {
            this->visibilityVertex[counter*3+0] = float( cntX);
            this->visibilityVertex[counter*3+1] = float( cntY);
            this->visibilityVertex[counter*3+2] = 0.0f;
            counter++;
        }
    }
    // generate the VBO, if necessary
    if( !glIsBuffer( this->visibilityVertexVBO) )
        glGenBuffers( 1, &this->visibilityVertexVBO);
    // fill VBO
    glBindBuffer( GL_ARRAY_BUFFER, this->visibilityVertexVBO);
    glBufferData( GL_ARRAY_BUFFER, counter*3*sizeof( float), this->visibilityVertex, GL_STATIC_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    
    // create VBO
    if( !glIsBuffer( this->visibilityTexVBO) )
        glGenBuffers( 1, &this->visibilityTexVBO);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, visibilityTexVBO);
    glBufferData( GL_PIXEL_PACK_BUFFER_EXT, this->visibilityTexWidth*this->visibilityTexHeight*4*sizeof(float), NULL, GL_DYNAMIC_DRAW_ARB );
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0);
}

/*
 * create the FBO for visible atoms
 */
void MoleculeCudaSESRenderer::CreateVisibleAtomsFBO( unsigned int atomCount) {
    // (re)create visible atoms mask array
    if( this->visibleAtomMask )
        delete[] this->visibleAtomMask;
    this->visibleAtomMask = new float[atomCount];
    // delete FBO & textures, if necessary
    if( this->visibleAtomsFBO )
        glDeleteFramebuffersEXT( 1, &this->visibleAtomsFBO);
    if( this->visibleAtomsColor )
        glDeleteTextures( 1, &this->visibleAtomsColor);
    if( this->visibleAtomsDepth )
        glDeleteTextures( 1, &this->visibleAtomsDepth);
    // generate FBO & textures
    glGenFramebuffersEXT( 1, &this->visibleAtomsFBO);
    glGenTextures( 1, &this->visibleAtomsColor);
    glGenTextures( 1, &this->visibleAtomsDepth);
    // color and depth FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->visibleAtomsFBO);
    // init texture0 (color)
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsColor);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, atomCount, 1, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture( GL_TEXTURE_2D, 0);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->visibleAtomsColor, 0);
    // init depth texture
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsDepth);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, atomCount, 1, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glBindTexture( GL_TEXTURE_2D, 0);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->visibleAtomsDepth, 0);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}


/*
 * creates the FBO for reduced surface triangle generation
 */
void MoleculeCudaSESRenderer::CreateTriangleFBO( unsigned int atomCount, unsigned int vicinityCount) {

    if( !this->triangleFBO )
    {
        // generate FBO
        glGenFramebuffersEXT( 1, &this->triangleFBO);
        // generate textures, if necessary
        if( !glIsTexture( this->triangleColor0) )
            glGenTextures( 1, &this->triangleColor0);
        if( !glIsTexture( this->triangleColor1) )
            glGenTextures( 1, &this->triangleColor1);
        if( !glIsTexture( this->triangleColor2) )
            glGenTextures( 1, &this->triangleColor2);
        if( !glIsTexture( this->triangleNormal) )
            glGenTextures( 1, &this->triangleNormal);
        if( !glIsTexture( this->triangleDepth) )
            glGenTextures( 1, &this->triangleDepth);
        
        // color and depth FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->triangleFBO);
        // init color texture 0
        glBindTexture( GL_TEXTURE_2D, this->triangleColor0);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, vicinityCount*vicinityCount, atomCount, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture( GL_TEXTURE_2D, 0);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->triangleColor0, 0);
        // init color texture 1
        glBindTexture( GL_TEXTURE_2D, this->triangleColor1);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, vicinityCount*vicinityCount, atomCount, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture( GL_TEXTURE_2D, 0);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, this->triangleColor1, 0);
        // init color texture 2
        glBindTexture( GL_TEXTURE_2D, this->triangleColor2);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, vicinityCount*vicinityCount, atomCount, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture( GL_TEXTURE_2D, 0);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, this->triangleColor2, 0);
        // init color texture 3 for storing normals
        glBindTexture( GL_TEXTURE_2D, this->triangleNormal);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, vicinityCount*vicinityCount, atomCount, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture( GL_TEXTURE_2D, 0);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT3_EXT, GL_TEXTURE_2D, this->triangleNormal, 0);
        // init depth texture
        glBindTexture( GL_TEXTURE_2D, this->triangleDepth);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, vicinityCount*vicinityCount, atomCount, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->triangleDepth, 0);
        glBindTexture( GL_TEXTURE_2D, 0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    }
}


/*
 * creates the FBO for reduced surface triangle generation
 */
void MoleculeCudaSESRenderer::CreateVisibleTriangleFBO( unsigned int atomCount, unsigned int vicinityCount) {
    
    if( !glIsFramebufferEXT( this->visibleTriangleFBO) )
    {
        // generate FBO
        glGenFramebuffersEXT( 1, &this->visibleTriangleFBO);
        // generate textures, if necessary
        if( !glIsTexture( this->visibleTriangleColor) )
            glGenTextures( 1, &this->visibleTriangleColor);
        if( !glIsTexture( this->visibleTriangleDepth) )
            glGenTextures( 1, &this->visibleTriangleDepth);
        
        // color and depth FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->visibleTriangleFBO);
        // init color texture
        glBindTexture( GL_TEXTURE_2D, this->visibleTriangleColor);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F, vicinityCount*vicinityCount, atomCount, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture( GL_TEXTURE_2D, 0);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->visibleTriangleColor, 0);
        // init depth texture
        glBindTexture( GL_TEXTURE_2D, this->visibleTriangleDepth);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, vicinityCount*vicinityCount, atomCount, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->visibleTriangleDepth, 0);
        glBindTexture( GL_TEXTURE_2D, 0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    }
}


/*
 * Find all visible atoms
 */
void MoleculeCudaSESRenderer::FindVisibleAtoms( MolecularDataCall *protein) {
    // counter
    unsigned int cnt;

    // use interpolated positions, if availabe
    const float *positions = protein->AtomPositions();
    if( posInter ) positions = posInter;

    // ---------- render all atoms to a FBO with color set to atom Id ----------
    // set viewport
    glViewport( 0, 0, this->visibilityTexWidth, this->visibilityTexHeight);
    // set camera information
    this->cameraInfo->SetVirtualViewSize( (float)this->visibilityTexWidth, (float)this->visibilityTexHeight);
    // start rendering to visibility FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibilityFBO);
    // set clear color & clear
    glClearColor( -1.0f, 0.0f, 0.0f, 1.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // render all atoms as spheres and write the Id to the red color channel
    this->RenderAtomIdGPU( protein);
    // read FBO to VBO
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, this->visibilityTexVBO);
    glReadPixels( 0, 0, this->visibilityTexWidth, this->visibilityTexHeight, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0 );
    // stop rendering to FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);
    // set camera information
    this->cameraInfo->SetVirtualViewSize( (float)this->width, (float)this->height);

    //glDisable( GL_DEPTH_TEST );

    // ---------- create a histogram of visible atom Ids ----------
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glOrtho( 0.0, double( protein->AtomCount()), 0.0, 1.0, 0.0, 1.0);
    glDisable( GL_LIGHTING);

    // set viewport, get and set clearcolor, start rendering to framebuffer
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
    glViewport( 0, 0, protein->AtomCount(), 1);
    // render to framebuffer
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibleAtomsFBO);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    /*
    // bind texture
    glBindTexture( GL_TEXTURE_2D, this->visibilityColor);
    // enable and set up point drawing shader
    this->drawPointShader.Enable();
    glUniform1i( this->drawPointShader.ParameterLocation( "positionTex"), 0);
	// set vertex pointer and draw it
	glEnableClientState( GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, this->visibilityVertexVBO);
    glVertexPointer( 3, GL_FLOAT, 0, 0 );
    glDrawArrays( GL_POINTS, 0, this->visibilityTexWidth*this->visibilityTexHeight);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
	glDisableClientState( GL_VERTEX_ARRAY);
    this->drawPointShader.Disable();
    glBindTexture( GL_TEXTURE_2D, 0);
    */

    // render visibility texture VBO
    glColor3f( 1.0f, 1.0f, 1.0f);
    glBindBuffer( GL_ARRAY_BUFFER, this->visibilityTexVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer( 2, GL_FLOAT, sizeof(float)*4, (float*)0);
    glDrawArrays( GL_POINTS, 0, this->visibilityTexWidth*this->visibilityTexHeight);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    
    // stop drawing to fbo
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    //glEnable( GL_DEPTH_TEST );

    // ---------- use the histogram for generating a list of visible atoms ----------
    // read the visible atoms texture (visibility mask)
    memset( this->visibleAtomMask, 0, sizeof(float)*protein->AtomCount());
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsColor);
    glGetTexImage( GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, this->visibleAtomMask);
    glBindTexture( GL_TEXTURE_2D, 0);
    // TODO test alternative method:
    //    - render one vertex per atom
    //    - use GeomShader to emit only visible atoms
    //    - use TransformFeedback to get atom positions

    // copy only the visible atom
    this->visibleAtomCount = 0;
    for( cnt = 0; cnt < protein->AtomCount(); ++cnt ) {
        // check for atom visibility in mask
        if( this->visibleAtomMask[cnt] > 0.5f ) {
            //  write atoms pos (x,y,z) and radius
            this->visibleAtomsList[this->visibleAtomCount*4+0] = positions[cnt*3+0];
            this->visibleAtomsList[this->visibleAtomCount*4+1] = positions[cnt*3+1];
            this->visibleAtomsList[this->visibleAtomCount*4+2] = positions[cnt*3+2];
            this->visibleAtomsList[this->visibleAtomCount*4+3] = protein->AtomTypes()[protein->AtomTypeIndices()[cnt]].Radius();
            this->visibleAtomsIdList[this->visibleAtomCount] = cnt;
            // write atom Id
            this->visibleAtomCount++;
        }
    }

    //// generate textures for visible atoms, if necessary
    //if( !glIsTexture( this->visibleAtomsTex) )
    //    glGenTextures( 1, &this->visibleAtomsTex);
    //if( !glIsTexture( this->visibleAtomsIdTex) )
    //    glGenTextures( 1, &this->visibleAtomsIdTex);
    //// create visible atoms texture
    //glBindTexture( GL_TEXTURE_2D, this->visibleAtomsTex);
    //glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGBA16F, visibleAtomCount, 1, 0, GL_RGBA, GL_FLOAT, this->visibleAtomsList);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glBindTexture( GL_TEXTURE_2D, 0);
    //// create visible atoms Id texture
    //glBindTexture( GL_TEXTURE_2D, visibleAtomsIdTex);
    //glTexImage2D( GL_TEXTURE_2D, 0,  GL_R16F, visibleAtomCount, 1, 0, GL_R, GL_UNSIGNED_INT, this->visibleAtomsIdList); NOT TESTED!!!
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glBindTexture( GL_TEXTURE_2D, 0);
}


/*
 * Find for each visible atom all atoms that are in the proximity.
 */
void MoleculeCudaSESRenderer::ComputeVisibleAtomsVicinityTable( MolecularDataCall *protein) {
    // use interpolated positions, if availabe
    const float *positions = protein->AtomPositions();
    if( posInter ) positions = posInter;

    // temporary variables
    unsigned int cnt1, cnt2, vicinityCounter;
    vislib::math::Vector<float, 3> tmpVec, pos;
    float radius, rad, distance, threshold;
	unsigned int cnt, maxXId, maxYId, maxZId;
	int cntX, cntY, cntZ, xId, yId, zId;
    unsigned int xCoord, yCoord, zCoord, voxelIdx, firstIdx, atomCnt;
    
    // set up vicinity table, if necessary --> reserve one aditional vec for # of vicinity atoms
    if( !this->vicinityTable )
        this->vicinityTable = new float[protein->AtomCount() * ( MAX_ATOM_VICINITY + 1) * 4];
    // reset all items in vicinity table to zero
    memset( this->vicinityTable, 0, sizeof(float) * protein->AtomCount() * ( MAX_ATOM_VICINITY + 1) * 4);

	// set voxel length, if necessary --> diameter of the probe + maximum atom diameter
    unsigned int tmpVoxelLength = (unsigned int)(2.0f * this->probeRadius + 6.0f);
    if( tmpVoxelLength != this->voxelLength ) {
        this->voxelLength = tmpVoxelLength;
    }

    // compute bounding box dimensions
    unsigned int vW = (unsigned int)ceilf( this->bBox.Width() / float(this->voxelLength));
    unsigned int vH = (unsigned int)ceilf( this->bBox.Height() / float(this->voxelLength));
    unsigned int vD = (unsigned int)ceilf( this->bBox.Depth() / float(this->voxelLength));
    unsigned int dimensions = vW * vH * vD;
    maxXId = vW - 1;
    maxYId = vH - 1;
    maxZId = vD - 1;
    
    // resize voxel map, if necessary
    if( ( dimensions * this->numAtomsPerVoxel ) > this->voxelMapSize ) {
        if( this->voxelMap )
            delete[] this->voxelMap;
        this->voxelMapSize = dimensions;
        this->voxelMap = new unsigned int[this->voxelMapSize * this->numAtomsPerVoxel];
    }
    memset( this->voxelMap, 0, sizeof(unsigned int)*this->voxelMapSize*this->numAtomsPerVoxel);
    
    for( cnt = 0; cnt < protein->AtomCount(); ++cnt )
    {
        // get position of current atom
        tmpVec.SetX( positions[cnt*3+0]);
        tmpVec.SetY( positions[cnt*3+1]);
        tmpVec.SetZ( positions[cnt*3+2]);
        // compute coordinates for new atom
        xCoord = (unsigned int)std::min( maxXId,
                 (unsigned int)std::max( 0, (int)floorf( (tmpVec.GetX() - bBox.Left()) / float(voxelLength))));
        yCoord = (unsigned int)std::min( maxYId,
                 (unsigned int)std::max( 0, (int)floorf( (tmpVec.GetY() - bBox.Bottom()) / float(voxelLength))));
        zCoord = (unsigned int)std::min( maxZId,
                 (unsigned int)std::max( 0, (int)floorf( (tmpVec.GetZ() - bBox.Back()) / float(voxelLength))));
        // add atom to voxel texture
        voxelIdx = vW * vH * zCoord + vW * yCoord + xCoord;
        firstIdx = voxelIdx * numAtomsPerVoxel;
        this->voxelMap[firstIdx]++;
        atomCnt = this->voxelMap[firstIdx];
        this->voxelMap[firstIdx+atomCnt] = cnt;
    }

    // loop over all atoms
    for( cnt1 = 0; cnt1 < protein->AtomCount(); ++cnt1 ) {
        // continue if atom is not visible
        if( this->visibleAtomMask[cnt1] < 0.5 ) continue;

		// get position of current atom
		tmpVec.SetX( positions[cnt1*3+0]);
		tmpVec.SetY( positions[cnt1*3+1]);
		tmpVec.SetZ( positions[cnt1*3+2]);
		// get the radius of current atom
        radius = protein->AtomTypes()[protein->AtomTypeIndices()[cnt1]].Radius();
        // reset id
        vicinityCounter = 1;

	    xId = (int)std::max( 0, (int)floorf( ( tmpVec.GetX() - bBox.Left()) / float(voxelLength)));
	    xId = (int)std::min( (int)maxXId, xId);
	    yId = (int)std::max( 0, (int)floorf( ( tmpVec.GetY() - bBox.Bottom()) / float(voxelLength)));
	    yId = (int)std::min( (int)maxYId, yId);
	    zId = (int)std::max( 0, (int)floorf( ( tmpVec.GetZ() - bBox.Back()) / float(voxelLength)));
	    zId = (int)std::min( (int)maxZId, zId);

	    // loop over all atoms to find vicinity
	    for( cntX = ((xId > 0)?(-1):0); cntX < ((xId < (int)maxXId)?2:1); ++cntX )
	    {
		    for( cntY = ((yId > 0)?(-1):0); cntY < ((yId < (int)maxYId)?2:1); ++cntY )
		    {
			    for( cntZ = ((zId > 0)?(-1):0); cntZ < ((zId < (int)maxZId)?2:1); ++cntZ )
			    {
                    voxelIdx = vW * vH * (zId+cntZ) + vW * (yId+cntY) + (xId+cntX);
                    firstIdx = voxelIdx * this->numAtomsPerVoxel;
                    for( cnt = 0; cnt < this->voxelMap[firstIdx]; ++cnt )
				    {
                        cnt2 = voxelMap[firstIdx+cnt+1];
					    // don't check the same atom --> continue
					    if( cnt2 == cnt1 )
						    continue;
                        pos.SetX( positions[cnt2*3+0]);
		                pos.SetY( positions[cnt2*3+1]);
		                pos.SetZ( positions[cnt2*3+2]);
                        rad = protein->AtomTypes()[protein->AtomTypeIndices()[cnt2]].Radius();
					    // compute distance
					    distance = ( pos - tmpVec).Length();
					    // compute threshold
					    threshold = rad + radius + 2.0f * this->probeRadius;
					    // if distance < threshold --> add atom 'cnt' to vicinity
                        if( distance < threshold && vicinityCounter <= MAX_ATOM_VICINITY )
					    {
                            this->vicinityTable[cnt1 * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 0] = pos.GetX();
                            this->vicinityTable[cnt1 * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 1] = pos.GetY();
                            this->vicinityTable[cnt1 * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 2] = pos.GetZ();
                            this->vicinityTable[cnt1 * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 3] = rad;
                            vicinityCounter++;
					    }
				    }
			    }
		    }
	    } // loop over all atoms to find vicinity
        this->vicinityTable[cnt1 * ( MAX_ATOM_VICINITY + 1) * 4] = float(vicinityCounter - 1);
    } // loop over all atoms

    // generate vicinity table texture if necessary
    if( !glIsTexture( this->vicinityTableTex ) ) 
        glGenTextures( 1, &this->vicinityTableTex);
    // create texture containing the vicinity table
    glBindTexture( GL_TEXTURE_2D, this->vicinityTableTex);
    glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGBA32F, ( MAX_ATOM_VICINITY + 1), protein->AtomCount(), 0, GL_RGBA, GL_FLOAT, this->vicinityTable);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);
    
}
/*
 * Find for each visible atom all atoms that are in the proximity.
 */
void MoleculeCudaSESRenderer::ComputeVicinityTableCUDA( MolecularDataCall *protein) {

    // temporary variables
	//unsigned int cnt, neighborCnt, vicinityCounter;
    
    // set up vicinity table, if necessary --> reserve one aditional vec for # of vicinity atoms
    //if( !this->vicinityTable )
    //    this->vicinityTable = new float[protein->AtomCount() * ( MAX_ATOM_VICINITY + 1) * 4];
    //// reset all items in vicinity table to zero
    //memset( this->vicinityTable, 0, sizeof(float) * protein->AtomCount() * ( MAX_ATOM_VICINITY + 1) * 4);

	// execute CUDA kernels, if initialization succeeded
	if( this->cudaInitalized ) {
		// write atom positions to array
		this->writeAtomPositions( protein);

		// update constants
		setParameters( &this->params);

		// calculate grid hash
		calcHash(
			m_dGridParticleHash,
			m_dGridParticleIndex,
			m_dPos,
			this->numAtoms);

		// sort particles based on hash
        //cudppSort( this->sortHandle, m_dGridParticleHash, m_dGridParticleIndex, this->gridSortBits, this->numAtoms);
        sortParticles(m_dGridParticleHash, m_dGridParticleIndex, this->numAtoms);

		// reorder particle arrays into sorted order and
		// find start and end of each cell
		reorderDataAndFindCellStart(
			m_dCellStart,
			m_dCellEnd,
			m_dSortedPos,
			m_dGridParticleHash,
			m_dGridParticleIndex,
			m_dPos,
			this->numAtoms,
			this->numGridCells);

		// count neighbors of all atoms
		countNeighbors2(
			m_dNeighborCount,
			m_dNeighbors,
			m_dSortedPos,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			this->numAtoms,
            MAX_ATOM_VICINITY,
			this->numGridCells);

        //copyArrayFromDevice( m_hNeighborCount, m_dNeighborCount, 0, sizeof(uint)*protein->AtomCount());
        //copyArrayFromDevice( m_hNeighbors, m_dNeighbors, 0, sizeof(uint)*protein->AtomCount()*MAX_ATOM_VICINITY);
        //copyArrayFromDevice( m_hPos, m_dSortedPos, 0, sizeof(float)*4*protein->AtomCount());
	}

    //// loop over all atoms
    //for( cnt = 0; cnt < protein->AtomCount(); ++cnt ) {
    //    // continue if atom is not visible
    //    if( this->visibleAtomMask[cnt] < 0.5 ) continue;
    //    // reset id
    //    vicinityCounter = 1;

    //    // loop over all neighbors and write them to the table
    //    for( neighborCnt = 0; neighborCnt < m_hNeighborCount[cnt]; ++neighborCnt )
    //    {
    //        this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 0] = 
    //            m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+0];
    //        this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 1] = 
    //            m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+1];
    //        this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 2] = 
    //            m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+2];
    //        this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 3] = 
    //            m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+3];
    //        vicinityCounter++;
    //    }
    //    this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4] = float(vicinityCounter - 1);
    //} // loop over all atoms

    //// generate vicinity table texture if necessary
    //if( !glIsTexture( this->vicinityTableTex ) ) 
    //    glGenTextures( 1, &this->vicinityTableTex);
    //// create texture containing the vicinity table
    //glBindTexture( GL_TEXTURE_2D, this->vicinityTableTex);
    //glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGBA32F, ( MAX_ATOM_VICINITY + 1), protein->AtomCount(), 0, GL_RGBA, GL_FLOAT, this->vicinityTable);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glBindTexture( GL_TEXTURE_2D, 0);
    
}


/*
 * Write atom positions and radii to an array for processing in CUDA
 */
void MoleculeCudaSESRenderer::writeAtomPositions( MolecularDataCall *protein ) {
    // use interpolated positions, if availabe
    const float *positions = protein->AtomPositions();
    if( posInter ) positions = posInter;

	// write atoms to array
    /*
	int p = 0;
	for( unsigned int cnt = 0; cnt < protein->AtomCount(); ++cnt ) {
		// write pos and rad to array
		m_hPos[p++] = positions[cnt*3+0];
		m_hPos[p++] = positions[cnt*3+1];
		m_hPos[p++] = positions[cnt*3+2];
        m_hPos[p++] = protein->AtomTypes()[protein->AtomTypeIndices()[cnt]].Radius();
	}
    */
    int cnt;
//#pragma omp parallel for
	for( cnt = 0; cnt < int( protein->AtomCount()); ++cnt ) {
		// write pos and rad to array
		m_hPos[cnt*4+0] = positions[cnt*3+0];
		m_hPos[cnt*4+1] = positions[cnt*3+1];
		m_hPos[cnt*4+2] = positions[cnt*3+2];
        m_hPos[cnt*4+3] = protein->AtomTypes()[protein->AtomTypeIndices()[cnt]].Radius();
//        printf("wrote %f %f %f (%f)\n",
//                positions[cnt*3+0], positions[cnt*3+1], positions[cnt*3+2],
//                protein->AtomTypes()[protein->AtomTypeIndices()[cnt]].Radius());
	}
	//setArray( POSITION, m_hPos, 0, this->numAtoms);
	copyArrayToDevice( this->m_dPos, this->m_hPos, 0, this->numAtoms*4*sizeof(float));
	//checkCudaErrors( cudaMemcpyAsync( this->m_dPos, this->m_hPos, this->numAtoms*4*sizeof(float), cudaMemcpyHostToDevice, 0));
	//checkCudaErrors( cudaMemcpy( this->m_dPos, this->m_hPos, this->numAtoms*4*sizeof(float), cudaMemcpyHostToDevice));
}


/*
 * Compute the Reduced Surface using the Fragment Shader.
 */
void MoleculeCudaSESRenderer::ComputeRSFragShader( MolecularDataCall *protein) {
    // create FBO
    this->CreateTriangleFBO( protein->AtomCount(), MAX_ATOM_VICINITY);
    
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_LIGHTING);
        
    // start rendering to FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->triangleFBO);

    // set up multiple render targets
    GLenum mrt[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT };
    glDrawBuffers( 4, mrt);

    // set viewport, set clearcolor
    glViewport( 0, 0, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, this->visibleAtomCount);
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, this->vicinityTableTex);
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsTex);
    glActiveTexture( GL_TEXTURE2);
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsIdTex);
    glActiveTexture( GL_TEXTURE0);
    // enable reduced surface shader
    this->reducedSurfaceShader.Enable();
    // set uniform variables for the reduced surface shader
    glUniform1i( this->reducedSurfaceShader.ParameterLocation("vicinityTex"), 0);
    glUniform1i( this->reducedSurfaceShader.ParameterLocation("visibleAtomsTex"), 1);
    glUniform1i( this->reducedSurfaceShader.ParameterLocation("visibleAtomsIdTex"), 2);
    glUniform1f( this->reducedSurfaceShader.ParameterLocation("probeRadius"), this->probeRadius);
    // draw
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    glBegin(GL_QUADS);
    glVertex2f( 0.0f, 0.0f);
    glVertex2f( 1.0f, 0.0f);
    glVertex2f( 1.0f, 1.0f);
    glVertex2f( 0.0f, 1.0f);
    glEnd();
    glBindTexture( GL_TEXTURE_2D, 0);
    // disable reduced surface shader
    this->reducedSurfaceShader.Disable();

    // stop rendering to FBO, reset clearcolor
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}


/*
 * render all potential RS-faces as triangles using a vertex shader
 */
void MoleculeCudaSESRenderer::RenderTriangles( MolecularDataCall *protein) {
    // write VBO for fast drawing, if necessary
    if( !glIsBuffer( this->triangleVBO ) ) {
        // generate triangle VBO
        glGenBuffers( 1, &this->triangleVBO);
        // --- write triangle vertex positions (texture coordinates, respectively) ---
        // counter
        unsigned int cntX, cntY;
        unsigned int cnt = 0;
        // number of triangle vertices (3 floats per vertex, 3 vertices per triangle)
        unsigned int numVerts = MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*protein->AtomCount()*9;
        // delete triangle vertex list, if necessary
        if( this->triangleVertex )
            delete[] this->triangleVertex;
        this->triangleVertex = new float[numVerts];
        // for each pixel, write the coordinates to the vertex list
        for( cntY = 0; cntY < protein->AtomCount(); ++cntY ) {
            for( cntX = 0; cntX < MAX_ATOM_VICINITY*MAX_ATOM_VICINITY; ++cntX ) {
                // write pixel/texel pos (x,y) and vertex index [0,1,2]
                this->triangleVertex[cnt+0] = float( cntX);
                this->triangleVertex[cnt+1] = float( cntY);
                this->triangleVertex[cnt+2] = 0.0f;
                this->triangleVertex[cnt+3] = float( cntX);
                this->triangleVertex[cnt+4] = float( cntY);
                this->triangleVertex[cnt+5] = 1.0f;
                this->triangleVertex[cnt+6] = float( cntX);
                this->triangleVertex[cnt+7] = float( cntY);
                this->triangleVertex[cnt+8] = 2.0f;
                cnt += 9;
            }
        }
        // fill triangle VBO
        glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
        glBufferData( GL_ARRAY_BUFFER, numVerts*sizeof( float), this->triangleVertex, GL_STATIC_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
    }

    // render the triangles (i.e. the potential RS-facs) using the vertex shader
    this->drawTriangleShader.Enable();
    // enable and set up triangle drawing shader
    glUniform1i( this->drawTriangleShader.ParameterLocation( "positionTex0"), 0);
    glUniform1i( this->drawTriangleShader.ParameterLocation( "positionTex1"), 1);
    glUniform1i( this->drawTriangleShader.ParameterLocation( "positionTex2"), 2);
    // bind textures
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor0);
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor1);
    glActiveTexture( GL_TEXTURE2);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor2);

	// set vertex pointer and draw it
	glEnableClientState( GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
    glVertexPointer( 3, GL_FLOAT, 0, 0 );
    glDrawArrays( GL_TRIANGLES, 0, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*this->visibleAtomCount*3);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
	glDisableClientState( GL_VERTEX_ARRAY);
    
    // unbind texture and deactivate shader
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, 0);
    this->drawTriangleShader.Disable();
}


/*
 * find visible triangles (i.e. visible RS-faces)
 */
void MoleculeCudaSESRenderer::FindVisibleTriangles( MolecularDataCall *protein) {
    // generate FBO
    this->CreateVisibleTriangleFBO( protein->AtomCount(), MAX_ATOM_VICINITY);

    // set viewport
    glViewport( 0, 0, this->visibilityTexWidth, this->visibilityTexHeight);
    // start render to FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibilityFBO);
    // set clear color & clear
    glClearColor( -1.0f, 0.0f, 0.0f, 0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // render all triangles
    this->RenderTriangles( protein);

    // read FBO to VBO
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, this->visibilityTexVBO);
    glReadPixels( 0, 0, this->visibilityTexWidth, this->visibilityTexHeight, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0 );

    // stop render to FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);

    // ********** START find visible triangles **********
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_LIGHTING);

    // set ortho mode
    glOrtho( 0.0, double( MAX_ATOM_VICINITY*MAX_ATOM_VICINITY), 0.0, double( this->visibleAtomCount), 0.0, 1.0);
    // set viewport
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
    glViewport( 0, 0, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, this->visibleAtomCount);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibleTriangleFBO);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render visibility texture VBO
    glColor3f( 1.0f, 1.0f, 1.0f);
    glBindBuffer( GL_ARRAY_BUFFER, this->visibilityTexVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer( 2, GL_FLOAT, sizeof(float)*4, (float*)0);
    glDrawArrays( GL_POINTS, 0, this->visibilityTexWidth*this->visibilityTexHeight);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
    glClearColor( clearCol[0], clearCol[1], clearCol[2], clearCol[3]);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);
    // ********** END find visible triangles **********
}


/*
 * create fbo for adjacent triangles of visible triangles
 */
void MoleculeCudaSESRenderer::CreateAdjacentTriangleFBO( unsigned int atomCount, unsigned int vicinityCount) {

    if( !this->adjacentTriangleFBO )
    {
        // generate FBO
        glGenFramebuffersEXT( 1, &this->adjacentTriangleFBO);
        // generate textures, if necessary
        if( !glIsTexture( this->adjacentTriangleColor) )
            glGenTextures( 1, &this->adjacentTriangleColor);
        if( !glIsTexture( this->adjacentTriangleDepth) )
            glGenTextures( 1, &this->adjacentTriangleDepth);
        
        // color and depth FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->adjacentTriangleFBO);
        // init color texture
        glBindTexture( GL_TEXTURE_2D, this->adjacentTriangleColor);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, vicinityCount, atomCount, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture( GL_TEXTURE_2D, 0);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->adjacentTriangleColor, 0);
        // init depth texture
        glBindTexture( GL_TEXTURE_2D, this->adjacentTriangleDepth);
        glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, vicinityCount, atomCount, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameterf( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->adjacentTriangleDepth, 0);
        glBindTexture( GL_TEXTURE_2D, 0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    
        // delete vertex array, if necessary
        if( this->adjacentTriangleVertex ) delete[] this->adjacentTriangleVertex;
        // create new vertex array
        this->adjacentTriangleVertex = new float[vicinityCount*atomCount*3];
        // fill vertex array with coordinates
        unsigned int counter = 0;
        for( unsigned int cntX = 0; cntX < vicinityCount; ++cntX ) {
            for( unsigned int cntY = 0; cntY < atomCount; ++cntY ) {
                this->adjacentTriangleVertex[counter*3+0] = float(cntX);
                this->adjacentTriangleVertex[counter*3+1] = float(cntY);
                this->adjacentTriangleVertex[counter*3+2] = 0.0f;
                counter++;
            }
        }
        // generate VBO, if necessary
        if( !glIsBuffer( this->adjacentTriangleVBO) )
            glGenBuffers( 1, &this->adjacentTriangleVBO);
        // fill VBO
        glBindBuffer( GL_ARRAY_BUFFER, this->adjacentTriangleVBO);
        glBufferData( GL_ARRAY_BUFFER, counter*3*sizeof( float), this->adjacentTriangleVertex, GL_STATIC_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
        
        // create VBO
        if( !glIsBuffer( this->adjacentTriangleTexVBO) )
            glGenBuffers( 1, &this->adjacentTriangleTexVBO);
        glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, this->adjacentTriangleTexVBO);
        glBufferData( GL_PIXEL_PACK_BUFFER_EXT, atomCount*vicinityCount*4*sizeof(float), NULL, GL_DYNAMIC_DRAW_ARB );
        glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0);
    }

}


/*
 * Find the adjacent triangles to all visible triangles.
 */
void MoleculeCudaSESRenderer::FindAdjacentTriangles( MolecularDataCall *protein) {
    // 1.) create fbo for offscreen rendering ( MAX_ATOM_VICINITY * protein->AtomCount() )
    // 2.) shader: for each edge:
    //      2.1) test how many visible triangles are connected
    //      2.2) 2 --> discard; 0 --> discard; 1 --> continue with 2.3)
    //      2.3) get all invisible triangles and compute angle to visible triangle
    //      2.4) mark the invisible triangle with the least angle as visible
    // 3.) write all found adjacent triangles to visible triangles lookup texture

    // ----- create FBO (1.) -----
    this->CreateAdjacentTriangleFBO( protein->AtomCount(), MAX_ATOM_VICINITY);

    // ----- find adjacent triangles (2.) -----
    // START draw overlay 
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_LIGHTING);    
    // start rendering to FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->adjacentTriangleFBO);
    // set viewport, set clearcolor
    glViewport( 0, 0, MAX_ATOM_VICINITY, this->visibleAtomCount);
    glClearColor( -1.0f, 0.0f, 0.0f, 0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor0);
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor1);
    glActiveTexture( GL_TEXTURE2);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor2);
    glActiveTexture( GL_TEXTURE3);
    glBindTexture( GL_TEXTURE_2D, this->triangleNormal);
    glActiveTexture( GL_TEXTURE4);
    glBindTexture( GL_TEXTURE_2D, this->visibleTriangleColor);
    glActiveTexture( GL_TEXTURE0);
    // enable adjacent triangle shader
    this->adjacentTriangleShader.Enable();
    // set uniform variables for the adjacent triangle shader
    glUniform1i( this->adjacentTriangleShader.ParameterLocation( "positionTex0"), 0);
    glUniform1i( this->adjacentTriangleShader.ParameterLocation( "positionTex1"), 1);
    glUniform1i( this->adjacentTriangleShader.ParameterLocation( "positionTex2"), 2);
    glUniform1i( this->adjacentTriangleShader.ParameterLocation( "normalTex"), 3);
    glUniform1i( this->adjacentTriangleShader.ParameterLocation( "markerTex"), 4);
    glUniform1f( this->adjacentTriangleShader.ParameterLocation( "probeRadius"), this->probeRadius);
    // draw
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    glBegin(GL_QUADS);
    glVertex2f( 0.0f, 0.0f);
    glVertex2f( 1.0f, 0.0f);
    glVertex2f( 1.0f, 1.0f);
    glVertex2f( 0.0f, 1.0f);
    glEnd();
    glBindTexture( GL_TEXTURE_2D, 0);
    // disable adjacent triangle shader
    this->adjacentTriangleShader.Disable();

    // read FBO to VBO
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, this->adjacentTriangleTexVBO);
    glReadPixels( 0, 0, MAX_ATOM_VICINITY, this->visibleAtomCount, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0 );

    // stop rendering to FBO, reset clearcolor
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();


    // ----- write all found adjacent triangles to visible triangles lookup texture (3.) -----
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_LIGHTING);

    // set ortho mode
    glOrtho( 0.0, double( MAX_ATOM_VICINITY*MAX_ATOM_VICINITY), 0.0, double( this->visibleAtomCount), 0.0, 1.0);
    // set viewport
    glViewport( 0, 0, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, this->visibleAtomCount);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibleTriangleFBO);

    glDisable( GL_DEPTH_TEST);
    // render visibility texture VBO
    glColor3f( 1.0f, 0.0f, 1.0f);
    glBindBuffer( GL_ARRAY_BUFFER, this->adjacentTriangleTexVBO);
    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer( 2, GL_FLOAT, sizeof(float)*4, (const GLfloat *)0);
    glDrawArrays( GL_POINTS, 0, MAX_ATOM_VICINITY*this->visibleAtomCount);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    glEnable( GL_DEPTH_TEST);
    
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);
}
/*
 * Find the adjacent triangles to all visible triangles.
 */
void MoleculeCudaSESRenderer::FindAdjacentTrianglesCUDA( MolecularDataCall *mol) {
    // write VBO for fast drawing, if necessary
    if( !glIsBuffer( this->visibilityPbo ) ) {
        // generate PBO
        glGenBuffers( 1, &this->visibilityPbo);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, this->visibilityPbo);
        glBufferData( GL_PIXEL_UNPACK_BUFFER, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*sizeof(float), 0, GL_DYNAMIC_DRAW);
        cudaGLRegisterBufferObject( this->visibilityPbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

	// register this texture with CUDA
	if( !cudaTexResource ) {
		checkCudaErrors(cudaGraphicsGLRegisterImage( &cudaTexResource,
			this->visibleTriangleColor,	GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
	}

    cudaArray *in_array;

	// map Texture buffer objects to get CUDA device pointers
	checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexResource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&in_array, cudaTexResource, 0, 0));

    // map PBO
    float *ptr;
    cudaGLMapBufferObject( (void **)&ptr, this->visibilityPbo);
	
    // find adjacent triangles using CUDA
    findAdjacentTrianglesCuda( 
        ptr, 
        in_array,
        m_dPoint1, 
        m_dProbePosTable, 
        m_dNeighborCount,
        m_dNeighbors,
        m_dSortedPos,
        m_dVisibleAtoms, 
        m_dVisibleAtomsId,
        mol->AtomCount(), 
        this->visibleAtomCount, 
        MAX_ATOM_VICINITY);

	// unmap buffer object (PBO)
    checkCudaErrors( cudaGLUnmapBufferObject( this->visibilityPbo));
	// unmap buffer object (tex)
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaTexResource, 0));

    // copy PBO to texture
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, this->visibilityPbo);
    glBindTexture( GL_TEXTURE_2D, this->visibleTriangleColor);
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, mol->AtomCount(), GL_RED, GL_FLOAT, NULL); 
    glBindTexture( GL_TEXTURE_2D, 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
    CHECK_FOR_OGL_ERROR();
}


/*
 * create the VBO for transform feedback
 */
void MoleculeCudaSESRenderer::CreateTransformFeedbackVBO( MolecularDataCall *mol) {
    /////////////////
    // set up VBOs //
    /////////////////
    
    if( glIsBuffer( this->sphericalTriaVBO) ) return;

    // TODO: this should be the maximum number of atoms times the maximum number of neighborhood atoms!!
    unsigned int max_buffer_verts = mol->AtomCount() * MAX_ATOM_VICINITY;
    CHECK_FOR_OGL_ERROR();
    // create buffer objects
    glGenBuffers( 1, &this->sphericalTriaVBO);
    CHECK_FOR_OGL_ERROR();
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO);
    glBufferData( GL_ARRAY_BUFFER, max_buffer_verts*4*sizeof(GLfloat), 0, GL_DYNAMIC_COPY);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    CHECK_FOR_OGL_ERROR();
    // delete query objects, if necessary
    if( this->query ) {
        glDeleteQueries( 1, &this->query);
        CHECK_FOR_OGL_ERROR();
    }

    this->visibleTriangleIdxShader.Enable();
    GLint varyingLocCuda[] = { glGetVaryingLocationNV( this->visibleTriangleIdxShader, "gl_Position") };
    glTransformFeedbackVaryingsNV( this->visibleTriangleIdxShader, 1, varyingLocCuda, GL_SEPARATE_ATTRIBS_NV);
    CHECK_FOR_OGL_ERROR();
    this->visibleTriangleIdxShader.Disable();
    
    // create query objects
    glGenQueries( 1, &this->query);
    CHECK_FOR_OGL_ERROR();
}


/*
 * create the singularity texture
 */
void MoleculeCudaSESRenderer::CreateSingularityTextureCuda( MolecularDataCall *mol, unsigned int numProbes) {
    // update the RS parameters
    this->rsParams.probeCount = numProbes;
    setRSParameters( &this->rsParams);

    // create singularity texture
    if( !glIsTexture( this->singTex) )
        glGenTextures( 1, &this->singTex);
    // create singularity texture
    glBindTexture( GL_TEXTURE_2D, this->singTex);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F, MAX_PROBE_VICINITY, numProbes, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);

    // allocate CUDA arrays
    if( this->m_dProbePos )
        freeArray( this->m_dProbePos);
    allocateArray((void**)&m_dProbePos, numProbes*4*sizeof(float));
    if( this->m_dSortedProbePos )
        freeArray( this->m_dSortedProbePos);
    allocateArray((void**)&m_dSortedProbePos, numProbes*4*sizeof(float));
    if( this->m_dProbeNeighborCount )
        freeArray( m_dProbeNeighborCount);
    allocateArray((void**)&m_dProbeNeighborCount, numProbes*sizeof(uint));
    if( this->m_dProbeNeighbors )
        freeArray( this->m_dProbeNeighbors);
    allocateArray((void**)&m_dProbeNeighbors, numProbes*MAX_PROBE_VICINITY*sizeof(uint));
    if( this->m_dGridProbeHash )
        freeArray( this->m_dGridProbeHash);
    allocateArray((void**)&m_dGridProbeHash, numProbes*sizeof(uint));
    if( this->m_dGridProbeIndex )
        freeArray( this->m_dGridProbeIndex);
    allocateArray((void**)&m_dGridProbeIndex, numProbes*sizeof(uint));

    // register the vertex buffer object with CUDA
    if( !this->cudaSTriaResource )
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &this->cudaSTriaResource, this->sTriaVbo, cudaGraphicsMapFlagsReadOnly));
    float4 *outSTriaPtr;
    size_t num_bytes;
    // map OpenGL buffer object for reading from CUDA
    checkCudaErrors( cudaGraphicsMapResources( 1, &this->cudaSTriaResource, 0));
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&outSTriaPtr, &num_bytes, this->cudaSTriaResource));
    // execute kernel (copy probe positions from VBO to CUDA array
    writeProbePositionsCuda( this->m_dProbePos, outSTriaPtr, numProbes);
    // unmap buffer object
    checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaSTriaResource, 0));

    // calculate grid hash
    calcHash(
        m_dGridProbeHash,
        m_dGridProbeIndex,
        m_dProbePos,
        numProbes);

    // Create the CUDPP radix sort
    //cudppDestroyPlan( this->sortHandleProbe);
    //CUDPPConfiguration sortConfig;
    //sortConfig.algorithm = CUDPP_SORT_RADIX;
    //sortConfig.datatype = CUDPP_UINT;
    //sortConfig.op = CUDPP_ADD;
    //sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    //cudppPlan( &this->sortHandleProbe, sortConfig, numProbes, 1, 0);

    // sort particles based on hash
    //cudppSort( this->sortHandleProbe, m_dGridProbeHash, m_dGridProbeIndex, this->gridSortBits, numProbes);
    sortParticles(m_dGridProbeHash, m_dGridProbeIndex, numProbes);

    // reorder particle arrays into sorted order and find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedProbePos,
        m_dGridProbeHash,
        m_dGridProbeIndex,
        m_dProbePos,
        numProbes,
        this->numGridCells);

    // create PBO
    if( !glIsBuffer( this->singCoordsPbo) )
        glGenBuffers( 1, &this->singCoordsPbo);
    glBindBuffer( GL_ARRAY_BUFFER, this->singCoordsPbo);
    glBufferData( GL_ARRAY_BUFFER, numProbes*3*sizeof(float), 0, GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject( this->singCoordsPbo);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // map VBO
    float3 *coordsPtr;
    cudaGLMapBufferObject( (void **)&coordsPtr, this->singCoordsPbo);

    // create PBO
    if( !glIsBuffer( this->singPbo) )
        glGenBuffers( 1, &this->singPbo);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, this->singPbo);
    glBufferData( GL_PIXEL_UNPACK_BUFFER, numProbes*MAX_PROBE_VICINITY*3*sizeof(float), 0, GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject( this->singPbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // map PBO
    float3 *ptr;
    cudaGLMapBufferObject( (void **)&ptr, this->singPbo);

    // count neighbors of all atoms
    countProbeNeighbors(
        //m_dProbeNeighborCount,
        coordsPtr,
        //m_dProbeNeighbors,
        ptr,
        m_dSortedProbePos,
        m_dGridProbeIndex,
        m_dCellStart,
        m_dCellEnd,
        numProbes,
        MAX_PROBE_VICINITY,
        this->numGridCells);
    
    cudaGLUnmapBufferObject( this->singCoordsPbo);
    cudaGLUnregisterBufferObject( this->singCoordsPbo);

    cudaGLUnmapBufferObject( this->singPbo);
    cudaGLUnregisterBufferObject( this->singPbo);

    // copy PBO to texture
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, this->singPbo);
    glBindTexture( GL_TEXTURE_2D, this->singTex);
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, MAX_PROBE_VICINITY, numProbes, GL_RGB, GL_FLOAT, NULL); 
    glBindTexture( GL_TEXTURE_2D, 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);


    // DEBUG
    /*
    float* hProbePos = new float[numProbes*4];
    copyArrayFromDevice( hProbePos, m_dProbePos, 0, sizeof(float)*numProbes*4);

    // set viewport
	float viewportStuff[4] = { cameraInfo->TileRect().Left(), cameraInfo->TileRect().Bottom(), cameraInfo->TileRect().Width(), cameraInfo->TileRect().Height() };
	if ( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if ( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];
	// enable sphere shader
	this->sphereShader.Enable();
	// set shader variables
    glUniform4fv( this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    glUniform3f( this->sphereShader.ParameterLocation( "zValues"), this->fogStart, cameraInfo->NearClip(), cameraInfo->FarClip());
    glUniform3f( this->sphereShader.ParameterLocation( "fogCol"), 1.0f,  1.0f, 1.0f);
    glUniform1f( this->sphereShader.ParameterLocation( "alpha"), this->transparency);

	// draw probes
    unsigned int cnt = this->debugParam.Param<param::IntParam>()->Value();
    if( cnt >= numProbes ) cnt = numProbes-1;

    //glColor3f( 1.0f, 0.5f, 0.0f);
    //glBegin( GL_POINTS );
    //for( cnt = 0; cnt < numProbes; ++cnt ) {
    //    glVertex4f(
    //        hProbePos[cnt*4+0],
    //        hProbePos[cnt*4+1],
    //        hProbePos[cnt*4+2],
    //        this->probeRadius );
    //}
    //glEnd(); // GL_POINTS
    
    // bind buffers...
    glEnableClientState( GL_VERTEX_ARRAY);
    glColor3f( 1.0f, 1.0f, 0.0f);
    glBindBuffer( GL_ARRAY_BUFFER, this->singPbo);
    glVertexPointer( 3, GL_FLOAT, 0, (const GLfloat *)0+(3*MAX_PROBE_VICINITY)*cnt);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    // ...and draw arrays
    glDrawArrays( GL_POINTS, 0, MAX_PROBE_VICINITY);
    glDisableClientState( GL_VERTEX_ARRAY);

    glEnable( GL_BLEND);
    // draw probe
    glColor3f( 1.0f, 0.5f, 0.0f);
    glBegin( GL_POINTS );
    glVertex4f(
        hProbePos[cnt*4+0],
        hProbePos[cnt*4+1],
        hProbePos[cnt*4+2],
        this->probeRadius );
    glEnd(); // GL_POINTS
    glDisable( GL_BLEND);

	// disable sphere shader
	this->sphereShader.Disable();
    // delete array
    delete[] hProbePos;

    glBindBuffer( GL_ARRAY_BUFFER, this->singCoordsPbo);
    float* map = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_ONLY);
    std::cout << "test " << cnt << ": " << map[3*cnt] << ", " << map[3*cnt+1] << ", " << map[3*cnt+2] << ", " << std::endl;
    glUnmapBuffer( GL_ARRAY_BUFFER);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    */
}


/*
 * render all visible atoms using GPU ray casting.
 */
void MoleculeCudaSESRenderer::RenderVisibleAtomsGPU( MolecularDataCall *protein) {
    // use interpolated positions, if availabe
    const float *positions = protein->AtomPositions();
    if( posInter ) positions = posInter;

    // set viewport
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

	// get clear color (i.e. background color) for fogging
	vislib::math::Vector<float, 3> fogCol( this->clearCol[0], this->clearCol[1], this->clearCol[2]);
	
	// enable sphere shader
	this->sphereShader.Enable();
	// set shader variables
    glUniform4fv( this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    glUniform3f( this->sphereShader.ParameterLocation( "zValues"), this->fogStart, cameraInfo->NearClip(), cameraInfo->FarClip());
    glUniform3f( this->sphereShader.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
    glUniform1f( this->sphereShader.ParameterLocation( "alpha"), this->transparency);
	// draw visible atoms
    unsigned int atomId;
    glBegin( GL_POINTS );
    for( unsigned int cnt = 0; cnt < this->visibleAtomCount; ++cnt ) {
        atomId = (unsigned int)this->visibleAtomsIdList[cnt];
        glColor3f( 1.0f, 0.0f, 0.0f);
        glVertex4f( positions[atomId*3+0],
            positions[atomId*3+1],
            positions[atomId*3+2],
            protein->AtomTypes()[protein->AtomTypeIndices()[atomId]].Radius() );
    }
    glEnd(); // GL_POINTS
	// disable sphere shader
	this->sphereShader.Disable();
}


/*
 * render the SES using GPU ray casting.
 */
void MoleculeCudaSESRenderer::RenderSESCuda( MolecularDataCall *mol, unsigned int primitiveCount) {
    // use interpolated positions, if availabe
    const float *positions = mol->AtomPositions();
    if( posInter ) positions = posInter;
	
    // set viewport
    float viewportStuff[4] = { cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(), cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height() };
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];
	// get clear color (i.e. background color) for fogging
    vislib::math::Vector<float, 3> fogCol( this->clearCol[0], this->clearCol[1], clearCol[2]);
    
    ///////////////////////////////////////////////////////////////////////////
    // render the atoms as spheres
    ///////////////////////////////////////////////////////////////////////////
#if 1
	// enable sphere shader
	this->sphereShader.Enable();
	// set shader variables
    glUniform4fv( this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    glUniform3f( this->sphereShader.ParameterLocation( "zValues"), this->fogStart, cameraInfo->NearClip(), cameraInfo->FarClip());
    glUniform3f( this->sphereShader.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
    glUniform1f( this->sphereShader.ParameterLocation( "alpha"), this->transparency);
	// draw visible atoms
    glBegin( GL_POINTS );
    glColor3f( 1.0f, 0.0f, 0.0f);
    for( unsigned int cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
        glVertex4f( positions[cnt*3+0],
            positions[cnt*3+1],
            positions[cnt*3+2],
            mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius() );
    }
    glEnd(); // GL_POINTS
	// disable sphere shader
	this->sphereShader.Disable();
#endif

    /////////////////////////////////////////////////
    // ray cast the spherical triangles on the GPU //
    /////////////////////////////////////////////////
    // bind singularity texture
    glBindTexture( GL_TEXTURE_2D, this->singTex);

    // enable spherical triangle shader
    this->sphericalTriangleShader.Enable();
    // set shader variables
    glUniform4fv(this->sphericalTriangleShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->sphericalTriangleShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv(this->sphericalTriangleShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv(this->sphericalTriangleShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    glUniform3f( this->sphericalTriangleShader.ParameterLocation( "zValues"), this->fogStart, cameraInfo->NearClip(), cameraInfo->FarClip());
    glUniform3f( this->sphericalTriangleShader.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
    glUniform2f( this->sphericalTriangleShader.ParameterLocation( "texOffset"), 1.0f, 1.0f);
    // get attribute locations
    GLuint attribVec1 = glGetAttribLocation( this->sphericalTriangleShader, "attribVec1");
    GLuint attribVec2 = glGetAttribLocation( this->sphericalTriangleShader, "attribVec2");
    GLuint attribVec3 = glGetAttribLocation( this->sphericalTriangleShader, "attribVec3");
    GLuint attribColors = glGetAttribLocation( this->sphericalTriangleShader, "attribColors");
    GLuint attribTexCoord1 = glGetAttribLocation( this->sphericalTriangleShader, "attribTexCoord1");
    // enable vertex attribute arrays for the attribute locations
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableVertexAttribArray( attribVec1);
    glEnableVertexAttribArray( attribVec2);
    glEnableVertexAttribArray( attribVec3);
    glEnableVertexAttribArray( attribTexCoord1);

    glVertexAttrib3f( attribColors, 230, 240, 250);

    // bind buffers...
    glBindBuffer( GL_ARRAY_BUFFER, this->sTriaVbo);
    glVertexAttribPointer( attribVec1, 4, GL_FLOAT, 0, sizeof(float)*4*4, (const GLfloat *)0+4);
    glVertexAttribPointer( attribVec2, 4, GL_FLOAT, 0, sizeof(float)*4*4, (const GLfloat *)0+8);
    glVertexAttribPointer( attribVec3, 4, GL_FLOAT, 0, sizeof(float)*4*4, (const GLfloat *)0+12);
	glVertexPointer( 4, GL_FLOAT, sizeof(float)*4*4, 0);
    glBindBuffer( GL_ARRAY_BUFFER, this->singCoordsPbo);
    glVertexAttribPointer( attribTexCoord1, 3, GL_FLOAT, 0, 0, 0);
    //glVertexAttribPointer( attribTexCoord1, 3, GL_FLOAT, 0, 0, this->singTexCoords);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    // ...and draw arrays
    glDrawArrays( GL_POINTS, 0, primitiveCount*2);

    // disable vertex attribute arrays for the attribute locations
    glDisableVertexAttribArray( attribVec1);
    glDisableVertexAttribArray( attribVec2);
    glDisableVertexAttribArray( attribVec3);
    glDisableVertexAttribArray( attribTexCoord1);
    glDisableClientState(GL_VERTEX_ARRAY);
    // disable spherical triangle shader
    this->sphericalTriangleShader.Disable();
    glBindTexture( GL_TEXTURE_2D, 0);

	//////////////////////////////////
	// ray cast the tori on the GPU //
	//////////////////////////////////
	// enable torus shader
	this->torusShader.Enable();
	// set shader variables
	glUniform4fv( this->torusShader.ParameterLocation( "viewAttr"), 1, viewportStuff);
	glUniform3fv( this->torusShader.ParameterLocation( "camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fv( this->torusShader.ParameterLocation( "camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fv( this->torusShader.ParameterLocation( "camUp"), 1, cameraInfo->Up().PeekComponents());
	glUniform3f( this->torusShader.ParameterLocation( "zValues"), this->fogStart, cameraInfo->NearClip(), cameraInfo->FarClip());
	glUniform3f( this->torusShader.ParameterLocation( "fogCol"), fogCol.GetX(), fogCol.GetY(), fogCol.GetZ() );
	glUniform1f( this->torusShader.ParameterLocation( "alpha"), this->transparency);
	// get attribute locations
	GLuint attribInParams = glGetAttribLocation( this->torusShader, "inParams");
	GLuint attribQuatC = glGetAttribLocation( this->torusShader, "quatC");
	GLuint attribInSphere = glGetAttribLocation( this->torusShader, "inSphere");
	GLuint attribInColors = glGetAttribLocation( this->torusShader, "inColors");
	//GLuint attribInCuttingPlane = glGetAttribLocation( this->torusShader, "inCuttingPlane");
	// set color to orange
	glColor3f( 1.0f, 0.75f, 0.0f);
	glEnableClientState( GL_VERTEX_ARRAY);
	// enable vertex attribute arrays for the attribute locations
	glEnableVertexAttribArrayARB( attribInParams);
	glEnableVertexAttribArrayARB( attribQuatC);
	glEnableVertexAttribArrayARB( attribInSphere);
	//glEnableVertexAttribArrayARB( attribInColors);
	//glEnableVertexAttribArrayARB( attribInCuttingPlane);

	// set vertex and attribute pointers...
    glBindBuffer( GL_ARRAY_BUFFER, this->torusVbo);
	glVertexAttribPointer( attribInParams, 4, GL_FLOAT, 0, sizeof(float)*4*4, (const GLfloat *)0+4);
	glVertexAttribPointer( attribQuatC, 4, GL_FLOAT, 0, sizeof(float)*4*4, (const GLfloat *)0+8);
	glVertexAttribPointer( attribInSphere, 4, GL_FLOAT, 0, sizeof(float)*4*4, (const GLfloat *)0+12);
    glVertexAttrib4d( attribInColors, 255000, 255000, 255000, 255000);
	//glVertexAttribPointer( attribInColors, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
	//glVertexAttribPointer( attribInCuttingPlane, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
	glVertexPointer( 4, GL_FLOAT, sizeof(float)*4*4, 0);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    // ...and draw arrays
    glDrawArrays( GL_POINTS, 0, primitiveCount*3);
    
	// disable vertex attribute arrays for the attribute locations
	glDisableVertexAttribArray( attribInParams);
	glDisableVertexAttribArray( attribQuatC);
	glDisableVertexAttribArray( attribInSphere);
	//glDisableVertexAttribArray( attribInColors);
	//glDisableVertexAttribArray( attribInCuttingPlane);
	glDisableClientState(GL_VERTEX_ARRAY);
	// disable torus shader
	this->torusShader.Disable();
}

/*
 * mark all atoms which are vertices of adjacent triangles as visible
 */
void MoleculeCudaSESRenderer::MarkAdjacentAtoms( MolecularDataCall *protein) {

    // ----- write all found adjacent triangles to visible triangles lookup texture -----
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_LIGHTING);

    // set ortho mode
    glOrtho( 0.0, double( protein->AtomCount()), 0.0, 1.0, 0.0, 1.0);
    // set viewport
    glViewport( 0, 0, protein->AtomCount(), 1);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibleTriangleFBO);

    glDisable( GL_DEPTH_TEST);

    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsColor);
    glActiveTexture( GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_2D, this->visibleTriangleColor);
    glActiveTexture( GL_TEXTURE0);

    this->adjacentAtomsShader.Enable();

    // render visibility texture VBO
    glColor3f( 1.0f, 0.0f, 1.0f);
    glBindBuffer( GL_ARRAY_BUFFER, this->adjacentTriangleTexVBO);
    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer( 2, GL_FLOAT, sizeof(float)*4, (const GLfloat *)0);
    glDrawArrays( GL_POINTS, 0, MAX_ATOM_VICINITY*this->visibleAtomCount);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    glEnable( GL_DEPTH_TEST);

    this->adjacentAtomsShader.Disable();
    
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);


}


/*
 * Compute the Reduced Surface using CUDA
 */
void MoleculeCudaSESRenderer::ComputeRSCuda( MolecularDataCall *mol) {
    // allocate the arrays
    if( !m_dPoint1 ) {
        allocateArray( (void**)&m_dPoint1, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(uint));
        //allocateArray( (void**)&m_dPoint2, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(float));
        //allocateArray( (void**)&m_dPoint3, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(float));
        allocateArray( (void**)&m_dProbePosTable, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(float));
        allocateArray( (void**)&m_dVisibleAtoms, mol->AtomCount()*4*sizeof(float));
        allocateArray( (void**)&m_dVisibleAtomsId, mol->AtomCount()*sizeof(uint));
    }

    copyArrayToDevice( this->m_dVisibleAtoms, this->visibleAtomsList, 0, this->visibleAtomCount*4*sizeof(float));
    copyArrayToDevice( this->m_dVisibleAtomsId, this->visibleAtomsIdList, 0, this->visibleAtomCount*sizeof(uint));

    // set RS parameters
    this->rsParams.visibleAtomCount = this->visibleAtomCount;
    this->rsParams.maxNumProbeNeighbors = MAX_PROBE_VICINITY;
    // set RS parameters
    setRSParameters( &this->rsParams);

    cudaMemset( m_dPoint1, 0, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(uint));
    //cudaMemset( m_dPoint2, 0, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(float));
    //cudaMemset( m_dPoint3, 0, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(float));
    cudaMemset( m_dProbePosTable, 0, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(float));

	// count neighbors of all atoms
    computeReducedSurfaceCuda(
        m_dPoint1, 
        //m_dPoint2, 
        //m_dPoint3, 
        m_dProbePosTable, 
        m_dNeighborCount,
        m_dNeighbors,
        m_dSortedPos,
        m_dGridParticleIndex,
        m_dVisibleAtoms, 
        m_dVisibleAtomsId,
        mol->AtomCount(), 
        this->visibleAtomCount, 
        MAX_ATOM_VICINITY);

    /*
    // DEBUG ...
    // generate textures, if necessary
    if( !glIsTexture( this->triangleColor0) )
        glGenTextures( 1, &this->triangleColor0);
    if( !glIsTexture( this->triangleColor1) )
        glGenTextures( 1, &this->triangleColor1);
    if( !glIsTexture( this->triangleColor2) )
        glGenTextures( 1, &this->triangleColor2);
    if( !glIsTexture( this->triangleNormal) )
        glGenTextures( 1, &this->triangleNormal);
    
    float* hResults = new float[mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4];
    // init color texture 0
    copyArrayFromDevice( hResults, m_dPoint1, 0, mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4*sizeof(float));
    glBindTexture( GL_TEXTURE_2D, this->triangleColor0);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, mol->AtomCount(), 0, GL_RGBA, GL_FLOAT, hResults);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture( GL_TEXTURE_2D, 0);
    // init color texture 1
    copyArrayFromDevice( hResults, m_dPoint2, 0, sizeof(float)*mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor1);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, mol->AtomCount(), 0, GL_RGBA, GL_FLOAT, hResults);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture( GL_TEXTURE_2D, 0);
    // init color texture 2
    copyArrayFromDevice( hResults, m_dPoint3, 0, sizeof(float)*mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4);
    glBindTexture( GL_TEXTURE_2D, this->triangleColor2);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, mol->AtomCount(), 0, GL_RGBA, GL_FLOAT, hResults);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture( GL_TEXTURE_2D, 0);
    // init color texture 3 for storing normals
    copyArrayFromDevice( hResults, m_dProbePosTable, 0, sizeof(float)*mol->AtomCount()*MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*4);
    glBindTexture( GL_TEXTURE_2D, this->triangleNormal);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, mol->AtomCount(), 0, GL_RGBA, GL_FLOAT, hResults);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture( GL_TEXTURE_2D, 0);

    delete[] hResults;
    // ... DEBUG
    */
}

/*
 * find visible triangles (i.e. visible RS-faces)
 */
void MoleculeCudaSESRenderer::FindVisibleTrianglesCuda( MolecularDataCall *mol) {
    // generate FBO
    this->CreateVisibleTriangleFBO( mol->AtomCount(), MAX_ATOM_VICINITY);

    // set viewport
    glViewport( 0, 0, this->visibilityTexWidth, this->visibilityTexHeight);
    // start render to FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibilityFBO);
    // set clear color & clear
    glClearColor( -1.0f, 0.0f, 0.0f, 0.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // render all triangles
    this->RenderTrianglesCuda2( mol);

    // read FBO to VBO
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, this->visibilityTexVBO);
    glReadPixels( 0, 0, this->visibilityTexWidth, this->visibilityTexHeight, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0 );

    // stop render to FBO
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);

    // ********** START find visible triangles **********
    // START draw overlay
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable( GL_LIGHTING);

    // set ortho mode
    glOrtho( 0.0, double( MAX_ATOM_VICINITY*MAX_ATOM_VICINITY), 0.0, double( this->visibleAtomCount), 0.0, 1.0);
    // set viewport
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
    glViewport( 0, 0, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY, this->visibleAtomCount);
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->visibleTriangleFBO);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render visibility texture VBO
    glColor3f( 1.0f, 1.0f, 1.0f);
    glBindBuffer( GL_ARRAY_BUFFER, this->visibilityTexVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer( 2, GL_FLOAT, sizeof(float)*4, (float*)0);
    glDrawArrays( GL_POINTS, 0, this->visibilityTexWidth*this->visibilityTexHeight);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0);
    glClearColor( clearCol[0], clearCol[1], clearCol[2], clearCol[3]);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // reset viewport to normal view
    glViewport( 0, 0, this->width, this->height);
    // ********** END find visible triangles **********
}

/*
 * render all potential RS-faces as triangles using CUDA
 */
void MoleculeCudaSESRenderer::RenderTrianglesCuda( MolecularDataCall *mol) {
    // write VBO for fast drawing, if necessary
    if( !glIsBuffer( this->triangleVBO ) ) {
        // generate triangle VBO
        glGenBuffers( 1, &this->triangleVBO);
        // element count (3 vertices + 3 colors ) * 3 floats per triangle
        unsigned int count = MAX_ATOM_VICINITY * MAX_ATOM_VICINITY * mol->AtomCount() * 6 * 3;
        // fill triangle VBO
        glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
        glBufferData( GL_ARRAY_BUFFER, count * sizeof( float), 0, GL_DYNAMIC_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0);

	    // register this buffer object with CUDA
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &this->cudaVboResource, this->triangleVBO, cudaGraphicsMapFlagsWriteDiscard));
    }

    // map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    checkCudaErrors( cudaGraphicsMapResources( 1, &this->cudaVboResource, 0));
    size_t num_bytes;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, this->cudaVboResource));
    // write VBO using CUDA
    computeTriangleVBOCuda( dptr, this->m_dPoint1, 
        //this->m_dPoint2, this->m_dPoint3,
        m_dSortedPos, m_dVisibleAtoms,
        mol->AtomCount(), this->visibleAtomCount, MAX_ATOM_VICINITY, 0);

    // unmap buffer object
    checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaVboResource, 0));

    // enable triangle drawing shader
    this->drawCUDATriangleShader.Enable();
    GLuint attribCol = glGetAttribLocation( this->drawCUDATriangleShader, "attribCol");
    glEnableVertexAttribArray( attribCol);
	// set vertex pointer and draw it
    glEnableClientState( GL_VERTEX_ARRAY);
    //glEnableClientState( GL_COLOR_ARRAY);
    glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
    //glColorPointer( 3, GL_FLOAT, sizeof(float)*6, (const GLfloat *)0+3);
    glVertexAttribPointer( attribCol, 3, GL_FLOAT, 0, sizeof(float)*6, (const GLfloat *)0+3);
	glVertexPointer( 3, GL_FLOAT, sizeof(float)*6, (const GLfloat *)0);
    //glInterleavedArrays( GL_C3F_V3F, 0, 0);
    glDrawArrays( GL_TRIANGLES, 0, MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*this->visibleAtomCount*3);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    //glDisableClientState( GL_COLOR_ARRAY);
    glDisableClientState( GL_VERTEX_ARRAY);
    // disable triangle drawing shader
    glDisableVertexAttribArray( attribCol);
    this->drawCUDATriangleShader.Disable();
    
}

/*
 * render all potential RS-faces as triangles using a vertex shader
 */
void MoleculeCudaSESRenderer::RenderTrianglesCuda2( MolecularDataCall *mol) {
    // element count (3 vertices + 3 colors ) * 3 floats per triangle
    unsigned int count = MAX_ATOM_VICINITY * MAX_ATOM_VICINITY * 
        std::min( mol->AtomCount(), 512U) * 6 * 3;
    // write VBO for fast drawing, if necessary
    if( !glIsBuffer( this->triangleVBO ) ) {
        // generate triangle VBO
        glGenBuffers( 1, &this->triangleVBO);
        // fill triangle VBO
        glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
        glBufferData( GL_ARRAY_BUFFER, count * sizeof( float), 0, GL_DYNAMIC_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0);

	    // register this buffer object with CUDA
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &this->cudaVboResource, this->triangleVBO, cudaGraphicsMapFlagsWriteDiscard));
    }

    float3 *dptr;
    size_t num_bytes;

    unsigned int numRuns = this->visibleAtomCount / 512U + ( ( this->visibleAtomCount % 512U ) == 0 ? 0:1);

    for( unsigned int cnt = 0; cnt < numRuns; ++cnt ) {
		// map OpenGL buffer object for writing from CUDA
		checkCudaErrors( cudaGraphicsMapResources( 1, &this->cudaVboResource, 0));
		checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, this->cudaVboResource));
		
        // write VBO using CUDA
		computeTriangleVBOCuda( dptr, this->m_dPoint1,
            //this->m_dPoint2, this->m_dPoint3,
            m_dSortedPos, m_dVisibleAtoms,
            mol->AtomCount(), std::min( this->visibleAtomCount, 512U), MAX_ATOM_VICINITY, cnt * 512U);

		// unmap buffer object (vbo)
		checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaVboResource, 0));

        // enable triangle drawing shader
        this->drawCUDATriangleShader.Enable();
        GLuint attribCol = glGetAttribLocation( this->drawCUDATriangleShader, "attribCol");
        glEnableVertexAttribArray( attribCol);
	    // set vertex pointer and draw it
        glEnableClientState( GL_VERTEX_ARRAY);
        //glEnableClientState( GL_COLOR_ARRAY);
        glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
        //glColorPointer( 3, GL_FLOAT, sizeof(float)*6, (const GLfloat *)0+3);
        glVertexAttribPointer( attribCol, 3, GL_FLOAT, 0, sizeof(float)*6, (const GLfloat *)0+3);
	    glVertexPointer( 3, GL_FLOAT, sizeof(float)*6, (const GLfloat *)0);
        //glInterleavedArrays( GL_C3F_V3F, 0, 0);
        glDrawArrays( GL_TRIANGLES, 0, count / 6);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
        //glDisableClientState( GL_COLOR_ARRAY);
        glDisableClientState( GL_VERTEX_ARRAY);
        // disable triangle drawing shader
        glDisableVertexAttribArray( attribCol);
        this->drawCUDATriangleShader.Disable();
    }


}

/*
 * render all potential RS-faces as triangles using a vertex shader
 */
void MoleculeCudaSESRenderer::RenderVisibleTrianglesCuda( MolecularDataCall *mol) {
    // element count (3 vertices + 3 colors ) * 3 floats per triangle
    unsigned int count = MAX_ATOM_VICINITY * MAX_ATOM_VICINITY * 
        std::min( mol->AtomCount(), 512U) * 6 * 3;
    // write VBO for fast drawing, if necessary
    if( !glIsBuffer( this->triangleVBO ) ) {
        // generate triangle VBO
        glGenBuffers( 1, &this->triangleVBO);
        // fill triangle VBO
        glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
        glBufferData( GL_ARRAY_BUFFER, count * sizeof( float), 0, GL_DYNAMIC_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0);

	    // register this buffer object with CUDA
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &this->cudaVboResource, this->triangleVBO, cudaGraphicsMapFlagsWriteDiscard));
    }

	// register this texture with CUDA
	if( !cudaTexResource ) {
		checkCudaErrors(cudaGraphicsGLRegisterImage( &cudaTexResource,
			this->visibleTriangleColor,	GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
	}

    cudaArray *in_array; 
    float3 *dptr;
    size_t num_bytes;

    unsigned int numRuns = this->visibleAtomCount / 512U + ( ( this->visibleAtomCount % 512U ) == 0 ? 0:1);

    for( unsigned int cnt = 0; cnt < numRuns; ++cnt ) {
		// map Texture buffer objects to get CUDA device pointers
		checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexResource, 0));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&in_array, cudaTexResource, 0, 0));
		// map OpenGL buffer object for writing from CUDA
		checkCudaErrors( cudaGraphicsMapResources( 1, &this->cudaVboResource, 0));
		checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, this->cudaVboResource));
		
        // write VBO using CUDA
		computeVisibleTriangleVBOCuda( dptr, this->m_dPoint1, in_array, 
            //this->m_dPoint2, this->m_dPoint3,
            m_dSortedPos, m_dVisibleAtoms,
            mol->AtomCount(), std::min( this->visibleAtomCount, 512U), MAX_ATOM_VICINITY, cnt * 512U);

		// unmap buffer object (vbo)
		checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaVboResource, 0));
		// unmap buffer object (tex)
		checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaTexResource, 0));

        // enable triangle drawing shader
        this->drawCUDATriangleShader.Enable();
        GLuint attribCol = glGetAttribLocation( this->drawCUDATriangleShader, "attribCol");
        glEnableVertexAttribArray( attribCol);
	    // set vertex pointer and draw it
        glEnableClientState( GL_VERTEX_ARRAY);
        //glEnableClientState( GL_COLOR_ARRAY);
        glBindBuffer( GL_ARRAY_BUFFER, this->triangleVBO);
        //glColorPointer( 3, GL_FLOAT, sizeof(float)*6, (const GLfloat *)0+3);
        glVertexAttribPointer( attribCol, 3, GL_FLOAT, 0, sizeof(float)*6, (const GLfloat *)0+3);
	    glVertexPointer( 3, GL_FLOAT, sizeof(float)*6, (const GLfloat *)0);
        //glInterleavedArrays( GL_C3F_V3F, 0, 0);
        glDrawArrays( GL_TRIANGLES, 0, count / 6);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
        //glDisableClientState( GL_COLOR_ARRAY);
        glDisableClientState( GL_VERTEX_ARRAY);
        // disable triangle drawing shader
        glDisableVertexAttribArray( attribCol);
        this->drawCUDATriangleShader.Disable();
    }


}

/*
 * create geometric primitives for ray casting
 */
unsigned int MoleculeCudaSESRenderer::CreateGeometricPrimitivesCuda( MolecularDataCall *mol) {

    /*
    // render all visible triangles using CUDA to prepare the VBOs
    this->RenderVisibleTrianglesCuda( mol);
    return 0;
    */

    // create the VBOs for transform feedback of visible triangles
    this->CreateTransformFeedbackVBO( mol);

    // prepare the vertex array for fast rendering of the points
    if( !this->visTriaTestVerts ) {
        this->visTriaTestVerts = new float[MAX_ATOM_VICINITY * MAX_ATOM_VICINITY * mol->AtomCount() * 4];
        unsigned int idx = 0;
        for( unsigned int cntY = 0; cntY < mol->AtomCount(); ++cntY ) {
            for( unsigned int cntX = 0; cntX < MAX_ATOM_VICINITY*MAX_ATOM_VICINITY; ++cntX ) {
                this->visTriaTestVerts[idx] = float(cntX);
                this->visTriaTestVerts[idx+1] = float(cntY);
                this->visTriaTestVerts[idx+2] = 0.0f;
                this->visTriaTestVerts[idx+3] = 1.0f;
                idx += 4;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // use a geometry shader to draw all array indices as points and loop up in
    // the texture wether they are visible, use transform feedback to get the 
    // indices of the visible triangles
    ///////////////////////////////////////////////////////////////////////////

    // enable and set up visible index drawing shader
    this->visibleTriangleIdxShader.Enable();
    glUniform1i( this->drawVisibleTriangleShader.ParameterLocation( "markerTex"), 0);

    // bind textures
    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, this->visibleTriangleColor);

    // temp variable for the number of primitives
    GLuint primitiveCount = 0;
    // set base for transform feedback
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, this->sphericalTriaVBO);
    // start transform feedback
    glBeginTransformFeedbackNV( GL_POINTS );
    // disable rasterization
    glEnable( GL_RASTERIZER_DISCARD_NV);
    // start querying the number of primitives
    glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query);

    // draw triangles
    glEnableClientState( GL_VERTEX_ARRAY);
    glVertexPointer( 4, GL_FLOAT, 0, this->visTriaTestVerts);
    glDrawArrays( GL_POINTS, 0, MAX_ATOM_VICINITY * MAX_ATOM_VICINITY * this->visibleAtomCount);
    glDisableClientState( GL_VERTEX_ARRAY);

    // stop querying the number of primitives
    glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV);
    // enable rasterization
    glDisable( GL_RASTERIZER_DISCARD_NV);
    // stop transform feedback
    glEndTransformFeedbackNV();
    // read back query results
    glGetQueryObjectuiv( query, GL_QUERY_RESULT, &primitiveCount);
    // unbind base for transform feedback
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0);
    //std::cout << "# of primitives read back: " << primitiveCount << std::endl;

    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, 0);
    this->visibleTriangleIdxShader.Disable();

    ///////////////////////////////////////////////////////////////////////////
    // use CUDA to write the VBOs for drawing the primitives of the SES
    ///////////////////////////////////////////////////////////////////////////

    // register this buffer object with CUDA
    if( !this->cudaVisTriaVboResource )
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &this->cudaVisTriaVboResource, this->sphericalTriaVBO, cudaGraphicsMapFlagsReadOnly));

    if( !glIsBuffer( this->torusVbo) ) {
        // the maximum number of atoms times the maximum number of neighborhood atoms times three (one for each edge)
        unsigned int max_torus_count = mol->AtomCount() * MAX_ATOM_VICINITY * 3;
        // create buffer objects
        glGenBuffers( 1, &this->torusVbo);
        CHECK_FOR_OGL_ERROR();
        glBindBuffer( GL_ARRAY_BUFFER, this->torusVbo);
        // buffer size: max_torus_count times 4 float per vertex times 4 vertices
        glBufferData( GL_ARRAY_BUFFER, max_torus_count*4*4*sizeof(GLfloat), 0, GL_DYNAMIC_COPY);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
        CHECK_FOR_OGL_ERROR();
        // register the vertex buffer object with CUDA
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &this->cudaTorusVboResource, this->torusVbo, cudaGraphicsMapFlagsWriteDiscard));
    }
    
    if( !glIsBuffer( this->sTriaVbo) ) {
        // the maximum number of atoms times the maximum number of neighborhood atoms times two (one for each probe pos)
        unsigned int max_st_count = mol->AtomCount() * MAX_ATOM_VICINITY * 2;
        // create buffer objects
        glGenBuffers( 1, &this->sTriaVbo);
        CHECK_FOR_OGL_ERROR();
        glBindBuffer( GL_ARRAY_BUFFER, this->sTriaVbo);
        // buffer size: max_st_count times 4 float per vertex times 4 vertices
        glBufferData( GL_ARRAY_BUFFER, max_st_count*4*4*sizeof(GLfloat), 0, GL_DYNAMIC_COPY);
        glBindBuffer( GL_ARRAY_BUFFER, 0);
        CHECK_FOR_OGL_ERROR();
        // register the vertex buffer object with CUDA
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &this->cudaSTriaVboResource, this->sTriaVbo, cudaGraphicsMapFlagsWriteDiscard));
    }
    
    float4 *outTorusVboPtr;
    float4 *outSTriaVboPtr;
    float4 *inVboPtr;
    size_t num_bytes;

	// map OpenGL buffer object for writing from CUDA
	checkCudaErrors( cudaGraphicsMapResources( 1, &this->cudaTorusVboResource, 0));
	checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&outTorusVboPtr, &num_bytes, this->cudaTorusVboResource));
	// map OpenGL buffer object for writing from CUDA
	checkCudaErrors( cudaGraphicsMapResources( 1, &this->cudaSTriaVboResource, 0));
	checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&outSTriaVboPtr, &num_bytes, this->cudaSTriaVboResource));
	// map OpenGL buffer object for reading from CUDA
	checkCudaErrors( cudaGraphicsMapResources( 1, &this->cudaVisTriaVboResource, 0));
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&inVboPtr, &num_bytes, this->cudaVisTriaVboResource));

    // compute tori using CUDA
    computeSESPrimiticesVBOCuda( outTorusVboPtr, outSTriaVboPtr, inVboPtr, 
        this->m_dSortedPos, this->m_dVisibleAtoms, this->m_dPoint1, this->m_dProbePosTable, 
        mol->AtomCount(), this->visibleAtomCount, MAX_ATOM_VICINITY, primitiveCount);

	// unmap buffer object (vbo)
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaTorusVboResource, 0));
	// unmap buffer object (vbo)
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaSTriaVboResource, 0));
	// unmap buffer object (vbo)
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &this->cudaVisTriaVboResource, 0));

    // return the number of visible triangles
    return (unsigned int)primitiveCount;
}