/*
 * ProteinRendererSESGPUCuda.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ProteinRendererSESGPUCuda.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))

#define _USE_MATH_DEFINES 1

// define the maximum allowed number of atoms in the vicinity of an atom
#define MAX_ATOM_VICINITY 64
// define the maximum dimension for the visibility fbo
#define VISIBILITY_FBO_DIM 512

#include "vislib/ShaderSource.h"
#include "CoreInstance.h"
#include "glh/glh_extensions.h"
#include <iostream>

#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
        
/*
 * ProteinRendererSESGPUCuda::ProteinRendererSESGPUCuda
 */
ProteinRendererSESGPUCuda::ProteinRendererSESGPUCuda( void ) : Renderer3DModule(),
    protDataCallerSlot( "getData", "Connects the protein SES rendering with protein data storage" ),
    proteinAtomId( 0), proteinAtomCount( 0),
    visibilityFBO( 0), visibilityColor( 0), visibilityDepth( 0),
    visibilityVertex( 0), visibilityVertexVBO( 0), visibleAtomMask( 0),
    vicinityTable( 0), fogStart( 0.5f), transparency( 1.0), probeRadius( 1.4f),
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
    delta( 0.01f), first( true), cudaInitalized( false)
{
    this->protDataCallerSlot.SetCompatibleCall<CallProteinDataDescription>();
    this->MakeSlotAvailable ( &this->protDataCallerSlot );

    // set frame counter to zero
    this->printFps = 0;

}
/*
 * ProteinRendererSESGPUCuda::~ProteinRendererSESGPUCuda
 */
ProteinRendererSESGPUCuda::~ProteinRendererSESGPUCuda(void)
{
    this->Release();
}


/*
 * ProteinRendererSESGPUCuda::release
 */
void protein::ProteinRendererSESGPUCuda::release( void )
{
    this->drawPointShader.Release();
    this->writeSphereIdShader.Release();
    this->sphereShader.Release();
    this->reducedSurfaceGeomShader.Release();

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
}


/*
 * ProteinRendererSESGPUCuda::create
 */
bool ProteinRendererSESGPUCuda::create( void )
{
    if( !glh_init_extensions( "GL_VERSION_2_0 GL_EXT_framebuffer_object GL_ARB_texture_float GL_EXT_gpu_shader4 GL_EXT_geometry_shader4 GL_EXT_bindable_uniform") )
        return false;

    if( !glh_init_extensions( "GL_NV_transform_feedback") )
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
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::writeSphereIdVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for sphere id writing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::writeSphereIdFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for sphere id writing shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->writeSphereIdShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create sphere Id writing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // load the shader source for point drawing renderer (fetch vertex position from texture //
    ///////////////////////////////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawPointVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for point drawing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawPointFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for point drawing shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->drawPointShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create point drawing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
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
    try
    {
        if( !this->sphereShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    ////////////////////////////////////////////////
    // load the shader source for reduced surface //
    ////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::reducedSurfaceVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for reduced surface (geom)", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::reducedSurfaceGeometry", geomSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load geometry shader source for reduced surface (geom)", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::reducedSurfaceFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for reduced surface (geom)", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->reducedSurfaceGeomShader.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count()) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
        else
        {    
            // set INPUT primitive type
            this->reducedSurfaceGeomShader.SetProgramParameter( GL_GEOMETRY_INPUT_TYPE_EXT , GL_POINTS);
            // set OUTPUT primitive type
            this->reducedSurfaceGeomShader.SetProgramParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
            // set maximum number of vertices to be generated by geometry shader
            this->reducedSurfaceGeomShader.SetProgramParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 1024);
            // link the shader
            if( !this->reducedSurfaceGeomShader.Link() )
            {
                throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
            }
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create reduced surface (geom) shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    ////////////////////////////////////////////////////////////////
    // load the shader source for the reduced surface computation //
    ////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::reducedSurface2Vertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for reduced surface computation shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::reducedSurface2Fragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for reduced surface computation shader", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->reducedSurfaceShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create reduced surface computation shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }

    /////////////////////////////////////////////////////////////////
    // load the shader source for reduced surface triangle drawing //
    /////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawTriangleVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for RS triangle drawing shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawTriangleFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for RS triangle drawing shader", this->ClassName() );
        return false;
    }
    try
    {
        fragSrc.Count();

        if( !this->drawTriangleShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create RS triangle drawing shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    /////////////////////////////////////////////////////////
    // load the shader source for visible triangle drawing //
    /////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawVisibleTriangleVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for visible triangle drawing", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawVisibleTriangleGeometry", geomSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load geometry shader source for visible triangle drawing", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::drawVisibleTriangleFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for visible triangle drawing", this->ClassName() );
        return false;
    }
    try
    {
        if( !this->drawVisibleTriangleShader.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count()) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
        else
        {    
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
            if( !this->drawVisibleTriangleShader.Link() )
            {
                throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
            }
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create visible triangle drawing geometry shader: %s\n", this->ClassName(), e.GetMsgA() );
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
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::torusFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for torus shader", this->ClassName() );
        return false;
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

    ////////////////////////////////////////////////////////////////
    // load the shader source for the spherical triangle renderer //
    ////////////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphericaltriangleVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for spherical triangle shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::ses::sphericaltriangleFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for spherical triangle shader", this->ClassName() );
        return false;
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
    
    //////////////////////////////////////////////////////////
    // load the shader source for adjacent triangle finding //
    //////////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentTriangleVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for adjacent triangle finding shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentTriangleFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for adjacent triangle finding shader", this->ClassName() );
        return false;
    }
    try
    {
        fragSrc.Count();

        if( !this->adjacentTriangleShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create adjacent triangle finding shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    //////////////////////////////////////////////////////
    // load the shader source for adjacent atom finding //
    //////////////////////////////////////////////////////
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentAtomVertex", vertSrc ) )
    {
        Log::DefaultLog.WriteMsg ( Log::LEVEL_ERROR, "%: Unable to load vertex shader source for adjacent atom marking shader", this->ClassName() );
        return false;
    }
    if( !ci->ShaderSourceFactory().MakeShaderSource( "reducedsurface::ses::adjacentAtomFragment", fragSrc ) )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for adjacent atom marking shader", this->ClassName() );
        return false;
    }
    try
    {
        fragSrc.Count();

        if( !this->adjacentAtomsShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) )
        {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch( vislib::Exception e )
    {
        Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create adjacent atom marking shader: %s\n", this->ClassName(), e.GetMsgA() );
        return false;
    }
    
    // create the VBOs for transform feedback of visible triangles
    this->CreateTransformFeedbackVBO();

    // everything went well, return success
    return true;
}


/*
 * Initialize CUDA
 */
bool ProteinRendererSESGPUCuda::initCuda( const CallProteinData *protein, uint gridDim) {
	// set number of atoms
	this->numAtoms = protein->ProteinAtomCount();

	// set grid dimensions
    this->gridSize.x = this->gridSize.y = this->gridSize.z = gridDim;
    this->numGridCells = this->gridSize.x * this->gridSize.y * this->gridSize.z;
    //float3 worldSize = make_float3( 2.0f, 2.0f, 2.0f);
	float3 worldSize = make_float3(
		protein->BoundingBox().Width(),
		protein->BoundingBox().Height(),
		protein->BoundingBox().Depth() );
    this->gridSortBits = 18;    // increase this for larger grids

    // set parameters
    this->params.gridSize = this->gridSize;
    this->params.numCells = this->numGridCells;
    this->params.numBodies = this->numAtoms;
    //this->params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	this->params.worldOrigin = make_float3(
		protein->BoundingBox().GetLeftBottomBack().GetX(),
		protein->BoundingBox().GetLeftBottomBack().GetY(),
		protein->BoundingBox().GetLeftBottomBack().GetZ());
	this->params.cellSize = make_float3( worldSize.x / this->gridSize.x, worldSize.y / this->gridSize.y, worldSize.z / this->gridSize.z);
	this->params.probeRadius = this->probeRadius;
    this->params.maxNumNeighbors = MAX_ATOM_VICINITY;

    // allocate host storage
    m_hPos = new float[this->numAtoms*4];
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
    //cudaMalloc(  (void**)&m_dPos, memSize);
    //cudaError e;
    //e = cudaGetLastError();

	// array for sorted atom positions
    allocateArray((void**)&m_dSortedPos, memSize);
	// array for the counted number of atoms
	allocateArray((void**)&m_dNeighborCount, this->numAtoms*sizeof(uint));
	// array for the neighbor atoms
	allocateArray((void**)&m_dNeighbors, this->numAtoms*MAX_ATOM_VICINITY*sizeof(uint));
	// array for the small circles
	allocateArray((void**)&m_dSmallCircles, this->numAtoms*MAX_ATOM_VICINITY*4*sizeof(float));

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

	return true;
}


/*
 * ProteinRendererSESGPUCuda::GetCapabilities
 */
bool ProteinRendererSESGPUCuda::GetCapabilities( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * ProteinRendererSESGPUCuda::GetExtents
 */
bool ProteinRendererSESGPUCuda::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    protein::CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();
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

    return true;
}


/*
 * ProteinRendererSESGPUCuda::Render
 */
bool ProteinRendererSESGPUCuda::Render( Call& call )
{
    /*
    if( this->probeRadius > 2.2f )
        delta = -delta;
    else if( this->probeRadius < 0.95f )
        delta = -delta;
    this->probeRadius += delta;
    */

    // start frame for frame rate counter
    this->fpsCounter.FrameBegin();

    // get pointer to CallProteinData
    protein::CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();
    // if something went wrong --> return
    if( !protein) return false;
    // execute the call
    if ( ! ( *protein )() )
        return false;
    
    // get camera information
    this->cameraInfo = dynamic_cast<view::CallRender3D*>( &call )->GetCameraParameters();
    // get bounding box of the protein
    this->bBox = protein->BoundingBox();
    
    // get clear color
    glGetFloatv( GL_COLOR_CLEAR_VALUE, this->clearCol);

    // ==================== Scale & Translate ====================
    glPushMatrix();
    float scale, xoff, yoff, zoff;
    vislib::math::Point<float, 3> bbc = protein->BoundingBox().CalcCenter();
    xoff = -bbc.X();
    yoff = -bbc.Y();
    zoff = -bbc.Z();
    scale = 2.0f / vislib::math::Max( vislib::math::Max ( protein->BoundingBox().Width(),
        protein->BoundingBox().Height() ), protein->BoundingBox().Depth() );
    glScalef( scale, scale, scale );
    glTranslatef( xoff, yoff, zoff );

    // =============== Query Camera View Dimensions ===============
    if( static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
        static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height )
    {
        this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
        this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
        this->CreateVisibilityFBO( VISIBILITY_FBO_DIM);
    }

    // ========================== Render ==========================
    glDisable( GL_BLEND);

    // Step 1: Find all visible atoms (using GPU-based visibility query)
	this->FindVisibleAtoms( protein);
	
    // Step 2a: Compute the vicinity table for all visible atoms
	//this->ComputeVisibleAtomsVicinityTable( protein);
    this->ComputeVicinityTableCUDA( protein);

    //////////////note////// START - NOT USED: THIS IS WAY TOO SLOW! ////////////////////
    // Step 2b: Compute the Reduced Surface via Geom Shader
    //this->ComputeRSGeomShader( protein);
    ////////////////////  END  - NOT USED: THIS IS WAY TOO SLOW! ////////////////////

    // Step 2b: Compute the Reduced Surface via Fragment Shader
    this->ComputeRSFragShader( protein);

    // Step 3a: Find all visible potential RS-faces
    this->FindVisibleTriangles( protein);
    // Step 3b: Find adjacent, occluded RS-faces
    this->FindAdjacentTriangles( protein);

    // Step 4a: Extract geometric primitives for ray casting
    unsigned int primitiveCount = this->CreateGeometricPrimitives( protein);
    // Step 4b: Search intersecting probes for singularity handling
    //this->CreateSingularityTexture( protein, primitiveCount/5);

    // Step 5: Render the SES using GPU ray casting
    this->RenderSES( protein, primitiveCount);

    // check for GL errors
    CHECK_FOR_OGL_ERROR();

    // reset clear color
    glClearColor( clearCol[0], clearCol[1], clearCol[2], clearCol[3]);

    ////////////////////////////////////////////////////////////////////
    // TEST render atoms as points START
    /*
	float viewportStuff[4] = {
		cameraInfo->TileRect().Left(), cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(), cameraInfo->TileRect().Height()
	};
	if( viewportStuff[2] < 1.0f ) viewportStuff[2] = 1.0f;
	if( viewportStuff[3] < 1.0f ) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];
	// enable sphere shader
	this->sphereShader.Enable();
	// set shader variables
    glUniform4fv( this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv( this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fv( this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    glUniform3f( this->sphereShader.ParameterLocation( "zValues"), 1.0f, cameraInfo->NearClip(), cameraInfo->FarClip());
    //render 
    unsigned int cnt;
    glDisable( GL_LIGHTING);
    glEnable( GL_DEPTH_TEST);
    glColor4f( 1, 0, 0, 1);
    glBegin( GL_POINTS );
    for( cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {
        if( this->visibleAtomMask[cnt] > 0.5f ) {
            glVertex4f( protein->ProteinAtomPositions()[cnt*3+0],
                protein->ProteinAtomPositions()[cnt*3+1],
                protein->ProteinAtomPositions()[cnt*3+2],
                protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius()*0.2f );
        }
    }
    glEnd(); // GL_POINTS
    this->sphereShader.Disable();
    */
    // TEST render atoms as points END
    ////////////////////////////////////////////////////////////////////

    // START draw overlay
    /*
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
    glColor4f( 1.0f,  1.0f,  1.0f,  1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f( 0, 0);
    glVertex2f( 0, 0);
    glTexCoord2f( 1, 0);
    glVertex2f( 1, 0);
    glTexCoord2f( 1, 1);
    glVertex2f( 1, 0.1f);
    glTexCoord2f( 0, 1);
    glVertex2f( 0, 0.1f);
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
    */

    // ================= Undo Scale & Translate =================
    glPopMatrix();

    glEnable( GL_BLEND);

    // end frame for frame rate counter and output average fps
    this->fpsCounter.FrameEnd();
    this->printFps++;
    if( this->printFps >= 10 ) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO,
            "%s: Average frame rate: %f", this->ClassName(), this->fpsCounter.FPS() );
        this->printFps = 0;
    }

    // everything went well
    return true;
}

/*
 * ProteinRendererSESGPUCuda::deinitialise
 */
void ProteinRendererSESGPUCuda::deinitialise(void)
{
}


/*
 * renders all atoms using GPU ray casting and write atom ID
 */
void ProteinRendererSESGPUCuda::RenderAtomIdGPU( const CallProteinData *protein) {
    // initialize Id array if necessary
    if( protein->ProteinAtomCount() != this->proteinAtomCount ) {
        // set correct number of protein atoms
        this->proteinAtomCount = protein->ProteinAtomCount();
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
            this->proteinAtomId[cnt*3+1] = protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius();
        }
        // resize the lists for visible atoms and the Ids of the visible atoms
        if( this->visibleAtomsList )
            delete[] this->visibleAtomsList;
        this->visibleAtomsList = new float[this->proteinAtomCount*4];
        if( this->visibleAtomsIdList )
            delete[] this->visibleAtomsIdList;
        this->visibleAtomsIdList = new float[this->proteinAtomCount*4];
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
    glVertexPointer( 3, GL_FLOAT, 0, protein->ProteinAtomPositions());
    glDrawArrays( GL_POINTS, 0, protein->ProteinAtomCount());
	// disable sphere shader
	glDisableClientState( GL_COLOR_ARRAY);
	glDisableClientState( GL_VERTEX_ARRAY);

	// disable sphere shader
	this->writeSphereIdShader.Disable();
}


/*
 * create the FBO for visibility test
 */
void ProteinRendererSESGPUCuda::CreateVisibilityFBO( unsigned int maxSize) {
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
void ProteinRendererSESGPUCuda::CreateVisibleAtomsFBO( unsigned int atomCount) {
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
void ProteinRendererSESGPUCuda::CreateTriangleFBO( unsigned int atomCount, unsigned int vicinityCount) {

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
void ProteinRendererSESGPUCuda::CreateVisibleTriangleFBO( unsigned int atomCount, unsigned int vicinityCount) {
    
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
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, vicinityCount*vicinityCount, atomCount, 0, GL_RGBA, GL_FLOAT, NULL);
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
void ProteinRendererSESGPUCuda::FindVisibleAtoms( const CallProteinData *protein) {
    // counter
    unsigned int cnt;

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
    glOrtho( 0.0, double( protein->ProteinAtomCount()), 0.0, 1.0, 0.0, 1.0);
    glDisable( GL_LIGHTING);

    // set viewport, get and set clearcolor, start rendering to framebuffer
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
    glViewport( 0, 0, protein->ProteinAtomCount(), 1);
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
    memset( this->visibleAtomMask, 0, sizeof(float)*protein->ProteinAtomCount());
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsColor);
    glGetTexImage( GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, this->visibleAtomMask);
    glBindTexture( GL_TEXTURE_2D, 0);
    // TODO test alternative method:
    //    - render one vertex per atom
    //    - use GeomShader to emit only visible atoms
    //    - use TransformFeedback to get atom positions

    // copy only the visible atom
    this->visibleAtomCount = 0;
    for( cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {
        // check for atom visibility in mask
        if( this->visibleAtomMask[cnt] > 0.5f ) {
            //  write atoms pos (x,y,z) and radius
            this->visibleAtomsList[this->visibleAtomCount*4+0] = protein->ProteinAtomPositions()[cnt*3+0];
            this->visibleAtomsList[this->visibleAtomCount*4+1] = protein->ProteinAtomPositions()[cnt*3+1];
            this->visibleAtomsList[this->visibleAtomCount*4+2] = protein->ProteinAtomPositions()[cnt*3+2];
            this->visibleAtomsList[this->visibleAtomCount*4+3] = protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius();
            this->visibleAtomsIdList[this->visibleAtomCount*4+0] = float( cnt);
            // write atom Id
            this->visibleAtomCount++;
        }
    }
    // generate textures for visible atoms, if necessary
    if( !glIsTexture( this->visibleAtomsTex) )
        glGenTextures( 1, &this->visibleAtomsTex);
    if( !glIsTexture( this->visibleAtomsIdTex) )
        glGenTextures( 1, &this->visibleAtomsIdTex);
    // create visible atoms texture
    glBindTexture( GL_TEXTURE_2D, this->visibleAtomsTex);
    glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGBA16F, visibleAtomCount, 1, 0, GL_RGBA, GL_FLOAT, this->visibleAtomsList);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);
    // create visible atoms Id texture
    glBindTexture( GL_TEXTURE_2D, visibleAtomsIdTex);
    glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGBA16F, visibleAtomCount, 1, 0, GL_RGBA, GL_FLOAT, this->visibleAtomsIdList);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);

}


/*
 * Find for each visible atom all atoms that are in the proximity.
 */
void ProteinRendererSESGPUCuda::ComputeVisibleAtomsVicinityTable( const CallProteinData *protein) {

    // temporary variables
    unsigned int cnt1, cnt2, vicinityCounter;
    vislib::math::Vector<float, 3> tmpVec, pos;
    float radius, rad, distance, threshold;
	unsigned int cnt, maxXId, maxYId, maxZId;
	int cntX, cntY, cntZ, xId, yId, zId;
    unsigned int xCoord, yCoord, zCoord, voxelIdx, firstIdx, atomCnt;
    
    // set up vicinity table, if necessary --> reserve one aditional vec for # of vicinity atoms
    if( !this->vicinityTable )
        this->vicinityTable = new float[protein->ProteinAtomCount() * ( MAX_ATOM_VICINITY + 1) * 4];
    // reset all items in vicinity table to zero
    memset( this->vicinityTable, 0, sizeof(float) * protein->ProteinAtomCount() * ( MAX_ATOM_VICINITY + 1) * 4);

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
    
    for( cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt )
    {
        // get position of current atom
        tmpVec.SetX( protein->ProteinAtomPositions()[cnt*3+0]);
        tmpVec.SetY( protein->ProteinAtomPositions()[cnt*3+1]);
        tmpVec.SetZ( protein->ProteinAtomPositions()[cnt*3+2]);
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
    for( cnt1 = 0; cnt1 < protein->ProteinAtomCount(); ++cnt1 ) {
        // continue if atom is not visible
        if( this->visibleAtomMask[cnt1] < 0.5 ) continue;

		// get position of current atom
		tmpVec.SetX( protein->ProteinAtomPositions()[cnt1*3+0]);
		tmpVec.SetY( protein->ProteinAtomPositions()[cnt1*3+1]);
		tmpVec.SetZ( protein->ProteinAtomPositions()[cnt1*3+2]);
		// get the radius of current atom
		radius = protein->AtomTypes()[protein->ProteinAtomData()[cnt1].TypeIndex()].Radius();
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
		                pos.SetX( protein->ProteinAtomPositions()[cnt2*3+0]);
		                pos.SetY( protein->ProteinAtomPositions()[cnt2*3+1]);
		                pos.SetZ( protein->ProteinAtomPositions()[cnt2*3+2]);
                        rad = protein->AtomTypes()[protein->ProteinAtomData()[cnt2].TypeIndex()].Radius();
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
    glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGBA32F, ( MAX_ATOM_VICINITY + 1), protein->ProteinAtomCount(), 0, GL_RGBA, GL_FLOAT, this->vicinityTable);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);
    
}
/*
 * Find for each visible atom all atoms that are in the proximity.
 */
void ProteinRendererSESGPUCuda::ComputeVicinityTableCUDA( const CallProteinData *protein) {

    // temporary variables
	unsigned int cnt, neighborCnt, vicinityCounter;
    
    // set up vicinity table, if necessary --> reserve one aditional vec for # of vicinity atoms
    if( !this->vicinityTable )
        this->vicinityTable = new float[protein->ProteinAtomCount() * ( MAX_ATOM_VICINITY + 1) * 4];
    // reset all items in vicinity table to zero
    memset( this->vicinityTable, 0, sizeof(float) * protein->ProteinAtomCount() * ( MAX_ATOM_VICINITY + 1) * 4);

	// try to initialize CUDA
	if( !this->cudaInitalized ) {
		cudaInitalized = this->initCuda( protein, 16);
		vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO, 
			"%s: CUDA initialization: %i", this->ClassName(), cudaInitalized );
	}
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
        sortParticles( m_dGridParticleHash, m_dGridParticleIndex, this->numAtoms);

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
		countNeighbors(
			m_dNeighborCount,
			m_dNeighbors,
			m_dSmallCircles,
			m_dSortedPos,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			this->numAtoms,
            MAX_ATOM_VICINITY,
			this->numGridCells);

        copyArrayFromDevice( m_hNeighborCount, m_dNeighborCount, 0, sizeof(uint)*protein->ProteinAtomCount());
        copyArrayFromDevice( m_hNeighbors, m_dNeighbors, 0, sizeof(uint)*protein->ProteinAtomCount()*MAX_ATOM_VICINITY);
        copyArrayFromDevice( m_hPos, m_dSortedPos, 0, sizeof(float)*4*protein->ProteinAtomCount());
	}

    // loop over all atoms
    for( cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {
        // continue if atom is not visible
        if( this->visibleAtomMask[cnt] < 0.5 ) continue;
        // reset id
        vicinityCounter = 1;

	    // loop over all neighbors and write them to the table
        for( neighborCnt = 0; neighborCnt < m_hNeighborCount[cnt]; ++neighborCnt )
	    {
            this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 0] = 
                m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+0];
            this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 1] = 
                m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+1];
            this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 2] = 
                m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+2];
            this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4 + 4 * vicinityCounter + 3] = 
                m_hPos[m_hNeighbors[cnt * MAX_ATOM_VICINITY + neighborCnt]*4+3];
            vicinityCounter++;
	    }
        this->vicinityTable[cnt * ( MAX_ATOM_VICINITY + 1) * 4] = float(vicinityCounter - 1);
    } // loop over all atoms

    // generate vicinity table texture if necessary
    if( !glIsTexture( this->vicinityTableTex ) ) 
        glGenTextures( 1, &this->vicinityTableTex);
    // create texture containing the vicinity table
    glBindTexture( GL_TEXTURE_2D, this->vicinityTableTex);
    glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGBA32F, ( MAX_ATOM_VICINITY + 1), protein->ProteinAtomCount(), 0, GL_RGBA, GL_FLOAT, this->vicinityTable);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);
    
}


/*
 * Write atom positions and radii to an array for processing in CUDA
 */
void ProteinRendererSESGPUCuda::writeAtomPositions( const CallProteinData *protein ) {
	// write atoms to array
	int p = 0;
	for( unsigned int cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {
		// write pos and rad to array
		m_hPos[p++] = protein->ProteinAtomPositions()[cnt*3+0];
		m_hPos[p++] = protein->ProteinAtomPositions()[cnt*3+1];
		m_hPos[p++] = protein->ProteinAtomPositions()[cnt*3+2];
		m_hPos[p++] = protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius();
	}

	// setArray( POSITION, m_hPos, 0, this->numAtoms);
	copyArrayToDevice( this->m_dPos, this->m_hPos, 0, this->numAtoms*4*sizeof(float));
}


/*
 * Compute the Reduced Surface using the Geometry Shader.
 */
void ProteinRendererSESGPUCuda::ComputeRSGeomShader( const CallProteinData *protein) {
    glDisable( GL_LIGHTING);
    glEnable( GL_DEPTH_TEST);
    // bind textures
    glEnable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, this->vicinityTableTex);
	// enable geometry shader
	this->reducedSurfaceGeomShader.Enable();
	// set shader variables
    glUniform1i( this->reducedSurfaceGeomShader.ParameterLocation("vicinityTex"), 0);
    glUniform1f( this->reducedSurfaceGeomShader.ParameterLocation("probeRadius"), this->probeRadius);
    // create and fill vertex and color arrays
    if( !this->atomPosRSGS ) this->atomPosRSGS = new float[protein->ProteinAtomCount()*3];
    if( !this->atomColRSGS ) this->atomColRSGS = new float[protein->ProteinAtomCount()*3];
    unsigned int cntA, counterA;
    counterA = 0;
    for( cntA = 0; cntA < protein->ProteinAtomCount(); ++cntA ) {
        if( this->visibleAtomMask[cntA] > 0.5f ) {
            this->atomColRSGS[counterA*3+0] = float( cntA);
            this->atomColRSGS[counterA*3+1] = protein->AtomTypes()[protein->ProteinAtomData()[cntA].TypeIndex()].Radius();
            this->atomColRSGS[counterA*3+2] = 0.0f;
            this->atomPosRSGS[counterA*3+0] = protein->ProteinAtomPositions()[cntA*3+0];
            this->atomPosRSGS[counterA*3+1] = protein->ProteinAtomPositions()[cntA*3+1];
            this->atomPosRSGS[counterA*3+2] = protein->ProteinAtomPositions()[cntA*3+2];
            counterA++;
        }
    }
    // render the arrays
    glEnableClientState( GL_VERTEX_ARRAY);
    glEnableClientState( GL_COLOR_ARRAY);
    // set vertex and color pointers and draw them
    glColorPointer( 3, GL_FLOAT, 0, this->atomColRSGS );
    glVertexPointer( 3, GL_FLOAT, 0, this->atomPosRSGS );
    glDrawArrays( GL_POINTS, 0, counterA);
    // disable sphere shader
    glDisableClientState( GL_COLOR_ARRAY);
    glDisableClientState( GL_VERTEX_ARRAY);
    
    // disable shader and unbind texture
    this->reducedSurfaceGeomShader.Disable();
    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable( GL_TEXTURE_2D );
}

/*
 * Compute the Reduced Surface using the Fragment Shader.
 */
void ProteinRendererSESGPUCuda::ComputeRSFragShader( const CallProteinData *protein) {
    // create FBO
    this->CreateTriangleFBO( protein->ProteinAtomCount(), MAX_ATOM_VICINITY);
    
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
void ProteinRendererSESGPUCuda::RenderTriangles( const CallProteinData *protein) {
    // write VBO for fast drawing, if necessary
    if( !glIsBuffer( this->triangleVBO ) ) {
        // generate triangle VBO
        glGenBuffers( 1, &this->triangleVBO);
        // --- write triangle vertex positions (texture coordinates, respectively) ---
        // counter
        unsigned int cntX, cntY;
        unsigned int cnt = 0;
        // number of triangle vertices (3 floats per vertex, 3 vertices per triangle)
        unsigned int numVerts = MAX_ATOM_VICINITY*MAX_ATOM_VICINITY*protein->ProteinAtomCount()*9;
        // delete triangle vertex list, if necessary
        if( this->triangleVertex )
            delete[] this->triangleVertex;
        this->triangleVertex = new float[numVerts];
        // for each pixel, write the coordinates to the vertex list
        for( cntY = 0; cntY < protein->ProteinAtomCount(); ++cntY ) {
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
void ProteinRendererSESGPUCuda::FindVisibleTriangles( const CallProteinData *protein) {
    // generate FBO
    this->CreateVisibleTriangleFBO( protein->ProteinAtomCount(), MAX_ATOM_VICINITY);

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
void ProteinRendererSESGPUCuda::CreateAdjacentTriangleFBO( unsigned int atomCount, unsigned int vicinityCount) {

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
void ProteinRendererSESGPUCuda::FindAdjacentTriangles( const CallProteinData *protein) {
    // 1.) create fbo for offscreen rendering ( MAX_ATOM_VICINITY * protein->ProteinAtomCount() )
    // 2.) shader: for each edge:
    //      2.1) test how many visible triangles are connected
    //      2.2) 2 --> discard; 0 --> discard; 1 --> continue with 2.3)
    //      2.3) get all invisible triangles and compute angle to visible triangle
    //      2.4) mark the invisible triangle with the least angle as visible
    // 3.) write all found adjacent triangles to visible triangles lookup texture

    // ----- create FBO (1.) -----
    this->CreateAdjacentTriangleFBO( protein->ProteinAtomCount(), MAX_ATOM_VICINITY);

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
 * create the VBO for transform feedback
 */
void ProteinRendererSESGPUCuda::CreateTransformFeedbackVBO() {
    /////////////////
    // set up VBOs //
    /////////////////
    
    // TODO: this should be the maximum number of atoms!!
    unsigned int max_buffer_verts = 100000;
    // delete VBO, if necessary
    if( glIsBuffer( this->sphericalTriaVBO[0]) ||
        glIsBuffer( this->sphericalTriaVBO[1]) ||
        glIsBuffer( this->sphericalTriaVBO[2]) ||
        glIsBuffer( this->sphericalTriaVBO[3]) )
        glDeleteBuffers( 4, this->sphericalTriaVBO);
    CHECK_FOR_OGL_ERROR();
    // create buffer objects
    glGenBuffers( 4, this->sphericalTriaVBO);
    CHECK_FOR_OGL_ERROR();
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[0]);
    glBufferData( GL_ARRAY_BUFFER, max_buffer_verts*4*sizeof(GLfloat), 0, GL_DYNAMIC_COPY);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[1]);
    glBufferData( GL_ARRAY_BUFFER, max_buffer_verts*4*sizeof(GLfloat), 0, GL_DYNAMIC_COPY);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[2]);
    glBufferData( GL_ARRAY_BUFFER, max_buffer_verts*4*sizeof(GLfloat), 0, GL_DYNAMIC_COPY);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[3]);
    glBufferData( GL_ARRAY_BUFFER, max_buffer_verts*4*sizeof(GLfloat), 0, GL_DYNAMIC_COPY);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    CHECK_FOR_OGL_ERROR();
    // delete query objects, if necessary
    if( this->query ) {
        glDeleteQueries( 1, &this->query);
        CHECK_FOR_OGL_ERROR();
    }

    this->drawVisibleTriangleShader.Enable();
    // set varyings for transform feedback
    
    // ---> ACCORDING TO SPECIFICATION, BUT NOT WORKING!!!
    //const char *varyings[] = { "gl_Position", "attribVec1", "attribVec2", "attribVec3" };
    //glTransformFeedbackVaryingsNV( this->drawVisibleTriangleShader, 4, varyings, GL_SEPARATE_ATTRIBS_NV);
    
    // ---> works only with older glext.h
    GLint varyingLoc[] = { glGetVaryingLocationNV( this->drawVisibleTriangleShader, "gl_Position"),
        glGetVaryingLocationNV( this->drawVisibleTriangleShader, "attribVec1"),
        glGetVaryingLocationNV( this->drawVisibleTriangleShader, "attribVec2"),
        glGetVaryingLocationNV( this->drawVisibleTriangleShader, "attribVec3") };
    CHECK_FOR_OGL_ERROR();
    glTransformFeedbackVaryingsNV( this->drawVisibleTriangleShader, 4, varyingLoc, GL_SEPARATE_ATTRIBS_NV);
    
    CHECK_FOR_OGL_ERROR();
    this->drawVisibleTriangleShader.Disable();
    
    // create query objects
    glGenQueries( 1, &this->query);
    CHECK_FOR_OGL_ERROR();
}


/*
 * create geometric primitives for ray casting
 */
unsigned int ProteinRendererSESGPUCuda::CreateGeometricPrimitives( const CallProteinData *protein) {
    // enable and set up triangle drawing shader
    this->drawVisibleTriangleShader.Enable();
    glUniform1i( this->drawVisibleTriangleShader.ParameterLocation( "positionTex0"), 0);
    glUniform1i( this->drawVisibleTriangleShader.ParameterLocation( "positionTex1"), 1);
    glUniform1i( this->drawVisibleTriangleShader.ParameterLocation( "positionTex2"), 2);
    glUniform1i( this->drawVisibleTriangleShader.ParameterLocation( "normalTex"), 3);
    glUniform1i( this->drawVisibleTriangleShader.ParameterLocation( "markerTex"), 4);
    glUniform1f( this->drawVisibleTriangleShader.ParameterLocation( "probeRadius"), this->probeRadius);

    // bind textures
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

    // temp variable for the number of primitives
    GLuint primitiveCount = 0;
    // set base for transform feedback
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, this->sphericalTriaVBO[0]);
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 1, this->sphericalTriaVBO[1]);
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 2, this->sphericalTriaVBO[2]);
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 3, this->sphericalTriaVBO[3]);
    // start transform feedback
    glBeginTransformFeedbackNV( GL_POINTS );
    // disable rasterization
    glEnable( GL_RASTERIZER_DISCARD_NV);
    // start querying the number of primitives
    glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query);

    // draw triangles
    glBegin( GL_POINTS);
    for( unsigned int cntX = 0; cntX < MAX_ATOM_VICINITY*MAX_ATOM_VICINITY; ++cntX ) {
        for( unsigned int cntY = 0; cntY < this->visibleAtomCount; ++cntY ) {
            glVertex3f( float(cntX), float(cntY), 0.0f);
        }
    }
    glEnd();

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
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 1, 0);
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 2, 0);
    glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 3, 0);
    //std::cout << "# of primitives read back: " << primitiveCount << std::endl;

    glActiveTexture( GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_2D, 0);
    this->drawVisibleTriangleShader.Disable();

    return (unsigned int)primitiveCount;
}


/*
 * create the singularity texture
 */
void ProteinRendererSESGPUCuda::CreateSingularityTexture( const CallProteinData *protein, unsigned int numProbes) {
    // bind the vertex buffer object containing the probe positions
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[0]);
    // map the vertex buffer to a pointer
	float *pos = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_ONLY);
    
    if( this->singTexData )
        delete[] this->singTexData;
    this->singTexData = new float[MAX_ATOM_VICINITY*numProbes*3];
    if( this->singTexCoords )
        delete[] this->singTexCoords;
    this->singTexCoords = new float[numProbes*3];
    if( !glIsTexture( this->singTex) )
        glGenTextures( 1, &this->singTex);
    
    /*
    // temporary variables
    unsigned int cnt1, cnt2;
    vislib::math::Vector<float, 3> tmpVec1, tmpVec2, pos1, pos2;
    unsigned int probeCnt;
    // search for intersecting probes
    // TODO: replace this by a grid-based search for efficiency!!!!
    for( cnt1 = 0; cnt1 < numProbes; ++cnt1 ) {
        probeCnt = 0;
        // fetch the first probe
        tmpVec1.SetX( pos[cnt1*20+0]);
        tmpVec1.SetY( pos[cnt1*20+1]);
        tmpVec1.SetZ( pos[cnt1*20+2]);
        // fetch the second probe
        tmpVec2.SetX( pos[cnt1*20+4]);
        tmpVec2.SetY( pos[cnt1*20+5]);
        tmpVec2.SetZ( pos[cnt1*20+6]);
        // test all probes against the two probes
        for( cnt2 = 0; cnt2 < numProbes; ++cnt2 ) {
            if( cnt1 != cnt2 ) {
                // fetch the first probe pos
                pos1.SetX( pos[cnt2*20+0]);
                pos1.SetY( pos[cnt2*20+1]);
                pos1.SetZ( pos[cnt2*20+2]);
                // fetch the second probe pos
                pos2.SetX( pos[cnt2*20+4]);
                pos2.SetY( pos[cnt2*20+5]);
                pos2.SetZ( pos[cnt2*20+6]);
                // don't add more than MAX_ATOM_VICINITY intersecting probes
                if( ( ( tmpVec1 - pos1).Length() < this->probeRadius*2.0f || ( tmpVec2 - pos1).Length() < this->probeRadius*2.0f )
                    && probeCnt < MAX_ATOM_VICINITY ) {
                    this->singTexData[cnt1*MAX_ATOM_VICINITY*3 + 3 * probeCnt + 0] = pos1.GetX();
                    this->singTexData[cnt1*MAX_ATOM_VICINITY*3 + 3 * probeCnt + 1] = pos1.GetY();
                    this->singTexData[cnt1*MAX_ATOM_VICINITY*3 + 3 * probeCnt + 2] = pos1.GetZ();
                    probeCnt++;
                }
                if( ( pos1 - pos2).Length() > vislib::math::FLOAT_EPSILON ) {
                    if( ( ( tmpVec1 - pos2).Length() < this->probeRadius*2.0f || ( tmpVec2 - pos2).Length() < this->probeRadius*2.0f )
                        && probeCnt < MAX_ATOM_VICINITY ) {
                        this->singTexData[cnt1*MAX_ATOM_VICINITY*3 + 3 * probeCnt + 0] = pos2.GetX();
                        this->singTexData[cnt1*MAX_ATOM_VICINITY*3 + 3 * probeCnt + 1] = pos2.GetY();
                        this->singTexData[cnt1*MAX_ATOM_VICINITY*3 + 3 * probeCnt + 2] = pos2.GetZ();
                        probeCnt++;
                    }
                }
            }
        }
        this->singTexCoords[cnt1*3+0] = float( probeCnt);
        this->singTexCoords[cnt1*3+1] = 0.0f;
        this->singTexCoords[cnt1*3+2] = float( cnt1);
    }
    */
    
    // temporary variables
    unsigned int cnt1, cnt2, vicinityCounter;
    vislib::math::Vector<float, 3> tmpVec1, tmpVec2, pos1, pos2;
    float distance11, distance12, distance21, distance22, threshold;
	unsigned int cnt, maxXId, maxYId, maxZId;
	int cntX, cntY, cntZ, xId, yId, zId;
    unsigned int xCoord, yCoord, zCoord, voxelIdx, firstIdx, probeCnt;
    
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
        if( this->probeVoxelMap )
            delete[] this->probeVoxelMap;
        this->voxelMapSize = dimensions;
        this->probeVoxelMap = new unsigned int[this->voxelMapSize * this->numAtomsPerVoxel];
    }
    memset( this->probeVoxelMap, 0, sizeof(unsigned int)*this->voxelMapSize*this->numAtomsPerVoxel);
    
    // fill probe voxel map with probe positions
    for( cnt = 0; cnt < numProbes; ++cnt )
    {
        // fetch the first probe
        tmpVec1.SetX( pos[cnt*20+0]);
        tmpVec1.SetY( pos[cnt*20+1]);
        tmpVec1.SetZ( pos[cnt*20+2]);
        // compute coordinates for the current probe
        xCoord = (unsigned int)std::min( maxXId,
                 (unsigned int)std::max( 0, (int)floorf( (tmpVec1.GetX() - bBox.Left()) / float(voxelLength))));
        yCoord = (unsigned int)std::min( maxYId,
                 (unsigned int)std::max( 0, (int)floorf( (tmpVec1.GetY() - bBox.Bottom()) / float(voxelLength))));
        zCoord = (unsigned int)std::min( maxZId,
                 (unsigned int)std::max( 0, (int)floorf( (tmpVec1.GetZ() - bBox.Back()) / float(voxelLength))));
        // add probe to voxel texture
        voxelIdx = vW * vH * zCoord + vW * yCoord + xCoord;
        firstIdx = voxelIdx * numAtomsPerVoxel;
        this->probeVoxelMap[firstIdx]++;
        probeCnt = this->probeVoxelMap[firstIdx];
        this->probeVoxelMap[firstIdx+probeCnt] = cnt;
        // fetch the second probe
        tmpVec2.SetX( pos[cnt*20+4]);
        tmpVec2.SetY( pos[cnt*20+5]);
        tmpVec2.SetZ( pos[cnt*20+6]);
        // add second probe only to voxel map if it is not equal to the first probe
        if( ( tmpVec1 - tmpVec2).Length() > vislib::math::FLOAT_EPSILON ) {
            // compute coordinates for the current probe
            xCoord = (unsigned int)std::min( maxXId,
                     (unsigned int)std::max( 0, (int)floorf( (tmpVec2.GetX() - bBox.Left()) / float(voxelLength))));
            yCoord = (unsigned int)std::min( maxYId,
                     (unsigned int)std::max( 0, (int)floorf( (tmpVec2.GetY() - bBox.Bottom()) / float(voxelLength))));
            zCoord = (unsigned int)std::min( maxZId,
                     (unsigned int)std::max( 0, (int)floorf( (tmpVec2.GetZ() - bBox.Back()) / float(voxelLength))));
            // add probe to voxel texture
            voxelIdx = vW * vH * zCoord + vW * yCoord + xCoord;
            firstIdx = voxelIdx * numAtomsPerVoxel;
            this->probeVoxelMap[firstIdx]++;
            probeCnt = this->probeVoxelMap[firstIdx];
            this->probeVoxelMap[firstIdx+probeCnt] = cnt;
        }
    }

    // set distance threshold
    threshold = 2.0f * this->probeRadius;

    // loop over all probes
    for( cnt1 = 0; cnt1 < numProbes; ++cnt1 ) {
        // reset id
        vicinityCounter = 0;
        // fetch the first probe
        tmpVec1.SetX( pos[cnt1*20+0]);
        tmpVec1.SetY( pos[cnt1*20+1]);
        tmpVec1.SetZ( pos[cnt1*20+2]);
	    xId = (int)std::max( 0, (int)floorf( ( tmpVec1.GetX() - bBox.Left()) / float(voxelLength)));
	    xId = (int)std::min( (int)maxXId, xId);
	    yId = (int)std::max( 0, (int)floorf( ( tmpVec1.GetY() - bBox.Bottom()) / float(voxelLength)));
	    yId = (int)std::min( (int)maxYId, yId);
	    zId = (int)std::max( 0, (int)floorf( ( tmpVec1.GetZ() - bBox.Back()) / float(voxelLength)));
	    zId = (int)std::min( (int)maxZId, zId);
	    // loop over all probes to find vicinity of first probe
	    for( cntX = ((xId > 0)?(-1):0); cntX < ((xId < (int)maxXId)?2:1); ++cntX )
	    {
		    for( cntY = ((yId > 0)?(-1):0); cntY < ((yId < (int)maxYId)?2:1); ++cntY )
		    {
			    for( cntZ = ((zId > 0)?(-1):0); cntZ < ((zId < (int)maxZId)?2:1); ++cntZ )
			    {
                    voxelIdx = vW * vH * (zId+cntZ) + vW * (yId+cntY) + (xId+cntX);
                    firstIdx = voxelIdx * this->numAtomsPerVoxel;
                    for( cnt = 0; cnt < this->probeVoxelMap[firstIdx]; ++cnt )
				    {
                        cnt2 = probeVoxelMap[firstIdx+cnt+1];
					    // don't check the same probe --> continue
					    if( cnt2 == cnt1 ) continue;
                        // get position of the second probe
                        // fetch the first probe pos
                        pos1.SetX( pos[cnt2*20+0]);
                        pos1.SetY( pos[cnt2*20+1]);
                        pos1.SetZ( pos[cnt2*20+2]);
                        // fetch the second probe pos
                        pos2.SetX( pos[cnt2*20+4]);
                        pos2.SetY( pos[cnt2*20+5]);
                        pos2.SetZ( pos[cnt2*20+6]);
					    // compute distances
					    distance11 = ( pos1 - tmpVec1).Length();
					    distance21 = ( pos2 - tmpVec1).Length();
					    // if distance < threshold --> add probe to vicinity
                        if( distance11 < threshold && vicinityCounter <= MAX_ATOM_VICINITY ) {
                            this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 0] = pos1.GetX();
                            this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 1] = pos1.GetY();
                            this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 2] = pos1.GetZ();
                            vicinityCounter++;
					    }
                        if( ( pos1 - pos2).Length() > vislib::math::FLOAT_EPSILON
                            && distance21 < threshold
                            && vicinityCounter <= MAX_ATOM_VICINITY ) {
                            this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 0] = pos2.GetX();
                            this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 1] = pos2.GetY();
                            this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 2] = pos2.GetZ();
                            vicinityCounter++;
					    }
				    }
			    }
		    }
	    } // END loop over all probes to find vicinity of first probe

        // fetch the second probe
        tmpVec2.SetX( pos[cnt1*20+4]);
        tmpVec2.SetY( pos[cnt1*20+5]);
        tmpVec2.SetZ( pos[cnt1*20+6]);
        // search vicinity for second probe only if it is not equal to the first probe
        if( ( tmpVec1 - tmpVec2).Length() > vislib::math::FLOAT_EPSILON ) {
	        xId = (int)std::max( 0, (int)floorf( ( tmpVec2.GetX() - bBox.Left()) / float(voxelLength)));
	        xId = (int)std::min( (int)maxXId, xId);
	        yId = (int)std::max( 0, (int)floorf( ( tmpVec2.GetY() - bBox.Bottom()) / float(voxelLength)));
	        yId = (int)std::min( (int)maxYId, yId);
	        zId = (int)std::max( 0, (int)floorf( ( tmpVec2.GetZ() - bBox.Back()) / float(voxelLength)));
	        zId = (int)std::min( (int)maxZId, zId);
	        // loop over all probes to find vicinity of second probe
	        for( cntX = ((xId > 0)?(-1):0); cntX < ((xId < (int)maxXId)?2:1); ++cntX )
	        {
		        for( cntY = ((yId > 0)?(-1):0); cntY < ((yId < (int)maxYId)?2:1); ++cntY )
		        {
			        for( cntZ = ((zId > 0)?(-1):0); cntZ < ((zId < (int)maxZId)?2:1); ++cntZ )
			        {
                        voxelIdx = vW * vH * (zId+cntZ) + vW * (yId+cntY) + (xId+cntX);
                        firstIdx = voxelIdx * this->numAtomsPerVoxel;
                        for( cnt = 0; cnt < this->probeVoxelMap[firstIdx]; ++cnt )
				        {
                            cnt2 = probeVoxelMap[firstIdx+cnt+1];
					        // don't check the same probe --> continue
					        if( cnt2 == cnt1 ) continue;
                            // get position of the second probe
                            // fetch the first probe pos
                            pos1.SetX( pos[cnt2*20+0]);
                            pos1.SetY( pos[cnt2*20+1]);
                            pos1.SetZ( pos[cnt2*20+2]);
                            // fetch the second probe pos
                            pos2.SetX( pos[cnt2*20+4]);
                            pos2.SetY( pos[cnt2*20+5]);
                            pos2.SetZ( pos[cnt2*20+6]);
					        // compute distances
					        distance12 = ( pos1 - tmpVec2).Length();
					        distance22 = ( pos2 - tmpVec2).Length();
					        // if distance < threshold --> add probe to vicinity
                            if( distance12 < threshold && vicinityCounter <= MAX_ATOM_VICINITY ) {
                                this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 0] = pos1.GetX();
                                this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 1] = pos1.GetY();
                                this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 2] = pos1.GetZ();
                                vicinityCounter++;
					        }
                            if( ( pos1 - pos2).Length() > vislib::math::FLOAT_EPSILON
                                && distance22 < threshold
                                && vicinityCounter <= MAX_ATOM_VICINITY ) {
                                this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 0] = pos2.GetX();
                                this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 1] = pos2.GetY();
                                this->singTexData[cnt1 * MAX_ATOM_VICINITY * 3 + 3 * vicinityCounter + 2] = pos2.GetZ();
                                vicinityCounter++;
					        }
				        }
			        }
		        }
	        } // END loop over all probes to find vicinity of second probe
        } // END search vicinity for second probe only if it is not equal to the first probe

        // write number of vicinity probes and texture coordinates
        this->singTexCoords[cnt1*3+0] = float( vicinityCounter);
        this->singTexCoords[cnt1*3+1] = 0.0f;
        this->singTexCoords[cnt1*3+2] = float( cnt1);
    } // loop over all atoms

    // unmap the vertex buffer
	glUnmapBuffer( GL_ARRAY_BUFFER);
    // unbind the vertex buffer object
	glBindBuffer( GL_ARRAY_BUFFER, 0);

    // create singularity texture
    glBindTexture( GL_TEXTURE_2D, this->singTex);
    glTexImage2D( GL_TEXTURE_2D, 0,  GL_RGB32F, MAX_ATOM_VICINITY, numProbes, 0, GL_RGB, GL_FLOAT, this->singTexData);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture( GL_TEXTURE_2D, 0);
}


/*
 * render all visible atoms using GPU ray casting.
 */
void ProteinRendererSESGPUCuda::RenderVisibleAtomsGPU( const CallProteinData *protein) {
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
        atomId = (unsigned int)this->visibleAtomsIdList[cnt*4];
        glColor3f( 1.0f, 0.0f, 0.0f);
        glVertex4f( protein->ProteinAtomPositions()[atomId*3+0],
            protein->ProteinAtomPositions()[atomId*3+1],
            protein->ProteinAtomPositions()[atomId*3+2],
            protein->AtomTypes()[protein->ProteinAtomData()[atomId].TypeIndex()].Radius() );
    }
    glEnd(); // GL_POINTS
	// disable sphere shader
	this->sphereShader.Disable();
}


/*
 * render the SES using GPU ray casting.
 */
void ProteinRendererSESGPUCuda::RenderSES( const CallProteinData *protein, unsigned int primitiveCount) {
    // render the atoms as spheres
    this->RenderVisibleAtomsGPU( protein);

    // set viewport
    float viewportStuff[4] =
    {
        cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(),
        cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height()
    };
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];
    // set fog color
    vislib::math::Vector<float, 3> fogCol( this->clearCol[0], this->clearCol[1], clearCol[2]);
    
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
    GLuint attribTexCoord1 = glGetAttribLocation( this->sphericalTriangleShader, "attribTexCoord1");
    // enable vertex attribute arrays for the attribute locations
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableVertexAttribArray( attribVec1);
    glEnableVertexAttribArray( attribVec2);
    glEnableVertexAttribArray( attribVec3);
    glEnableVertexAttribArray( attribTexCoord1);

    // bind buffers...
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[1]);
    glVertexAttribPointer( attribVec1, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[2]);
    glVertexAttribPointer( attribVec2, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[3]);
    glVertexAttribPointer( attribVec3, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[0]);
    glVertexPointer( 4, GL_FLOAT, sizeof(float)*20, NULL);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer( attribTexCoord1, 3, GL_FLOAT, 0, 0, this->singTexCoords);
    // ...and draw arrays
    glDrawArrays( GL_POINTS, 0, primitiveCount/5);

    // bind buffers...
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[1]);
    glVertexAttribPointer( attribVec1, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+4);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[2]);
    glVertexAttribPointer( attribVec2, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+4);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[3]);
    glVertexAttribPointer( attribVec3, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+4);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[0]);
    glVertexPointer( 4, GL_FLOAT, sizeof(float)*20, (const GLfloat *)0+4);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer( attribTexCoord1, 3, GL_FLOAT, 0, 0, this->singTexCoords);
    // ...and draw arrays
    glDrawArrays( GL_POINTS, 0, primitiveCount/5);

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
	//GLuint attribInColors = glGetAttribLocation( this->torusShader, "inColors");
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
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[1]);
	glVertexAttribPointer( attribInParams, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+8);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[2]);
	glVertexAttribPointer( attribQuatC, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+8);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[3]);
	glVertexAttribPointer( attribInSphere, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+8);
	//glVertexAttribPointer( attribInColors, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
	//glVertexAttribPointer( attribInCuttingPlane, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[0]);
	glVertexPointer( 4, GL_FLOAT, sizeof(float)*20, (const GLfloat *)0+8);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    // ...and draw arrays
	glDrawArrays( GL_POINTS, 0, primitiveCount/5);
    
	// set vertex and attribute pointers...
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[1]);
	glVertexAttribPointer( attribInParams, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+12);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[2]);
	glVertexAttribPointer( attribQuatC, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+12);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[3]);
	glVertexAttribPointer( attribInSphere, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+12);
	//glVertexAttribPointer( attribInColors, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
	//glVertexAttribPointer( attribInCuttingPlane, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[0]);
	glVertexPointer( 4, GL_FLOAT, sizeof(float)*20, (const GLfloat *)0+12);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    // ...and draw arrays
	glDrawArrays( GL_POINTS, 0, primitiveCount/5);
    
	// set vertex and attribute pointers...
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[1]);
	glVertexAttribPointer( attribInParams, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+16);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[2]);
	glVertexAttribPointer( attribQuatC, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+16);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[3]);
	glVertexAttribPointer( attribInSphere, 4, GL_FLOAT, 0, sizeof(float)*20, (const GLfloat *)0+16);
	//glVertexAttribPointer( attribInColors, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
	//glVertexAttribPointer( attribInCuttingPlane, 4, GL_FLOAT, 0, sizeof(float)*8, NULL);
    glBindBuffer( GL_ARRAY_BUFFER, this->sphericalTriaVBO[0]);
	glVertexPointer( 4, GL_FLOAT, sizeof(float)*20, (const GLfloat *)0+16);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    // ...and draw arrays
	glDrawArrays( GL_POINTS, 0, primitiveCount/5);

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
void ProteinRendererSESGPUCuda::MarkAdjacentAtoms( const CallProteinData *protein) {

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
    glOrtho( 0.0, double( protein->ProteinAtomCount()), 0.0, 1.0, 0.0, 1.0);
    // set viewport
    glViewport( 0, 0, protein->ProteinAtomCount(), 1);
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

#endif /* (defined(WITH_CUDA) && (WITH_CUDA)) */
