/*
 * ProteinRendererCBCUDA.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ProteinRendererCBCUDA.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))

#define _USE_MATH_DEFINES 1

#include "vislib/assert.h"
#include "CoreInstance.h"
#include "param/EnumParam.h"
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "param/IntParam.h"
#include "vislib/File.h"
#include "vislib/Path.h"
#include "vislib/sysfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/String.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/ShaderSource.h"
#include "vislib/AbstractOpenGLShader.h"
#include "glh/glh_genext.h"
#include "glh/glh_extensions.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>
#include "vislib/Matrix.h"

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
 * ProteinRendererCBCUDA::ProteinRendererCBCUDA
 */
ProteinRendererCBCUDA::ProteinRendererCBCUDA( void ) : Renderer3DModule (),
	protDataCallerSlot ( "getData", "Connects the protein SES rendering with protein data storage" ),
	probeRadius( 1.4f), atomNeighborCount( 64), cudaInitalized( false)
{
	this->protDataCallerSlot.SetCompatibleCall<CallProteinDataDescription>();
	this->MakeSlotAvailable ( &this->protDataCallerSlot );
}


/*
 * ProteinRendererCBCUDA::~ProteinRendererCBCUDA
 */
ProteinRendererCBCUDA::~ProteinRendererCBCUDA(void) {
	this->Release();
}


/*
 * protein::ProteinRendererCBCUDA::release
 */
void protein::ProteinRendererCBCUDA::release( void ) {

    //cudppDestroyPlan( this->sortHandle);
}


/*
 * ProteinRendererCBCUDA::create
 */
bool ProteinRendererCBCUDA::create( void ) {
	using namespace vislib::sys;
	using namespace vislib::graphics::gl;
	// try to initialize the necessary extensions for GLSL shader support
	if ( !GLSLShader::InitialiseExtensions() )
		return false;
	
	glEnable( GL_DEPTH_TEST);
	glDepthFunc( GL_LEQUAL);
	glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
	glEnable( GL_VERTEX_PROGRAM_TWO_SIDE);
	
	float spec[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, spec);
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.0f);

	ShaderSource vertSrc;
	ShaderSource fragSrc;

	CoreInstance *ci = this->GetCoreInstance();
	if( !ci ) return false;
	
	////////////////////////////////////////////////////
	// load the shader source for the sphere renderer //
	////////////////////////////////////////////////////
	if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::sphereVertex", vertSrc ) ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
			"%s: Unable to load vertex shader source for sphere shader", this->ClassName() );
		return false;
	}
	if( !ci->ShaderSourceFactory().MakeShaderSource( "protein::std::sphereFragment", fragSrc ) ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
			"%s: Unable to load fragment shader source for sphere shader", this->ClassName() );
		return false;
	}
	try {
		if( !this->sphereShader.Create( vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count() ) ) {
			throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );
		}
	} catch( vislib::Exception e ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
			"%s: Unable to create sphere shader: %s\n", this->ClassName(), e.GetMsgA() );
		return false;
	}

	return true;
}


/*
 * ProteinRendererCBCUDA::GetCapabilities
 */
bool ProteinRendererCBCUDA::GetCapabilities( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * ProteinRendererCBCUDA::GetExtents
 */
bool ProteinRendererCBCUDA::GetExtents( Call& call) {
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
 * ProteinRendererCBCUDA::Render
 */
bool ProteinRendererCBCUDA::Render( Call& call ) {
	// temporary variables
	
	// get pointer to CallProteinData
	protein::CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();

	// if something went wrong --> return
	if( !protein) return false;

	// execute the call
	if ( !( *protein )() )
		return false;
	
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
		countNeighbors(
			m_dNeighborCount,
			m_dNeighbors,
			m_dSmallCircles,
			m_dSortedPos,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			this->numAtoms,
            this->atomNeighborCount,
			this->numGridCells);

        //copyArrayFromDevice( m_hNeighborCount, m_dNeighborCount, 0, sizeof(uint)*this->numAtoms);
        //unsigned int maxNeighbors = 0;
        //unsigned int maxNeighborsCount = 0;
        //for( unsigned int i = 0; i < this->numAtoms; ++i ) {
        //    printf( "Atom %i, Number of Neighbors: %i\n", i, m_hNeighborCount[i]);
        //    maxNeighbors = maxNeighbors < m_hNeighborCount[i]? m_hNeighborCount[i] : maxNeighbors;
        //    maxNeighborsCount = m_hNeighborCount[i] >= 64 ? maxNeighborsCount + 1 : maxNeighborsCount;
        //}

        // compute the arcs
        computeArcsCUDA( m_dArcs, 
                         m_dNeighborCount, 
                         m_dNeighbors, 
                         m_dSmallCircles, 
                         m_dSortedPos, 
                         m_dGridParticleIndex, 
                         this->numAtoms,
                         this->atomNeighborCount);
	}

	// get camera information
	this->cameraInfo = dynamic_cast<view::CallRender3D*>( &call )->GetCameraParameters();

	
	// ==================== Scale & Translate ====================
	glPushMatrix();

	this->bBox = protein->BoundingBox();

	float scale, xoff, yoff, zoff;
	vislib::math::Point<float, 3> bbc = this->bBox.CalcCenter();

	xoff = -bbc.X();
	yoff = -bbc.Y();
	zoff = -bbc.Z();

	scale = 2.0f / vislib::math::Max ( vislib::math::Max ( this->bBox.Width(),
		this->bBox.Height() ), this->bBox.Depth() );

	glScalef ( scale, scale, scale );
	glTranslatef ( xoff, yoff, zoff );
	
	// ==================== Start actual rendering ====================
	glDisable( GL_BLEND);
	glEnable( GL_DEPTH_TEST);
	glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
	glEnable( GL_VERTEX_PROGRAM_TWO_SIDE);
	
	// TODO: do actual rendering
	
	float viewportStuff[4] = {
		cameraInfo->TileRect().Left(), cameraInfo->TileRect().Bottom(),
		cameraInfo->TileRect().Width(), cameraInfo->TileRect().Height()};
	if( viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
	if( viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

#define ATOM_ID 0
	// get CUDA stuff
    copyArrayFromDevice( m_hNeighborCount, m_dNeighborCount, 0, sizeof(uint)*this->numAtoms);
    //copyArrayFromDevice( m_hNeighbors, m_dNeighbors, 0, sizeof(uint)*this->numAtoms*this->atomNeighborCount);
    //copyArrayFromDevice( m_hParticleIndex, m_dGridParticleIndex, 0, sizeof(uint)*this->numAtoms);
    copyArrayFromDevice( m_hArcs, m_dArcs, 0, sizeof(float)*4*4*this->numAtoms*this->atomNeighborCount);
    //copyArrayFromDevice( m_hPos, m_dSortedPos, 0, sizeof(float)*4*this->numAtoms);
    //copyArrayFromDevice( m_hSmallCircles, m_dSmallCircles, 0, sizeof(float)*4*this->numAtoms*this->atomNeighborCount);

    // DEBUG computation ...
    /*
    std::vector<vislib::math::Vector<float, 3>> points;
    vislib::math::Vector<float, 3> ai, aj, ak, rj, rk, rm, tmp, p1, p2;
    unsigned int akIdx, ajIdx, atmpIdx;
    float Ri = protein->AtomTypes()[protein->ProteinAtomData()[ATOM_ID].TypeIndex()].Radius() + this->probeRadius;
    float Ri2 = Ri * Ri;
    float Rj, Rk, Rtmp;
    bool add1, add2;

    ai.Set(
        protein->ProteinAtomPositions()[ATOM_ID * 3 + 0],
        protein->ProteinAtomPositions()[ATOM_ID * 3 + 1],
        protein->ProteinAtomPositions()[ATOM_ID * 3 + 2]);

    // loop over all neighbors
    for( unsigned int k = 0; k < m_hNeighborCount[ATOM_ID]; ++k ) {
        akIdx = m_hParticleIndex[m_hNeighbors[this->atomNeighborCount*ATOM_ID + k]];
        Rk = protein->AtomTypes()[protein->ProteinAtomData()[akIdx].TypeIndex()].Radius() + this->probeRadius;
        rk.Set(
            m_hSmallCircles[this->atomNeighborCount * ATOM_ID * 4 + k * 4 + 0],
            m_hSmallCircles[this->atomNeighborCount * ATOM_ID * 4 + k * 4 + 1],
            m_hSmallCircles[this->atomNeighborCount * ATOM_ID * 4 + k * 4 + 2]);

        // intersect each small circle with all other small circles 
        for( unsigned int j = k + 1; j < m_hNeighborCount[ATOM_ID]; ++j ) {
            ajIdx = m_hParticleIndex[m_hNeighbors[this->atomNeighborCount*ATOM_ID + j]];
            Rj = protein->AtomTypes()[protein->ProteinAtomData()[ajIdx].TypeIndex()].Radius() + this->probeRadius;
            rj.Set(
                m_hSmallCircles[this->atomNeighborCount * ATOM_ID * 4 + j * 4 + 0],
                m_hSmallCircles[this->atomNeighborCount * ATOM_ID * 4 + j * 4 + 1],
                m_hSmallCircles[this->atomNeighborCount * ATOM_ID * 4 + j * 4 + 2]);
            
            rm = rj *
                ( ( rj.Dot( rj) - rj.Dot( rk) ) * rk.Dot( rk)) /
                ( rj.Dot( rj) * rk.Dot( rk) - rj.Dot( rk) * rj.Dot( rk)) +
                rk *
                ( ( rk.Dot( rk) - rj.Dot( rk) ) * rj.Dot( rj)) /
                ( rj.Dot( rj) * rk.Dot( rk) - rj.Dot( rk) * rj.Dot( rk));

            if( rm.Dot( rm) <= Ri2 ) {
                tmp = rj.Cross( rk);
                p1 = rm + tmp * sqrt( ( Ri2 - rm.Dot( rm)) / ( tmp.Dot( tmp) ) );
                p2 = rm - tmp * sqrt( ( Ri2 - rm.Dot( rm)) / ( tmp.Dot( tmp) ) );
            } else {
                continue;
            }
            
            // test each result with all other neighbor spheres
            add1 = add2 = true;
            for( unsigned int i = 0; i < m_hNeighborCount[ATOM_ID]; ++i ) {
                if( i == j || i == k ) continue;
                atmpIdx = m_hParticleIndex[m_hNeighbors[this->atomNeighborCount*ATOM_ID + i]];
                tmp.Set(
                    protein->ProteinAtomPositions()[atmpIdx * 3 + 0],
                    protein->ProteinAtomPositions()[atmpIdx * 3 + 1],
                    protein->ProteinAtomPositions()[atmpIdx * 3 + 2]);
                Rtmp = protein->AtomTypes()[protein->ProteinAtomData()[atmpIdx].TypeIndex()].Radius() + this->probeRadius;
                // test p1
                if( ( ( p1 + ai) - tmp).Length() < Rtmp ) add1 = false;
                // test p2
                if( ( ( p2 + ai) - tmp).Length() < Rtmp ) add2 = false;
            }

            // add p1, if it is not within any neighbor atom
            if( add1 ) points.push_back( p1);
            // add p2, if it is not within any neighbor atom
            if( add2 ) points.push_back( p2);
        }
    }

    glDisable( GL_BLEND);
    // enable sphere shader
	this->sphereShader.Enable();
	// set shader variables
	glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
	// set vertex and color pointers and draw them
	glBegin( GL_POINTS);
    // draw all arc start- and endpoints as small blue spheres
    for( unsigned int c = 0; c < points.size(); ++c ) {
        glColor3f( 0.0, 0.0, 1.0);
        tmp = ai + points[c];
        glVertex4f( tmp.GetX(), tmp.GetY(), tmp.GetZ(), 0.2f);
    }
    glEnd();
    this->sphereShader.Disable();
    */
    // ... DEBUG computation

	// DEBUG ...
    //glEnable( GL_BLEND);
    //glDisable( GL_CULL_FACE);
	// enable sphere shader
	this->sphereShader.Enable();
	// set shader variables
	glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    
    glDisable( GL_BLEND);
    // draw all arc start- and endpoints as small blue spheres ...
    vislib::math::Vector<float, 3> tmpArc, tmpAtom;
    glBegin( GL_POINTS);
    glColor3f( 0.0, 0.0, 1.0);
    for( unsigned int cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {	
        tmpAtom.Set( 
            protein->ProteinAtomPositions()[cnt*3+0],
            protein->ProteinAtomPositions()[cnt*3+1],
            protein->ProteinAtomPositions()[cnt*3+2]);
        for( unsigned int k = 0; k < m_hNeighborCount[cnt]; ++k ) {
            tmpArc.Set(
                m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 0],
                m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 1],
                m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 2]);
            if( tmpArc.Length() > 0.0f )
                glVertex4f( tmpAtom.X() + tmpArc.X(), tmpAtom.Y() + tmpArc.Y(), tmpAtom.Z() + tmpArc.Z(), 0.2f);
            tmpArc.Set(
                m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 4],
                m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 5],
                m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 6]);
            if( tmpArc.Length() > 0.0f )
                glVertex4f( tmpAtom.X() + tmpArc.X(), tmpAtom.Y() + tmpArc.Y(), tmpAtom.Z() + tmpArc.Z(), 0.2f);
        }
	}
	glEnd(); // GL_POINTS
    // ... draw all arc start- and endpoints as small blue spheres

    glEnable( GL_BLEND);
	// set vertex and color pointers and draw them
	glBegin( GL_POINTS);
    for( unsigned int cnt = 0; cnt < protein->ProteinAtomCount(); ++cnt ) {	
        // draw all arc start- and endpoints as small blue spheres
        //for( unsigned int k = 0; k < m_hNeighborCount[cnt]; ++k ) {
        //    glColor3f( 0.0, 0.0, 1.0);
        //    glVertex4f(
        //        protein->ProteinAtomPositions()[cnt*3+0] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 0],
        //        protein->ProteinAtomPositions()[cnt*3+1] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 1],
        //        protein->ProteinAtomPositions()[cnt*3+2] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 2],
        //        //this->probeRadius);
        //        0.2f);
        //    glVertex4f(
        //        protein->ProteinAtomPositions()[cnt*3+0] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 4],
        //        protein->ProteinAtomPositions()[cnt*3+1] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 5],
        //        protein->ProteinAtomPositions()[cnt*3+2] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 6],
        //        //this->probeRadius);
        //        0.2f);
        //    glVertex4f(
        //        protein->ProteinAtomPositions()[cnt*3+0] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 8],
        //        protein->ProteinAtomPositions()[cnt*3+1] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 9],
        //        protein->ProteinAtomPositions()[cnt*3+2] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 10],
        //        //this->probeRadius);
        //        0.2f);
        //    glVertex4f(
        //        protein->ProteinAtomPositions()[cnt*3+0] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 12],
        //        protein->ProteinAtomPositions()[cnt*3+1] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 13],
        //        protein->ProteinAtomPositions()[cnt*3+2] + m_hArcs[this->atomNeighborCount*cnt*4*4 + k*4*4 + 14],
        //        //this->probeRadius);
        //        0.2f);
        //}
		glColor3f( 1.0, 0.0, 0.0);
        // draw atom #ATOM_ID big and green
        //if( cnt == ATOM_ID ) {
        //    glColor3f( 0.0, 1.0, 0.0);
        ////    glVertex4f(
        ////        protein->ProteinAtomPositions()[cnt*3+0],
        ////        protein->ProteinAtomPositions()[cnt*3+1],
        ////        protein->ProteinAtomPositions()[cnt*3+2],
        ////        protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius() + this->probeRadius);
        //}
        // draw neighboring atoms to atom #ATOM_ID in yellow
        //for( unsigned int j = 0; j < m_hNeighborCount[ATOM_ID]; ++j ) {
        //    if( m_hParticleIndex[m_hNeighbors[j + this->atomNeighborCount*ATOM_ID]] == cnt ) {
        //        glColor3f( 1.0, 1.0, 0.0);
        //    }
        //}
        // draw atom
		glVertex4f(
			protein->ProteinAtomPositions()[cnt*3+0],
			protein->ProteinAtomPositions()[cnt*3+1],
			protein->ProteinAtomPositions()[cnt*3+2],
            protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius() + this->probeRadius);
            //protein->AtomTypes()[protein->ProteinAtomData()[cnt].TypeIndex()].Radius());
            //0.3f);
	}
	glEnd(); // GL_POINTS
	this->sphereShader.Disable();
	// ... DEBUG

    /*
    unsigned int idx;
    vislib::math::Vector<float, 3> a1, a2, a3, rj, rk, ri, s1, ex, d1, d2, rm, rjat, rkat, n1, n2, n3;
    float Ri2, Rj2, Rk2, b1, b2, b3;

    glBegin( GL_LINES);
    a1.Set(
        protein->ProteinAtomPositions()[0],
        protein->ProteinAtomPositions()[1],
        protein->ProteinAtomPositions()[2]);
    a2.Set(
        protein->ProteinAtomPositions()[3],
        protein->ProteinAtomPositions()[4],
        protein->ProteinAtomPositions()[5]);
    a3.Set(
        protein->ProteinAtomPositions()[6],
        protein->ProteinAtomPositions()[7],
        protein->ProteinAtomPositions()[8]);
    
    Ri2 = ( protein->AtomTypes()[protein->ProteinAtomData()[0].TypeIndex()].Radius() + this->probeRadius) * 
        (protein->AtomTypes()[protein->ProteinAtomData()[0].TypeIndex()].Radius() + this->probeRadius );
    
    //rjat = a2 - a1;
    //rkat = a3 - a1;

    rj.Set(
        m_hSmallCircles[0],
        m_hSmallCircles[1],
        m_hSmallCircles[2]);
    rk.Set(
        m_hSmallCircles[4],
        m_hSmallCircles[5],
        m_hSmallCircles[6]);

    rm = rj *
        ( ( rj.Dot( rj) - rj.Dot( rk) ) * rk.Dot( rk)) /
        ( rj.Dot( rj) * rk.Dot( rk) - rj.Dot( rk) * rj.Dot( rk)) +
        rk *
        ( ( rk.Dot( rk) - rj.Dot( rk) ) * rj.Dot( rj)) /
        ( rj.Dot( rj) * rk.Dot( rk) - rj.Dot( rk) * rj.Dot( rk));

	glColor3f( 0.0, 0.0, 1.0);
    glVertex3fv( ( a1).PeekComponents() );
    glVertex3fv( ( a1 + rm).PeekComponents() );

    //if( rm.Dot( rm) <= Ri2 )
    //{
    //    ex = rj.Cross( rk);
    //    d1 = rm + 1.1f * ex * sqrt( ( Ri2 - rm.Dot( rm)) / ( ex.Dot( ex) ) );
    //    d2 = rm - 1.1f * ex * sqrt( ( Ri2 - rm.Dot( rm)) / ( ex.Dot( ex) ) );
    //    glColor3f( 1.0, 0.0, 1.0);
    //    glVertex3fv( ( a1 + d1).PeekComponents() );
    //    glVertex3fv( ( a1 + d2).PeekComponents() );
    //}
    
	glColor3f( 0.0, 1.0, 1.0);
    glVertex3f(
        protein->ProteinAtomPositions()[0] + m_hArcs[0],
        protein->ProteinAtomPositions()[1] + m_hArcs[1],
        protein->ProteinAtomPositions()[2] + m_hArcs[2]);
    glVertex3f(
        protein->ProteinAtomPositions()[0] + m_hArcs[4],
        protein->ProteinAtomPositions()[1] + m_hArcs[5],
        protein->ProteinAtomPositions()[2] + m_hArcs[6]);

    glEnd(); // GL_LINES
    */

    /*
    glBegin( GL_QUADS);
    ex.Set( 1.0f, 0.0f, 0.0f);
    d1 = ex.Cross( rj);
    d1.Normalise();
    d2 = rj.Cross( d1);
    d2.Normalise();
    glColor3f( 1.0, 1.0, 1.0);
    glVertex3fv( ( a1 + rj + 4.0f * d1 + 4.0f * d2 ).PeekComponents() );
    glVertex3fv( ( a1 + rj + 4.0f * d1 - 4.0f * d2 ).PeekComponents() );
    glVertex3fv( ( a1 + rj - 4.0f * d1 - 4.0f * d2 ).PeekComponents() );
    glVertex3fv( ( a1 + rj - 4.0f * d1 + 4.0f * d2 ).PeekComponents() );
    glEnd();
    */


	glPopMatrix();

	return true;
}


/*
 * ProteinRendererCBCUDA::deinitialise
 */
void ProteinRendererCBCUDA::deinitialise(void) {
	// release shaders
	this->sphereShader.Release();
	
	if( this->cudaInitalized ) {
		delete[] m_hPos;
		delete[] m_hNeighborCount;
		delete[] m_hNeighbors;
		delete[] m_hSmallCircles;
		delete[] m_hArcs;
		delete[] m_hCellStart;
		delete[] m_hCellEnd;
        delete[] m_hParticleIndex;

		freeArray( m_dPos);
		freeArray( m_dSortedPos);
		freeArray( m_dNeighborCount);
		freeArray( m_dNeighbors);
		freeArray( m_dSmallCircles);
		freeArray( m_dArcs);
		freeArray( m_dGridParticleHash);
		freeArray( m_dGridParticleIndex);
		freeArray( m_dCellStart);
		freeArray( m_dCellEnd);
    }
}

/*
 * Initialize CUDA
 */
bool ProteinRendererCBCUDA::initCuda( const CallProteinData *protein, uint gridDim) {
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
    this->params.maxNumNeighbors = this->atomNeighborCount;

    // allocate host storage
    m_hPos = new float[this->numAtoms*4];
    memset(m_hPos, 0, this->numAtoms*4*sizeof(float));

	m_hNeighborCount = new uint[this->numAtoms];
	memset( m_hNeighborCount, 0, this->numAtoms*sizeof(uint));

	m_hNeighbors = new uint[this->numAtoms*this->atomNeighborCount];
	memset( m_hNeighbors, 0, this->numAtoms*this->atomNeighborCount*sizeof(uint));

	m_hSmallCircles = new float[this->numAtoms*this->atomNeighborCount*4];
	memset( m_hSmallCircles, 0, this->numAtoms*this->atomNeighborCount*4*sizeof(float));

	m_hArcs = new float[this->numAtoms*this->atomNeighborCount*4*4];
	memset( m_hArcs, 0, this->numAtoms*this->atomNeighborCount*4*4*sizeof(float));

    m_hParticleIndex = new uint[this->numAtoms];
	memset( m_hParticleIndex, 0, this->numAtoms*sizeof(uint));

    m_hCellStart = new uint[this->numGridCells];
    memset( m_hCellStart, 0, this->numGridCells*sizeof(uint));

    m_hCellEnd = new uint[this->numGridCells];
    memset( m_hCellEnd, 0, this->numGridCells*sizeof(uint));

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
	allocateArray((void**)&m_dNeighbors, this->numAtoms*this->atomNeighborCount*sizeof(uint));
	// array for the small circles
	allocateArray((void**)&m_dSmallCircles, this->numAtoms*this->atomNeighborCount*4*sizeof(float));
	// array for the arcs
	allocateArray((void**)&m_dArcs, this->numAtoms*this->atomNeighborCount*4*4*sizeof(float));

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
 * Write atom positions and radii to an array for processing in CUDA
 */
void ProteinRendererCBCUDA::writeAtomPositions( const CallProteinData *protein ) {
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

#endif /* (defined(WITH_CUDA) && (WITH_CUDA)) */
