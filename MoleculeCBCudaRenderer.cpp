/*
 * MoleculeCBCudaRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MoleculeCBCudaRenderer.h"

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
#include "vislib/Matrix.h"
#include "vislib/ShallowVector.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>

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
 * MoleculeCBCudaRenderer::MoleculeCBCudaRenderer
 */
MoleculeCBCudaRenderer::MoleculeCBCudaRenderer( void ) : Renderer3DModule (),
	molDataCallerSlot ( "getData", "Connects the protein SES rendering with PDB data source" ),
    probeRadiusParam( "probeRadius", "Probe Radius"),
    opacityParam( "opacity", "Atom opacity"),
    stepsParam( "steps", "Drawing steps"),
	probeRadius( 1.4f), atomNeighborCount( 64), cudaInitalized( false)
{
	this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable ( &this->molDataCallerSlot );

    // setup probe radius parameter
    this->probeRadiusParam.SetParameter(new param::FloatParam( this->probeRadius, 0.0f, 10.0f));
    this->MakeSlotAvailable( &this->probeRadiusParam);

    // setup atom opacity parameter
    this->opacityParam.SetParameter(new param::FloatParam( 0.4f, 0.0f, 1.0f));
    this->MakeSlotAvailable( &this->opacityParam);

    // DEBUG
    this->stepsParam.SetParameter(new param::IntParam( 5, 0, 100));
    this->MakeSlotAvailable( &this->stepsParam);

}


/*
 * MoleculeCBCudaRenderer::~MoleculeCBCudaRenderer
 */
MoleculeCBCudaRenderer::~MoleculeCBCudaRenderer(void) {
	this->Release();
}


/*
 * protein::MoleculeCBCudaRenderer::release
 */
void protein::MoleculeCBCudaRenderer::release( void ) {

    cudppDestroyPlan( this->sortHandle);
}


/*
 * MoleculeCBCudaRenderer::create
 */
bool MoleculeCBCudaRenderer::create( void ) {
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
 * MoleculeCBCudaRenderer::GetCapabilities
 */
bool MoleculeCBCudaRenderer::GetCapabilities( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        | view::CallRender3D::CAP_ANIMATION );

    return true;
}


/*
 * MoleculeCBCudaRenderer::GetExtents
 */
bool MoleculeCBCudaRenderer::GetExtents( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

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


/*
 * MoleculeCBCudaRenderer::Render
 */
bool MoleculeCBCudaRenderer::Render( Call& call ) {	
    // cast the call to Render3D
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

	// get pointer to call
	MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
	// if something went wrong --> return
	if( !mol) return false;

	// execute the call
	if ( !(*mol)(MolecularDataCall::CallForGetData) )
		return false;

    // update parameters
    this->UpdateParameters( mol);

	// try to initialize CUDA
	if( !this->cudaInitalized ) {
		cudaInitalized = this->initCuda( mol, 16);
		vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_INFO, 
			"%s: CUDA initialization: %i", this->ClassName(), cudaInitalized );
	}
	// execute CUDA kernels, if initialization succeeded
	if( this->cudaInitalized ) {
		// write atom positions to array
		this->writeAtomPositions( mol);

		this->params.probeRadius = this->probeRadius;

		// update constants
		setParameters( &this->params);

		// calculate grid hash
		calcHash(
			m_dGridParticleHash,
			m_dGridParticleIndex,
			m_dPos,
			this->numAtoms);

		// sort particles based on hash
        cudppSort( this->sortHandle, m_dGridParticleHash, m_dGridParticleIndex, this->gridSortBits, this->numAtoms);

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

	}
	
	// ==================== Scale & Translate ====================
    glPushMatrix();
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

	// get CUDA stuff
    copyArrayFromDevice( m_hNeighborCount, m_dNeighborCount, 0, sizeof(uint)*this->numAtoms);
    copyArrayFromDevice( m_hNeighbors, m_dNeighbors, 0, sizeof(uint)*this->numAtoms*this->atomNeighborCount);
    copyArrayFromDevice( m_hParticleIndex, m_dGridParticleIndex, 0, sizeof(uint)*this->numAtoms);
    //copyArrayFromDevice( m_hPos, m_dSortedPos, 0, sizeof(float)*4*this->numAtoms);
    copyArrayFromDevice( m_hSmallCircles, m_dSmallCircles, 0, sizeof(float)*4*this->numAtoms*this->atomNeighborCount);

	// ========= RENDERING =========
	unsigned int cnt1, cnt2, cnt3; 
	vislib::math::Vector<float, 3> tmpVec1, tmpVec2, tmpVec3, ey( 0, 1, 0);
	vislib::math::Quaternion<float> tmpQuat;
	
	// draw neighbor connections
	for( cnt1 = 0; cnt1 < mol->AtomCount(); ++cnt1 ) {
		for( cnt2 = 0; cnt2 < m_hNeighborCount[cnt1]; ++cnt2 ) {
			cnt3 = m_hParticleIndex[m_hNeighbors[cnt1 * atomNeighborCount + cnt2]];
			glColor3f( 0.0f, 1.0f, 1.0f);
			glBegin( GL_LINES);
			glVertex3fv( &mol->AtomPositions()[cnt1 * 3]);
			glVertex3fv( &mol->AtomPositions()[cnt3 * 3]);
			glEnd();
		}
	}

    // START ... remove all unnecessary small circles
    if( smallCircles.Count() < mol->AtomCount() )
        smallCircles.SetCount( mol->AtomCount());
    if( smallCircleRadii.Count() < mol->AtomCount() )
        smallCircleRadii.SetCount( mol->AtomCount());
    if( neighbors.Count() < mol->AtomCount() )
        neighbors.SetCount( mol->AtomCount());
    for( unsigned int iCnt = 0; iCnt < mol->AtomCount(); ++iCnt ) {
        smallCircles[iCnt].Clear();
        smallCircleRadii[iCnt].Clear();
        neighbors[iCnt].Clear();
        smallCircles[iCnt].AssertCapacity( this->atomNeighborCount);
        smallCircleRadii[iCnt].AssertCapacity( this->atomNeighborCount);
        neighbors[iCnt].AssertCapacity( this->atomNeighborCount);
        // pi = center of atom i
        vislib::math::Vector<float, 3> pi( &mol->AtomPositions()[3*iCnt]);
        float r = mol->AtomTypes()[mol->AtomTypeIndices()[iCnt]].Radius();
        float R = r + this->probeRadius;
        // go over all neighbors j
        for( unsigned int jCnt = 0; jCnt < m_hNeighborCount[iCnt]; ++jCnt ) {
            // flag wether j sould be added (true) is cut off (false)
            bool addJ = true;
            // the atom index of j
            unsigned int j = m_hParticleIndex[m_hNeighbors[iCnt * atomNeighborCount + jCnt]];
            // vj = the small circle center
            vislib::math::Vector<float, 3> vj( &m_hSmallCircles[iCnt * 4 * atomNeighborCount + jCnt * 4]);
            // pj = center of atom j
            vislib::math::Vector<float, 3> pj( &mol->AtomPositions()[j * 3]);
            // check each neighbor j with all other neighbors k
            for( unsigned int kCnt = 0; kCnt < m_hNeighborCount[iCnt]; ++kCnt ) {
                // don't compare the circle with itself
                if( jCnt == kCnt ) 
                    continue;
                // the atom index of k
                unsigned int k = m_hParticleIndex[m_hNeighbors[iCnt * atomNeighborCount + kCnt]];
                // vk = the small circle center
                vislib::math::Vector<float, 3> vk( &m_hSmallCircles[iCnt * 4 * atomNeighborCount + kCnt * 4]);
                // pk = center of atom k
                vislib::math::Vector<float, 3> pk( &mol->AtomPositions()[k * 3]);
                // vj * vk
                float vjvk = vj.Dot( vk);
                // denominator
                float denom = vj.Dot( vj) * vk.Dot( vk) - vjvk * vjvk;
                vislib::math::Vector<float, 3> h = vj * ( vj.Dot( vj - vk) * vk.Dot( vk) ) / denom + 
                    vk * ( ( vk - vj).Dot( vk) * vj.Dot( vj) ) / denom;
                // compute cases
                vislib::math::Vector<float, 3> nj( pi - pj);
                nj.Normalise();
                vislib::math::Vector<float, 3> nk( pi - pk);
                nk.Normalise();
                vislib::math::Vector<float, 3> q( vk - vj);
                // if normals are the same (unrealistic, yet theoretically possible)
                if( vislib::math::IsEqual( nj.Dot( nk), 1.0f) ) {
                    if( nj.Dot( nk) > 0 ) {
                        if( nj.Dot( q) > 0 ) {
                            // k cuts off j --> remove j
                            addJ = false;
                            break;
                        }
                    }
                } else if( h.Length() > R ) {
                    vislib::math::Vector<float, 3> mj( vj - h);
                    vislib::math::Vector<float, 3> mk( vk - h);
                    if( nj.Dot( nk) > 0 ) {
                        if( mj.Dot( mk) > 0 && nj.Dot( q) > 0 ) {
                            // k cuts off j --> remove j
                            addJ = false;
                            break;
                        }
                    }
                }
            }
            // all k were tested, see if j is cut of or sould be added
            if( addJ ) {
                this->smallCircles[iCnt].Add( vj);
				this->smallCircleRadii[iCnt].Add( 1.0f);
                this->neighbors[iCnt].Add( j);
            }
        }
    }
    // ... END remove all unnecessary small circles 

	this->sphereShader.Enable();
	glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
	// draw small circles
	glBegin( GL_POINTS);
	for( cnt1 = 0; cnt1 < mol->AtomCount(); ++cnt1 ) {
		tmpVec1.Set( mol->AtomPositions()[cnt1 * 3 + 0], 
			mol->AtomPositions()[cnt1 * 3 + 1], 
			mol->AtomPositions()[cnt1 * 3 + 2]);
        for( cnt2 = 0; cnt2 < this->neighbors[cnt1].Count(); ++cnt2 ) {
            tmpVec2 = this->smallCircles[cnt1][cnt2];
            // center of small circle
			glColor3f( 0.0f, 0.0f, 1.0f);
			glVertex4f(
				tmpVec1.X() + tmpVec2.X(),
				tmpVec1.Y() + tmpVec2.Y(),
				tmpVec1.Z() + tmpVec2.Z(),
				0.1f);
			// point on small circle
			glColor3f( 1.0f, 1.0f, 0.0f);
			tmpVec3 = tmpVec2.Cross( ey);
			tmpVec3.Normalise();
			tmpVec3 *= sqrt(
					(mol->AtomTypes()[mol->AtomTypeIndices()[cnt1]].Radius() + this->probeRadius) *
					(mol->AtomTypes()[mol->AtomTypeIndices()[cnt1]].Radius() + this->probeRadius) -
					tmpVec2.SquareLength());
			tmpQuat.Set( float( vislib::math::PI_DOUBLE / 50.0), tmpVec2 / tmpVec2.Length());
			for( cnt3 = 0; cnt3 < 100; ++cnt3 ) {
				tmpVec3 = tmpQuat * tmpVec3;
				glVertex4f(
					tmpVec1.X() + tmpVec2.X() + tmpVec3.X(),
					tmpVec1.Y() + tmpVec2.Y() + tmpVec3.Y(),
					tmpVec1.Z() + tmpVec2.Z() + tmpVec3.Z(),
					0.1f);
			}
		}
	}
	glEnd();

    // ========== contour buildup ==========

    glBegin( GL_POINTS);
    vislib::Array<vislib::Pair<vislib::math::Vector<float, 3>, vislib::math::Vector<float, 3>>> arcs;
    arcs.SetCapacityIncrement( 100);
    vislib::Array<vislib::Pair<float, float>> arcAngles;
	arcAngles.SetCapacityIncrement( 100);
    // go over all atoms
    //for( unsigned int iCnt = 0; iCnt < mol->AtomCount(); ++iCnt ) {
	for( unsigned int iCnt = 0; iCnt < 1; ++iCnt ) {
        // pi = center of atom i
        vislib::math::Vector<float, 3> pi( &mol->AtomPositions()[3*iCnt]);
        float r = mol->AtomTypes()[mol->AtomTypeIndices()[iCnt]].Radius();
        float R = r + this->probeRadius;
        // go over all neighbors j
        //for( unsigned int jCnt = 0; jCnt < m_hNeighborCount[iCnt]; ++jCnt ) {
        //for( unsigned int jCnt = 0; jCnt < this->neighbors[iCnt].Count(); ++jCnt ) {
		//for( unsigned int jCnt = 0; jCnt < 1; ++jCnt ) {
		for( unsigned int jCnt = 1; jCnt < this->neighbors[iCnt].Count(); ++jCnt ) {
            // the atom index of j
            //unsigned int j = m_hParticleIndex[m_hNeighbors[iCnt * atomNeighborCount + jCnt]];
            unsigned int j = this->neighbors[iCnt][jCnt];
            // vj = the small circle center
            //vislib::math::Vector<float, 3> vj( &m_hSmallCircles[iCnt * 4 * atomNeighborCount + jCnt * 4]);
            vislib::math::Vector<float, 3> vj( this->smallCircles[iCnt][jCnt]);
            // pj = center of atom j
            vislib::math::Vector<float, 3> pj( &mol->AtomPositions()[j * 3]);
            // clear the arc array
            arcs.Clear();
			// compute axes of local coordinate system
			vislib::math::Vector<float, 3> xAxis = vj.Cross( ey);
			xAxis.Normalise();
			vislib::math::Vector<float, 3> yAxis = xAxis.Cross( vj);
			yAxis.Normalise();
            // clear the arc angles array
            arcAngles.Clear();
            // check each neighbor j with all other neighbors k
            //for( unsigned int kCnt = 0; kCnt < m_hNeighborCount[iCnt]; ++kCnt ) {
            //for( unsigned int kCnt = 0; kCnt < this->neighbors[iCnt].Count(); ++kCnt ) {
			for( unsigned int kCnt = 0; kCnt < this->neighbors[iCnt].Count(); ++kCnt ) {
                // don't compare the circle with itself
                if( jCnt == kCnt ) 
                    continue;
                // the atom index of k
                //unsigned int k = m_hParticleIndex[m_hNeighbors[iCnt * atomNeighborCount + kCnt]];
                unsigned int k = this->neighbors[iCnt][kCnt];
                // vk = the small circle center
                //vislib::math::Vector<float, 3> vk( &m_hSmallCircles[iCnt * 4 * atomNeighborCount + kCnt * 4]);
                vislib::math::Vector<float, 3> vk( this->smallCircles[iCnt][kCnt]);
                // pk = center of atom k
                vislib::math::Vector<float, 3> pk( &mol->AtomPositions()[k * 3]);
#if 0
                // vj * vk
                float vjvk = vj.Dot( vk);
                // denominator
                float denom = vj.Dot( vj) * vk.Dot( vk) - vjvk * vjvk;
                vislib::math::Vector<float, 3> h = vj * ( vj.Dot( vj - vk) * vk.Dot( vk) ) / denom +
                    vk * ( ( vk - vj).Dot( vk) * vj.Dot( vj) ) / denom;
                // do nothing if h is outside of the extended sphere of atom i
                if( h.Length() > R ) 
                    continue;
                // DEBUG: draw purple sphere for h
                glColor3f( 1, 0, 1);
                glVertex4f( h.X() + pi.X(), 
                    h.Y() + pi.Y(), 
                    h.Z() + pi.Z(), 0.1f);
                // compute the root
                float root = sqrtf( ( R*R - h.Dot( h)) / ( ( vj.Cross( vk)).Dot( vj.Cross( vk))));
                // compute the two intersection points
                vislib::math::Vector<float, 3> x1 = h + vj.Cross( vk) * root;
                vislib::math::Vector<float, 3> x2 = h - vj.Cross( vk) * root;
                // swap x1 & x2 if vj points in the opposit direction of pj-pi
                if( vj.Dot( pj - pi) < 0.0f ) {
                    vislib::math::Vector<float, 3> tmpVec = x1;
                    x1 = x2;
                    x2 = tmpVec;
                }
                if( arcs.Count() == 0 ) {
                    arcs.SetCount( 1);
                    arcs[0].SetFirst( x1);
                    arcs[0].SetSecond( x2);
                } else {
                    for( int aCnt = 0; aCnt < arcs.Count(); aCnt++ ) {
                        float d1 = ( pk - pi).Dot( arcs[aCnt].First() - vk);
                        float d2 = ( pk - pi).Dot( arcs[aCnt].Second() - vk);
                        float d3 = ( vj.Dot( pj - pi)) * ( arcs[aCnt].First().Cross( x1)).Dot( arcs[aCnt].Second());
                        //if( vislib::math::IsEqual( d1, 0.0f) || vislib::math::IsEqual( d2, 0.0f) || vislib::math::IsEqual( d3, 0.0f) ) {
                        //    std::cout << "error i=" << iCnt << 
                        //        " " << vislib::math::IsEqual( d1, 0.0f) << 
                        //        " " << vislib::math::IsEqual( d2, 0.0f) << 
                        //        " " << vislib::math::IsEqual( d3, 0.0f) << std::endl;
                        //}
                        if( d1 > 0 ) {
                            if( d2 > 0 ) {
                                if( d3 > 0 ) {
                                    // der kreisbogen wird geloescht
                                    arcs.RemoveAt( aCnt);
                                    aCnt--;
                                } else {
                                    // start- & endvektor werden geändert: s = x1, e = x2
                                    arcs[aCnt].SetFirst( x1);
                                    arcs[aCnt].SetSecond( x2);
                                }
                            } else {
                                if( d3 > 0 ) {
                                    // startvektor wird geändert: s = x1
                                    arcs[aCnt].SetFirst( x1);
                                } else {
                                    // startvektor wird geändert: s = x2
                                    arcs[aCnt].SetFirst( x2);
                                }
                            }
                        } else {
                            if( d2 > 0 ) {
                                if( d3 > 0 ) {
                                    // endvektor wird geändert: e = x1
                                    arcs[aCnt].SetSecond( x1);
                                } else {
                                    // endvektor wird geändert: e = x2
                                    arcs[aCnt].SetSecond( x2);
                                }
                            } else {
                                if( d3 > 0 ) {
                                    // kreisbogen wird aufgeteilt
                                    arcs.SetCount( arcs.Count() + 1);
                                    arcs.Last().SetFirst( x1);
                                    arcs.Last().SetSecond( x2);
                                } else {
                                    // keine auswirkung
                                }
                            }
                        }
                    }
                }
#else
                // vk * vj
                float vkvj = vk.Dot( vj);
                // denominator
                float denom = vk.Dot( vk) * vj.Dot( vj) - vkvj * vkvj;
                vislib::math::Vector<float, 3> h = vk * ( vk.Dot( vk - vj) * vj.Dot( vj) ) / denom +
                    vj * ( ( vj - vk).Dot( vj) * vk.Dot( vk) ) / denom;
                // do nothing if h is outside of the extended sphere of atom i
                if( h.Length() > R ) 
                    continue;
                // DEBUG: draw purple sphere for h
                glColor3f( 1, 0, 1);
                glVertex4f( h.X() + pi.X(), 
                    h.Y() + pi.Y(), 
                    h.Z() + pi.Z(), 0.1f);
                // compute the root
                float root = sqrtf( ( R*R - h.Dot( h)) / ( ( vk.Cross( vj)).Dot( vk.Cross( vj))));
                // compute the two intersection points
                vislib::math::Vector<float, 3> x1 = h + vk.Cross( vj) * root;
                vislib::math::Vector<float, 3> x2 = h - vk.Cross( vj) * root;
                // swap x1 & x2 if vj points in the opposit direction of pj-pi
                if( vk.Dot( pk - pi) < 0.0f ) {
                    vislib::math::Vector<float, 3> tmpVec = x1;
                    x1 = x2;
                    x2 = tmpVec;
                }
#if 0
                if( arcs.Count() == 0 ) {
                    arcs.SetCount( 1);
                    arcs[0].SetFirst( x1);
                    arcs[0].SetSecond( x2);
                } else {
                    for( int aCnt = 0; aCnt < arcs.Count(); aCnt++ ) {
                        float d1 = ( pk - pi).Dot( arcs[aCnt].First() - vk);
                        float d2 = ( pk - pi).Dot( arcs[aCnt].Second() - vk);
                        float d3 = ( vj.Dot( pj - pi)) * ( arcs[aCnt].First().Cross( x1)).Dot( arcs[aCnt].Second());
                        if( vislib::math::IsEqual( d1, 0.0f) || vislib::math::IsEqual( d2, 0.0f) || vislib::math::IsEqual( d3, 0.0f) ) {
                            std::cout << "error i=" << iCnt << 
                                " " << vislib::math::IsEqual( d1, 0.0f) << 
                                " " << vislib::math::IsEqual( d2, 0.0f) << 
                                " " << vislib::math::IsEqual( d3, 0.0f) << std::endl;
                        }
                        if( d1 > 0 ) {
                            if( d2 > 0 ) {
                                if( d3 > 0 ) {
                                    // der kreisbogen wird geloescht
                                    arcs.RemoveAt( aCnt);
                                    aCnt--;
                                } else {
                                    // start- & endvektor werden geändert: s = x1, e = x2
                                    arcs[aCnt].SetFirst( x1);
                                    arcs[aCnt].SetSecond( x2);
                                }
                            } else {
                                if( d3 > 0 ) {
                                    // startvektor wird geändert: s = x1
                                    arcs[aCnt].SetFirst( x1);
                                } else {
                                    // startvektor wird geändert: s = x2
                                    arcs[aCnt].SetFirst( x2);
                                }
                            }
                        } else {
                            if( d2 > 0 ) {
                                if( d3 > 0 ) {
                                    // endvektor wird geändert: e = x1
                                    arcs[aCnt].SetSecond( x1);
                                } else {
                                    // endvektor wird geändert: e = x2
                                    arcs[aCnt].SetSecond( x2);
                                }
                            } else {
                                if( d3 > 0 ) {
                                    // kreisbogen wird aufgeteilt
									vislib::math::Vector<float, 3> tmpVecE( arcs.Last().Second());
                                    arcs.Last().SetSecond( x2);
                                    arcs.SetCount( arcs.Count() + 1);
                                    arcs.Last().SetFirst( x1);
									arcs.Last().SetSecond( tmpVecE);
                                } else {
                                    // keine auswirkung
                                }
                            }
                        }
                    }
                }
#else
				// compute small circle radius
				this->smallCircleRadii[iCnt][jCnt] = ( x1 - vj).Length();
				// transform x1 and x2 to small circle coordinate system
				float xX1 = ( x1 - vj).Dot( xAxis);
				float yX1 = ( x1 - vj).Dot( yAxis);
				float xX2 = ( x2 - vj).Dot( xAxis);
				float yX2 = ( x2 - vj).Dot( yAxis);
				float angleX1 = atan2( yX1, xX1);
				float angleX2 = atan2( yX2, xX2);
				if( angleX2 < angleX1 ) {
					angleX2 += float( vislib::math::PI_DOUBLE) * 2.0f;
				}
				if( arcAngles.Count() == 0 ) {
					arcAngles.Add( vislib::Pair<float, float>( angleX1, angleX2));
				}
#endif
#endif
			}
#if 0
			// draw small circles
			tmpVec1.Set( mol->AtomPositions()[iCnt*3], mol->AtomPositions()[iCnt*3+1], mol->AtomPositions()[iCnt*3+2]);
			//for( cnt2 = 0; cnt2 < arcs.Count(); ++cnt2 ) {
			for( cnt2 = 0; cnt2 < arcs.Count(); ++cnt2 ) {
				vislib::math::Vector<float, 3> tmpVec2 = this->smallCircles[iCnt][jCnt];
				// point on small circle
				glColor3f( 0.0f, 1.0f, 1.0f);
				vislib::math::Vector<float, 3> tmpVec3( arcs[cnt2].First() - tmpVec2);
				vislib::math::Quaternion<float> tmpQuat( float( vislib::math::PI_DOUBLE / 50.0), tmpVec2 / tmpVec2.Length());
				for( cnt3 = 0; cnt3 < this->stepsParam.Param<param::IntParam>()->Value(); ++cnt3 ) {
					tmpVec3 = tmpQuat * tmpVec3;
					glVertex4f(
						tmpVec1.X() + tmpVec2.X() + tmpVec3.X(),
						tmpVec1.Y() + tmpVec2.Y() + tmpVec3.Y(),
						tmpVec1.Z() + tmpVec2.Z() + tmpVec3.Z(),
						0.15f);
				}
			}
            // draw all arc start & end points
            for( unsigned int aCnt = 0; aCnt < arcs.Count(); aCnt++ ) {
                glColor3f( 0, 1, 0);
                glVertex4f( arcs[aCnt].First().X() + pi.X(), 
                    arcs[aCnt].First().Y() + pi.Y(), 
                    arcs[aCnt].First().Z() + pi.Z(), 0.2f);
                glColor3f( 1, 0.5f, 0);
                glVertex4f( arcs[aCnt].Second().X() + pi.X(), 
                    arcs[aCnt].Second().Y() + pi.Y(), 
                    arcs[aCnt].Second().Z() + pi.Z(), 0.2f);
            }
#else
			tmpVec1.Set( mol->AtomPositions()[iCnt*3], mol->AtomPositions()[iCnt*3+1], mol->AtomPositions()[iCnt*3+2]);
            // draw all arc start & end points from angles
			vislib::math::Quaternion<float> rotQuat;
            for( unsigned int aCnt = 0; aCnt < arcAngles.Count(); aCnt++ ) {
				glColor3f( 0, 1, 0);
				rotQuat.Set( -arcAngles[aCnt].First(), vj / vj.Length());
				tmpVec2 = tmpVec1 + vj + ( rotQuat * xAxis) * this->smallCircleRadii[iCnt][jCnt];
				glVertex4f( tmpVec2.X(), tmpVec2.Y(), tmpVec2.Z(), 0.2f);
				glColor3f( 1, 0.5f, 0);
				rotQuat.Set( -arcAngles[aCnt].Second(), vj / vj.Length());
				tmpVec2 = tmpVec1 + vj + ( rotQuat * xAxis) * this->smallCircleRadii[iCnt][jCnt];
				glVertex4f( tmpVec2.X(), tmpVec2.Y(), tmpVec2.Z(), 0.2f);
            }
			// draw small circles
			for( unsigned int aCnt = 0; aCnt < arcAngles.Count(); ++aCnt ) {
				glColor3f( 0.0f, 1.0f, 1.0f);
				rotQuat.Set( -arcAngles[aCnt].First(), vj / vj.Length());
				tmpVec2 = ( rotQuat * xAxis) * this->smallCircleRadii[iCnt][jCnt];
				rotQuat.Set( -( arcAngles[aCnt].Second() - arcAngles[aCnt].First()) / this->stepsParam.Param<param::IntParam>()->Value(), vj / vj.Length());
				for( cnt3 = 0; cnt3 < this->stepsParam.Param<param::IntParam>()->Value() / 2; ++cnt3 ) {
					tmpVec2 = rotQuat * tmpVec2;
					glVertex4f(
						tmpVec1.X() + tmpVec2.X() + vj.X(),
						tmpVec1.Y() + tmpVec2.Y() + vj.Y(),
						tmpVec1.Z() + tmpVec2.Z() + vj.Z(),
						0.15f);
				}
			}
#endif
        }
    }
    glEnd(); // GL_POINTS
    
	// START draw atoms ...
    //glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);   // standard transparency
    //glBlendFunc( GL_DST_COLOR, GL_ONE_MINUS_DST_ALPHA);   // pretty cool & useful...
    glBlendFunc( GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA);     // very useful
	glEnable( GL_BLEND);
    glDisable( GL_CULL_FACE);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
	glBegin( GL_POINTS);
	for( cnt1 = 0; cnt1 < mol->AtomCount(); ++cnt1 ) {
		if( cnt1 == 0 )
			glColor4f( 1.0, 0.0, 1.0, this->opacityParam.Param<param::FloatParam>()->Value());
		else if( cnt1 == this->neighbors[0][0] )
			glColor4f( 1.0, 0.0, 0.5, this->opacityParam.Param<param::FloatParam>()->Value());
		else
			glColor4f( 1.0, 0.0, 0.0, this->opacityParam.Param<param::FloatParam>()->Value());
		glVertex4f(
			mol->AtomPositions()[cnt1*3+0],
			mol->AtomPositions()[cnt1*3+1],
			mol->AtomPositions()[cnt1*3+2],
			//mol->AtomTypes()[mol->AtomTypeIndices()[cnt1]].Radius());
            //mol->AtomTypes()[mol->AtomTypeIndices()[cnt1]].Radius() * 0.1f);
			mol->AtomTypes()[mol->AtomTypeIndices()[cnt1]].Radius() + this->probeRadius);
	}
	glEnd();
    // ... END draw atoms

	this->sphereShader.Disable();
	glDisable( GL_BLEND);


	return true;
}


/*
 * update parameters
 */
void MoleculeCBCudaRenderer::UpdateParameters( const MolecularDataCall *mol) {
    // color table param
    if( this->probeRadiusParam.IsDirty() ) {
        this->probeRadius = this->probeRadiusParam.Param<param::FloatParam>()->Value();
        this->probeRadiusParam.ResetDirty();
    }
}




/*
 * MoleculeCBCudaRenderer::deinitialise
 */
void MoleculeCBCudaRenderer::deinitialise(void) {
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

        cudppDestroyPlan( this->sortHandle);
	}
}

/*
 * Initialize CUDA
 */
bool MoleculeCBCudaRenderer::initCuda( MolecularDataCall *mol, uint gridDim) {
	// set number of atoms
	this->numAtoms = mol->AtomCount();

	// set grid dimensions
    this->gridSize.x = this->gridSize.y = this->gridSize.z = gridDim;
    this->numGridCells = this->gridSize.x * this->gridSize.y * this->gridSize.z;
	float3 worldSize = make_float3(
		mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
		mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
		mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth() );
    this->gridSortBits = 18;    // increase this for larger grids

    // set parameters
    this->params.gridSize = this->gridSize;
    this->params.numCells = this->numGridCells;
    this->params.numBodies = this->numAtoms;
    //this->params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	this->params.worldOrigin = make_float3(
		mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetX(),
		mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetY(),
		mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetZ());
	this->params.cellSize = make_float3( 
		worldSize.x / this->gridSize.x, 
		worldSize.y / this->gridSize.y, 
		worldSize.z / this->gridSize.z);
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
    CUDPPConfiguration sortConfig;
    sortConfig.algorithm = CUDPP_SORT_RADIX;
    sortConfig.datatype = CUDPP_UINT;
    sortConfig.op = CUDPP_ADD;
    sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    cudppPlan( &this->sortHandle, sortConfig, this->numAtoms, 1, 0);

	setParameters( &this->params);

	return true;
}

/*
 * Write atom positions and radii to an array for processing in CUDA
 */
void MoleculeCBCudaRenderer::writeAtomPositions( const MolecularDataCall *mol ) {
	// write atoms to array
	int p = 0;
	for( unsigned int cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
		// write pos and rad to array
		m_hPos[p++] = mol->AtomPositions()[cnt*3+0];
		m_hPos[p++] = mol->AtomPositions()[cnt*3+1];
		m_hPos[p++] = mol->AtomPositions()[cnt*3+2];
		m_hPos[p++] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
	}

	// setArray( POSITION, m_hPos, 0, this->numAtoms);
	copyArrayToDevice( this->m_dPos, this->m_hPos, 0, this->numAtoms*4*sizeof(float));
}

#endif /* (defined(WITH_CUDA) && (WITH_CUDA)) */
