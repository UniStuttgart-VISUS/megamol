/*
 * ProteinRendererCBOpenCL.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ProteinRendererCBOpenCL.h"

#if (defined(WITH_OPENCL) && (WITH_OPENCL))

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

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

/*
 * ProteinRendererCBOpenCL::ProteinRendererCBOpenCL
 */
ProteinRendererCBOpenCL::ProteinRendererCBOpenCL( void ) : Renderer3DModule (),
	protDataCallerSlot ( "getData", "Connects the protein SES rendering with protein data storage" ),
	probeRadius( 1.4f), atomNeighbors( 0), atomNeighborsSize( 0), atomNeighborCount( 100)
{
	this->protDataCallerSlot.SetCompatibleCall<CallProteinDataDescription>();
	this->MakeSlotAvailable ( &this->protDataCallerSlot );
	
}


/*
 * ProteinRendererCBOpenCL::~ProteinRendererCBOpenCL
 */
ProteinRendererCBOpenCL::~ProteinRendererCBOpenCL(void) {
	this->Release();
}


/*
 * protein::ProteinRendererCBOpenCL::release
 */
void protein::ProteinRendererCBOpenCL::release( void ) {
	
}


/*
 * ProteinRendererCBOpenCL::create
 */
bool ProteinRendererCBOpenCL::create( void ) {
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

	// get the device ID
	cl_uint num_devices;
	ciErrNum = clGetDeviceIDs( NULL, CL_DEVICE_TYPE_GPU, 1, &cdDevice, &num_devices);
	//std::cout << "error device:" << ciErrNum << std::endl;
	if( ciErrNum != CL_SUCCESS ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to get OCL device: %i\n", 
			this->ClassName(), ciErrNum );
		return false;
	}

	// create a context
	cxGPUContext = clCreateContext( 0, 1, &cdDevice, NULL, NULL, &ciErrNum);
	//std::cout << "error context:" << ciErrNum << std::endl;
	if( ciErrNum != CL_SUCCESS ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create OCL context: %i\n", 
			this->ClassName(), ciErrNum );
		return false;
	}

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue( cxGPUContext, cdDevice, 0, &ciErrNum);
	//std::cout << "error queue:" << ciErrNum << std::endl;
	if( ciErrNum != CL_SUCCESS ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, "%s: Unable to create OCL command-queue: %i\n", 
			this->ClassName(), ciErrNum );
		return false;
	}

	// load OpenCL kernel file
	vislib::StringW kernelFilename;
	bool kernelFileRead = false;
	size_t shaderDirCnt = ci->Configuration().ShaderDirectories().Count();
	// search for file in all shader directories
	for( size_t dirCnt = 0; dirCnt < shaderDirCnt; ++dirCnt ) {
		kernelFilename = vislib::sys::Path::Concatenate( 
			ci->Configuration().ShaderDirectories()[dirCnt], 
			vislib::StringW( "contourbuildupKernel.cl"));
		// check if the file exits
		if( vislib::sys::File::Exists( kernelFilename) ) {
			// try to read the file content
			if( !ReadTextFile( cSourceCL, vislib::StringA( kernelFilename).PeekBuffer()) ) {
				Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR,
					"%s: Unable to read contour-buildup kernel file: %s.",
					this->ClassName(), vislib::StringA( kernelFilename));
				return false;
			}
			kernelFileRead = true;
			break;
		}
	}
	// return false if the kernel file was not found
	if( !kernelFileRead ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
			"%s: Unable to find contour-buildup kernel file.\n", this->ClassName());
		return false;
	}

    // Create the program
    size_t program_length = cSourceCL.Length();
	char *kernelSource = new char[program_length];
	memcpy( kernelSource, cSourceCL.PeekBuffer(), program_length);
    cpProgram = clCreateProgramWithSource( cxGPUContext, 1, 
		(const char **)&kernelSource, &program_length, &ciErrNum);
	delete[] kernelSource;
	if( ciErrNum != CL_SUCCESS ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
			"%s: Unable to create the OpenCL kernel program: %i\n", this->ClassName(), ciErrNum );
		return false;
	}

	// Build the program
    ciErrNum = clBuildProgram( cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
	if( ciErrNum != CL_SUCCESS ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
			"%s: Unable to build the OpenCL kernel program: %i\n", this->ClassName(), ciErrNum );
		return false;
	}

	// Create kernels
    spherecutKernel = clCreateKernel( cpProgram, "sphereCut", &ciErrNum);
	if( ciErrNum != CL_SUCCESS ) {
		Log::DefaultLog.WriteMsg( Log::LEVEL_ERROR, 
			"%s: Unable to create the sphere cut kernel: %i\n", this->ClassName(), ciErrNum );
		return false;
	}

	return true;
}


/*
 * ProteinRendererCBOpenCL::GetCapabilities
 */
bool ProteinRendererCBOpenCL::GetCapabilities( Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * ProteinRendererCBOpenCL::GetExtents
 */
bool ProteinRendererCBOpenCL::GetExtents( Call& call) {
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
 * ProteinRendererCBOpenCL::Render
 */
bool ProteinRendererCBOpenCL::Render( Call& call ) {
	// temporary variables
	
	// get pointer to CallProteinData
	protein::CallProteinData *protein = this->protDataCallerSlot.CallAs<protein::CallProteinData>();

	// if something went wrong --> return
	if( !protein) return false;

	// execute the call
	if ( ! ( *protein )() )
		return false;
	
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
	
	// compute the neigborhood of all atoms
	time_t t = clock();
	this->writeNeighborAtoms( protein);
	std::cout << "time for computing the neigborhood of all atoms: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;

	// TODO: do actual rendering
	
	glPopMatrix();

	return true;
}


/*
 * ProteinRendererCBOpenCL::deinitialise
 */
void ProteinRendererCBOpenCL::deinitialise(void) {
	// release shaders
	this->sphereShader.Release();
	// clean up OpenCL	
	if( this->spherecutKernel )
		clReleaseKernel( this->spherecutKernel);
    if( this->cpProgram )
		clReleaseProgram( this->cpProgram);
    if( this->cqCommandQueue )
		clReleaseCommandQueue( this->cqCommandQueue);
    if( this->cxGPUContext )
		clReleaseContext( this->cxGPUContext);
}

/*
 * ProteinRendererCBOpenCL::writeNeighborAtoms
 */
void ProteinRendererCBOpenCL::writeNeighborAtoms( CallProteinData *protein) {
	// counter variables
	unsigned int cnt1, cnt2;
	// temporary position vector	
	vislib::math::Vector<float, 3> tmpVec1, tmpVec2;
	// temporary radius
	float radius1, radius2;
	// set the number of atoms to the total number of protein atoms
	unsigned int numberOfAtoms = protein->ProteinAtomCount();
	// set voxel lenght --> diameter of the probe + maximum atom diameter
	float voxelLength = 2.0f * this->probeRadius + 2.0f * 3.0f;
	unsigned int tmpSize = (unsigned int)ceilf( this->bBox.Width() / voxelLength);
	// voxel array for atom indices
	std::vector<std::vector<std::vector<std::vector<unsigned int>>>> voxelMap;
	voxelMap.resize( tmpSize);
	// resize voxel array
	for( cnt1 = 0; cnt1 < voxelMap.size(); ++cnt1 ) {
		voxelMap[cnt1].resize( (unsigned int)ceilf( this->bBox.Height() / voxelLength) );
		for( cnt2 = 0; cnt2 < voxelMap[cnt1].size(); ++cnt2 ) {
			voxelMap[cnt1][cnt2].resize(
				(unsigned int)ceilf( this->bBox.Depth() / voxelLength) );
		}
	}
	// get all protein atom positions
	for( cnt1 = 0; cnt1 < numberOfAtoms; ++cnt1 ) {
		// get position of current atom
		tmpVec1.SetX( protein->ProteinAtomPositions()[cnt1*3+0]);
		tmpVec1.SetY( protein->ProteinAtomPositions()[cnt1*3+1]);
		tmpVec1.SetZ( protein->ProteinAtomPositions()[cnt1*3+2]);
		
		// add atom index to voxel map cell
		voxelMap[(unsigned int)floorf( (tmpVec1.GetX() - bBox.Left()) / voxelLength)]
			[(unsigned int)floorf( (tmpVec1.GetY() - bBox.Bottom()) / voxelLength)]
			[(unsigned int)floorf( (tmpVec1.GetZ() - bBox.Back()) / voxelLength)].push_back( cnt1 );
	}

	// resize atom neighbor array, if necessary
	if( this->atomNeighborsSize != numberOfAtoms ) {
		delete[] this->atomNeighbors;
		this->atomNeighborsSize = 0;
	}
	if( !this->atomNeighborsSize ) {
		this->atomNeighbors = new float[numberOfAtoms*atomNeighborCount*4];
		this->atomNeighborsSize = numberOfAtoms;
	}
	// temp vars for neighbor search
	unsigned int cnt, xId, yId, zId, maxXId, maxYId, maxZId, neighborId;
	int cntX, cntY, cntZ;
	float distance, threshold;
	//unsigned int neighborcounter;	// DEBUG
	// get max voxel ID
	maxXId = (unsigned int)voxelMap.size()-1;
	maxYId = (unsigned int)voxelMap[0].size()-1;
	maxZId = (unsigned int)voxelMap[0][0].size()-1;
	// for each atom: compute neighbors
	for( cnt1 = 0; cnt1 < numberOfAtoms; ++cnt1 ) {
		// reset neighbor ID
		neighborId = 0;
		// get current atom position
		tmpVec1.SetX( protein->ProteinAtomPositions()[cnt1*3+0]);
		tmpVec1.SetY( protein->ProteinAtomPositions()[cnt1*3+1]);
		tmpVec1.SetZ( protein->ProteinAtomPositions()[cnt1*3+2]);
		// get the radius of current atom
		radius1 = protein->AtomTypes()[protein->ProteinAtomData()[cnt1].TypeIndex()].Radius();
		// get current atom voxel ID
		xId = (unsigned int)floorf( ( tmpVec1.GetX() - bBox.Left()) / voxelLength);
		yId = (unsigned int)floorf( ( tmpVec1.GetY() - bBox.Bottom()) / voxelLength);
		zId = (unsigned int)floorf( ( tmpVec1.GetZ() - bBox.Back()) / voxelLength);

		//neighborcounter = 0;	// DEBUG

		// loop over all atoms to find vicinity
		for( cntX = ((xId > 0)?(-1):0); cntX < ((xId < maxXId)?2:1); ++cntX ) {
			for( cntY = ((yId > 0)?(-1):0); cntY < ((yId < maxYId)?2:1); ++cntY ) {
				for( cntZ = ((zId > 0)?(-1):0); cntZ < ((zId < maxZId)?2:1); ++cntZ ) {
					for( cnt = 0; cnt < voxelMap[xId+cntX][yId+cntY][zId+cntZ].size(); ++cnt ) {
						// don't check the atom itself
						if( voxelMap[xId+cntX][yId+cntY][zId+cntZ][cnt] == cnt1 )
							continue;
						// get current atom position
						cnt2 = voxelMap[xId+cntX][yId+cntY][zId+cntZ][cnt];
						tmpVec2.SetX( protein->ProteinAtomPositions()[cnt2*3+0]);
						tmpVec2.SetY( protein->ProteinAtomPositions()[cnt2*3+1]);
						tmpVec2.SetZ( protein->ProteinAtomPositions()[cnt2*3+2]);
						// get the radius of current atom
						radius2 = protein->AtomTypes()[protein->ProteinAtomData()[cnt2].TypeIndex()].Radius();
						
						// compute distance
						distance = ( tmpVec1 - tmpVec2).Length();
						// compute threshold
						threshold = radius1 + radius2 + 2.0f*this->probeRadius;
						// if extended atoms are intersecting, add atom 'cnt' to neigborhood
						if( distance < threshold ) {
							this->atomNeighbors[cnt1 * atomNeighborCount + neighborId + 0] = tmpVec2.GetX();
							this->atomNeighbors[cnt1 * atomNeighborCount + neighborId + 1] = tmpVec2.GetY();
							this->atomNeighbors[cnt1 * atomNeighborCount + neighborId + 2] = tmpVec2.GetZ();
							this->atomNeighbors[cnt1 * atomNeighborCount + neighborId + 3] = radius2;
							//neighborcounter++; // DEBUG
						}
					}
				}
			}
		}
		//std::cout << "atom " << cnt1 << " neighbors: " << neighborcounter << std::endl;	// DEBUG
	}

}

#endif /* (defined(WITH_OPENCL) && (WITH_OPENCL)) */
