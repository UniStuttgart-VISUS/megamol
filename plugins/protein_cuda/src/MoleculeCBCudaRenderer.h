/*
 * MoleculeCBCudaRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MOLSURFREN_CONTOURBUILDUP_CUDA_H_INCLUDED
#define MEGAMOL_MOLSURFREN_CONTOURBUILDUP_CUDA_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/math/Quaternion.h"
#include <vector>
#include <set>
#include <algorithm>
#include <list>
#include "vislib/graphics/FpsCounter.h"

#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "cuda_runtime_api.h"
//#include "cudpp/cudpp.h"

namespace megamol {
namespace protein_cuda {

	/**
	 * Molecular Surface Renderer class.
	 * Computes and renders the solvent excluded (Connolly) surface 
	 * using the Contour-Buildup Algorithm by Totrov & Abagyan.
	 */
	class MoleculeCBCudaRenderer : public megamol::core::view::Renderer3DModule
	{
	public:

		/**
		 * Answer the name of this module.
		 *
		 * @return The name of this module.
		 */
		static const char *ClassName(void)
		{
			return "MoleculeCBCudaRenderer";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) 
		{
			return "Offers molecular surface renderings.";
		}

		/**
		 * Answers whether this module is available on the current system.
		 *
		 * @return 'true' if the module is available, 'false' otherwise.
		 */
		static bool IsAvailable(void) {
			//return true;
			return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
		}
		
		/** ctor */
		MoleculeCBCudaRenderer(void);
		
		/** dtor */
		virtual ~MoleculeCBCudaRenderer(void);

	   /**********************************************************************
		 * 'get'-functions
		 **********************************************************************/
		
		/** Get probe radius */
		const float GetProbeRadius() const { return probeRadius; };

		/**********************************************************************
		 * 'set'-functions
		 **********************************************************************/

		/** Set probe radius */
		void SetProbeRadius( const float rad) { probeRadius = rad; };
		
	protected:
		
		/**
		 * Implementation of 'Create'.
		 *
		 * @return 'true' on success, 'false' otherwise.
		 */
		virtual bool create(void);
		
		/**
		 * Implementation of 'release'.
		 */
		virtual void release(void);

		/**
		 * Initialize CUDA
		 */
        bool initCuda(megamol::protein_calls::MolecularDataCall *mol, uint gridDim, core::view::CallRender3D *cr3d);

		/**
		 * Write atom positions and radii to an array for processing in CUDA
		 */
        void writeAtomPositions(const megamol::protein_calls::MolecularDataCall *mol);

		/**
		 * Write atom positions and radii to a VBO for processing in CUDA
		 */
		void writeAtomPositionsVBO(megamol::protein_calls::MolecularDataCall *mol);

    private:
        
        // This function returns the best GPU (with maximum GFLOPS)
        VISLIB_FORCEINLINE int cudaUtilGetMaxGflopsDeviceId() const {
            int device_count = 0;
            cudaGetDeviceCount( &device_count );

            cudaDeviceProp device_properties;
            int max_gflops_device = 0;
            int max_gflops = 0;
    
            int current_device = 0;
            cudaGetDeviceProperties( &device_properties, current_device );
            max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
            ++current_device;

            while( current_device < device_count ) {
                cudaGetDeviceProperties( &device_properties, current_device );
                int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
                if( gflops > max_gflops ) {
                    max_gflops        = gflops;
                    max_gflops_device = current_device;
                }
                ++current_device;
            }

            return max_gflops_device;
        }

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents( megamol::core::Call& call);

        /**
		 * Open GL Render call.
		 *
		 * @param call The calling call.
		 * @return The return value of the function.
		 */
		virtual bool Render( megamol::core::Call& call);

        /**
         * CUDA version of contour buildup algorithm
         *
         * TODO
         *
         */
        void ContourBuildupCuda(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * CPU version of contour buildup algorithm
         *
         * TODO
         *
         */
        void ContourBuildupCPU(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
        void UpdateParameters(const megamol::protein_calls::MolecularDataCall *mol);

		/**
		 * Deinitialises this renderer. This is only called if there was a 
		 * successful call to "initialise" before.
		 */
		virtual void deinitialise(void);
		
		/**********************************************************************
		 * variables
		 **********************************************************************/
		
		// caller slot
		megamol::core::CallerSlot molDataCallerSlot;
		
        // parameter slots
        megamol::core::param::ParamSlot probeRadiusParam;
        megamol::core::param::ParamSlot opacityParam;
        megamol::core::param::ParamSlot stepsParam;

		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

		// shader for the sphere raycasting
		vislib::graphics::gl::GLSLShader sphereShader;
		// shader for the spherical triangle raycasting
		vislib::graphics::gl::GLSLShader sphericalTriangleShader;
		// shader for the torus raycasting
		vislib::graphics::gl::GLSLShader torusShader;

		// the bounding box of the protein
		vislib::math::Cuboid<float> bBox;

		// radius of the probe atom
		float probeRadius;

		// max number of neighbors per atom
		const unsigned int atomNeighborCount;

		// CUDA Radix sort
        //CUDPPHandle sortHandle;
		// CUDA Scan
        //CUDPPHandle scanHandle;
		// CUDA Radix sort
        //CUDPPHandle probeSortHandle;

		// params
		bool cudaInitalized;
		uint numAtoms;
		SimParams params;
		uint3 gridSize;
		uint numGridCells;

		// CPU data
		float* m_hPos;              // particle positions
		uint*  m_hNeighborCount;    // atom neighbor count
		uint*  m_hNeighbors;        // atom neighbor count
		float* m_hSmallCircles;     // small circles
		uint*  m_hParticleHash;
		uint*  m_hParticleIndex;
		uint*  m_hCellStart;
		uint*  m_hCellEnd;
        float* m_hArcs;
        uint*  m_hArcCount;
        uint*  m_hArcCountScan;

		// GPU data
		float* m_dPos;
		float* m_dSortedPos;
		float* m_dSortedProbePos;
		uint*  m_dNeighborCount;
		uint*  m_dNeighbors;
		float* m_dSmallCircles;
		uint*  m_dSmallCircleVisible;
		uint*  m_dSmallCircleVisibleScan;
        float* m_dArcs;
        uint*  m_dArcIdxK;
        uint*  m_dArcCount;
        uint*  m_dArcCountScan;

		// grid data for sorting method
		uint*  m_dGridParticleHash; // grid hash value for each particle
		uint*  m_dGridParticleIndex;// particle index for each particle
		uint*  m_dGridProbeHash;    // grid hash value for each probe
		uint*  m_dGridProbeIndex;   // particle index for each probe
		uint*  m_dCellStart;        // index of start of each cell in sorted list
		uint*  m_dCellEnd;          // index of end of cell
		uint   gridSortBits;
		uint   m_colorVBO;          // vertex buffer object for colors
		float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
		float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

        
        vislib::Array<vislib::Array<vislib::math::Vector<float, 3> > > smallCircles;
        vislib::Array<vislib::Array<float> > smallCircleRadii;
        vislib::Array<vislib::Array<unsigned int> > neighbors;

		// VBO for all atoms
		GLuint atomPosVBO;
		// VBO for probe positions
		GLuint probePosVBO;
		// VBO for spherical triangle vector 1
		GLuint sphereTriaVec1VBO;
		// VBO for spherical triangle vector 2
		GLuint sphereTriaVec2VBO;
		// VBO for spherical triangle vector 3
		GLuint sphereTriaVec3VBO;
		// VBO for torus center
		GLuint torusPosVBO;
		// VBO for torus visibility sphere
		GLuint torusVSVBO;
		// VBO for torus axis
		GLuint torusAxisVBO;

        // singularity texture
        GLuint singTex;
        // singularity texture pixel buffer object
        GLuint singTexPBO;
        // texture coordinates
        GLuint texCoordVBO;
        // maximum number of probe neighbors
        uint probeNeighborCount;
        unsigned int texHeight;
        unsigned int texWidth;

        bool setCUDAGLDevice;
	};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MEGAMOL_MOLSURFACERENDERERCONTOURBUILDUP_CUDA_H_INCLUDED */
