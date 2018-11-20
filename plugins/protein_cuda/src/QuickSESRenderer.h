/*
 * QuickSESRenderer.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINCUDAPLUGIN_QUICKSESRENDERER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_QUICKSESRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/MolecularDataCall.h"
#include "Color.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"

#include "WKFUtils.h"
#include "CUDAQuickSES.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <cuda_gl_interop.h>

namespace megamol {
namespace protein_cuda {

    /*
     * Simple Molecular Renderer class
     */

    class QuickSESRenderer : public megamol::core::view::Renderer3DModule
    {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "QuickSESRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void)
        {
            return "Offers grid-based solvent excluded surface renderings.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void)
        {
            return true;
        }

        /** Ctor. */
        QuickSESRenderer(void);

        /** Dtor. */
        virtual ~QuickSESRenderer(void);

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
		 * Update the parameter values.
		 * @param mol The molecular data call.
		 */
		void UpdateParameters(const protein_calls::MolecularDataCall *mol);

		/**
		* Calculate the SES density map and surface.
		* @param mol The molecular data call.
		* @param pos The atom positions and radii (xyzr).
		* @return true if successful, false otherwise.
		*/
		bool calcSurf(protein_calls::MolecularDataCall *mol, const float *pos);
    private:

       /**********************************************************************
        * 'render'-functions
        **********************************************************************/
        
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
         * The Open GL Render callback.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        virtual bool Render( megamol::core::Call& call);

        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataCallerSlot;

        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        /** parameter slot for color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam;
        /** parameter slot for min color of gradient color mode */
        megamol::core::param::ParamSlot minGradColorParam;
        /** parameter slot for mid color of gradient color mode */
        megamol::core::param::ParamSlot midGradColorParam;
        /** parameter slot for max color of gradient color mode */
        megamol::core::param::ParamSlot maxGradColorParam;
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
        // QuickSurf parameters
        megamol::core::param::ParamSlot probeRadiusParam;
        megamol::core::param::ParamSlot gridSpacingParam;
        /** Toggle offscreen rendering */
        megamol::core::param::ParamSlot offscreenRenderingParam;

        /** shader for the spheres (raycasting view) */
        vislib::graphics::gl::GLSLShader sphereShader;
        // shader for per pixel lighting (polygonal view)
        vislib::graphics::gl::GLSLShader lightShader;
        // OFFSCREEN RENDERING shader for per pixel lighting (polygonal view)
        vislib::graphics::gl::GLSLShader lightShaderOR;
        
        /** The current coloring mode */
        Color::ColoringMode currentColoringMode;

        /** The color lookup table (for chains, amino acids,...) */
        vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
        /** The color lookup table which stores the rainbow colors */
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;

        /** The atom color table for rendering */
        vislib::Array<float> atomColorTable;
        
        void *cudaqses;         ///< Pointer to CUDAQuickSurf object if it exists
		
		float *pos0;
		float *pos1;
		unsigned int posArraySize;
		vislib::Array<float> posInter;

        bool setCUDAGLDevice;
    };


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // MMPROTEINCUDAPLUGIN_QUICKSESRENDERER_H_INCLUDED
