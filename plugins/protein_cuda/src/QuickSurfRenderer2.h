/*
 * QuickSurfRenderer2.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINCUDAPLUGIN_QSR2_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_QSR2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "protein_calls/DiagramCall.h"
#include "Color.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "mmcore/view/CallClipPlane.h"

#include "WKFUtils.h"
#include "CUDAQuickSurf.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <cuda_gl_interop.h>

namespace megamol {
namespace protein_cuda {

    /*
     * Simple Molecular Renderer class
     */

    class QuickSurfRenderer2 : public megamol::core::view::Renderer3DModuleDS
    {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "QuickSurfRenderer2";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void)
        {
            return "Offers molecule renderings.";
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
        QuickSurfRenderer2(void);

        /** Dtor. */
        virtual ~QuickSurfRenderer2(void);

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
         * Calculate the density map and surface.
         * TODO
         *
         * @return 
         */
        int calcSurf( megamol::core::moldyn::MultiParticleDataCall *mol, float *posInter,
                         int quality, float radscale, float gridspacing,
                         float isoval, bool useCol);

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

        /**
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
        void UpdateParameters( const megamol::core::moldyn::MultiParticleDataCall *mol);

        /**
         * The callback that returns the data for the area diagram.
         * 
         * @param call The calling call.
         */
        virtual bool GetAreaDiagramData(core::Call& call);

        bool recomputeAreaDiagramCallback(core::param::ParamSlot& slot);

        void getClipData(float *clipDat, float *clipCol);

        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataCallerSlot;
        
        /** callee slot */
        megamol::core::CalleeSlot areaDiagramCalleeSlot;

        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        /** parameter slot for color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
        // QuickSurf parameters
        megamol::core::param::ParamSlot qualityParam;
        megamol::core::param::ParamSlot radscaleParam;
        megamol::core::param::ParamSlot gridspacingParam;
        megamol::core::param::ParamSlot isovalParam;
        // paramater to turn two sided lighting on and off
        megamol::core::param::ParamSlot twoSidedLightParam;
        // paramater to turn two sided lighting on and off
        megamol::core::param::ParamSlot surfaceColorParam;
        // paramater to recompute the area array
        megamol::core::param::ParamSlot recomputeAreaDiagramParam;

        // Parameter to google offscreen rendering
        // TODO
        //megamol::core::param::ParamSlot toggleOffscreenRenderingParam;

        /** shader for the spheres (raycasting view) */
        vislib::graphics::gl::GLSLShader sphereShader;
        // shader for per pixel lighting (polygonal view)
        vislib::graphics::gl::GLSLShader lightShader;
        vislib::graphics::gl::GLSLShader lightShaderOR;
        
        /** The current coloring mode */
        Color::ColoringMode currentColoringMode;

        /** The color lookup table (for chains, amino acids,...) */
        vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
        /** The color lookup table which stores the rainbow colors */
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;

        /** The atom color table for rendering */
        vislib::Array<float> atomColorTable;
        
        float *volmap;           ///< Density map
        float *voltexmap;        ///< Volumetric texture map in RGB format
        float isovalue;          ///< Isovalue of the surface to extract
        float solidcolor[3];     ///< RGB color to use when not using per-atom colors
        int numvoxels[3];        ///< Number of voxels in each dimension
        float origin[3];         ///< Origin of the volumetric map
        float xaxis[3];          ///< X-axis of the volumetric map
        float yaxis[3];          ///< Y-axis of the volumetric map
        float zaxis[3];          ///< Z-axis of the volumetric map

        void *cudaqsurf;         ///< Pointer to CUDAQuickSurf object if it exists
        int gpuvertexarray;      ///< Flag indicating if we're getting mesh from GPU 
        int gpunumverts;         ///< GPU vertex count
        float *gv;               ///< GPU vertex coordinates
        float *gn;               ///< GPU vertex normals
        float *gc;               ///< GPU vertex colors
        int gpunumfacets;        ///< GPU face count
        int   *gf;               ///< GPU facet index list

        wkf_timerhandle timer;   ///< Internal timer for performance instrumentation
        double pretime;          ///< Internal timer for performance instrumentation
        double voltime;          ///< Internal timer for performance instrumentation
        double gradtime;         ///< Internal timer for performance instrumentation
        double mctime;           ///< Internal timer for performance instrumentation
        double mcverttime;       ///< Internal timer for performance instrumentation
        double reptime;          ///< Internal timer for performance instrumentation
        
        GLuint v3f_vbo;
        GLuint n3f_vbo;
        GLuint c3f_vbo;

        cudaGraphicsResource *v3f_vbo_res;
        cudaGraphicsResource *n3f_vbo_res;
        cudaGraphicsResource *c3f_vbo_res;

        
        // CPU data
        float* m_hPos;              // particle positions
        size_t m_hPosSize;
        UINT64 numParticles;
        float currentSurfaceArea;
        bool recomputeAreaDiagram;
        float callTime;
        /** array for surface area */
		protein_calls::DiagramCall::DiagramSeries *areaDiagramData;
        
        bool setCUDAGLDevice;

        /** The call for clipping plane */
        core::CallerSlot getClipPlaneSlot;
    };


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // MMPROTEINCUDAPLUGIN_QSR2_H_INCLUDED
