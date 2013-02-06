/*
 * VolumeMeshRenderer.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VOLUMEMESHRENDERER_H_INCLUDED
#define MEGAMOLCORE_VOLUMEMESHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "VolumeMeshRenderer.cuh"
#include "view/Renderer3DModuleDS.h"
#include "DiagramCall.h"
#include "SplitMergeCall.h"
#include "Color.h"
#include "param/ParamSlot.h"
#include "vislib/GLSLGeometryShader.h"
#include "vislib/Cuboid.h"
#include "MolecularAOShader.h"
#include "CallerSlot.h"
#include "WKFUtils.h"
#include "CUDAQuickSurf.h"
#include <cuda_runtime.h>

namespace megamol {
namespace protein {

    /**
     * Volume Mesh Renderer class
     */
    class VolumeMeshRenderer : public megamol::core::view::Renderer3DModuleDS
    {
    public:

        enum PolygonMode 
        {
            POINT,
            LINE,
            FILL
        };

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) {
            return "VolumeMeshRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description(void) {
            return "Offers volume mesh renderings.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        VolumeMeshRenderer(void);

        /** Dtor. */
        virtual ~VolumeMeshRenderer(void);

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
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities(megamol::core::Call& call);

        virtual bool GetDiagramData(core::Call& call);

        virtual bool GetSplitMergeData(core::Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(megamol::core::Call& call);

        /**
         * The Open GL Render callback.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        virtual bool Render(megamol::core::Call& call);
        
        /**
         * Calculate the density map and surface.
         * TODO
         *
         * @return 
         */
        int calcMap(MolecularDataCall *mol, float *posInter,
                         int quality, float radscale, float gridspacing,
                         float isoval, bool useCol);

    private:
        // This function returns the best GPU (with maximum GFLOPS)
        int cudaUtilGetMaxGflopsDeviceId() const {
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
         * Marches the isosurface and updates all associated vertex buffer objects.
         */
        //bool UpdateMesh(GLuint volumeTextureId, 
        bool UpdateMesh(float* densityMap, 
            vislib::math::Vector<float, 3> translation, 
            vislib::math::Vector<float, 3> scale, const float* aoVolumeHost, 
            MolecularDataCall *mol);

        /**
         *
         */
        float4 GetNextColor();
        
        /**
         *
         */
        void ParameterRefresh( const MolecularDataCall *mol);
        
        /**
         *
         */
        void ValidateCubeMemory();

        /**
         *
         */
        void ValidateOldCubeMemory();

        /**
         *
         */
        void ValidateActiveCubeMemory(uint activeCubeCount);

        /**
         *
         */
        void ValidateVertexMemory(uint vertexCount);

        /**
         *
         */
        void ValidateCentroidMemory(uint centroidCount);

        /**
         * Creates an vertex buffer object.
         *
         * @param vbo
         * @param size
         * @param cudaResource
         */
        void CreateVbo(GLuint* vbo, size_t size, struct cudaGraphicsResource** cudaResource);

        /**
         * Destroys an vertex buffer object.
         *
         * @param vbo
         * @param cudaResource
         */
        void DestroyVbo(GLuint* vbo, struct cudaGraphicsResource** cudaResource);

        /**
         * Gets the last element from an scan.
         * @param buffer
         * @param scanBuffer
         * @param size
         */
        uint CudaGetLast(uint* buffer, uint* scanBuffer, size_t size);

        /**
         *
         */
        void CudaVerify(cudaError error, const int line);


        /** caller slot */
        megamol::core::CallerSlot molDataCallerSlot;

        megamol::core::CallerSlot selectionCallerSlot;
        megamol::core::CallerSlot hiddenCallerSlot;

        /** callee slot */
        megamol::core::CalleeSlot diagramCalleeSlot;

        megamol::core::CalleeSlot splitMergeCalleeSlot;

        /** polygon mode */
        megamol::core::param::ParamSlot polygonModeParam;
        PolygonMode polygonMode;

        /** blending flag */
        megamol::core::param::ParamSlot blendItParam;
        bool blendIt;

        /** show normals flag */
        megamol::core::param::ParamSlot showNormalsParam;
        bool showNormals;

        /** show centroids flag */
        megamol::core::param::ParamSlot showCentroidsParam;
        bool showCentroids;

        /** ao threshold */
        megamol::core::param::ParamSlot aoThresholdParam;
        float aoThreshold;

        /** iso value */
        megamol::core::param::ParamSlot isoValueParam;
        float isoValue;

        /** distance values */
        megamol::core::param::ParamSlot maxDistanceParam;
        megamol::core::param::ParamSlot maxDeltaDistanceParam;
        float maxDistance;
        float maxDeltaDistance;

        /** color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        vislib::Array< vislib::math::Vector<float, 3> > colorTable;

        /** maximum CUDA grid size */
        dim3 gridSize;

        /** last animation time */
        float lastTime;

        /** volume size */
        uint3 volumeSize;
        uint cubeCountAllocated;

        /** allocation strategy */
        uint activeCubeCountAllocted;
        uint vertexCountAllocted;
        uint centroidCountAllocated;

        /** number of active cubes */
        uint cubeCount;
        uint activeCubeCount;
        uint* cubeStates;   // 0 if cube is inactive (no triangle output)
        uint* cubeOffsets;  // the offset of the cube (depends on the cubeStates)
        uint* cubeMap;      // array of global cube indices (1..n) for the active cubes (1..m) : m <= n

        /** arrays from previous time step */
        uint cubeCountOld;
        uint activeCubeCountOld;
        uint* cubeStatesOld;   // 0 if cube is inactive (no triangle output)
        uint* cubeOffsetsOld;  // the offset of the cube (depends on the cubeStates)
        uint* cubeMapOld;      // array of global cube indices (1..n) for the active cubes (1..m) : m <= n
        uint* verticesPerTetrahedronOld;
        uint* eqListOld;

        /** number of extracted vertices */
        uint vertexCount;
        uint* verticesPerTetrahedron;
        uint* vertexOffsets;
        int2* tetrahedronLabelPair;

        /** labling **/
        uint* eqList;
        uint* refList;
        bool* modified;
        bool* segmentsRemoved;
        uint largestFeatureLabelOld;

        /** centroid map-reduce */
        uint* vertexLabels;
        uint* vertexLabelsCopy;
        float4* verticesCopy;
        float* triangleAO;
        float* triangleAreas;
        uint* centroidLabels;
        float* centroidAreas;
        float4* centroidSums;
        uint* centroidCounts;

        /** centroids */
        float4* centroids;
        float4* centroidColors;
        int centroidColorsIndex;
        uint centroidCountLast;
        float4* centroidsLast;
        float4* centroidColorsLast;
        uint* centroidLabelsLast;
        float* centroidAreasLast;
        
        unsigned int featureCounter;
        unsigned int* featureListIdx;
        uint2* featureStartEnd;

        /** AO */
        MolecularAOShader aoShader;
        cudaArray* aoVolume;

        /** vertex buffer for positions */
        GLuint positionVbo;
        struct cudaGraphicsResource* positionResource;

        /** vertex buffer for normals */
        GLuint normalVbo;
        struct cudaGraphicsResource* normalResource;

        /** vertex buffer for normals */
        GLuint colorVbo;
        struct cudaGraphicsResource* colorResource;

        /** */
        vislib::graphics::gl::GLSLGeometryShader normalShader;
        vislib::graphics::gl::GLSLShader lightShader;
        
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;

        // QuickSurf parameters
        megamol::core::param::ParamSlot qualityParam;
        megamol::core::param::ParamSlot radscaleParam;
        megamol::core::param::ParamSlot gridspacingParam;

        float solidcolor[3];     ///< RGB color to use when not using per-atom colors
        int numvoxels[3];        ///< Number of voxels in each dimension
        float origin[3];         ///< Origin of the volumetric map
        float xaxis[3];          ///< X-axis of the volumetric map
        float yaxis[3];          ///< Y-axis of the volumetric map
        float zaxis[3];          ///< Z-axis of the volumetric map

        void *cudaqsurf;         ///< Pointer to CUDAQuickSurf object if it exists

        wkf_timerhandle timer;   ///< Internal timer for performance instrumentation
        double pretime;          ///< Internal timer for performance instrumentation
        double voltime;          ///< Internal timer for performance instrumentation
        double gradtime;         ///< Internal timer for performance instrumentation
        double mctime;           ///< Internal timer for performance instrumentation
        double mcverttime;       ///< Internal timer for performance instrumentation
        double reptime;          ///< Internal timer for performance instrumentation
        
        /** The atom color table for rendering */
        vislib::Array<float> atomColorTable;
        /** The color lookup table which stores the rainbow colors */
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;
        /** The current coloring mode */
        Color::ColoringMode currentColoringMode;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam;
        /** parameter slot for min color of gradient color mode */
        megamol::core::param::ParamSlot minGradColorParam;
        /** parameter slot for mid color of gradient color mode */
        megamol::core::param::ParamSlot midGradColorParam;
        /** parameter slot for max color of gradient color mode */
        megamol::core::param::ParamSlot maxGradColorParam;        

        /** parameter slot for segment area threshold */
        megamol::core::param::ParamSlot areaThresholdParam;
        
        vislib::Array<bool> featureSelection;
        vislib::Array<bool> featureVisibility;

        /** array for surface features */
        vislib::PtrArray<DiagramCall::DiagramSeries> featureList;
        
        /** array for splitmerge series */
        vislib::PtrArray<SplitMergeCall::SplitMergeSeries> splitMergeList;
        
        /** array for feature transitions */
        vislib::PtrArray<SplitMergeCall::SplitMergeTransition> transitionList;

        /** data bounding box */
        vislib::math::Cuboid<float> dataBBox;

    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_VOLUMEMESHRENDERER_H_INCLUDED
