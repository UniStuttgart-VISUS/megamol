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
#include "mmcore/view/Renderer3DModuleDS.h"
#include "protein_calls/DiagramCall.h"
#include "protein_calls/SplitMergeCall.h"
#include "protein_calls/BindingSiteCall.h"
#include "Color.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/math/Cuboid.h"
#include "MolecularAOShader.h"
#include "mmcore/CallerSlot.h"
#include "WKFUtils.h"
#include "CUDAQuickSurf.h"
#include <cuda_runtime.h>
#include "CenterLineGenerator.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol {
namespace protein_cuda {

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

        virtual bool GetDiagramData(core::Call& call);

        virtual bool GetSplitMergeData(core::Call& call);

        virtual bool GetCenterLineDiagramData(core::Call& call);

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
        int calcMap(megamol::protein_calls::MolecularDataCall *mol, float *posInter,
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
         * Sort the triangles of the mesh for transparent rendering.
         */
        void SortTriangleMesh();

        /**
         * Marches the isosurface and updates all associated vertex buffer objects.
         */
        //bool UpdateMesh(GLuint volumeTextureId, 
        bool UpdateMesh(float* densityMap, 
            vislib::math::Vector<float, 3> translation, 
            vislib::math::Vector<float, 3> scale, const float* aoVolumeHost, 
            megamol::protein_calls::MolecularDataCall *mol, int* neighborMap);

        /**
         *
         */
        float4 GetNextColor();
        
        /**
         *
         */
		void ParameterRefresh(const megamol::protein_calls::MolecularDataCall *mol, const protein_calls::BindingSiteCall *bs = 0);
        
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

        
        /** MolecularDataCall caller slot */
        megamol::core::CallerSlot molDataCallerSlot;
        /** BindingSiteCall caller slot */
        megamol::core::CallerSlot bsDataCallerSlot;
        /** residue selection caller slot */
        core::CallerSlot resSelectionCallerSlot;

        megamol::core::CallerSlot selectionCallerSlot;
        megamol::core::CallerSlot hiddenCallerSlot;

        /** callee slot */
        megamol::core::CalleeSlot diagramCalleeSlot;

        megamol::core::CalleeSlot splitMergeCalleeSlot;
        
        megamol::core::CalleeSlot centerLineDiagramCalleeSlot;

        /** polygon mode */
        megamol::core::param::ParamSlot polygonModeParam;
        PolygonMode polygonMode;

        /** blending flag */
        megamol::core::param::ParamSlot blendItParam;
        megamol::core::param::ParamSlot alphaParam;
        bool blendIt;
        
        /** centerline param */
        megamol::core::param::ParamSlot showCenterlinesParam;

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
        
        /** parameter for minimum distance of center line node to molecular surface */
        megamol::core::param::ParamSlot minDistCenterLineParam;

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
        float4* featureVertices;
        float4* featureVerticesOut;
        uint* featureVertexIdx;
        uint* featureVertexCnt;
        uint* featureVertexCntOut;
        uint* featureVertexStartIdx;
        uint* featureVertexIdxOut;
        float* triangleAO;
        float* triangleAreas;
        uint* centroidLabels;
        float* centroidAreas;
        float4* centroidSums;
        uint* centroidCounts;
        uint* featureEdgeCnt;
        uint* featureEdgeCntOut;
        uint2 *featureEdges;
        uint2 *featureEdgesOut;
        uint edgeCount;

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
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam0;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam1;
        /** parameter slot for coloring mode weighting*/
        megamol::core::param::ParamSlot cmWeightParam;
        /** parameter slot for min color of gradient color mode */
        megamol::core::param::ParamSlot minGradColorParam;
        /** parameter slot for mid color of gradient color mode */
        megamol::core::param::ParamSlot midGradColorParam;
        /** parameter slot for max color of gradient color mode */
        megamol::core::param::ParamSlot maxGradColorParam;        
        /** parameter slot for segment area threshold */
        megamol::core::param::ParamSlot areaThresholdParam;
        
        /** The current coloring mode */
        int currentColoringMode0;
        int currentColoringMode1;
        
        vislib::Array<bool> featureSelection;
        vislib::Array<bool> featureVisibility;

        /** array for surface features */
        vislib::PtrArray<protein_calls::DiagramCall::DiagramSeries> featureList;
        
        /** array for splitmerge series */
		vislib::PtrArray<protein_calls::SplitMergeCall::SplitMergeSeries> splitMergeList;
        
        /** array for feature transitions */
		vislib::PtrArray<protein_calls::SplitMergeCall::SplitMergeTransition> transitionList;
        
        /** array for surface feature center lines */
		vislib::PtrArray<protein_calls::DiagramCall::DiagramSeries> featureCenterLines;

        /** data bounding box */
        vislib::math::Cuboid<float> dataBBox;
        
        /** host array for the nearest neighbor atom for each vertex */
        int *neighborAtomOfVertex;
        /** device array for the nearest neighbor atom for each vertex */
        int *neighborAtomOfVertex_d;
        /** the vertex color array */
        float* vertexColors;
        
        /** pinned memory for start and end of features */
        uint2 *featureStartEndHost;
        /** pinned memory for feature triangle vertices */
        float4 *featureTriangleVerticesHost;
        /** pinned memory for feature triangle vertices */
        uint *featureTriangleVertexIndicesHost;
        /** pinned memory for feature triangle vertices */
        uint2 *featureTriangleEdgesHost;
        /** pinned memory for feature triangle vertices */
        uint *featureTriangleEdgeCountHost;
        /** counter for feature triangles */
        unsigned int featureTrianglesCount;
        /** counter or compacted feature vertices */
        unsigned int featureVertexCntNew;

        /** center line variables */
        vislib::Array<CenterLineGenerator::CenterLineEdges> clEdges;
        vislib::Array<CenterLineGenerator::CenterLineNodes> clNodes;
        vislib::Array<CenterLineGenerator*> clg;
        
        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        
        cudaEvent_t start, stop;    // TIMING
        float time;                 // TIMING

        /** call ptr for residue selection */
		protein_calls::ResidueSelectionCall *resSelectionCall;
        bool *atomSelection;
        unsigned int atomSelectionCnt;

        // width and height of view
        unsigned int width, height;

		/** halo rendering of selected features **/
		vislib::graphics::gl::FramebufferObject haloFBO;
		vislib::graphics::gl::FramebufferObject haloBlurFBO;
		vislib::graphics::gl::FramebufferObject haloBlurFBO2;
        megamol::core::param::ParamSlot haloEnableParam;
        megamol::core::param::ParamSlot haloAlphaParam;
        megamol::core::param::ParamSlot haloColorParam;
        vislib::graphics::gl::GLSLShader haloGenerateShader;
        vislib::graphics::gl::GLSLShader haloGrowShader;	
		vislib::graphics::gl::GLSLShader haloGaussianHoriz;
		vislib::graphics::gl::GLSLShader haloGaussianVert;
		vislib::graphics::gl::GLSLShader haloDifferenceShader;
        
        bool setCUDAGLDevice;
	};


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // MEGAMOLCORE_VOLUMEMESHRENDERER_H_INCLUDED
