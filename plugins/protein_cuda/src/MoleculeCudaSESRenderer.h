/*
 * MoleculeCudaSESRenderer.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_MOLRENCUDASES_H_INCLUDED
#define MEGAMOL_MOLRENCUDASES_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include "vislib/graphics/gl/FramebufferObject.h"

#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "cuda_runtime_api.h"
//#include "cudpp/cudpp.h"

#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace protein_cuda {

    /**
     * Molecular Surface Renderer class.
     * Computes and renders the Solvent Excluded Surface on the GPU.
     */
    class MoleculeCudaSESRenderer : public megamol::core::view::Renderer3DModule
    {
    public:
        
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "MoleculeCudaSESRenderer";
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
        static bool IsAvailable(void) 
        {
            //return true;
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
        }
        
        /** ctor */
        MoleculeCudaSESRenderer(void);
        
        /** dtor */
        virtual ~MoleculeCudaSESRenderer(void);

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
         * Creates a FBO for visibility tests with the given size.
         * @param size      The maximum size for the FBO textures.
         * @param width     Out parameter: texture width.
         * @param height    Out parameter: texture height.
         */
        void createVisibilityFBO( unsigned int size, unsigned int &width,
            unsigned int &height );

        /**
         * Renders all atoms using GPU ray casting and write atom ID to red color channel.
         * @param protein The protein call.
         */
        void RenderAtomIdGPU(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Create the FBO for visibility test.
         * @param maxSize The maximum dimension for width/height.
         */
        void CreateVisibilityFBO( unsigned int maxSize);

        /**
         * Create the FBO for visible atoms.
         * @param atomCount The number of protein atoms.
         */
        void CreateVisibleAtomsFBO( unsigned int atomCount);

        /**
         * Find all visible atoms
         * @param protein The protein call.
         */
        void FindVisibleAtoms(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Find for each visible atom all atoms that are in the proximity.
         * @param protein The protein call.
         */
        void ComputeVisibleAtomsVicinityTable(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Use CUDA to find for each visible atom all atoms that are in the neighborhood.
         * @param protein The protein call.
         */
        void ComputeVicinityTableCUDA(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Compute the Reduced Surface using the Fragment Shader.
         * @param protein The protein call.
         */
        void ComputeRSFragShader(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Creates the FBO for reduced surface triangle generation.
         * @param atomCount The total number of atoms.
         * @param vicinityCount The maximum number of vicinity atoms.
         */
        void CreateTriangleFBO( unsigned int atomCount, unsigned int vicinityCount);

        /**
         * Creates the FBO for visible reduced surface triangles.
         */
        void CreateVisibleTriangleFBO( unsigned int atomCount, unsigned int vicinityCount);

        /**
         * Render all potential RS-faces as triangles using a vertex shader.
         * @param protein The protein call.
         */
        void RenderTriangles(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Find all visible triangles (i.e. visible RS-faces).
         * @param protein The protein call.
         */
        void FindVisibleTriangles(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Create fbo for adjacent triangles of visible triangles
         * @param atomCount The total number of atoms.
         * @param vicinityCount The maximum number of vicinity atoms.
         */
        void CreateAdjacentTriangleFBO( unsigned int atomCount, unsigned int vicinityCount);

        /**
         * Find the adjacent triangles to all visible triangles.
         * @param protein The protein call.
         */
        void FindAdjacentTriangles(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Find the adjacent triangles to all visible triangles.
         * @param mol The molecular data call.
         */
        void FindAdjacentTrianglesCUDA(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Create the VBO for transform feedback.
         */
        void CreateTransformFeedbackVBO(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Find all intersecting probes for each probe and create the singularity texture.
         * @param mol The molecular data call.
         * @param numProbes The number of RS-faces (i.e. probes in fixed positions).
         */
        void CreateSingularityTextureCuda(megamol::protein_calls::MolecularDataCall *mol, unsigned int numProbes);

        /**
         * Render the SES using GPU ray casting.
         * @param mol The molecular data call.
         * @param primitiveCount The number of primitives.
         */
        void RenderSESCuda(megamol::protein_calls::MolecularDataCall *mol, unsigned int primitiveCount);

        /**
         * Render all visible atoms using GPU ray casting.
         * @param protein The protein call.
         */
        void RenderVisibleAtomsGPU(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Mark all atoms which are vertices of adjacent triangles as visible
         */
        void MarkAdjacentAtoms(megamol::protein_calls::MolecularDataCall *protein);

        /**
         * Mark all atoms which are vertices of adjacent triangles as visible
         *
         * @param mol The molecular data call.
         */
        void MarkAdjacentAtomsCUDA(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Initialize CUDA
         * @param protein The molecular data call.
         * @param gridDim The grid dimension.
         * @param cr3d Pointer to the render call.
         * @return 'true' if initialization was successful, otherwise 'false'
         */
        bool initCuda(megamol::protein_calls::MolecularDataCall *protein, uint gridDim, core::view::CallRender3D *cr3d);

        /**
         * Write atom positions and radii to an array for processing in CUDA
         */
        void writeAtomPositions(megamol::protein_calls::MolecularDataCall *protein );

    private:

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
         * Deinitialises this renderer. This is only called if there was a 
         * successful call to "initialise" before.
         */
        virtual void deinitialise(void);

        /**
         * Compute the Reduced Surface using CUDA.
         *
         * @param mol The molecular data call.
         */
        void ComputeRSCuda(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Find all visible triangles (i.e. visible RS-faces).
         * @param mol The molecular call.
         */
        void FindVisibleTrianglesCuda(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Render all potential RS-faces as triangles.
         * @param mol The protein call.
         */
        void RenderTrianglesCuda(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Render all potential RS-faces as triangles.
         * @param mol The protein call.
         */
        void RenderTrianglesCuda2(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Render all visible RS-faces as triangles.
         * @param mol The protein call.
         */
        void RenderVisibleTrianglesCuda(megamol::protein_calls::MolecularDataCall *mol);

        /**
         * Create geometric primitives for ray casting.
         * @param mol The molecular data call.
         * @return The number of primitives which were read back.
         */
        unsigned int CreateGeometricPrimitivesCuda(megamol::protein_calls::MolecularDataCall *mol);

        /**********************************************************************
         * variables
         **********************************************************************/
        
        // caller slot
        megamol::core::CallerSlot protDataCallerSlot;
        
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;

        /** parameter slot for debugging */
        megamol::core::param::ParamSlot debugParam;
        /** parameter slot for probe radius */
        megamol::core::param::ParamSlot probeRadiusParam;
        
        // shaders
        vislib::graphics::gl::GLSLShader writeSphereIdShader;
        vislib::graphics::gl::GLSLShader drawPointShader;
        vislib::graphics::gl::GLSLShader sphereShader;
        vislib::graphics::gl::GLSLShader reducedSurfaceShader;
        vislib::graphics::gl::GLSLShader drawTriangleShader;
        vislib::graphics::gl::GLSLGeometryShader drawVisibleTriangleShader;
        vislib::graphics::gl::GLSLShader sphericalTriangleShader;
        vislib::graphics::gl::GLSLShader torusShader;
        vislib::graphics::gl::GLSLShader adjacentTriangleShader;
        vislib::graphics::gl::GLSLShader adjacentAtomsShader;
        vislib::graphics::gl::GLSLShader drawCUDATriangleShader;
        vislib::graphics::gl::GLSLGeometryShader visibleTriangleIdxShader;

        // camera information
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        
        // the bounding box of the protein
        vislib::math::Cuboid<float> bBox;

        // current width and height of camera view
        unsigned int width, height;

        // interpolated atom positions
        float *posInter;

        // the GL clear color
        float clearCol[4];

        // start value for fogging
        float fogStart;
        // transparency value
        float transparency;

        // the radius of the probe defining the SES
        float probeRadius;

        // ----- variables for atom visibility test -----
        // the ID array (contains IDs from 0 .. n)
        float *proteinAtomId;
        unsigned int proteinAtomCount;
        // visibility FBO, textures and parameters
        GLuint visibilityFBO;
        GLuint visibilityColor;
        GLuint visibilityDepth;
        unsigned int visibilityTexWidth, visibilityTexHeight;
        // variables for drawing visibility texture as vertices (data + VBO)
        float *visibilityVertex;
        GLuint visibilityVertexVBO;
        // FBO, textures and parameters for visible atoms
        GLuint visibleAtomsFBO;
        GLuint visibleAtomsColor;
        GLuint visibleAtomsDepth;
        // array for visible atoms maks
        float *visibleAtomMask;
        // ----- vicinity table for visible atoms -----
        float *vicinityTable;
        unsigned int voxelLength;
        unsigned int *voxelMap;
        unsigned int voxelMapSize;
        unsigned int numAtomsPerVoxel;
        GLuint vicinityTableTex;
        float *visibleAtomsList;
        unsigned int *visibleAtomsIdList;
        unsigned int visibleAtomCount;
        GLuint visibleAtomsTex;
        GLuint visibleAtomsIdTex;
        // vertex and color arrays for geometry shader implementation
        float* atomPosRSGS;
        float* atomColRSGS;
        // ----- FBO and textures for triangle (i.e. potential RS-faces) computation
        GLuint triangleFBO;
        GLuint triangleColor0;
        GLuint triangleColor1;
        GLuint triangleColor2;
        GLuint triangleNormal;
        GLuint triangleDepth;
        // VBO and data arrays for triangle drawing
        GLuint triangleVBO;
        float* triangleVertex;
        // ----- FBO and textures for visible triangles (i.e. visible RS-faces)
        GLuint visibleTriangleFBO;
        GLuint visibleTriangleColor;
        GLuint visibleTriangleDepth;
        // ----- render to VBO -----
        GLuint visibilityTexVBO;
        // ----- FBO and textures for finding adjacent triangles -----
        GLuint adjacentTriangleFBO;
        GLuint adjacentTriangleColor;
        GLuint adjacentTriangleDepth;
        float *adjacentTriangleVertex;
        GLuint adjacentTriangleVBO;
        GLuint adjacentTriangleTexVBO;

        // ----- vicinity table for probes -----
        unsigned int *probeVoxelMap;
        // ----- singularity handling -----
        float *singTexData;
        float *singTexCoords;
        GLuint singTex;

        // VBO for spherical triangle center transform feedback
        GLuint sphericalTriaVBO;
        // query for transform feedback
        GLuint query;

        float delta;

        bool first;

        // CUDA Radix sort
        //CUDPPHandle sortHandle;
        //CUDPPHandle sortHandleProbe;

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
        uint*  m_hParticleIndex;
        // GPU data
        float* m_dPos;
        float* m_dSortedPos;
        uint*  m_dNeighborCount;
        uint*  m_dNeighbors;
        // grid data for sorting method
        uint*  m_dGridParticleHash;  // grid hash value for each particle
        uint*  m_dGridParticleIndex; // particle index for each particle
        uint*  m_dCellStart;         // index of start of each cell in sorted list
        uint*  m_dCellEnd;           // index of end of cell
        uint   gridSortBits;
        // additional parameters for the reduced surface
        RSParams rsParams;
        // arrays for reduced surface computation
        uint* m_dPoint1;
        float* m_dPoint2;
        float* m_dPoint3;
        float* m_dProbePosTable;
        float* m_dVisibleAtoms;
        uint* m_dVisibleAtomsId;
        struct cudaGraphicsResource *cudaVboResource;

        struct cudaGraphicsResource *cudaTexResource;

        uint *pointIdx;

        // vertex array for fast drawing of texture coordinates (visible triangle testing)
        float *visTriaTestVerts;

        struct cudaGraphicsResource *cudaVisTriaVboResource;

        struct cudaGraphicsResource *cudaTorusVboResource;
        GLuint torusVbo;
        struct cudaGraphicsResource *cudaSTriaVboResource;
        GLuint sTriaVbo;

        // GPU data for probes
        float* m_dProbePos;
        float* m_dSortedProbePos;
        uint*  m_dProbeNeighborCount;
        uint*  m_dProbeNeighbors;
        uint*  m_dGridProbeHash;
        uint*  m_dGridProbeIndex;
        struct cudaGraphicsResource *cudaSTriaResource;
        GLuint singPbo;
        GLuint singCoordsPbo;

        GLuint visibilityPbo;

};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MEGAMOL_MOLRENCUDASES_H_INCLUDED */
