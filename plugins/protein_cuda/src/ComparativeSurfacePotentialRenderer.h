//
// ComparativeSurfacePotentialRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 13, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERENDERERCUDA_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERENDERERCUDA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/view/CallRender3D.h"
#include "protein_calls/VTIDataCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "CUDAMarchingCubes.h"
#include "CUDAQuickSurf.h"
#include "gridParams.h"
#include <GL/glu.h>
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Cuboid.h"

//#include "vislib_vector_typedefs.h"
typedef vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> Mat3f;
typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> Mat4f;
typedef vislib::math::Vector<float, 3> Vec3f;

#include "CudaDevArr.h"
#include "HostArr.h"

typedef unsigned int uint;

// Use distance field in addition to Gaussian volume - this might be
// necessary in some cases to overcome local minima inside the Gaussian volume
#define USE_DISTANCE_FIELD

// Use texture slices for debugging purposes (density field, potential texture)
#define USE_TEXTURE_SLICES

namespace megamol {
namespace protein_cuda {

class ComparativeSurfacePotentialRenderer : public core::view::Renderer3DModuleDS {

public:

    /// Render modes for the surfaces
    enum SurfaceRenderMode {SURFACE_NONE=0, SURFACE_POINTS, SURFACE_WIREFRAME,
        SURFACE_FILL};

    /// Color modes for the surfaces
    enum SurfaceColorMode {
        SURFACE_UNI=0,            // #0
        SURFACE_NORMAL,           // #1
        SURFACE_TEXCOORDS,        // #2
        SURFACE_POTENTIAL,        // #3
        SURFACE_DIST_TO_OLD_POS,  // #4
        SURFACE_POTENTIAL0,       // #5
        SURFACE_POTENTIAL1,       // #6
        SURFACE_POTENTIAL_DIFF,   // #7
        SURFACE_POTENTIAL_SIGN};  // #8

    /// Enum describing different ways of using RMS fitting
    enum RMSFittingMode {RMS_NONE=0, RMS_ALL, RMS_BACKBONE, RMS_C_ALPHA};

    /// Different modes for frame-by-frame comparison
    enum CompareMode {
        COMPARE_1_1=0,
        COMPARE_1_N,
        COMPARE_N_1,
        COMPARE_N_N
    };

    // Interpolation mode used when computing external forces based on gradient
    enum InterpolationMode {INTERP_LINEAR=0, INTERP_CUBIC};

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "ComparativeSurfacePotentialRenderer";
    }

    /** Ctor. */
    ComparativeSurfacePotentialRenderer(void);

    /** Dtor. */
    virtual ~ComparativeSurfacePotentialRenderer(void);

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Offers comparative rendering of two molecular surfaces.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        if (!vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()) {
            return false;
        }
        return true;
    }

protected:

    /**
     * Translate and rotate an array of positions according to the current
     * transformation obtained by RMS fitting (a translation vector and
     * rotation matrix).
     *
     * @param mol                A Molecular data call containing the particle
     *                           positions of the corresponding data set. This
     *                           is necessary to compute the centroid of the
     *                           particles.
     * @param cudaTokenVboMapped The vbo containing the vertex positions to be
     *                           transformed (mapped device memory)
     * @param vertexCnt          The number of vertices to be transformed
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool applyRMSFittingToPosArray(
            protein_calls::MolecularDataCall *mol,
            cudaGraphicsResource **cudaTokenVboMapped,
            uint vertexCnt);

    /**
     * (Re-)computes a smooth density map based on an array of givwen particle
     * positions using a CUDA implementation.
     *
     * @param mol           The data call containing the particle positions
     * @param cqs           The CUDAQuickSurf object used to compute the density
     *                      map
     * @param gridDensMap   Grid parameters for the resulting density map
     * @param bboxParticles The bounding box of the particle array
     * @param volume        The array holding the actual density map (only if
     *                      USE_TEXTURE_SLICES is defined)
     * @param volumeTex     Texture object associated with the density map (only
     *                      if USE_TEXTURE_SLICES is defined)
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool computeDensityMap(const protein_calls::MolecularDataCall *mol, CUDAQuickSurf *cqs,
                    gridParams &gridDensMap,
                    const vislib::math::Cuboid<float> &bboxParticles
#if defined(USE_TEXTURE_SLICES)
                    ,HostArr<float> &volume, GLuint &volumeTex
#endif // defined(USE_TEXTURE_SLICES)
                    );

#if defined(USE_DISTANCE_FIELD)
    /**
     * Computes a distance field based on a given set of vertices. For every
     * lattice point, the distance to the nearest vertex is stored.
     *
     * @param mol           The data call containing the particle positions
     * @param vertexPod_D   Array with the vertex positions (device memory)
     * @param distField_D   Array containing the distance field (device memory)
     * @param volume_D      The Gaussian volume
     * @param gridDistField Grid parameters of the distance field lattice
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool computeDistField(const protein_calls::MolecularDataCall *mol,
            cudaGraphicsResource **vboResource,
            uint vertexCnt,
            CudaDevArr<float> &distField_D,
            float *volume_D,
            gridParams &gridDistField);
#endif

    /**
     * Implementation of 'create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Creates a vertex buffer object of size s
     *
     * @param vbo    The vertex buffer object
     * @param size   The size of the vertex buffer object
     * @param target The target enum, can either be GL_ARRAY_BUFFER or
     *               GL_ELEMENT_ARRAY_BUFFER
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool createVbo(GLuint* vbo, size_t s, GLuint target);

    /**
     * Destroys the vertex buffer object 'vbo'
     *
     * @param vbo    The vertex buffer object
     * @param target The target enum, can either be GL_ARRAY_BUFFER or
     *               GL_ELEMENT_ARRAY_BUFFER
     */
    void destroyVbo(GLuint* vbo, GLuint target);

    /**
     * Computes the translation vector and the rotation matrix to minimize the
     * RMS (Root Mean Square) of the particles contained in data set 0 and data
     * set 1. Data set 1 is fitted to data set 0.
     *
     * @param mol0 The data call containing the particles of data set 0
     * @param mol1 The data call containing the particles of data set 1
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool fitMoleculeRMS(protein_calls::MolecularDataCall *mol0, protein_calls::MolecularDataCall *mol1);

    /**
     * Frees all dynamically allocated memory (host and device) and sets all
     * pointers to NULL.
     */
    void freeBuffers();

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param  call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * The get extent callback for vbo data.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool getVBOExtent(core::Call& call);

    /**
     * The get data callback for vbo data of data set #0
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool getVBOData0(core::Call& call);

    /**
     * The get data callback for vbo data of data set #1
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool getVBOData1(core::Call& call);

    /**
     * Initializes a texture associated with a given potential map
     *
     * @param cmd              The data call containing the potential map
     * @param gridPotentialMap Grid parameters of the potential map
     * @param potentialTex     The texture handle for the potential map
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool initPotentialMap(protein_calls::VTIDataCall *cmd, gridParams &gridPotentialMap,
                    GLuint &potentialTex);

    /**
     * Connectivity information of the isosurface vertices is used to compute
     * vertex normals.
     *
     * @param volume_D           The volume data (device memory)
     * @param gridDensMap        Grid parameters for the volume
     * @param vboResource        Vertex data buffer containingf positions and
     *                           normals for all vertices
     * @param vertexMap_D        Array for the vertex map (device memory)
     * @param vertexMapInv_D     Array for the inverse vertex map (device memory)
     * @param cubeMap_D          Array for the cube map (device memory)
     * @param cubeMapInv_D       Array for the inverse cube map (device memory)
     * @param vertexCnt          The number of vertex
     * @param arrDataOffsPos     Data offset for vertex positions
     * @param arrDataOffsNormals Data offset for vertex normals
     * @param arrDataSize        Vertex data stride
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool isosurfComputeNormals(float *volume_D,
            gridParams gridDensMap,
            cudaGraphicsResource **vboResource,
            CudaDevArr<uint> &vertexMap_D,
            CudaDevArr<uint> &vertexMapInv_D,
            CudaDevArr<uint> &cubeMap_D,
            CudaDevArr<uint> &cubeMapInv_D,
            uint vertexCnt,
            uint arrDataOffsPos,
            uint arrDataOffsNormals,
            uint arrDataSize);

    /**
     * Computed texture coordinates for an array of vertices based on the grid
     * information of the texture.
     *
     * @param vboResource     Vertex data buffer containing positions and
     *                        texture coordinates
     * @param vertexCnt       The number of active vertices
     * @param minC            The minimum coordinates of the bounding box
     * @param maxC            The maximum coordinates of the bounding box
     * @param arrDataOffsPos  Data offset for vertex positions
     * @param arrDataOffsTexCoords Data offset for vertex positions
     * @param arrDataSize     Vertex data stride
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool isosurfComputeTexCoords(
            cudaGraphicsResource **vboResource,
            uint vertexCnt,
            float3 minC,
            float3 maxC,
            uint arrDataOffsPos,
            uint arrDataOffsTexCoords,
            uint arrDataSize);

    /**
     * Extracts an isosurface from a given volume texture using Marching
     * Tetrahedra with the Freudenthal subdivision scheme.
     *
     * @param volume_D            The volume data (device memory)
     * @param cubeMap_D           Array for the cube map (device memory)
     * @param cubeMapInv_D        Array for the inverse cube map (device memory)
     * @param vertexMap_D         Array for the vertex map (device memory)
     * @param vertexMapInv_D      Array for the inverse vertex map (device
     *                            memory)
     * @param vertexNeighbours_D  Array containing neighbours of all vertices
     * @param gridDensMap         Grid parameters for the volume
     * @param vertexCount         The number of vertices
     * @param vbo                 Buffer for vertex data
     * @param vboResource         CUDA token for mapped memory
     * @param triangleCount       The number of triangles
     * @param vboTriangleIdx      Buffer for triangle indices
     * @param vboTriangleIdxResource CUDA token for mapped memory
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool isosurfComputeVertices(
            float *volume_D,
            CudaDevArr<uint> &cubeMap_D,
            CudaDevArr<uint> &cubeMapInv_D,
            CudaDevArr<uint> &vertexMap_D,
            CudaDevArr<uint> &vertexMapInv_D,
            CudaDevArr<int> &vertexNeighbours_D,
            gridParams gridDensMap,
            uint &vertexCount,
            GLuint &vbo,
            cudaGraphicsResource **vboResource,
            uint &triangleCount,
            GLuint &vboTriangleIdx,
            cudaGraphicsResource **vboTriangleIdxResource);

    /**
     * Maps an isosurface defined by an array of vertex positions with
     * connectivity information to a given volumetric isosurface (defined by
     * a texture and an isovalue). To achieve this a deformable model approach
     * is used which combines internal spring forces with an external force
     * obtained from the volume gradient.
     * The potential used for the external forces is a combination of a distance
     * field and a density map.
     *
     * @param volume_D                The volume the vertices are to be mapped
     *                                to (device memory)
     * @param gridDensMap             Grid parameters for the volume
     * @param vboCudaRes              CUDA token for mapped memory of vertex
     *                                data
     * @param vboCudaResTriangleIdx   CUDA token for mapped memory for triangle
     *                                indices
     * @param vertexCnt               The number of vertices
     * @param triangleCnt             The number of triangles
     * @param vertexNeighbours_D      Connectivity information of the vertices
     *                                (device memory)
     * @param maxIt                   The number of iterations for the mapping
     * @param springStiffness         The stiffness of the springs defining the
     *                                internal spring forces
     * @param forceScl                An overall scaling for the combined force
     * @param externalForcesWeight    The weighting of the external force. The
     *                                weight for the internal forces is
     *                                1.0 - externalForcesWeight
     * @param interpMode              Detemines whether linear or cubic
     *                                interpolation is to be used when computing
     *                                the external forces
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool mapIsosurfaceToVolume(
            float *volume_D,
            gridParams gridDensMap,
            cudaGraphicsResource **vboCudaRes,
            cudaGraphicsResource **vboCudaResTriangleIdx,
            uint vertexCnt,
            uint triangleCnt,
            CudaDevArr<int> &vertexNeighbours_D,
            uint maxIt,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            InterpolationMode interpMode);


    /**
     * Maps an isosurface defined by an array of vertex positions with
     * connectivity information to a given volumetric isosurface (defined by
     * a texture and an isovalue). To achieve this a deformable model approach
     * is used which combines internal spring forces with an external force
     * obtained from the volume gradient.
     *
     * @param volume_D                The volume the vertices are to be mapped
     *                                to (device memory)
     * @param gridDensMap             Grid parameters for the volume
     * @param vboCudaRes              CUDA token for mapped memory containing
     *                                vertex data
     * @param vertexCnt               The number of vertices
     * @param vertexNeighbours_D      Connectivity information of the vertices
     *                                (device memory)
     * @param maxIt                   The number of iterations for the mapping
     * @param springStiffness         The stiffness of the springs defining the
     *                                internal spring forces
     * @param forceScl                An overall scaling for the combined force
     * @param externalForcesWeight    The weighting of the external force. The
     *                                weight for the internal forces is
     *                                1.0 - externalForcesWeight
     * @param interpMode              Detemines whether linear or cubic
     *                                interpolation is to be used when computing
     *                                the external forces
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool regularizeSurface(float *volume_D,
            gridParams gridDensMap,
            cudaGraphicsResource **vboCudaRes,
            uint vertexCnt,
            CudaDevArr<int> &vertexNeighbours_D,
            uint maxIt,
            float springStiffness,
            float forceScl,
            float externalForcesWeight,
            InterpolationMode interpMode);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

#if defined(USE_TEXTURE_SLICES)
    /**
     * Render texture slices of different textures
     *
     * @param densityTex   Texture handle for the density texture
     * @param potentialTex Texture handle for the potential texture
     * @param potGrid      Grid parameters for potential grid
     * @param densGrid     Grid parameters for density grid
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool renderSlices(GLuint densityTex, GLuint potentialTex,
            gridParams potGrid, gridParams densGrid);
#endif // defined(USE_TEXTURE_SLICES)

    /**
     * Renders the isosurface using different rendering modes and surface
     * colorings.
     *
     * @param vbo                Vertex buffer object containing all vertex data
     * @param vertexCnt          The number of vertices on the isosurface
     * @param vboTriangleIdx     Vertex buffer object containing triangle
     *                           indices
     * @param renderMode         The surface render mode
     * @param colorMode          The surface coloring mode
     * @param potentialTex       Texture containing potential map
     * @param uniformColor       Color to be used as uniform coloring
     * @param alphaScl           Opacity scaling for the rendering
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderSurface(
            GLuint &vbo,
            uint vertexCnt,
            GLuint &vboTriangleIdx,
            uint triangleVertexCnt,
            SurfaceRenderMode renderMode,
            SurfaceColorMode colorMode,
            GLuint potentialTex,
            Vec3f uniformColor,
            float alphaScl);


    /**
     * Renders the mapped isosurface using different rendering modes and surface
     * colorings.
     *
     * @param vbo                Vertex buffer object containing all vertex data
     * @param vertexCnt          The number of vertices on the isosurface
     * @param vboTriangleIdx     Vertex buffer object containing triangle
     *                           indices
     * @param renderMode         The surface render mode
     * @param colorMode          The surface coloring mode
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderMappedSurface(
            GLuint &vbo,
            uint vertexCnt,
            GLuint &vboTriangleIdx,
            uint triangleVertexCnt,
            SurfaceRenderMode renderMode,
            SurfaceColorMode colorMode
            );


    /**
     * Sort triangles of one surface by diatnce to camera. This is necessary
     * for transparent rendering.
     *
     * @param vboResource            CUDA token for mapped memory containing
     *                               vertex data
     * @param vertexCnt              The number of vertices
     * @param vboTriangleIdxResource CUDA token for mapped memory containing
     *                               triangle indices
     * @param triangleCntn           The number of triangles
     * @param dataBuffSize           The stride in the vertex data buffer
     * @param dataBuffOffsPos        The offset for the positions in the vertex
     *                               data buffer
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool sortTriangles(cudaGraphicsResource **vboResource,
            uint vertexCount,
            cudaGraphicsResource **vboTriangleIdxResource,
            uint triangleCnt,
            uint dataBuffSize,
            uint dataBuffOffsPos);

    /**
     * Update all parameters if necessary.
     */
    void updateParams();

private:

    /* Callee slots for slave renderers */

    /// Callee slot for slave renderer #0
    core::CalleeSlot vboSlaveSlot0;

    /// Callee slot for slave renderer #1
    core::CalleeSlot vboSlaveSlot1;


    /* Caller slots */

    /// Data caller slot for potential maps of both data sets
    core::CallerSlot potentialDataCallerSlot0, potentialDataCallerSlot1;

    /// Data caller slot for particles of both data set
    core::CallerSlot particleDataCallerSlot0, particleDataCallerSlot1;

    /// Caller slot for additional render module
    core::CallerSlot rendererCallerSlot;


#if defined(USE_TEXTURE_SLICES)
    /* Parameters for slice rendering */

    /// Parameter for slice rendering mode
    core::param::ParamSlot sliceDataSetSlot;
    int sliceDataSet;

    /// Parameter for slice rendering mode
    core::param::ParamSlot sliceRMSlot;
    int sliceRM;

    /// Parameter for x-Plane
    core::param::ParamSlot xPlaneSlot;
    float xPlane;

    /// Parameter for y-Plane
    core::param::ParamSlot yPlaneSlot;
    float yPlane;

    /// Parameter for z-Plane
    core::param::ParamSlot zPlaneSlot;
    float zPlane;

    /// Parameter to toggle x-Plane
    core::param::ParamSlot toggleXPlaneSlot;
    bool showXPlane;

    /// Parameter to toggle y-Plane
    core::param::ParamSlot toggleYPlaneSlot;
    bool showYPlane;

    /// Parameter to toggle z-Plane
    core::param::ParamSlot toggleZPlaneSlot;
    bool showZPlane;

    /// Param for minimum potential on texture slice
    core::param::ParamSlot sliceMinValSlot;
    float sliceMinVal;

    /// Param for maximum potential on texture slice
    core::param::ParamSlot sliceMaxValSlot;
    float sliceMaxVal;
#endif //  defined(USE_TEXTURE_SLICES)


    /* Parameters for frame-by-frame comparison */

    /// Param slot for compare mode
    core::param::ParamSlot cmpModeSlot;
    CompareMode cmpMode;

    /// Param for single frame #0
    core::param::ParamSlot singleFrame0Slot;
    int singleFrame0;

    /// Param for single frame #1
    core::param::ParamSlot singleFrame1Slot;
    int singleFrame1;


    /* Global rendering options */

    /// Parameter for minimum potential value for the color map
    core::param::ParamSlot minPotentialSlot;
    float minPotential;

    /// Parameter for maximum potential value for the color map
    core::param::ParamSlot maxPotentialSlot;
    float maxPotential;


    /* Global mapping options */

    /// Interpolation method used when computing external forces based on
    /// gradient of the scalar field
    core::param::ParamSlot interpolModeSlot;
    InterpolationMode interpolMode;


    /* Parameters for the surface mapping */

    /// RMS fitting mode
    core::param::ParamSlot fittingModeSlot;
    RMSFittingMode fittingMode;

    /// Weighting for external forces when mapping the surface
    core::param::ParamSlot surfaceMappingExternalForcesWeightSclSlot;
    float surfaceMappingExternalForcesWeightScl;

    /// Overall scaling of the resulting forces
    core::param::ParamSlot surfaceMappingForcesSclSlot;
    float surfaceMappingForcesScl;

    /// Maximum number of iterations when mapping the surface
    core::param::ParamSlot surfaceMappingMaxItSlot;
    uint surfaceMappingMaxIt;

    /// Minimum displacement when mapping (this prevents unneeded computations
    /// for vertices that will not move anymore)
    core::param::ParamSlot surfMappedMinDisplSclSlot;
    float surfMappedMinDisplScl;

#if defined(USE_DISTANCE_FIELD)
    /// Maximum distance to use the density map instead of distance field
    core::param::ParamSlot surfMappedMaxDistSlot;
    float surfMappedMaxDist;
#endif

    /// Stiffness of the springs defining the spring forces
    core::param::ParamSlot surfMappedSpringStiffnessSlot;
    float surfMappedSpringStiffness;


    /* Parameters for the mapped surface rendering */

    /// Parameter for rendering mode of mapped surface
    core::param::ParamSlot surfaceMappedRMSlot;
    SurfaceRenderMode surfaceMappedRM;

    /// Parameter for coloring mode of mapped surface
    core::param::ParamSlot surfaceMappedColorModeSlot;
    SurfaceColorMode surfaceMappedColorMode;

    /// Maximum positional difference for the surface
    core::param::ParamSlot surfMaxPosDiffSlot;
    float surfMaxPosDiff;

    /// Transparency factor for the mapped surface
    core::param::ParamSlot surfMappedAlphaSclSlot;
    float surfMappedAlphaScl;


    /* Surface regularization */

    /// Maximum number of iterations when regularizing the mesh #0
    core::param::ParamSlot regMaxItSlot;
    uint regMaxIt;

    /// Stiffness of the springs defining the spring forces in surface #0
    core::param::ParamSlot regSpringStiffnessSlot;
    float regSpringStiffness;

    /// Weighting of the external forces in surface #0, note that the weight
    /// of the internal forces is implicitely defined by
    /// 1.0 - surf0ExternalForcesWeight
    core::param::ParamSlot regExternalForcesWeightSlot;
    float regExternalForcesWeight;

    /// Overall scaling for the forces acting upon surface #0
    core::param::ParamSlot regForcesSclSlot;
    float regForcesScl;


    /* Surface rendering options for surface #0 and #1 */

    /// Parameter for rendering mode of surface #0
    core::param::ParamSlot surface0RMSlot;
    SurfaceRenderMode surface0RM;

    /// Parameter for coloring mode of surface #0
    core::param::ParamSlot surface0ColorModeSlot;
    SurfaceColorMode surface0ColorMode;

    /// Transparency factor for surface #0
    core::param::ParamSlot surf0AlphaSclSlot;
    float surf0AlphaScl;

    /// Parameter for rendering mode of surface #1
    core::param::ParamSlot surface1RMSlot;
    SurfaceRenderMode surface1RM;

    /// Parameter for coloring mode of surface #1
    core::param::ParamSlot surface1ColorModeSlot;
    SurfaceColorMode surface1ColorMode;

    /// Transparency factor for surface #1
    core::param::ParamSlot surf1AlphaSclSlot;
    float surf1AlphaScl;


    /* Hardcoded parameters for the 'quicksurf' class */

    /// Parameter for assumed radius of density grid data
    static const float qsParticleRad;

    /// Parameter for the cutoff radius for the gaussian kernel
    static const float qsGaussLim;

    /// Parameter for assumed radius of density grid data
    static const float qsGridSpacing;

    /// Parameter to toggle scaling by van der Waals radius
    static const bool qsSclVanDerWaals;

    /// Parameter for iso value for volume rendering
    static const float qsIsoVal;


    /* Hardcoded colors for surface rendering */

    /// The uniform color for surface #0
    static const Vec3f uniformColorSurf0;

    /// The uniform color for surface #1
    static const Vec3f uniformColorSurf1;

    /// The uniform color for the mapped surface
    static const Vec3f uniformColorSurfMapped;

    /// Global color for maximum potential
    static const Vec3f colorMaxPotential;

    /// Global color for minimum potential
    static const Vec3f colorMinPotential;


    /* Rendering */

#if defined(USE_TEXTURE_SLICES)
    /// Shader for slice rendering
    vislib::graphics::gl::GLSLShader sliceShader;
#endif // defined(USE_TEXTURE_SLICES)

    /// Camera information
    vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

    /// The textures holding the potential maps
    GLuint potentialTex0, potentialTex1;


    /* Volume generation */

    void *cudaqsurf0, *cudaqsurf1;   ///> Pointer to CUDAQuickSurf objects
#if defined(USE_TEXTURE_SLICES)
    HostArr<float> volume0, volume1; ///> Arrays for volume data
    GLuint volumeTex0, volumeTex1;   ///> Volume textures
#endif // defined(USE_TEXTURE_SLICES)
    HostArr<float> gridDataPos;      ///> Data array for intermediate calculations
    float minAtomRad; ///> Minimum atom radius
    float maxAtomRad; ///> Maximum atom radius


    /* Bounding boxes and hash values of the data sets */

    core::BoundingBoxes bbox;                           ///> The union of all the data sets' bounding boxes
    core::BoundingBoxes bboxParticles0, bboxParticles1; ///> The bounding boxes for particle data
    gridParams gridPotential0, gridPotential1;          ///> The bounding boxes for potential maps
    gridParams gridDensMap0, gridDensMap1;              ///> The bounding boxes of the density maps

    SIZE_T datahashParticles0, datahashParticles1; ///> The data hash for the particle data
    SIZE_T datahashPotential0, datahashPotential1; ///> The data hash for the potential map

    float calltimeOld; ///> Calltime of the last frame


    /* Boolean flags */

    /// Triggers recomputation of the volume texture
    bool triggerComputeVolume;

    /// Triggers initialization of the potential texture
    bool triggerInitPotentialTex;

    /// Triggers recomputation of the vertices of
    bool triggerComputeSurfacePoints0, triggerComputeSurfacePoints1;

    /// Triggers recomputation of the surface mapping
    bool triggerSurfaceMapping;


    /* Surface triangulation */

    /// Array containing activity information for all grid cells (device memory)
    CudaDevArr<uint> cubeStates_D;

    /// Array containing activity information for all grid cells (device memory)
    CudaDevArr<uint> cubeOffsets_D;

    /// Mapping from list of active cells to generall cell list (and vice versa)
    CudaDevArr<uint> cubeMap0_D, cubeMapInv0_D;
    CudaDevArr<uint> cubeMap1_D, cubeMapInv1_D;

    /// Activity of vertices
    CudaDevArr<uint> vertexStates_D;

    /// Positions of active vertices
    CudaDevArr<float3> activeVertexPos_D;

    /// Index offsets for active vertices
    CudaDevArr<uint> vertexIdxOffs_D;

    /// Mapping from active vertex indices to general index list (and vice versa)
    CudaDevArr<uint> vertexMap0_D, vertexMapInv0_D;
    CudaDevArr<uint> vertexMap1_D, vertexMapInv1_D;

    /// Connectivity information for all vertices (at most 18 neighbours per vertex)
    CudaDevArr<int> vertexNeighbours0_D, vertexNeighbours1_D;

    /// Array containing number of vertices for each tetrahedron
    CudaDevArr<uint> verticesPerTetrahedron_D;

    /// Vertex index offsets for all tetrahedrons
    CudaDevArr<uint> tetrahedronVertexOffsets_D;


    /* Surface mapping */

    /// Device pointer to external forces for every vertex
    CudaDevArr<float> vertexExternalForcesScl_D;

    /// Device pointer to gradient field
    CudaDevArr<float4> volGradient_D;


    /* Triangle sorting */

    CudaDevArr<float> triangleCamDistance_D;


    /* Surface rendering */

    /// Shader implementing per pixel lighting
    vislib::graphics::gl::GLSLShader pplSurfaceShader;

    /// Shader implementing per pixel lighting
    vislib::graphics::gl::GLSLShader pplMappedSurfaceShader;


    /* RMSD fitting */

    HostArr<float> rmsPosVec0;  ///> Position vector #0 for rms fitting
    HostArr<float> rmsPosVec1;  ///> Position vector #1 for rms fitting
    HostArr<float> rmsWeights;  ///> Particle weights
    HostArr<int> rmsMask;       ///> Mask for particles
    float rmsValue;             ///> The calculated RMS value
    Mat3f rmsRotation;          ///> Rotation matrix for the fitting
    Vec3f rmsTranslation;       ///> Translation vector for the fitting
    bool toggleRMSFit;          ///> Toggles RMS fitting
    static const float maxRMSVal;  ///> Maximum RMS value to enable fitting

#if defined(USE_DISTANCE_FIELD)
    /* Distance field computation */

    CudaDevArr<float> distField_D;    ///> Array holding the distance field (device memory)

#if defined(USE_TEXTURE_SLICES)
    HostArr<float> distField;         ///> Array holding the distance field (host memory)
    GLuint distFieldTex;              ///> Texture handle for distance field
#endif // defined(USE_TEXTURE_SLICES)

    bool triggerComputeDistanceField; ///> Triggers computation of the distance field
#endif


    /* Vertex buffers and respecting CUDA mapped memory tokens */

    /// Vertex buffer object for surface #0.
    /// 3 float Position
    /// 3 float Normal
    /// 3 float Tex coords potential map
    GLuint vbo0;

    /// Cuda graphics ressource associated with vbo #0
    struct cudaGraphicsResource *vbo0Resource;

    /// Vertex buffer object for surface #1.
    /// 3 float Position
    /// 3 float Normal
    /// 3 float Tex coords potential map
    GLuint vbo1;

    /// Cuda graphics ressource associated with vbo #1
    struct cudaGraphicsResource *vbo1Resource;

    /// Vertex buffer object for mapped surface.
    /// 3 float New position
    /// 3 float Old position
    /// 3 float Tex coords potential map #0 new Position
    /// 3 float Tex coords potential map #1 old Position
    /// 3 float Tex coords density map #0 new position (= tex coords distance map)
    /// 1 float Marker for corrupt triangle vertices (1.0 = corrupt, 0.0 = normal)
    GLuint vboMapped;

    /// Cuda graphics ressource associated with mapped surface vbo
    struct cudaGraphicsResource *vboMappedResource;

    /// Vertex buffer object for triangle indices of surface #0
    GLuint vboTriangleIdx0;

    /// Cuda graphics ressource associated with the triangle index buffer #0
    struct cudaGraphicsResource *vboTriangleIdx0Resource;

    /// Vertex buffer object for triangle indices of surface #1
    GLuint vboTriangleIdx1;

    /// Cuda graphics ressource associated with the triangle index buffer #1
    struct cudaGraphicsResource *vboTriangleIdx1Resource;

    /// Vertex buffer object for triangle indices of mapped surface
    GLuint vboTriangleIdxMapped;

    /// Cuda graphics ressource associated with the triangle index buffer #0
    struct cudaGraphicsResource *vboTriangleIdxMappedResource;

    /// The number of vertices of surface #0
    uint vertexCnt0;

    /// The number of vertices of surface #1
    uint vertexCnt1;

    /// The number of triangles of surface #0
    uint triangleCnt0;

    /// The number of triangles of surface #1
    uint triangleCnt1;


    /* Constants defining the stride and offsets in the vertex buffer data */

    /// Vertex data buffer offset for positions
    static const uint vertexDataOffsPos;

    /// Vertex data buffer offset for normals
    static const uint vertexDataOffsNormal;

    /// Vertex data buffer offset for tex coords
    static const uint vertexDataOffsTexCoord;

    /// Vertex data buffer element size
    static const uint vertexDataStride;

    /// Vertex data buffer offset for new positions (mapped surface)
    static const uint vertexDataMappedOffsPosNew;

    /// Vertex data buffer offset for old positions (mapped surface)
    static const uint vertexDataMappedOffsPosOld;

    /// Vertex data buffer offset for normals (mapped surface)
    static const uint vertexDataMappedOffsNormal;

    /// Vertex data buffer offset for new tex coords (mapped surface)
    static const uint vertexDataMappedOffsTexCoordNew;

    /// Vertex data buffer offset for old tex coords (mapped surface)
    static const uint vertexDataMappedOffsTexCoordOld;

    /// Vertex data buffer offset for corrupt triangle flag (mapped surface)
    static const uint vertexDataMappedOffsCorruptTriangleFlag;

    /// Vertex data buffer element size (mapped surface)
    static const uint vertexDataMappedStride;

    /// Array for laplacian
    CudaDevArr<float3> laplacian_D;

    /// Array to safe displacement length
    CudaDevArr<float> displLen_D;

};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_ComparativeSurfacePotentialRenderer_H_INCLUDED
