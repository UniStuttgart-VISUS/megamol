//
// ComparativeMolSurfaceRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#if (defined(WITH_CUDA) && (WITH_CUDA))

#ifndef MMPROTEINPLUGIN_COMPARATIVEMOLSURFACERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_COMPARATIVEMOLSURFACERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer3DModuleDS.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "view/CallRender3D.h"
#include "vislib_vector_typedefs.h"
#include "MolecularDataCall.h"
#include "CUDAQuickSurf.h"
#include "CudaDevArr.h"
#include "HostArr.h"
#include "gridParams.h"
#include "VTIDataCall.h"
#include "DeformableGPUSurfaceMT.h"
#include "vislib/GLSLShader.h"
#include "HostArr.h"

namespace megamol {
namespace protein {

class ComparativeMolSurfaceRenderer : public core::view::Renderer3DModuleDS {

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

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "ComparativeMolSurfaceRenderer";
    }

    /** Ctor. */
    ComparativeMolSurfaceRenderer(void);

    /** Dtor. */
    virtual ~ComparativeMolSurfaceRenderer(void);

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
     * @param surf  The surface object
     * @return 'True' on success, 'false' otherwise
     */
    bool applyRMSFitting(MolecularDataCall *mol, DeformableGPUSurfaceMT *surf);

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
    bool computeDensityMap(
            const MolecularDataCall *mol,
            CUDAQuickSurf *cqs,
            gridParams &gridDensMap,
            const Cubef &bboxParticles);

    /**
     * Implementation of 'create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

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
    bool fitMoleculeRMS(MolecularDataCall *mol0, MolecularDataCall *mol1);

    /**
     * The get capabilities callback. The module should set the members
     * of 'call' to tell the caller its capabilities.
     *
     * @param  call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetCapabilities(core::Call& call);

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
     * Initializes a texture associated with potential map
     *
     * @param cmd              The data call containing the potential map
     * @param gridPotentialMap Grid parameters of the potential map
     * @param potentialTex     The texture handle for the potential map
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool initPotentialMap(VTIDataCall *cmd, gridParams &gridPotentialMap,
                    GLuint &potentialTex);

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

    /**
     * Renders the mapped isosurface using different rendering modes and surface
     * colorings.
     *
     * @param vboOld             Vertex buffer object containing all
     *                           unmapped vertex data
     * @param vboNew             Vertex buffer object containing all
     *                           mapped vertex data
     * @param vertexCnt          The number of vertices on the isosurface
     * @param vboTriangleIdx     Vertex buffer object containing triangle
     *                           indices
     * @param renderMode         The surface render mode
     * @param colorMode          The surface coloring mode
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderMappedSurface(
            GLuint vboOld, GLuint vboNew,
            uint vertexCnt,
            GLuint vboTriangleIdx,
            uint triangleVertexCnt,
            SurfaceRenderMode renderMode,
            SurfaceColorMode colorMode);

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
            GLuint vbo,
            uint vertexCnt,
            GLuint vboTriangleIdx,
            uint triangleVertexCnt,
            SurfaceRenderMode renderMode,
            SurfaceColorMode colorMode,
            GLuint potentialTex,
            Vec3f uniformColor,
            float alphaScl);

    /**
     * Render external forces as arrow glyphes (ray casted).
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderExternalForces();

    /**
     * Update all parameters if necessary.
     */
    void updateParams();

private:

    /* Data caller/callee slots */

    /// Callee slot to output the initial external forces (representing the
    /// target shape)
    core::CalleeSlot volOutputSlot;

    /// Caller slot for input molecule #1
    core::CallerSlot molDataSlot1;

    /// Caller slot for input molecule #2
    core::CallerSlot molDataSlot2;

    /// Data caller slot for surface attributes of both data sets
    core::CallerSlot volDataSlot1, volDataSlot2;

    /// Caller slot for additional render module
    core::CallerSlot rendererCallerSlot;


    /* Parameters for frame-by-frame comparison */

    /// Param slot for compare mode
    core::param::ParamSlot cmpModeSlot;
    CompareMode cmpMode;

    /// Param for single frame #1
    core::param::ParamSlot singleFrame1Slot;
    int singleFrame1;

    /// Param for single frame #2
    core::param::ParamSlot singleFrame2Slot;
    int singleFrame2;


    /* Global mapping options */

    /// Interpolation method used when computing external forces based on
    /// gradient of the scalar field
    core::param::ParamSlot interpolModeSlot;
    DeformableGPUSurfaceMT::InterpolationMode interpolMode;


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

    /// Stiffness of the springs defining the spring forces
    core::param::ParamSlot surfMappedSpringStiffnessSlot;
    float surfMappedSpringStiffness;

    /// GVF scale factor
    core::param::ParamSlot surfMappedGVFSclSlot;
    float surfMappedGVFScl;

    /// GVF iterations
    core::param::ParamSlot surfMappedGVFItSlot;
    unsigned int surfMappedGVFIt;


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


    /* Surface rendering options for surface #1 and #2 */

    /// Parameter for rendering mode of surface #1
    core::param::ParamSlot surface1RMSlot;
    SurfaceRenderMode surface1RM;

    /// Parameter for coloring mode of surface #1
    core::param::ParamSlot surface1ColorModeSlot;
    SurfaceColorMode surface1ColorMode;

    /// Transparency factor for surface #1
    core::param::ParamSlot surf1AlphaSclSlot;
    float surf1AlphaScl;

    /// Parameter for rendering mode of surface #2
    core::param::ParamSlot surface2RMSlot;
    SurfaceRenderMode surface2RM;

    /// Parameter for coloring mode of surface #2
    core::param::ParamSlot surface2ColorModeSlot;
    SurfaceColorMode surface2ColorMode;

    /// Transparency factor for surface #2
    core::param::ParamSlot surf2AlphaSclSlot;
    float surf2AlphaScl;


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


    /* Volume generation */

    /// Pointer to CUDAQuickSurf objects
    void *cudaqsurf1, *cudaqsurf2;

    /// Data array for intermediate calculations
    HostArr<float> gridDataPos;

    /// Minimum atom radius
    float minAtomRad;

    /// Maximum atom radius
    float maxAtomRad;


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

    /// The uniform color for surface #1
    static const Vec3f uniformColorSurf1;

    /// The uniform color for surface #2
    static const Vec3f uniformColorSurf2;

    /// The uniform color for the mapped surface
    static const Vec3f uniformColorSurfMapped;

    /// Global color for maximum potential
    static const Vec3f colorMaxPotential;

    /// Global color for minimum potential
    static const Vec3f colorMinPotential;


    /* Bounding boxes and hash values of the data sets */

    /// The union of all the data sets' bounding boxes
    core::BoundingBoxes bbox;

    /// The bounding boxes for particle data
    core::BoundingBoxes bboxParticles1, bboxParticles2;

    /// The bounding boxes for potential maps
    gridParams gridPotential1, gridPotential2;

    /// The bounding boxes of the density maps
    gridParams gridDensMap1, gridDensMap2;

    /// The data hash for the particle data
    SIZE_T datahashParticles1, datahashParticles2;

    /// The data hash for the potential map
    SIZE_T datahashPotential1, datahashPotential2;

    /// Calltime of the last frame
    float calltimeOld;


    /* Surface morphing */

    /// The deformable mesh representing the first molecular surface
    DeformableGPUSurfaceMT deformSurf1;

    /// The deformable mesh representing the second molecular surface
    DeformableGPUSurfaceMT deformSurf2;

    /// The deformable mesh representing the second molecular surface
    DeformableGPUSurfaceMT deformSurfMapped;


    /* Surface rendering */

    /// Camera information
    vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

    /// Shader implementing per pixel lighting
    vislib::graphics::gl::GLSLShader pplSurfaceShader;

    /// Shader implementing per pixel lighting
    vislib::graphics::gl::GLSLShader pplMappedSurfaceShader;

    /// The textures holding surface attributes (e.g. surface potential)
    GLuint surfAttribTex1, surfAttribTex2;


    /* RMSD fitting */

    HostArr<float> rmsPosVec0;  ///> Position vector #0 for rms fitting
    HostArr<float> rmsPosVec1;  ///> Position vector #1 for rms fitting
    HostArr<float> rmsWeights;  ///> Particle weights
    HostArr<int> rmsMask;       ///> Mask for particles
    float rmsValue;             ///> The calculated RMS value
    Mat3f rmsRotation;          ///> Rotation matrix for the fitting
    Vec3f rmsTranslation;       ///> Translation vector for the fitting
    static const float maxRMSVal;  ///> Maximum RMS value to enable fitting


    /* Boolean flags */

    /// Triggers recomputation of the volume texture
    bool triggerComputeVolume;

    /// Triggers initialization of the potential texture
    bool triggerInitPotentialTex;

    /// Triggers recomputation of the vertices of
    bool triggerComputeSurfacePoints1, triggerComputeSurfacePoints2;

    /// Triggers recomputation of the surface mapping
    bool triggerSurfaceMapping;

    /// Triggers recomputation of minimum RMSD
    bool triggerRMSFit;

    /// DEBUG
    vislib::Array<float> lines;
    bool triggerComputeLines;
    HostArr<float> gvf;

};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_COMPARATIVEMOLSURFACERENDERER_H_INCLUDED
#endif // (defined(WITH_CUDA) && (WITH_CUDA))
