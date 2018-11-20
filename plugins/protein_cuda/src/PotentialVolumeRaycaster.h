//
// PotentialVolumeRaycaster.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 16, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERAYCASTER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERAYCASTER_H_INCLUDED
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
#include "slicing.h"
#include <GL/glu.h>
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/sys/Log.h"
//#include "vislib_vector_typedefs.h"
typedef vislib::math::Vector<int, 2> Vec2i;
typedef vislib::math::Vector<double, 3> Vec3d;
typedef vislib::math::Cuboid<float> Cubef;
#include "gridParams.h"


namespace megamol {
namespace protein_cuda {

class PotentialVolumeRaycaster : public core::view::Renderer3DModuleDS {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "PotentialVolumeRaycaster";
    }

    /** Ctor. */
    PotentialVolumeRaycaster(void);

    /** Dtor. */
    virtual ~PotentialVolumeRaycaster(void);

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Offers volume rendering textured by potential map.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        if(!vislib::graphics::gl::GLSLShader::AreExtensionsAvailable())
            return false;
        return true;
    }

protected:

    /**
     * (Re-)computes the density maps using CUDA.
     *
     * @param mol The data call containing the particles
     * @return 'True' on success, 'false' otherwise
     */
    bool computeDensityMap(const megamol::protein_calls::MolecularDataCall *mol);

    /**
     * Implementation of 'create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Create fbo to hold depth values for ray casting.
     *
     * @param width The width of the fbo
     * @param height The height of the fbo
     * @return 'True', if the fbo could be created
     */
    bool createFbos(UINT width, UINT height);

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
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * Initializes the textures containing the potential maps
     *
     * @param cmd The data call with the potential map
     * @return 'True' on success, 'false' otherwise
     */
	bool initPotential(protein_calls::VTIDataCall *cmd);

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

    /**
     * Render cube for volume rendering.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool RenderVolCube();

    /**
     * Update all parameters if necessary.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool updateParams();

private:


    /* Caller slots */

    /// Data caller slot for the potential map
    core::CallerSlot potentialDataCallerSlot;

    /// Data caller slot for particles
    core::CallerSlot particleDataCallerSlot;

    /// Caller slot for additional render module
    core::CallerSlot rendererCallerSlot;


    /* Parameters for the GPU volume rendering */

    /// Parameter for minimum potential value for the color map
    core::param::ParamSlot colorMinPotentialSclSlot;
    float colorMinPotentialScl;

    /// Parameter for maximum potential value for the color map
    core::param::ParamSlot colorMaxPotentialSclSlot;
    float colorMaxPotentialScl;

    /// Parameter for the color of the minimum potential value
    core::param::ParamSlot colorMinPotentialSlot;
    Vec3f colorMinPotential;

    /// Parameter for the color of zero potential
    core::param::ParamSlot colorZeroPotentialSlot;
    Vec3f colorZeroPotential;

    /// Parameter for the color of the maximum potential value
    core::param::ParamSlot colorMaxPotentialSlot;
    Vec3f colorMaxPotential;

    /// Parameter for iso value for volume rendering
    core::param::ParamSlot volIsoValSlot;
    float volIsoVal;

    /// Parameters for delta for volume rendering
    core::param::ParamSlot volDeltaSlot;
    float volDelta;

    /// Parameter for alpha scale factor for the isosurface
    core::param::ParamSlot volAlphaSclSlot;
    float volAlphaScl;

    /// Parameter for the distance of the volume clipping plane
    core::param::ParamSlot volClipZSlot;
    float volClipZ;

    /// Parameter for the maximum number of iterations when raycasting
    core::param::ParamSlot volMaxItSlot;
    float volMaxIt;

    /// Gradient offset
    core::param::ParamSlot gradOffsSlot;
    float gradOffs;


    /* Parameters for the 'quicksurf' class */

    /// Parameter for assumed radius of density grid data
    core::param::ParamSlot qsParticleRadSlot;
    float qsParticleRad;

    /// Parameter for the cutoff radius for the gaussian kernel
    core::param::ParamSlot qsGaussLimSlot;
    float qsGaussLim;

    /// Parameter for assumed radius of density grid data
    core::param::ParamSlot qsGridSpacingSlot;
    float qsGridSpacing;

    /// Parameter to toggle scaling by van der Waals radius
    core::param::ParamSlot qsSclVanDerWaalsSlot;
    bool qsSclVanDerWaals;


    /* Parameters for slice rendering */

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

    /// Parameter for minimum potential value for the color map
    core::param::ParamSlot sliceColorMinPotentialSclSlot;
    float sliceColorMinPotentialScl;

    /// Parameter for mid potential value for the color map
    core::param::ParamSlot sliceColorMidPotentialSclSlot;
    float sliceColorMidPotentialScl;

    /// Parameter for maximum potential value for the color map
    core::param::ParamSlot sliceColorMaxPotentialSclSlot;
    float sliceColorMaxPotentialScl;

    /// Parameter for the color of the minimum potential value
    core::param::ParamSlot sliceColorMinPotentialSlot;
    Vec3f sliceColorMinPotential;

    /// Parameter for the color of zero potential
    core::param::ParamSlot sliceColorZeroPotentialSlot;
    Vec3f sliceColorZeroPotential;

    /// Parameter for the color of the maximum potential value
    core::param::ParamSlot sliceColorMaxPotentialSlot;
    Vec3f sliceColorMaxPotential;

    /// Param for minimum potential on texture slice
    core::param::ParamSlot sliceMinValSlot;
    float sliceMinVal;

    /// Param for maximum potential on texture slice
    core::param::ParamSlot sliceMaxValSlot;
    float sliceMaxVal;


    /* Raycasting */

    /// Frame buffer object for raycasting
    vislib::graphics::gl::FramebufferObject rcFbo;

    /// Frame buffer object for opaque objects of the scene
    vislib::graphics::gl::FramebufferObject srcFbo;

    /// Shader for raycasting
    vislib::graphics::gl::GLSLShader rcShader;

    /// Shader for rendering the cube backface
    vislib::graphics::gl::GLSLShader rcShaderRay;

    /// Current resolution of the fbos
    Vec2i fboDim;

    /// Camera information
    vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

    /// The texture holding the scalar data
    GLuint volumeTex;

    /// The texture holding the potential map
    GLuint potentialTex;

    /// The density map
    float *volume;

    /// View slicing object to draw view aligned plane
    ViewSlicing viewSlicing;


    /* Volume generation */

    /// Pointer to CUDAQuickSurf object
    void *cudaqsurf;


    /* The data */

    /// The union of all the data sets' bounding boxes
    megamol::core::BoundingBoxes bbox;

    /// The bounding box of the potential map
    megamol::core::BoundingBoxes bboxPotential;

    /// The bounding box of the particles
    megamol::core::BoundingBoxes bboxParticles;

    /// The data hash for the particle data
    SIZE_T datahashParticles;

    /// The data hash for the potential map
    SIZE_T datahashPotential;

    /// The grid origin of the density map
    Vec3f gridXAxis, gridYAxis, gridZAxis;

    gridParams gridPotentialMap; ///> The bounding boxes for potential maps
    gridParams gridDensMap;      ///> The bounding boxes of the density maps


    /* Boolean flags */

    /// Triggers recomputation of the volume texture
    bool computeVolume;

    /// Triggers initialization of the potential texture
    bool initPotentialTex;

    // Last frame id
    int frameOld;


    /* Slice rendering */

    /// Shader for slice rendering
    vislib::graphics::gl::GLSLShader sliceShader;


};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERAYCASTER_H_INCLUDED
