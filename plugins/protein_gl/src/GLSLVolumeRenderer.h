/*
 * GLSLVolumeRenderer.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <list>
#include <memory>

#include <glm/glm.hpp>
#include <glowl/glowl.h>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/RenderUtils.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ProteinColor.h"
#include "slicing.h"
#include "vislib_gl/graphics/gl/FramebufferObject.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include "vislib_gl/graphics/gl/SimpleFont.h"

#define CHECK_FOR_OGL_ERROR()                                                                 \
    do {                                                                                      \
        GLenum err;                                                                           \
        err = glGetError();                                                                   \
        if (err != GL_NO_ERROR) {                                                             \
            fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); \
        }                                                                                     \
    } while (0)

#define NUM 10000

namespace megamol::protein_gl {

/**
 * Protein Renderer class
 */
class GLSLVolumeRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "GLSLVolumeRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers volume renderings.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    GLSLVolumeRenderer();

    /** Dtor. */
    ~GLSLVolumeRenderer() override;

    /**********************************************************************
     * 'set'-functions
     **********************************************************************/

    /** Set current coloring mode */
    inline void SetColoringMode(protein_calls::ProteinColor::ColoringMode cm) {
        currentColoringMode = cm;
    };

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'release'.
     */
    void release() override;

private:
    /**********************************************************************
     * 'render'-functions
     **********************************************************************/

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

    /**
     * Volume rendering using molecular data.
     */
    bool RenderMolecularData(mmstd_gl::CallRender3DGL* call, megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Refresh all parameters.
     */
    void ParameterRefresh(mmstd_gl::CallRender3DGL* call);

    /**
     * Create a volume containing all molecule atoms.
     *
     * @param mol The data interface.
     */
    void UpdateVolumeTexture(megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Draw the volume.
     *
     * @param boundingbox The bounding box.
     */
    void RenderVolume(vislib::math::Cuboid<float> boundingbox);

    /**
     * Write the parameters of the ray to the textures.
     *
     * @param boundingbox The bounding box.
     */
    void RayParamTextures(vislib::math::Cuboid<float> boundingbox);

    /**
     * Draw the bounding box of the protein.
     *
     * @paramboundingbox The bounding box.
     */
    void DrawBoundingBoxTranslated(vislib::math::Cuboid<float> boundingbox);

    /**
     * Draw the bounding box of the protein around the origin.
     *
     * @param boundingbox The bounding box.
     */
    void DrawBoundingBox(vislib::math::Cuboid<float> boundingbox);

    /**
     * Draw the clipped polygon for correct clip plane rendering.
     *
     * @param boundingbox The bounding box.
     */
    void drawClippedPolygon(vislib::math::Cuboid<float> boundingbox);

    /**
     * Write the current volume as a raw file.
     */
    void writeVolumeRAW();

    /**********************************************************************
     * variables
     **********************************************************************/

    /** caller slot */
    megamol::core::CallerSlot protDataCallerSlot;
    /** caller slot */
    megamol::core::CallerSlot protRendererCallerSlot;

    // camera information
    core::view::Camera cameraInfo;
    // scaling factor for the scene
    float scale;
    // translation of the scene
    vislib::math::Vector<float, 3> translation;

    megamol::core::param::ParamSlot coloringModeParam;
    // parameters for the volume rendering
    megamol::core::param::ParamSlot volIsoValueParam;
    megamol::core::param::ParamSlot volFilterRadiusParam;
    megamol::core::param::ParamSlot volDensityScaleParam;
    megamol::core::param::ParamSlot volIsoOpacityParam;
    megamol::core::param::ParamSlot volClipPlaneFlagParam;
    megamol::core::param::ParamSlot volClipPlane0NormParam;
    megamol::core::param::ParamSlot volClipPlane0DistParam;
    megamol::core::param::ParamSlot volClipPlaneOpacityParam;

    /** parameter slot for color table filename */
    megamol::core::param::ParamSlot colorTableFileParam;
    /** parameter slot for min color of gradient color mode */
    megamol::core::param::ParamSlot minGradColorParam;
    /** parameter slot for mid color of gradient color mode */
    megamol::core::param::ParamSlot midGradColorParam;
    /** parameter slot for max color of gradient color mode */
    megamol::core::param::ParamSlot maxGradColorParam;

    // param slot for toggling volume rendering
    megamol::core::param::ParamSlot renderVolumeParam;
    // param slot for toggling protein rendering
    megamol::core::param::ParamSlot renderProteinParam;

    // shader for volume texture generation
    std::unique_ptr<glowl::GLSLProgram> updateVolumeShader;
    // shader for volume rendering
    std::unique_ptr<glowl::GLSLProgram> volumeShader;
    std::unique_ptr<glowl::GLSLProgram> volRayStartShader;
    std::unique_ptr<glowl::GLSLProgram> volRayStartEyeShader;
    std::unique_ptr<glowl::GLSLProgram> volRayLengthShader;
    std::unique_ptr<glowl::GLSLProgram> colorWriterShader;

    // current coloring mode
    protein_calls::ProteinColor::ColoringMode currentColoringMode;

    // attribute locations for GLSL-Shader
    GLint attribLocInParams;
    GLint attribLocQuatC;
    GLint attribLocColor1;
    GLint attribLocColor2;

    // color table for amino acids
    std::vector<glm::vec3> aminoAcidColorTable;
    /** The color lookup table (for chains, amino acids,...) */
    std::vector<glm::vec3> colorLookupTable;
    /** The color lookup table which stores the rainbow colors */
    std::vector<glm::vec3> rainbowColors;
    /** color table for protein atoms */
    std::vector<glm::vec3> atomColorTable;
    std::vector<glm::vec3> fileColorTable;

    // the Id of the current frame (for dynamic data)
    unsigned int currentFrameId;

    // the number of protein atoms
    unsigned int atomCount;

    // FBO for rendering the protein
    std::shared_ptr<glowl::FramebufferObject> proteinFBO;

    // volume texture
    GLuint volumeTex;
    unsigned int volumeSize;
    float texClearArray;
    // FBO for volume generation
    GLuint volFBO;
    // volume parameters
    float volFilterRadius;
    float volDensityScale;
    float volScale[3];
    float volScaleInv[3];
    // width and height of view
    unsigned int width, height;
    // current width and height of textures used for ray casting
    unsigned int volRayTexWidth, volRayTexHeight;
    // volume ray casting textures
    GLuint volRayStartTex;
    GLuint volRayLengthTex;
    GLuint volRayDistTex;

    // render the volume as isosurface
    bool renderIsometric;
    // the first iso value
    float isoValue;
    // the opacity of the isosurface
    float volIsoOpacity;

    vislib::math::Vector<float, 3> protrenTranslate;
    float protrenScale;

    // flag wether clipping planes are enabled
    bool volClipPlaneFlag;
    // the array of clipping planes
    vislib::Array<vislib::math::Vector<double, 4>> volClipPlane;
    // view aligned slicing
    ViewSlicing slices;
    // the opacity of the clipping plane
    float volClipPlaneOpacity;

    bool forceUpdateVolumeTexture;

    float callTime;

    float* p;
    float* c;

    // Render utility class used for drawing texture to framebuffer
    core_gl::utility::RenderUtils renderUtils;
};


} // namespace megamol::protein_gl
