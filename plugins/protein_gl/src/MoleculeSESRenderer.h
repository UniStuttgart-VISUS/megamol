/*
 * MoleculeSESRenderer.h
 *
 * Copyright (C) 2009-2021 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_MOLSESRENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_MOLSESRENDERER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "glowl/BufferObject.hpp"
#include "glowl/GLSLProgram.hpp"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Camera.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"
#include "protein/Color.h"
#include "protein/ReducedSurface.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib_gl/graphics/gl/GLSLComputeShader.h"
#include "vislib_gl/graphics/gl/GLSLGeometryShader.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#include "vislib_gl/graphics/gl/SimpleFont.h"
#include <algorithm>
#include <list>
#include <set>
#include <vector>

namespace megamol {
namespace protein_gl {

/**
 * Molecular Surface Renderer class.
 * Computes and renders the solvent excluded (Connolly) surface.
 */
class MoleculeSESRenderer : public megamol::core_gl::view::Renderer3DModuleGL {
public:
    /** postprocessing modi */
    enum PostprocessingMode { NONE = 0, AMBIENT_OCCLUSION = 1, SILHOUETTE = 2, TRANSPARENCY = 3 };

    /** render modi */
    enum RenderMode {
        GPU_RAYCASTING = 0,
        // POLYGONAL = 1,
        // POLYGONAL_GPU = 2,
        GPU_RAYCASTING_INTERIOR_CLIPPING = 3,
        GPU_SIMPLIFIED = 4
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MoleculeSESRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers protein surface renderings.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        // return true;
        return vislib_gl::graphics::gl::GLSLShader::AreExtensionsAvailable();
    }

    /** ctor */
    MoleculeSESRenderer(void);

    /** dtor */
    virtual ~MoleculeSESRenderer(void);

    /**********************************************************************
     * 'get'-functions
     **********************************************************************/

    /** Get probe radius */
    const float GetProbeRadius() const {
        return probeRadius;
    };

    /**********************************************************************
     * 'set'-functions
     **********************************************************************/

    /** Set probe radius */
    void SetProbeRadius(const float rad) {
        probeRadius = rad;
    };

    /** set the color of the silhouette */
    void SetSilhouetteColor(float r, float g, float b) {
        silhouetteColor.Set(r, g, b);
        codedSilhouetteColor = int(r * 255.0f) * 1000000 + int(g * 255.0f) * 1000 + int(b * 255.0f);
    };
    void SetSilhouetteColor(vislib::math::Vector<float, 3> color) {
        SetSilhouetteColor(color.GetX(), color.GetY(), color.GetZ());
    };

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
     * Render atoms as spheres using GLSL Raycasting shaders.
     *
     * @param protein The protein data interface.
     * @param scale The scale factor for the atom radius.
     */
    void RenderAtomsGPU(const megamol::protein_calls::MolecularDataCall* mol, const float scale = 1.0f);

    /**
     * Renders the probe atom at position 'm'.
     *
     * @param m The probe position.
     */
    // void RenderProbe(const vislib::math::Vector<float, 3> m);
    void RenderProbeGPU(const vislib::math::Vector<float, 3> m);

    /**
     * Compute all vertex, attribute and color arrays used for ray casting
     * all molecular surfaces (spheres, spherical triangles, tori).
     */
    void ComputeRaycastingArrays();

    /**
     * Compute all vertex, attribute and color arrays used for ray casting
     * the molecular surface 'ixdRS' (spheres, spherical triangles, tori).
     * @param idxRS The index of the reduced surface.
     */
    void ComputeRaycastingArrays(unsigned int idxRS);

    /**
     * Code a RGB-color into one float.
     * For each color channel, its representation in range 0..255 is computed
     * and stores as follows:
     * rrrgggbbb.0
     * Note that the minimum value for the coded color is 0 and the maximum
     * value is 255255255.0 .
     *
     * @param col Vector containing the color as float [0.0]..[1.0] .
     * @return The coded color value.
     */
    // float CodeColor( const vislib::math::Vector<float, 3> &col) const;
    float CodeColor(const float* col) const;

    /**
     * Decode a coded color to the original RGB-color.
     *
     * @param codedColor Integer value containing the coded color (rrrgggbbb).
     * @return The RGB-color value vector.
     */
    vislib::math::Vector<float, 3> DecodeColor(int codedColor) const;

    /**
     * Creates the frame buffer object and textures needed for offscreen rendering.
     */
    void CreateFBO();

    /**
     * Render the molecular surface using GPU raycasting.
     *
     * @param protein Pointer to the protein data interface.
     */
    void RenderSESGpuRaycasting(const megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Render debug stuff --- THIS IS ONLY FOR DEBUGGING PURPOSES, REMOVE IN FINAL VERSION!!!
     *
     * @param protein Pointer to the protein data interface.
     */
    void RenderDebugStuff(const megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Postprocessing: use screen space ambient occlusion
     */
    void PostprocessingSSAO();

    /**
     * Postprocessing: use silhouette shader
     */
    void PostprocessingSilhouette();

    /**
     * Postprocessing: transparency (blend two images)
     */
    void PostprocessingTransparency(float transparency);

    /**
     * returns the color of the atom 'idx' for the current coloring mode
     *
     * @param idx The index of the atom.
     * @return The color of the atom with the index 'idx'.
     */
    vislib::math::Vector<float, 3> GetProteinAtomColor(unsigned int idx);

    /**
     * Create the singularity textureS which stores for every RS-edge (of all
     * molecular surfaces) the positions of the probes that cut it.
     */
    void CreateSingularityTextures();

    /**
     * Create the singularity texture for the reduced surface 'idxRS' which
     * stores for every RS-edge the positions of the probes that cut it.
     */
    void CreateSingularityTexture(unsigned int idxRS);

private:
    /**
     * Update all parameter slots.
     *
     * @param mol   Pointer to the data call.
     */
    void UpdateParameters(
        const megamol::protein_calls::MolecularDataCall* mol, const protein_calls::BindingSiteCall* bs = 0);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(megamol::core_gl::view::CallRender3DGL& call);

    /**
     * Open GL Render call.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core_gl::view::CallRender3DGL& call);

    /**
     * Deinitialises this renderer. This is only called if there was a
     * successful call to "initialise" before.
     */
    virtual void deinitialise(void);

    /**********************************************************************
     * variables
     **********************************************************************/

    /** MolecularDataCall caller slot */
    megamol::core::CallerSlot molDataCallerSlot;
    /** BindingSiteCall caller slot */
    megamol::core::CallerSlot bsDataCallerSlot;
    /** Light data caller slot */
    megamol::core::CallerSlot getLightsSlot;

    /** camera information */
    // vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
    core::view::Camera camera;

    /** framebuffer information */
    std::shared_ptr<glowl::FramebufferObject> fbo;

    // camera information
    // vislib::SmartPtr<vislib::graphics::CameraParameters> MoleculeSESRenderercameraInfo;
    core::view::Camera MoleculeSESRenderercameraInfo;

    megamol::core::param::ParamSlot postprocessingParam;

    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam0;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam1;
    /** parameter slot for coloring mode weighting*/
    megamol::core::param::ParamSlot cmWeightParam;
    megamol::core::param::ParamSlot silhouettecolorParam;
    megamol::core::param::ParamSlot sigmaParam;
    megamol::core::param::ParamSlot lambdaParam;
    /** parameter slot for min color of gradient color mode */
    megamol::core::param::ParamSlot minGradColorParam;
    /** parameter slot for mid color of gradient color mode */
    megamol::core::param::ParamSlot midGradColorParam;
    /** parameter slot for max color of gradient color mode */
    megamol::core::param::ParamSlot maxGradColorParam;
    megamol::core::param::ParamSlot fogstartParam;
    megamol::core::param::ParamSlot debugParam;
    megamol::core::param::ParamSlot drawSESParam;
    megamol::core::param::ParamSlot drawSASParam;
    megamol::core::param::ParamSlot molIdxListParam;
    /** parameter slot for color table filename */
    megamol::core::param::ParamSlot colorTableFileParam;

    megamol::core::param::ParamSlot probeRadiusSlot;


    bool drawRS;
    bool drawSES;
    bool drawSAS;


    /** the reduced surface(s) */
    std::vector<std::vector<protein::ReducedSurface*>> reducedSurfaceAllFrames;
    /** the reduced surface(s) */
    std::vector<protein::ReducedSurface*> reducedSurface;

    // shader for the cylinders (raycasting view)
    vislib_gl::graphics::gl::GLSLShader cylinderShader;
    // shader for the spheres (raycasting view)
    vislib_gl::graphics::gl::GLSLShader sphereShader;
    // shader for the spheres with clipped interior (raycasting view)
    vislib_gl::graphics::gl::GLSLShader sphereClipInteriorShader;
    // shader for per pixel lighting (polygonal view)
    vislib_gl::graphics::gl::GLSLShader lightShader;
    // shader for 1D gaussian filtering (postprocessing)
    vislib_gl::graphics::gl::GLSLShader hfilterShader;
    vislib_gl::graphics::gl::GLSLShader vfilterShader;
    // shader for silhouette drawing (postprocessing)
    vislib_gl::graphics::gl::GLSLShader silhouetteShader;
    // shader for cheap transparency (postprocessing/blending)
    vislib_gl::graphics::gl::GLSLShader transparencyShader;

    std::shared_ptr<glowl::GLSLProgram> torusShader_;
    std::shared_ptr<glowl::GLSLProgram> sphereShader_;
    std::shared_ptr<glowl::GLSLProgram> sphericalTriangleShader_;

    /**
     * updates and uploads all arrays according to the incoming light information
     */
    void UpdateLights();

    ////////////

    // the bounding box of the protein
    vislib::math::Cuboid<float> bBox;

    // epsilon value for float-comparison
    float epsilon;

    // radius of the probe atom
    float probeRadius;

    vislib::Array<float> atomColorTable;
    unsigned int currentArray;

    /** 'true' if the data for the current render mode is computed, 'false' otherwise */
    bool preComputationDone;


    /** The current coloring mode */
    protein::Color::ColoringMode currentColoringMode0;
    protein::Color::ColoringMode currentColoringMode1;
    /** postprocessing mode */
    PostprocessingMode postprocessing;

    /** vertex and attribute arrays for raycasting the tori */
    std::vector<vislib::Array<float>> torusVertexArray;
    std::vector<vislib::Array<float>> torusInParamArray;
    std::vector<vislib::Array<float>> torusQuatCArray;
    std::vector<vislib::Array<float>> torusInSphereArray;
    std::vector<vislib::Array<float>> torusColors;
    std::vector<vislib::Array<float>> torusInCuttingPlaneArray;
    /** vertex ans attribute arrays for raycasting the spherical triangles */
    std::vector<vislib::Array<float>> sphericTriaVertexArray;
    std::vector<vislib::Array<float>> sphericTriaVec1;
    std::vector<vislib::Array<float>> sphericTriaVec2;
    std::vector<vislib::Array<float>> sphericTriaVec3;
    std::vector<vislib::Array<float>> sphericTriaProbe1;
    std::vector<vislib::Array<float>> sphericTriaProbe2;
    std::vector<vislib::Array<float>> sphericTriaProbe3;
    std::vector<vislib::Array<float>> sphericTriaTexCoord1;
    std::vector<vislib::Array<float>> sphericTriaTexCoord2;
    std::vector<vislib::Array<float>> sphericTriaTexCoord3;
    std::vector<vislib::Array<float>> sphericTriaColors;
    /** vertex and color array for raycasting the spheres */
    std::vector<vislib::Array<float>> sphereVertexArray;
    std::vector<vislib::Array<float>> sphereColors;

    // FBOs and textures for postprocessing
    GLuint colorFBO;
    GLuint blendFBO;
    GLuint horizontalFilterFBO;
    GLuint verticalFilterFBO;
    GLuint texture0;
    GLuint depthTex0;
    GLuint texture1;
    GLuint depthTex1;
    GLuint hFilter;
    GLuint vFilter;
    // width and height of view
    unsigned int width;
    unsigned int height;
    // sigma factor for screen space ambient occlusion
    float sigma;
    // lambda factor for screen space ambient occlusion
    float lambda;

    /** The color lookup table (for chains, amino acids,...) */
    vislib::Array<vislib::math::Vector<float, 3>> colorLookupTable;
    /** The color lookup table which stores the rainbow colors */
    vislib::Array<vislib::math::Vector<float, 3>> rainbowColors;

    // texture for singularity handling (concave triangles)
    std::vector<GLuint> singularityTexture;
    // sizes of singularity textures
    std::vector<unsigned int> singTexWidth, singTexHeight;
    // data of the singularity texture
    float* singTexData;

    // texture for interior clipping / cutting planes (convex spherical cutouts)
    std::vector<GLuint> cutPlanesTexture;
    // sizes of the cutting planes textures
    std::vector<unsigned int> cutPlanesTexWidth, cutPlanesTexHeight;
    // data of the cutting planes texture
    std::vector<vislib::Array<float>> cutPlanesTexData;

    // silhouette color
    vislib::math::Vector<float, 3> silhouetteColor;
    int codedSilhouetteColor;

    // start value for fogging
    float fogStart;
    // transparency value
    float transparency;

    // the list of molecular indices
    vislib::Array<vislib::StringA> molIdxList;
    // flag for SES computation (false = one SES per molecule)
    bool computeSesPerMolecule;
    glm::mat4 view_;
    glm::mat4 proj_;
    glm::mat4 invview_;
    glm::mat4 transview_;
    glm::mat4 invproj_;
    glm::mat4 invtransview_;
    glm::mat4 mvp_;
    glm::mat4 mvpinverse_;
    glm::mat4 mvptranspose_;

    std::unique_ptr<glowl::BufferObject> sphereVertexBuffer_;
    std::unique_ptr<glowl::BufferObject> sphereColorBuffer_;

    std::unique_ptr<glowl::BufferObject> torusVertexBuffer_;
    std::unique_ptr<glowl::BufferObject> torusColorBuffer_;
    std::unique_ptr<glowl::BufferObject> torusParamsBuffer_;
    std::unique_ptr<glowl::BufferObject> torusQuaternionBuffer_;
    std::unique_ptr<glowl::BufferObject> torusSphereBuffer_;
    std::unique_ptr<glowl::BufferObject> torusCuttingPlaneBuffer_;

    std::unique_ptr<glowl::BufferObject> triaVertexBuffer_;
    std::unique_ptr<glowl::BufferObject> triaColorBuffer_;
    std::unique_ptr<glowl::BufferObject> triaAttrib1Buffer_;
    std::unique_ptr<glowl::BufferObject> triaAttrib2Buffer_;
    std::unique_ptr<glowl::BufferObject> triaAttrib3Buffer_;
    std::unique_ptr<glowl::BufferObject> triaAttribTexCoord1Buffer_;
    std::unique_ptr<glowl::BufferObject> triaAttribTexCoord2Buffer_;
    std::unique_ptr<glowl::BufferObject> triaAttribTexCoord3Buffer_;

    std::unique_ptr<glowl::BufferObject> pointLightBuffer_;
    std::unique_ptr<glowl::BufferObject> directionalLightBuffer_;

    GLuint vertexArraySphere_;
    GLuint vertexArrayTorus_;
    GLuint vertexArrayTria_;

    struct LightParams {
        float x, y, z, intensity;
    };

    std::vector<LightParams> pointLights_;
    std::vector<LightParams> directionalLights_;
};

} // namespace protein_gl
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_MOLSESRENDERER_H_INCLUDED */
