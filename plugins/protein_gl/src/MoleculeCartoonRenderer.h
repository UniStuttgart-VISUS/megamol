/*
 * MoleculeCartoonRenderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <memory>
#include <vector>

#include <glowl/glowl.h>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "protein/BSpline.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ProteinColor.h"
#include "vislib/Array.h"
#include "vislib_gl/graphics/gl/SimpleFont.h"

namespace megamol {
namespace protein_gl {

/*
 * Protein Renderer class
 *
 * TODO:
 * - add Parameter:
 *    o number of segments per amino acids
 *    o number of tube segments for CARTOON_CPU
 * - add RenderMode CARTOON_GPU
 */

class MoleculeCartoonRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MoleculeCartoonRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers protein cartoon renderings.";
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
    MoleculeCartoonRenderer(void);

    /** Dtor. */
    ~MoleculeCartoonRenderer(void) override;

    enum class CartoonRenderMode {
        CARTOON = 0,
        CARTOON_SIMPLE = 1,
        CARTOON_CPU = 2,
        CARTOON_GPU = 3,
        CARTOON_LINE = 4,
        CARTOON_TUBE_ONLY = 5
    };

    enum class RenderSource { RENDER_NORMAL = 0, RENDER_COMPARISON_BASE = 1 };


    /**********************************************************************
     * 'get'-functions
     **********************************************************************/

    /** Get radius for cartoon rendering mode */
    inline float GetRadiusCartoon(void) const {
        return radiusCartoon;
    };

    /** Get number of spline segments per amino acid for cartoon rendering mode */
    inline unsigned int GetNumberOfSplineSegments(void) const {
        return numberOfSplineSeg;
    };

    /** Get number of tube segments per 390 degrees in CPU cartoon rendering mode */
    inline unsigned int GetNumberOfTubeSegments(void) const {
        return numberOfTubeSeg;
    };

    /**********************************************************************
     * 'set'-functions
     **********************************************************************/

    /** Set number of spline segments per amino acid for cartoon rendering mode */
    inline void SetNumberOfSplineSegments(unsigned int numSeg) {
        numberOfSplineSeg = numSeg;
    };

    /** Set number of tube segments per 390 degrees in CPU cartoon rendering mode */
    inline void SetNumberOfTubeSegments(unsigned int numSeg) {
        numberOfTubeSeg = numSeg;
    };

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'release'.
     */
    void release(void) override;

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
     * Render protein in hybrid CARTOON mode using the Geometry Shader.
     *
     * @param prot The data interface.
     */
    void RenderCartoonHybrid(const megamol::protein_calls::MolecularDataCall* mol, float* atomPos);

    /**
     * Render protein in CPU CARTOON mode using OpenGL primitives.
     *
     * @param prot The data interface.
     */
    void RenderCartoonCPU(const megamol::protein_calls::MolecularDataCall* mol, float* atomPos);

    /**
     * Render protein in CPU CARTOON mode using OpenGL lines.
     *
     * @param prot The data interface.
     */
    void RenderCartoonLineCPU(const megamol::protein_calls::MolecularDataCall* mol, float* atomPos);

    /**
     * Render protein in GPU CARTOON mode using OpenGL primitives.
     *
     * @param prot The data interface.
     */
    void RenderCartoonGPU(const megamol::protein_calls::MolecularDataCall* mol, float* atomPos);

    /**
     * Render protein in GPU CARTOON mode using OpenGL primitives.
     *
     * @param prot The data interface.
     */
    void RenderCartoonGPUTubeOnly(const megamol::protein_calls::MolecularDataCall* mol, float* atomPos);

    /**
     * Render the molecular data in stick mode.
     */
    void RenderStick(const megamol::protein_calls::MolecularDataCall* mol, const float* atomPos,
        const protein_calls::BindingSiteCall* bs = NULL);

    /**
     * Recompute all values.
     * This function has to be called after every change rendering attributes,
     * e.g. coloring or render mode.
     */
    void RecomputeAll(void);

    /**
     *  Update all parameter slots.
     *
     *  @param mol   Pointer to the data call.
     *  @param frameID The current frame id used for the data call.
     *  @param bs Pointer to the binding site call.
     */
    void UpdateParameters(megamol::protein_calls::MolecularDataCall* mol, unsigned int frameID,
        const protein_calls::BindingSiteCall* bs = 0);

    /**********************************************************************
     * variables
     **********************************************************************/

    // caller slot
    megamol::core::CallerSlot molDataCallerSlot;
    // caller slot
    megamol::core::CallerSlot molRendererCallerSlot;
    /** BindingSiteCall caller slot */
    megamol::core::CallerSlot bsDataCallerSlot;
    // caller slot for light input
    megamol::core::CallerSlot getLightsSlot;

    /** camera information */
    core::view::Camera camera;

    /** framebuffer information */
    std::shared_ptr<glowl::FramebufferObject> fbo;

    megamol::core::param::ParamSlot renderingModeParam;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam0;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam1;
    /** parameter slot for coloring mode weighting*/
    megamol::core::param::ParamSlot cmWeightParam;
    megamol::core::param::ParamSlot stickColoringModeParam;
    megamol::core::param::ParamSlot smoothCartoonColoringParam;
    megamol::core::param::ParamSlot compareParam;
    /** parameter slot for color table filename */
    megamol::core::param::ParamSlot colorTableFileParam;
    /** parameter slot for min color of gradient color mode */
    megamol::core::param::ParamSlot minGradColorParam;
    /** parameter slot for mid color of gradient color mode */
    megamol::core::param::ParamSlot midGradColorParam;
    /** parameter slot for max color of gradient color mode */
    megamol::core::param::ParamSlot maxGradColorParam;
    /** parameter slot for stick radius */
    megamol::core::param::ParamSlot stickRadiusParam;
    /** parameter slot for positional interpolation */
    megamol::core::param::ParamSlot interpolParam;
    /** parameter slot for disabling rendering except protein */
    megamol::core::param::ParamSlot proteinOnlyParam;
    /** parameter slot for stick radius */
    megamol::core::param::ParamSlot tubeRadiusParam;
    /** parameter slot for refreshing in every frame*/
    megamol::core::param::ParamSlot recomputeAlwaysParam;

    // shader for per pixel lighting (polygonal view)
    std::unique_ptr<glowl::GLSLProgram> lightShader;
    // shader for tube generation (cartoon view)
    std::unique_ptr<glowl::GLSLProgram> cartoonShader;
    std::unique_ptr<glowl::GLSLProgram> tubeShader;
    std::unique_ptr<glowl::GLSLProgram> arrowShader;
    std::unique_ptr<glowl::GLSLProgram> helixShader;
    std::unique_ptr<glowl::GLSLProgram> tubeSimpleShader;
    std::unique_ptr<glowl::GLSLProgram> arrowSimpleShader;
    std::unique_ptr<glowl::GLSLProgram> helixSimpleShader;
    std::unique_ptr<glowl::GLSLProgram> tubeSplineShader;
    std::unique_ptr<glowl::GLSLProgram> arrowSplineShader;
    std::unique_ptr<glowl::GLSLProgram> helixSplineShader;

    std::shared_ptr<glowl::GLSLProgram> sphereShader_;
    std::shared_ptr<glowl::GLSLProgram> cylinderShader_;

    // buffer objects
    // the 6 is missing to match the buffers of
    enum class Buffers : GLuint {
        POSITION = 0,
        COLOR = 1,
        CYL_PARAMS = 2,
        CYL_QUAT = 3,
        CYL_COL1 = 4,
        CYL_COL2 = 5,
        FILTER = 6,
        LIGHT_POSITIONAL = 7,
        LIGHT_DIRECTIONAL = 8,
        BUFF_COUNT = 9
    };

    GLuint vertex_array_spheres_;
    std::array<std::unique_ptr<glowl::BufferObject>, static_cast<int>(Buffers::BUFF_COUNT)> buffers_;

    glm::mat4 MVP;
    glm::mat4 MVinv;
    glm::mat4 MVPinv;
    glm::mat4 MVPtransp;
    glm::mat4 NormalM;
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 invProj;
    glm::vec2 planes;

    // current render mode
    CartoonRenderMode currentRenderMode;
    /** The current coloring mode */
    protein_calls::ProteinColor::ColoringMode currentColoringMode0;
    protein_calls::ProteinColor::ColoringMode currentColoringMode1;
    // smooth coloring of cartoon mode
    bool smoothCartoonColoringMode;

    // is comparison mode enabled?
    bool compare;

    // has the hybrid CARTOON render mode to be prepared?
    bool prepareCartoonHybrid;
    // has the CPU CARTOON render mode to be prepared?
    bool prepareCartoonCPU;
    // has the CARTOON LINE render mode to be prepared?
    bool prepareCartoonLine;

    // counters, vertex- and color-arrays for cartoon mode
    float* vertTube;
    float* normalTube;
    float* colorsParamsTube;
    unsigned int totalCountTube;
    float* vertArrow;
    float* normalArrow;
    float* colorsParamsArrow;
    unsigned int totalCountArrow;
    float* vertHelix;
    float* normalHelix;
    float* colorsParamsHelix;
    unsigned int totalCountHelix;

    // number of spline segments per amino acid
    unsigned int numberOfSplineSeg;
    // number of tube segments per 390 degrees (only used with cartoon GPU)
    unsigned int numberOfTubeSeg;
    // radius for secondary structure elements with CARTOON render modes
    float radiusCartoon;

    /** The color lookup table (for chains, amino acids,...) */
    std::vector<glm::vec3> colorLookupTable;
    std::vector<glm::vec3> fileLookupTable;
    /** The color lookup table which stores the rainbow colors */
    std::vector<glm::vec3> rainbowColors;

    /** The atom color table for rendering */
    std::vector<glm::vec3> atomColorTable;

    // the Id of the current frame (for dynamic data)
    unsigned int currentFrameId;
    // the current call time
    float oldCallTime;

    unsigned int atomCount;

    // coordinates of the first (center) b-spline (result of the spline computation)
    std::vector<std::vector<vislib::math::Vector<float, 3>>> bSplineCoordsCPU;
    // coordinates of the second (direction) b-spline (result of the spline computation)
    std::vector<std::vector<vislib::math::Vector<float, 3>>> bSplineCoordsDirCPU;
    // color of secondary structure b-spline
    std::vector<std::vector<vislib::math::Vector<float, 3>>> cartoonColorCPU;

    vislib::Array<bool> atomVisible;
};


} // namespace protein_gl
} /* end namespace megamol */
