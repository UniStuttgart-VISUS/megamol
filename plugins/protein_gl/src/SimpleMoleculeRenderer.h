/*
 * SimpleMoleculeRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_SIMPLEMOLECULERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_SIMPLEMOLECULERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/light/CallLight.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ProteinColor.h"
#include "protein_gl/DeferredRenderingProvider.h"

#include "glowl/BufferObject.hpp"
#include "glowl/FramebufferObject.hpp"
#include "glowl/GLSLProgram.hpp"

namespace megamol {
namespace protein_gl {

/*
 * Simple Molecular Renderer class
 */

class SimpleMoleculeRenderer : public megamol::core_gl::view::Renderer3DModuleGL {
public:
    /** The names of the render modes */
    enum RenderMode {
        LINES = 0,
        STICK = 1,
        BALL_AND_STICK = 2,
        SPACEFILLING = 3,
        SAS = 4,
        LINES_FILTER = 5,
        STICK_FILTER = 6,
        SPACEFILL_FILTER = 7
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SimpleMoleculeRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers molecule renderings.";
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
    SimpleMoleculeRenderer(void);

    /** Dtor. */
    virtual ~SimpleMoleculeRenderer(void);

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
    virtual bool GetExtents(core_gl::view::CallRender3DGL& call);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core_gl::view::CallRender3DGL& call);

    /**
     * Render the molecular data using lines and points.
     *
     * @param mol        Pointer to the data call.
     * @param atomPos    Pointer to the interpolated atom positions.
     */
    void RenderLines(const megamol::protein_calls::MolecularDataCall* mol, const float* atomPos,
        bool useFiltering = false, bool useClipplane = false);

    /**
     * Render the molecular data in stick mode.
     *
     * @param mol          Pointer to the data call.
     * @param atomPos      Pointer to the interpolated atom positions.
     * @param useFiltering Enables atom filtering in the renderer
     * @param useClipplane Enables the a clipplane cutting in the renderer
     */
    void RenderStick(const megamol::protein_calls::MolecularDataCall* mol, const float* atomPos,
        bool useFiltering = false, bool useClipplane = false);

    /**
     * Render the molecular data in ball-and-stick mode.
     *
     * @param mol        Pointer to the data call.
     * @param atomPos    Pointer to the interpolated atom positions.
     */
    void RenderBallAndStick(const megamol::protein_calls::MolecularDataCall* mol, const float* atomPos);


    /**
     * Render the molecular data in spacefilling mode.
     *
     * @param mol          Pointer to the data call.
     * @param atomPos      Pointer to the interpolated atom positions.
     * @param useFiltering Enables atom filtering in the renderer
     * @param useClipplane Enables the a clipplane cutting in the renderer
     */
    void RenderSpacefilling(const megamol::protein_calls::MolecularDataCall* mol, const float* atomPos,
        bool useFiltering = false, bool useClipplane = false);

    /**
     * Render the molecular data in solvent accessible surface mode.
     *
     * @param mol        Pointer to the data call.
     * @param atomPos    Pointer to the interpolated atom positions.
     */
    void RenderSAS(const megamol::protein_calls::MolecularDataCall* mol, const float* atomPos);

    /**
     * Update all parameter slots.
     *
     * @param mol   Pointer to the data call.
     */
    void UpdateParameters(
        const megamol::protein_calls::MolecularDataCall* mol, const protein_calls::BindingSiteCall* bs = 0);


    /**********************************************************************
     * variables
     **********************************************************************/
    struct LightParams {
        float x, y, z, intensity;
    };

    /** MolecularDataCall caller slot */
    megamol::core::CallerSlot molDataCallerSlot;
    /** BindingSiteCall caller slot */
    megamol::core::CallerSlot bsDataCallerSlot;
    /** Slot to get the lights */
    megamol::core::CallerSlot getLightsSlot;

    /** camera information */
    core::view::Camera cam;
    float viewportStuff[4];
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 invProj;
    glm::mat4 MVinv;
    glm::mat4 NormalM;
    glm::mat4 MVP;
    glm::mat4 MVPinv;
    glm::mat4 MVPtransp;

    /** parameter slot for color table filename */
    megamol::core::param::ParamSlot colorTableFileParam;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam0;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam1;
    /** parameter slot for coloring mode weighting*/
    megamol::core::param::ParamSlot cmWeightParam;
    /** parameter slot for rendering mode */
    megamol::core::param::ParamSlot renderModeParam;
    /** parameter slot for stick radius */
    megamol::core::param::ParamSlot stickRadiusParam;
    /** parameter slot for SAS probe radius */
    megamol::core::param::ParamSlot probeRadiusParam;
    /** parameter slot for min color of gradient color mode */
    megamol::core::param::ParamSlot minGradColorParam;
    /** parameter slot for mid color of gradient color mode */
    megamol::core::param::ParamSlot midGradColorParam;
    /** parameter slot for max color of gradient color mode */
    megamol::core::param::ParamSlot maxGradColorParam;
    /** list of molecule indices */
    megamol::core::param::ParamSlot molIdxListParam;
    /** parameter slot for special color */
    megamol::core::param::ParamSlot specialColorParam;
    /** parameter slot for positional interpolation */
    megamol::core::param::ParamSlot interpolParam;
    /** Toggle the use of geometry shaders for glyph raycasting */
    megamol::core::param::ParamSlot toggleZClippingParam;
    /**  */
    megamol::core::param::ParamSlot clipPlaneTimeOffsetParam;
    /**  */
    megamol::core::param::ParamSlot clipPlaneDurationParam;
    /** Toggle use of neighborhood colors for own color */
    megamol::core::param::ParamSlot useNeighborColors;

    float currentZClipPos;

    // shader programs
    std::shared_ptr<glowl::GLSLProgram> sphereShader_;
    std::shared_ptr<glowl::GLSLProgram> cylinderShader_;
    std::shared_ptr<glowl::GLSLProgram> lineShader_;

    // buffer objects
    enum class Buffers : GLuint {
        POSITION = 0,
        COLOR = 1,
        CYL_PARAMS = 2,
        CYL_QUAT = 3,
        CYL_COL1 = 4,
        CYL_COL2 = 5,
        FILTER = 6,
        BUFF_COUNT = 7
    };
    GLuint vertex_array_;
    std::array<std::unique_ptr<glowl::BufferObject>, static_cast<int>(Buffers::BUFF_COUNT)> buffers_;

    /** The current coloring mode */
    protein_calls::ProteinColor::ColoringMode currentColoringMode0;
    protein_calls::ProteinColor::ColoringMode currentColoringMode1;

    /** The color lookup table (for chains, amino acids,...) */
    std::vector<glm::vec3> colorLookupTable_;
    /** The color lookup table read from file */
    std::vector<glm::vec3> fileColorTable_;
    /** The color lookup table which stores the rainbow colors */
    std::vector<glm::vec3> rainbowColors_;

    bool tableFromFile_;

    /** The atom color table for rendering */
    std::vector<glm::vec3> atomColorTable_;

    /** The current rendering mode */
    RenderMode currentRenderMode;

    /** vertex array for spheres */
    std::vector<glm::vec4> vertSpheres_;
    /** vertex array for cylinders */
    std::vector<glm::vec4> vertCylinders_;
    /** attribute array for quaterinons of the cylinders */
    std::vector<glm::vec4> quatCylinders_;
    /** attribute array for inParam of the cylinders (radius and length) */
    std::vector<glm::vec2> inParaCylinders_;
    /** first color array for cylinder */
    std::vector<glm::vec3> color1Cylinders_;
    /** second color array for cylinder */
    std::vector<glm::vec3> color2Cylinders_;
    /** Connections filter */
    std::vector<int> conFilter_;
    /** vertex array for points */
    std::vector<glm::vec4> vertPoints_;
    /** vertex array for lines */
    std::vector<glm::vec4> vertLines_;
    /** color array for the lines */
    std::vector<glm::vec3> colorLines_;

    // the list of molecular indices
    vislib::Array<vislib::StringA> molIdxList;

    /** The module providing the fbo and the deferred shading */
    DeferredRenderingProvider deferredProvider_;

    /** The hash of the lastly rendered molecular data call*/
    SIZE_T lastDataHash;
};


} // namespace protein_gl
} // namespace megamol

#endif // MMPROTEINPLUGIN_SIMPLEMOLECULERENDERER_H_INCLUDED
