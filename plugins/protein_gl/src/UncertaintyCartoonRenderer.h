/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <map>
#include <utility>

#include <glowl/glowl.h>

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/light/CallLight.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/UncertaintyDataCall.h"
#include "protein_gl/DeferredRenderingProvider.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"


namespace megamol::protein_gl {

using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace vislib_gl::graphics::gl;

/**
 * Renderer for uncertain cartoon renderings
 */
class UncertaintyCartoonRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "UncertaintyCartoonRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers cartoon renderings for biomolecules (uses Tessellation Shaders).";
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
    UncertaintyCartoonRenderer();

    /** Dtor. */
    ~UncertaintyCartoonRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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
     * The ...
     *
     * @param t          The ...
     * @param outScaling The ...
     *
     * @return The pointer to the molecular data call.
     */
    MolecularDataCall* GetData(unsigned int t);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    /**
     * The ... .
     *
     * @param udc The uncertainty data call.
     * @param mol The molecular data call.
     *
     * @return The return value of the function.
     */
    bool GetUncertaintyData(UncertaintyDataCall* udc, MolecularDataCall* mol);

    /**
     * The ... .
     *
     * @param mol        The ...
     * @param colBytes   The ...
     * @param vertBytes  The ...
     * @param colStride  The ...
     * @param vertStride The ...
     */
    void GetBytesAndStride(MolecularDataCall& mol, unsigned int& colBytes, unsigned int& vertBytes,
        unsigned int& colStride, unsigned int& vertStride);

    /**
     * The ... .
     *
     * @param syncObj The ...
     */
    void QueueSignal(GLsync& syncObj);

    /**
     * The ... .
     *
     * @param syncObj The ...
     */
    void WaitSignal(GLsync& syncObj);

    void RefreshLights(core::view::light::CallLight* lightCall, glm::vec3 camDir);

    /** Strucutre to hold C-alpha data */
    struct CAlpha {
        float pos[4];      // position of the C-alpha atom
        float dir[3];      // direction of the amino-acid
        int colIdx;        // UNUSED - don't delete ... shader is corrupt otherwise - WHY?
        float col[4];      // the color of the amino-acid or chain (depending on coloring mode)
        float uncertainty; // the uncertainty
        int flag;          // UNUSED - the amino-acid flag (none, missing, heterogen)
        float unc[UncertaintyDataCall::secStructure::NOE];        // the uncertainties of the different sctructure
                                                                  // assignments          - used for dithering
        int sortedStruct[UncertaintyDataCall::secStructure::NOE]; // the sorted structure assignments: max=[0] to
                                                                  // min=[NOE]             - used for dithering
    };

    /**
     * ... .
     *
     * @return The ... .
     */
    bool loadTubeShader();

    /**
     * Enumeration of secondary structure colorings
     */
    enum coloringModes {
        COLOR_MODE_STRUCT = 0,
        COLOR_MODE_UNCERTAIN = 1,
        COLOR_MODE_CHAIN = 2,
        COLOR_MODE_AMINOACID = 3,
        COLOR_MODE_RESIDUE_DEBUG = 4
    };

    /**
     * Enumeration of uncertainty geometrical visualisations
     */
    enum uncVisualisations {
        UNC_VIS_NONE = 0,
        UNC_VIS_SIN_U = 1,
        UNC_VIS_SIN_V = 2,
        UNC_VIS_SIN_UV = 3,
        UNC_VIS_TRI_U = 4,
        UNC_VIS_TRI_UV = 5
    };

    /**
     * Enumeration of oulining visualisations
     */
    enum outlineOptions { OUTLINE_NONE = 0, OUTLINE_LINE = 1, OUTLINE_FULL_UNCERTAIN = 2, OUTLINE_FULL_CERTAIN = 3 };

    /**********************************************************************
     * variables
     **********************************************************************/

#ifdef FIRSTFRAME_CHECK
    bool firstFrame;
#endif
    /** The call for PDB data */
    core::CallerSlot getPdbDataSlot;
    /** The call for uncertainty data */
    core::CallerSlot uncertaintyDataSlot;
    /** The call for lighting data */
    core::CallerSlot getLightSlot;

    // paramter
    core::param::ParamSlot scalingParam;
    core::param::ParamSlot backboneParam;
    core::param::ParamSlot backboneWidthParam;
    core::param::ParamSlot materialParam;
    core::param::ParamSlot uncertainMaterialParam;
    core::param::ParamSlot lineDebugParam;
    core::param::ParamSlot buttonParam;
    core::param::ParamSlot colorInterpolationParam;
    core::param::ParamSlot tessLevelParam;
    core::param::ParamSlot colorModeParam;
    core::param::ParamSlot onlyTubesParam;
    core::param::ParamSlot colorTableFileParam;
    core::param::ParamSlot lightPosParam;
    core::param::ParamSlot uncVisParam;
    core::param::ParamSlot uncDistorGainParam;
    core::param::ParamSlot uncDistorRepParam;
    core::param::ParamSlot ditherParam;
    core::param::ParamSlot methodDataParam;
    core::param::ParamSlot outlineParam;
    core::param::ParamSlot outlineScalingParam;
    core::param::ParamSlot outlineColorParam;
    core::param::ParamSlot bFactorAsUncertaintyParam;
    core::param::ParamSlot showRMSFParam;
    core::param::ParamSlot useAlphaBlendingParam;
    core::param::ParamSlot maxRMSFParam;

    // local parameter values
    int currentTessLevel;
    uncVisualisations currentUncVis;
    coloringModes currentColoringMode;
    glm::vec4 currentLightPos;
    float currentScaling; // UNUSED ...
    float currentBackboneWidth;
    glm::vec4 currentMaterial;
    glm::vec4 currentUncertainMaterial;
    glm::vec2 currentUncDist;
    int currentDitherMode;
    UncertaintyDataCall::assMethod currentMethodData;
    outlineOptions currentOutlineMode;
    float currentOutlineScaling;
    glm::vec3 currentOutlineColor;

    GLuint vertArray;
    std::vector<GLsync> fences; // (?)
    GLuint theSingleBuffer;
    unsigned int currBuf;
    GLsizeiptr bufSize;
    int numBuffers;
    void* theSingleMappedMem;
    GLuint singleBufferCreationBits;
    GLuint singleBufferMappingBits;

    std::shared_ptr<glowl::GLSLProgram> tubeShader_;

    // the number of different structure types
    unsigned int structCount;

    // C-alpha main chain
    std::vector<CAlpha> mainChain;

    // the total number of amino-acids defined in molecular data
    unsigned int molAtomCount;

    // the total number of amino-acids defined in uncertainty data
    unsigned int aminoAcidCount;
    // The original PDB index
    vislib::Array<vislib::StringA> pdbIndex;
    // The synchronized index between molecular data and uncertainty data
    vislib::Array<unsigned int> synchronizedIndex;
    // the array for the residue flag
    vislib::Array<unsigned int> residueFlag;
    /** The uncertainty difference of secondary structure types */
    vislib::Array<float> uncertainty;
    // The values of the secondary structure uncertainty for each amino-acid
    vislib::Array<vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>>
        secStructUncertainty;
    // The sorted structure types of the uncertainty values
    vislib::Array<vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure,
        static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>>
        sortedSecStructAssignment;

    // color table for chain id per amino acid
    std::vector<glm::vec3> chainColors;
    // color table for amino acid per amino acid
    std::vector<glm::vec3> aminoAcidColors;
    // secondary structure type colors as RGB(A)
    vislib::Array<vislib::math::Vector<float, 4>> secStructColor;
    // color table
    std::vector<glm::vec3> colorTable;

    // positions of C-alpha-atoms and O-atoms
    vislib::Array<vislib::Array<float>> positionsCa;
    vislib::Array<vislib::Array<float>> positionsO;

    std::shared_ptr<glowl::BufferObject> pointLightBuffer_;
    std::shared_ptr<glowl::BufferObject> distantLightBuffer_;

    std::vector<DeferredRenderingProvider::LightParams> pointLights_;
    std::vector<DeferredRenderingProvider::LightParams> distantLights_;

    bool firstframe;
};

} // namespace megamol::protein_gl
