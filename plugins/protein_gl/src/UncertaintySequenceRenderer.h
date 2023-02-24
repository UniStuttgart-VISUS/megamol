/*
 * UncertaintySequenceRenderer.h
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "SequenceRenderer" in megamol protein plugin (svn revision 1500).
 *
 */


#pragma once

//#define USE_SIMPLE_FONT

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"

#include "vislib_gl/graphics/gl/OpenGLTexture2D.h"

#ifdef USE_SIMPLE_FONT
#include "vislib_gl/graphics/gl/SimpleFont.h"
#else //  USE_SIMPLE_FONT
#include "vislib_gl/graphics/gl/OutlineFont.h"
#include "vislib_gl/graphics/gl/Verdana.inc"
#endif //  USE_SIMPLE_FONT

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/UncertaintyDataCall.h"


namespace megamol::protein_gl {

class UncertaintySequenceRenderer : public megamol::mmstd_gl::Renderer2DModuleGL {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "UncertaintySequenceRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers sequence renderings of protein secondary structure uncertainty.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** ctor */
    UncertaintySequenceRenderer();

    /** dtor */
    ~UncertaintySequenceRenderer() override;

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
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags) override;

    /**
     * Prepares the data for rendering.
     *
     * @param mol The molecular data call.
     * @return true if preparation was successful, false otherwise
     */
    bool PrepareData(protein_calls::UncertaintyDataCall* udc, protein_calls::BindingSiteCall* bs);

    /**
     * TODO
     */
    bool LoadTexture(vislib::StringA filename);

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
    bool GetExtents(megamol::mmstd_gl::CallRender2DGL& call) override;

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(megamol::mmstd_gl::CallRender2DGL& call) override;

    /**********************************************************************
     * other functions
     **********************************************************************/

    /**
     * Draws the texture tiles for the secondary structure types.
     *
     * @param pre     The PREvious secondary structure type on position i-1
     * @param cur     The CURrent secondary structure type on position i
     * @param fol     The FOLlowing secondary structure type on position i+1
     * @param f       The residue flag
     * @param x       The x position
     * @param y       The y position
     * @param bgColor The the default color
     */
    void DrawSecStructTextureTiles(protein_calls::UncertaintyDataCall::secStructure pre,
        protein_calls::UncertaintyDataCall::secStructure cur, protein_calls::UncertaintyDataCall::secStructure fol,
        protein_calls::UncertaintyDataCall::addFlags f, float x, float y, float bgColor[4]);

    /**
     * Draws the geometry tiles for the secondary structure types.
     *
     * @param cur     The CURrent secondary structure type on position i
     * @param fol     The FOLlowing secondary structure type on position i+1
     * @param f       The residue flag
     * @param x       The x position
     * @param y       The y position
     * @param bgColor The the default color
     */
    void DrawSecStructGeometryTiles(protein_calls::UncertaintyDataCall::secStructure cur,
        protein_calls::UncertaintyDataCall::secStructure fol, protein_calls::UncertaintyDataCall::addFlags f, float x,
        float y, float bgColor[4]);

    /**
     * Draws the threshold/energy value tiles.
     *
     * @param str     The secondary structure type on position i
     * @param f       The residue flag
     * @param x       The x position
     * @param y       The y position
     * @param value   The ... .
     * @param min     The ... .
     * @param max     The ... .
     * @param thresh  The ... .
     */
    void DrawThresholdEnergyValueTiles(protein_calls::UncertaintyDataCall::secStructure str,
        protein_calls::UncertaintyDataCall::addFlags f, float x, float y, float value, float min, float max,
        float thresh, bool invert = false);

    /**
     * Renders a two lined tooltip text.
     *
     * @param start   The height the tooltip starts.
     * @param end     The height the tooltip ends.
     * @param str1    The string for the first line.
     * @param str2    The string for the second line.
     * @param fgColor The foreground color.
     * @param bgColor The background color.
     */
    void RenderToolTip(
        float start, float end, vislib::StringA str1, vislib::StringA str2, float fgColor[4], float bgColor[4]);

    /**
     * block chart color for certain structure assignment
     */
    enum certainBlockChartColor { CERTAIN_BC_NONE = 0, CERTAIN_BC_COLORED = 1 };
    /**
     * structure color for certain structure assignment
     */
    enum certainStructColor { CERTAIN_STRUCT_NONE = 0, CERTAIN_STRUCT_COLORED = 1, CERTAIN_STRUCT_GRAY = 2 };
    /**
     * block chart color for uncertain structure assignment
     */
    enum uncertainBlockChartColor { UNCERTAIN_BC_NONE = 0, UNCERTAIN_BC_COLORED = 1 };
    /**
     * block chart orientation for uncertain structure assignment
     */
    enum uncertainBlockChartOrientation { UNCERTAIN_BC_VERTI = 0, UNCERTAIN_BC_HORIZ = 1 };
    /**
     * structure color for uncertain structure assignment
     */
    enum uncertainStructColor { UNCERTAIN_STRUCT_NONE = 0, UNCERTAIN_STRUCT_COLORED = 1, UNCERTAIN_STRUCT_GRAY = 2 };
    /**
     * structure geometry for uncertain structure assignment
     */
    enum uncertainStructGeometry {
        UNCERTAIN_STRUCT_STAT_VERTI = 0,
        UNCERTAIN_STRUCT_STAT_HORIZ = 1,
        UNCERTAIN_STRUCT_STAT_MORPH = 2,
        UNCERTAIN_STRUCT_DYN_TIME = 3,
        UNCERTAIN_STRUCT_DYN_SPACE = 4,
        UNCERTAIN_STRUCT_DYN_EQUAL = 5,
        UNCERTAIN_GLYPH = 6,
        UNCERTAIN_GLYPH_WITH_AXES = 7,
    };
    /**
     * Color interpolation methods available for: UNCERTAIN_STRUCT_STAT_MORPH
     *                                            UNCERTAIN_BC_HORIZ
     */
    enum uncertainColorInterpol { UNCERTAIN_COLOR_RGB = 0, UNCERTAIN_COLOR_HSL = 1, UNCERTAIN_COLOR_HSL_HP = 2 };

    /**
     * Uncertainty View Modes
     *
     */
    enum viewModes { VIEWMODE_NORMAL_SEQUENCE = 0, VIEWMODE_UNFOLDED_SEQUENCE = 1, VIEWMODE_UNFOLDED_AMINOACID = 2 };


    /**
     * Renders the uncertainty visualization.
     *
     * @param yPos The y position the rendering should start.
     * @param fgColor The foreground color.
     * @param bgColor The background color.
     */
    void RenderUncertainty(float yPos, float fgColor[4], float bgColor[4]);


    /**
     * Convert color from RGB(A) to HSL(A).
     *
     * @param rgba The color as RGB(A).
     * @return The color in HSL(A).
     *
     * Source: http://easyrgb.com/index.php?X=MATH
     */
    vislib::math::Vector<float, 4> rgb2hsl(vislib::math::Vector<float, 4> rgba);

    /**
     * Convert color from HSL(A) to RGB(A).
     *
     * @param rgba The color as HSL(A).
     * @return The color in RGB(A).
     *
     * Source: http://easyrgb.com/index.php?X=MATH
     */
    vislib::math::Vector<float, 4> hsl2rgb(vislib::math::Vector<float, 4> hsla);

    /**
     * Convert hue to RGB(A). Helper function for hsl2rgb.
     *
     * @param v1 The first interim value for hue of the hsl2rgb function
     * @param v2 The second interim value for hue of the hsl2rgb function.
     * @param vH The current hue.
     * @return The corresponding component for RGB color.
     *
     * Source: http://easyrgb.com/index.php?X=MATH
     */
    float hue2rgb(float v1, float v2, float vH);

    /**
     * Hue preserving color blending for two colors in RGB(A).
     *
     * @param c1 The first color as RGB(A).
     * @param c2 The second color as RGB(A).
     * @return The color in RGB(A).
     *
     * Source: http://www.vis.uni-stuttgart.de/~weiskopf/publications/vis09_blending.pdf
     *         https://www.w3.org/TR/compositing-1/#blendingluminosity
     */
    vislib::math::Vector<float, 4> HuePreservingColorBlending(
        vislib::math::Vector<float, 4> c1, vislib::math::Vector<float, 4> c2);

    /**
     * ... .
     *
     * @return The ... .
     */
    bool LoadShader();

    /**
     * ... .
     *
     */
    void calculateGeometryVertices(int samples);


    /**********************************************************************
     * variables
     **********************************************************************/

    /** The call for uncertainty data */
    core::CallerSlot uncertaintyDataSlot;

    /** binding site caller slot */
    core::CallerSlot bindingSiteCallerSlot;
    /** residue selection caller slot */
    core::CallerSlot resSelectionCallerSlot;
    /** ramachandran plot caller slot */
    core::CallerSlot ramachandranCallerSlot;

    // the number of residues in one row
    megamol::core::param::ParamSlot resCountPerRowParam;
    // the file name for the color table
    megamol::core::param::ParamSlot colorTableFileParam;
    // parameter to show/hide row legend/key
    megamol::core::param::ParamSlot toggleLegendParam;
    // clear the current residue selection
    megamol::core::param::ParamSlot clearResSelectionParam;

    // parameter to show/hide tooltip
    megamol::core::param::ParamSlot toggleTooltipParam;

    // parameter to show/hide stride secondary structure
    megamol::core::param::ParamSlot toggleStrideParam;
    // parameter to show/hide dssp secondary structure
    megamol::core::param::ParamSlot toggleDsspParam;
    // parameter to show/hide pdb secondary structure
    megamol::core::param::ParamSlot togglePdbParam;
    // parameter to show/hide prosign secondary structure
    megamol::core::param::ParamSlot toggleProsignParam;

    // parameter to show/hide thresholds of stride
    megamol::core::param::ParamSlot toggleStrideThreshParam;
    unsigned int strideThresholdCount;
    // parameter to show/hide threshold of dssp
    megamol::core::param::ParamSlot toggleDsspThreshParam;
    unsigned int dsspThresholdCount;
    // parameter to show/hide threshold of prosign
    megamol::core::param::ParamSlot toggleProsignThreshParam;
    unsigned int prosignThresholdCount;
    // Wireframe rendering.
    megamol::core::param::ParamSlot toggleWireframeParam;

    ////////////////////////////////////////////////
    // INSERT CODE FROM OBOVE FOR NEW METHOD HERE //
    ////////////////////////////////////////////////

    // parameter to show/hide disagreements in secondary structure assignment
    megamol::core::param::ParamSlot toggleUncertaintyParam;
    // parameter to show/hide secondary structure uncertainty visualization
    megamol::core::param::ParamSlot toggleUncertainStructParam;
    // parameter to show/hide separator line for amino-acids in uncertainty visualization
    megamol::core::param::ParamSlot toggleUncSeparatorParam;

    // parameter to choose different visualizations
    megamol::core::param::ParamSlot certainBlockChartColorParam;
    megamol::core::param::ParamSlot certainStructColorParam;
    megamol::core::param::ParamSlot uncertainBlockChartColorParam;
    megamol::core::param::ParamSlot uncertainBlockChartOrientationParam;
    megamol::core::param::ParamSlot uncertainStructColorParam;
    megamol::core::param::ParamSlot uncertainStructGeometryParam;
    megamol::core::param::ParamSlot uncertainColorInterpolParam;
    megamol::core::param::ParamSlot uncertainGardientIntervalParam;
    megamol::core::param::ParamSlot geometryTessParam;
    megamol::core::param::ParamSlot alternativeMouseHoverParam;
    megamol::core::param::ParamSlot flipUncertaintyVisParam;

    megamol::core::param::ParamSlot viewModeParam;

    // the current uncertainty visualization selection
    certainBlockChartColor currentCertainBlockChartColor;
    certainStructColor currentCertainStructColor;
    uncertainBlockChartColor currentUncertainBlockChartColor;
    uncertainBlockChartOrientation currentUncertainBlockChartOrientation;
    uncertainStructColor currentUncertainStructColor;
    uncertainStructGeometry currentUncertainStructGeometry;
    uncertainColorInterpol currentUncertainColorInterpol;
    float currentUncertainGardientInterval;
    bool showSeparatorLine;
    int currentGeometryTess;

    viewModes currentViewMode;

    // parameter to reload shader
    megamol::core::param::ParamSlot reloadShaderParam;

    // the number of secondary structure rows which can be shown/hidden
    unsigned int secStructRows;
    // the number of method rows which can be shown/hidden
    unsigned int methodRows;
    // the number of residue columns
    unsigned int resCols;
    // the number of residue rows
    unsigned int resRows;
    // the height of a row
    float rowHeight;
    // data preparation flag
    bool dataPrepared;
    // the total number of amino-acids
    unsigned int aminoAcidCount;
    // the total number of binding sites
    unsigned int bindingSiteCount;
    // the array of amino acid 1-letter codes
    vislib::Array<char> aminoAcidName;
    // the array of pdb amino-acid indices
    vislib::Array<vislib::StringA> aminoAcidIndex;
    // the array of the chain IDs
    vislib::Array<char> chainID;
    // the array for the residue flag
    vislib::Array<protein_calls::UncertaintyDataCall::addFlags> residueFlag;

    // the array of binding site names
    vislib::Array<vislib::StringA> bindingSiteNames;
    // the array of descriptons for the binding sites
    vislib::Array<vislib::StringA> bindingSiteDescription;

    // the vertex buffer array for the tiles
    vislib::Array<float> vertices;
    // the vertex buffer array secondary structure
    vislib::Array<vislib::Array<vislib::math::Vector<float, 2>>> secStructVertices;
    // the vertex buffer array for the amino-acid separator lines
    vislib::Array<float> aminoacidSeparatorVertices;
    // the vertex buffer array for the chain tiles
    vislib::Array<float> chainVertices;
    // the color buffer array for the chain tiles
    vislib::Array<float> chainColors;
    // the vertex buffer array for the chain separator lines
    vislib::Array<float> chainSeparatorVertices;
    // the vertex buffer array for the binding site tiles
    vislib::Array<float> bsVertices;
    // the index array for the binding site tiles
    vislib::Array<unsigned int> bsIndices;
    // the color array for the binding site tiles
    vislib::Array<vislib::math::Vector<float, 3>> bsColors;

    // axes for uncertainty glyph
    vislib::Array<vislib::math::Vector<float, 2>> glyphAxis;

    // different possible secondary structure assignments per amino-acid
    vislib::Array<unsigned int> diffStrucCount;

    // The values of the secondary structure uncertainty for each amino-acid
    vislib::Array<vislib::Array<
        vislib::math::Vector<float, static_cast<int>(protein_calls::UncertaintyDataCall::secStructure::NOE)>>>
        secStructUncertainty;
    // The sorted structure types of the uncertainty values
    vislib::Array<vislib::Array<vislib::math::Vector<protein_calls::UncertaintyDataCall::secStructure,
        static_cast<int>(protein_calls::UncertaintyDataCall::secStructure::NOE)>>>
        sortedSecStructAssignment;
    // the array of the secondary structure type difference for each amino-acid tile
    vislib::Array<float> uncertainty;

    /** The 5 STRIDE threshold values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 7>> strideStructThreshold;
    /** The 4 DSSP energy values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 4>> dsspStructEnergy;
    /** THe 4 PROSIGN threshold values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 6>> prosignStructThreshold;

    // color table
    std::vector<glm::vec3> colorTable;
    std::vector<glm::vec3> fileColorTable;

    // secondary structure type colors as RGB(A)
    std::vector<glm::vec4> secStructColor;
    // secondary structure type descriptions
    vislib::Array<vislib::StringA> secStructDescription;

    // textures
    vislib::Array<vislib::SmartPtr<vislib_gl::graphics::gl::OpenGLTexture2D>> markerTextures;

    // mouse hover
    vislib::math::Vector<float, 2> mousePos;
    vislib::math::Vector<float, 2> mousePosDetail;
    int mousePosResIdx;
    bool rightMouseDown;
    bool initialClickSelection;

    // font rendering
#ifdef USE_SIMPLE_FONT
    vislib_gl::graphics::gl::SimpleFont theFont;
#else
    vislib_gl::graphics::gl::OutlineFont theFont;
#endif
    // selection
    vislib::Array<bool> selection;
    protein_calls::ResidueSelectionCall* resSelectionCall;

    // PDB ID
    vislib::StringA pdbID;
    // PDB legend string with methods used for secondary structure assignment
    vislib::StringA pdbLegend;

    // timer for animation of geometry
    clock_t animTimer;

    // shader for per fragment color interpolation
    std::unique_ptr<glowl::GLSLProgram> shader;
};

} // namespace megamol::protein_gl
