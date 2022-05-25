/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <optional>

#include <glowl/FramebufferObject.hpp>

#include "Renderer2D.h"
#include "datatools/table/TableDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/flags/FlagStorage.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmcore_gl/utility/SSBOBufferArray.h"
#include "mmstd_gl/flags/FlagCallsGL.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"
#include "vislib/math/Matrix.h"


namespace megamol::infovis_gl {

class ScatterplotMatrixRenderer2D : public Renderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ScatterplotMatrixRenderer2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Scatterplot matrix renderer for generic tables.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    ScatterplotMatrixRenderer2D();

    /**
     * Finalises an instance.
     */
    ~ScatterplotMatrixRenderer2D() override;

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

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

private:
    enum ValueMapping {
        VALUE_MAPPING_KERNEL_BLEND = 0,
        VALUE_MAPPING_KERNEL_DENSITY,
        VALUE_MAPPING_WEIGHTED_KERNEL_DENSITY
    };
    enum GeometryType {
        GEOMETRY_TYPE_POINT = 0,
        GEOMETRY_TYPE_LINE,
        GEOMETRY_TYPE_TEXT,
        GEOMETRY_TYPE_TRIANGULATION,
        GEOMETRY_TYPE_POINT_TRIANGLE_SPRITE
    };
    enum KernelType { KERNEL_TYPE_BOX = 0, KERNEL_TYPE_GAUSSIAN };
    enum AxisMode { AXIS_MODE_NONE = 0, AXIS_MODE_MINIMALISTIC, AXIS_MODE_SCIENTIFIC };

    struct ParamState {
        std::optional<size_t> valueIdx;
        size_t labelIdx;
    };

    enum class BrushState {
        NOP,
        ADD,
        REMOVE,
    };

    struct MouseState {
        float x;
        float y;
        BrushState selector;
    };

    struct PlotInfo {
        GLuint indexX;
        GLuint indexY;
        GLfloat offsetX;
        GLfloat offsetY;
        GLfloat sizeX;
        GLfloat sizeY;
        GLfloat minX;
        GLfloat minY;
        GLfloat maxX;
        GLfloat maxY;
        GLfloat smallTickX;
        GLfloat smallTickY;
    };


    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    bool hasDirtyData() const;

    void resetDirtyData();

    bool hasDirtyScreen() const;

    void resetDirtyScreen();

    bool validate(mmstd_gl::CallRender2DGL& call, bool ignoreMVP);

    void updateColumns();

    void drawMinimalisticAxis(glm::mat4 ortho);

    void drawScientificAxis(glm::mat4 ortho);

    void bindMappingUniforms(std::unique_ptr<glowl::GLSLProgram>& shader);

    void bindFlagsAttribute();

    void drawPoints();

    void drawPointTriangleSprites();

    void drawLines();

    void validateTriangulation();

    void drawTriangulation();

    void validateText(glm::mat4 ortho);

    void drawText(glm::mat4 ortho);

    void drawPickIndicator();

    void drawMouseLabels(glm::mat4 ortho);

    void unbindScreen();

    void bindAndClearScreen();

    void drawScreen();

    void updateSelection();

    bool resetSelectionCallback(core::param::ParamSlot& caller);

    core::CallerSlot floatTableInSlot;

    core::CallerSlot transferFunctionInSlot;

    core::CallerSlot readFlagStorageSlot;

    core::CallerSlot writeFlagStorageSlot;

    core::param::ParamSlot valueMappingParam;

    core::param::ParamSlot valueSelectorParam;

    core::param::ParamSlot labelSelectorParam;

    core::param::ParamSlot labelSizeParam;

    core::param::ParamSlot geometryTypeParam;

    core::param::ParamSlot kernelWidthParam;

    core::param::ParamSlot kernelTypeParam;

    core::param::ParamSlot splitLinesByValueParam;

    core::param::ParamSlot lineConnectedValueSelectorParam;

    core::param::ParamSlot pickRadiusParam;

    core::param::ParamSlot pickColorParam;

    core::param::ParamSlot resetSelectionParam;

    core::param::ParamSlot drawPickIndicatorParam;

    core::param::ParamSlot drawMouseLabelsParam;

    core::param::ParamSlot triangulationSmoothnessParam;

    core::param::ParamSlot axisModeParam;

    core::param::ParamSlot axisColorParam;

    core::param::ParamSlot axisWidthParam;

    core::param::ParamSlot axisTicksParam;

    core::param::ParamSlot axisTicksRedundantParam;

    core::param::ParamSlot axisTickLengthParam;

    core::param::ParamSlot axisTickMarginParam;

    core::param::ParamSlot axisTickSizeParam;

    core::param::ParamSlot axisTickPrecisionX;

    core::param::ParamSlot axisTickPrecisionY;

    core::param::ParamSlot drawOuterLabelsParam;

    core::param::ParamSlot drawDiagonalLabelsParam;

    core::param::ParamSlot cellInvertYParam;

    core::param::ParamSlot cellSizeParam;

    core::param::ParamSlot cellMarginParam;

    core::param::ParamSlot cellNameSizeParam;

    core::param::ParamSlot outerXLabelMarginParam;

    core::param::ParamSlot outerYLabelMarginParam;

    core::param::ParamSlot alphaScalingParam;

    core::param::ParamSlot alphaAttenuateSubpixelParam;

    core::param::ParamSlot smoothFontParam;

    core::param::ParamSlot forceRedrawDebugParam;

    size_t dataHash;
    unsigned int dataTime;

    datatools::table::TableDataCall* floatTable;

    mmstd_gl::CallGetTransferFunctionGL* transferFunction;

    mmstd_gl::FlagCallRead_GL* readFlags;

    ParamState map;

    MouseState mouse;

    std::vector<PlotInfo> plots;

    core::BoundingBoxes_2 bounds;

    std::unique_ptr<glowl::GLSLProgram> minimalisticAxisShader;

    std::unique_ptr<glowl::GLSLProgram> scientificAxisShader;

    std::unique_ptr<glowl::GLSLProgram> pointShader;

    std::unique_ptr<glowl::GLSLProgram> pointTriangleSpriteShader;

    std::unique_ptr<glowl::GLSLProgram> lineShader;

    std::unique_ptr<glowl::GLSLProgram> triangleShader;

    std::unique_ptr<glowl::GLSLProgram> pickIndicatorShader;

    std::unique_ptr<glowl::GLSLProgram> screenShader;

    std::unique_ptr<glowl::GLSLProgram> pickProgram;

    GLint pickWorkgroupSize[3];
    GLint maxWorkgroupCount[3];

    core::utility::SSBOBufferArray plotSSBO;
    // GLsizeiptr plotDstOffset;
    // GLsizeiptr plotDstLength;

    core::utility::SSBOBufferArray valueSSBO;

    core::FlagStorageTypes::flag_version_type flagsBufferVersion;

    bool selectionNeedsUpdate = false;
    bool selectionNeedsReset = false;

    GLuint triangleVBO;
    GLuint triangleIBO;
    GLsizei triangleVertexCount;
    bool trianglesValid;

    std::optional<core::view::Camera> currentCamera;
    glm::ivec2 currentViewRes;

    std::unique_ptr<glowl::FramebufferObject> screenFBO;
    glm::mat4 screenLastMVP;
    GLint screenRestoreFBO;
    bool screenValid;

    megamol::core::utility::SDFFont axisFont;
    megamol::core::utility::SDFFont textFont;
    bool textValid;

    std::vector<::megamol::core::param::ParamSlot*> dataParams;
    std::vector<::megamol::core::param::ParamSlot*> screenParams;
};

} // namespace megamol::infovis_gl
