#ifndef MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagCall_GL.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/utility/SSBOBufferArray.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include <glowl/FramebufferObject.hpp>
#include <memory>
#include <optional>
#include "Renderer2D.h"
#include "mmcore/FlagStorage.h"

namespace megamol::infovis {

class ScatterplotMatrixRenderer2D : public Renderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "ScatterplotMatrixRenderer2D"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "Scatterplot matrix renderer for generic tables."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return true; }

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
    enum GeometryType { GEOMETRY_TYPE_POINT = 0, GEOMETRY_TYPE_LINE, GEOMETRY_TYPE_TEXT, GEOMETRY_TYPE_TRIANGULATION };
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
    bool Render(core::view::CallRender2D& call) override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core::view::CallRender2D& call) override;

    bool hasDirtyData() const;

    void resetDirtyData();

    bool hasDirtyScreen() const;

    void resetDirtyScreen();

    bool validate(core::view::CallRender2D& call, bool ignoreMVP);

    void updateColumns();

    void drawMinimalisticAxis();

    void drawScientificAxis();

    void bindMappingUniforms(vislib::graphics::gl::GLSLShader& shader);

    void bindFlagsAttribute();

    void drawPoints();

    void drawLines();

    void validateTriangulation();

    void drawTriangulation();

    void validateText();

    void drawText();

    void unbindScreen();

    void bindAndClearScreen();

    void drawScreen();

    void updateSelection();

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

    core::param::ParamSlot triangulationSmoothnessParam;

    core::param::ParamSlot axisModeParam;

    core::param::ParamSlot axisColorParam;

    core::param::ParamSlot axisWidthParam;

    core::param::ParamSlot axisTicksParam;

    core::param::ParamSlot axisTicksRedundantParam;

    core::param::ParamSlot axisTickLengthParam;

    core::param::ParamSlot axisTickSizeParam;

    core::param::ParamSlot cellSizeParam;

    core::param::ParamSlot cellMarginParam;

    core::param::ParamSlot cellNameSizeParam;

    core::param::ParamSlot alphaScalingParam;

    core::param::ParamSlot alphaAttenuateSubpixelParam;

    size_t dataHash;
    unsigned int dataTime;

    stdplugin::datatools::table::TableDataCall* floatTable;

    core::view::CallGetTransferFunction* transferFunction;

    core::FlagCallRead_GL* readFlags;

    ParamState map;

    MouseState mouse;

    std::vector<PlotInfo> plots;

    vislib::math::Rectangle<float> bounds;

    vislib::graphics::gl::GLSLShader minimalisticAxisShader;

    vislib::graphics::gl::GLSLShader scientificAxisShader;

    vislib::graphics::gl::GLSLShader pointShader;

    vislib::graphics::gl::GLSLGeometryShader lineShader;

    vislib::graphics::gl::GLSLShader triangleShader;

    vislib::graphics::gl::GLSLShader screenShader;

    vislib::graphics::gl::GLSLComputeShader pickProgram;

    GLint pickWorkgroupSize[3];
    GLint maxWorkgroupCount[3];

    core::utility::SSBOBufferArray plotSSBO;
    // GLsizeiptr plotDstOffset;
    // GLsizeiptr plotDstLength;

    core::utility::SSBOBufferArray valueSSBO;

    core::FlagStorage::FlagVersionType flagsBufferVersion;

    bool selectionNeedsUpdate = false;

    GLuint triangleVBO;
    GLuint triangleIBO;
    GLsizei triangleVertexCount;
    bool trianglesValid;

    std::unique_ptr<glowl::FramebufferObject> screenFBO;
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> screenLastMVP;
    GLint screenRestoreFBO;
    bool screenValid;

    megamol::core::utility::SDFFont axisFont;
    megamol::core::utility::SDFFont textFont;
    bool textValid;

    std::vector<::megamol::core::param::ParamSlot*> dataParams;
    std::vector<::megamol::core::param::ParamSlot*> screenParams;
};

} // namespace megamol::infovis

#endif // MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED
