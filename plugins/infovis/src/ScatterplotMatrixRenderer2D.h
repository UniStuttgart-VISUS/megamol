#ifndef MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/utility/SSBOStreamer.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include "FlagCall.h"
#include "Renderer2D.h"

namespace megamol {
namespace infovis {

class ScatterplotMatrixRenderer2D : public Renderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ScatterplotMatrixRenderer2D"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Scatterplot matrix renderer for generic tables."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /**
     * Initialises a new instance.
     */
    ScatterplotMatrixRenderer2D();

    /**
     * Finalises an instance.
     */
    virtual ~ScatterplotMatrixRenderer2D();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    virtual bool MouseEvent(float x, float y, core::view::MouseFlags flags);

private:
    enum GeometryType { GEOMETRY_TYPE_POINT = 0, GEOMETRY_TYPE_LINE, GEOMETRY_TYPE_TEXT, GEOMETRY_TYPE_TRIANGULATION };
    enum KernelType { KERNEL_TYPE_BOX = 0, KERNEL_TYPE_GAUSSIAN };
    enum AxisMode { AXIS_MODE_NONE = 0, AXIS_MODE_MINIMALISTIC, AXIS_MODE_SCIENTIFIC };

    struct ParamState {
        size_t colorIdx;
        size_t labelIdx;
    };

    struct MouseState {
        float x;
        float y;
        bool selects;
        bool inspects;
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
    virtual bool Render(core::view::CallRender2D& call);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::view::CallRender2D& call);

    bool isDirty(void) const;

    void resetDirty(void);

    bool validate(void);

    void updateColumns(void);

    void drawMinimalisticAxis(void);

    void drawScientificAxis(void);

    void drawPoints(void);

    void drawLines(void);

    void validateTriangulation();

    void drawTriangulation();

    void validateText();

    void drawText();

    int itemAt(const float x, const float y);

    core::CallerSlot floatTableInSlot;

    core::CallerSlot transferFunctionInSlot;

    core::CallerSlot flagStorageInSlot;

    core::param::ParamSlot colorSelectorParam;

    core::param::ParamSlot labelSelectorParam;

    core::param::ParamSlot labelSizeParam;

    core::param::ParamSlot geometryTypeParam;

    core::param::ParamSlot kernelWidthParam;

    core::param::ParamSlot kernelTypeParam;

    core::param::ParamSlot triangulationSelectorParam;

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

    stdplugin::datatools::table::TableDataCall* floatTable;

    core::view::CallGetTransferFunction* transferFunction;

    FlagCall* flagStorage;

    ParamState map;

    MouseState mouse;

    std::vector<PlotInfo> plots;

    vislib::math::Rectangle<float> bounds;

    megamol::core::utility::SDFFont axisFont;

    vislib::graphics::gl::GLSLShader minimalisticAxisShader;

    vislib::graphics::gl::GLSLShader scientificAxisShader;

    vislib::graphics::gl::GLSLShader pointShader;

    vislib::graphics::gl::GLSLGeometryShader lineShader;

    vislib::graphics::gl::GLSLShader triangleShader;

    core::utility::SSBOStreamer valueSSBO;

    core::utility::SSBOStreamer plotSSBO;
    GLsizeiptr plotDstOffset;
    GLsizeiptr plotDstLength;

    GLuint triangleVBO;
    GLuint triangleIBO;
    GLsizei triangleVertexCount;

    megamol::core::utility::SDFFont labelFont;
    bool trianglesValid;
    bool textValid;
};

} // end namespace infovis
} // end namespace megamol

#endif // MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED
