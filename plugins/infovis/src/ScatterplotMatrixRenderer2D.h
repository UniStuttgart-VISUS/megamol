#ifndef MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/utility/SSBOStreamer.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include <glowl/FramebufferObject.hpp>
#include <memory>
#include <nanoflann.hpp>
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
        size_t valueIdx;
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


    struct SPLOMPoints {
        SPLOMPoints(const std::vector<PlotInfo>& plots, const stdplugin::datatools::table::TableDataCall* floatTable)
            : plots(plots), floatTable(floatTable) {}

        inline size_t idx_to_row(size_t idx) const {
            const size_t rowCount = floatTable->GetRowsCount();
            return idx % rowCount;
        }

        inline size_t kdtree_get_point_count() const { return floatTable->GetRowsCount() * plots.size(); }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            const size_t rowCount = floatTable->GetRowsCount();
            const size_t rowIdx = idx % rowCount;
            const size_t plotIdx = idx / rowCount;
            const PlotInfo& plot = this->plots[plotIdx];
            if (dim == 0) {
                const float xValue = this->floatTable->GetData(plot.indexX, rowIdx);
                const float xPos = (xValue - plot.minX) / (plot.maxX - plot.minX);
                return xPos * plot.sizeX + plot.offsetX;
            } else if (dim == 1) {
                const float yValue = this->floatTable->GetData(plot.indexY, rowIdx);
                const float yPos = (yValue - plot.minY) / (plot.maxY - plot.minY);
                return yPos * plot.sizeY + plot.offsetY;
            } else {
                assert(false && "Invalid dimension");
            }
        }

        template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }

    private:
        const std::vector<PlotInfo>& plots;
        const stdplugin::datatools::table::TableDataCall* floatTable;
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

    bool hasDirtyData(void) const;

    void resetDirtyData(void);

    bool hasDirtyScreen(void) const;

    void resetDirtyScreen(void);

    bool validate(core::view::CallRender2D& call);

    void updateColumns(void);

    void drawMinimalisticAxis(void);

    void drawScientificAxis(void);

    void bindMappingUniforms(vislib::graphics::gl::GLSLShader& shader);

    void bindFlagsAttribute();

    void drawPoints(void);

    void drawLines(void);

    void validateTriangulation();

    void drawTriangulation();

    void unbindScreen();

    void bindAndClearScreen();

    void drawScreen();

    void validateText();

    void drawText();

    void updateSelection();

    core::CallerSlot floatTableInSlot;

    core::CallerSlot transferFunctionInSlot;

    core::CallerSlot flagStorageInSlot;

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

    core::FlagCall* flagStorage;

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

    core::utility::SSBOStreamer plotSSBO;
    GLsizeiptr plotDstOffset;
    GLsizeiptr plotDstLength;

    core::utility::SSBOStreamer valueSSBO;

    GLuint flagsBuffer;
    core::FlagStorage::FlagVersionType flagsBufferVersion;

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

    typedef nanoflann::L2_Simple_Adaptor<float, SPLOMPoints> SPLOMDistance;
    typedef nanoflann::KDTreeSingleIndexAdaptor<SPLOMDistance, SPLOMPoints, 2> TreeIndex;

    std::unique_ptr<SPLOMPoints> indexPoints;
    std::unique_ptr<TreeIndex> index;
};

} // end namespace infovis
} // end namespace megamol

#endif // MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED
