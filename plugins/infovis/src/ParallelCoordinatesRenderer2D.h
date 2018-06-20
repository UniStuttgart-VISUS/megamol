#ifndef MEGAMOL_INFOVIS_PARCORENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_PARCORENDERER2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLTesselationShader.h"
//#include "vislib/graphics/gl/SimpleFont.h"
#include <map>
#include "mmcore/utility/SDFFont.h"

namespace megamol {
namespace infovis {

using namespace megamol::core;

class ParallelCoordinatesRenderer2D : public view::Renderer2DModule {
public:
    enum // draw mode
    { DRAW_DISCRETE = 0,
        DRAW_CONTINUOUS,
        DRAW_HISTOGRAM };

    enum // selection mode
    { SELECT_PICK = 0,
        SELECT_STROKE };

    struct DimensionFilter {
        uint32_t dimension; // useless but good padding
        float lower;
        float upper;
        uint32_t flags;
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName(void) { return "ParallelCoordinatesRenderer2D"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) { return "Parallel coordinates renderer for generic float tables"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) {
        // TODO unknown yet
        return true;
    }

    /**
     * Initialises a new instance.
     */
    ParallelCoordinatesRenderer2D(void);

    /**
     * Finalises an instance.
     */
    virtual ~ParallelCoordinatesRenderer2D(void);

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
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core::view::CallRender2D& call);

    virtual bool GetExtents(core::view::CallRender2D& call);

    virtual bool MouseEvent(float x, float y, ::megamol::core::view::MouseFlags flags);

    bool selectedItemsColorSlotCallback(::megamol::core::param::ParamSlot& caller);
    bool otherItemsColorSlotCallback(::megamol::core::param::ParamSlot& caller);
    bool axesColorSlotCallback(::megamol::core::param::ParamSlot& caller);
    bool filterIndicatorColorSlotCallback(::megamol::core::param::ParamSlot& caller);
    bool selectionIndicatorColorSlotCallback(::megamol::core::param::ParamSlot& caller);
    bool scalingChangedCallback(::megamol::core::param::ParamSlot& caller);
    bool resetFlagsSlotCallback(::megamol::core::param::ParamSlot& caller);
    bool resetFiltersSlotCallback(::megamol::core::param::ParamSlot& caller);

private:
    inline float relToAbsValue(int axis, float val) {
        return (val * (this->maximums[axis] - this->minimums[axis])) + this->minimums[axis];
    }

    void pickIndicator(float x, float y, int& axis, int& index);

    void assertData(void);

    void computeScaling(void);

    void drawAxes(void);

    void drawDiscrete(const float otherColor[4], const float selectedColor[4], float tfColorFactor);

    void drawItemsDiscrete(uint32_t testMask, uint32_t passMask, const float color[4], float tfColorFactor);

    void drawPickIndicator(float x, float y, float pickRadius, const float color[4]);

    void drawStrokeIndicator(float x0, float y0, float x1, float y1, const float color[4]);

    void drawItemsContinuous();

    void drawItemsHistogram();

    void doPicking(float x, float y, float pickRadius);

    void doStroking(float x0, float y0, float x1, float y1);

    void doFragmentCount();

    void drawParcos(void);

    int mouseXtoAxis(float x);

    bool makeProgram(std::string prefix, vislib::graphics::gl::GLSLShader& program);
    bool makeComputeProgram(std::string prefix, vislib::graphics::gl::GLSLComputeShader& program);
    bool makeTessellationProgram(std::string prefix, vislib::graphics::gl::GLSLTesselationShader& program);

    bool enableProgramAndBind(vislib::graphics::gl::GLSLShader& program);

    CallerSlot getDataSlot;

    CallerSlot getTFSlot;

    CallerSlot getFlagsSlot;

    size_t currentHash;

    ::vislib::graphics::gl::FramebufferObject densityFBO;

    float mousePressedX;
    float mousePressedY;
    float mouseReleasedX;
    float mouseReleasedY;
    float mouseX;
    float mouseY;
    ::megamol::core::view::MouseFlags mouseFlags;

    ::megamol::core::param::ParamSlot drawModeSlot;

    ::megamol::core::param::ParamSlot drawSelectedItemsSlot;
    ::megamol::core::param::ParamSlot selectedItemsColorSlot;
    ::megamol::core::param::ParamSlot selectedItemsAlphaSlot;
    float selectedItemsColor[4];

    ::megamol::core::param::ParamSlot drawOtherItemsSlot;
    ::megamol::core::param::ParamSlot otherItemsColorSlot;
    ::megamol::core::param::ParamSlot otherItemsAlphaSlot;
    float otherItemsColor[4];

    ::megamol::core::param::ParamSlot drawAxesSlot;
    ::megamol::core::param::ParamSlot axesColorSlot;
    float axesColor[4];

    ::megamol::core::param::ParamSlot filterIndicatorColorSlot;
    float filterIndicatorColor[4];


    ::megamol::core::param::ParamSlot selectionModeSlot;
    ::megamol::core::param::ParamSlot drawSelectionIndicatorSlot;
    ::megamol::core::param::ParamSlot selectionIndicatorColorSlot;
    float selectionIndicatorColor[4];

    ::megamol::core::param::ParamSlot pickRadiusSlot;

    ::megamol::core::param::ParamSlot scaleToFitSlot;
    //::megamol::core::param::ParamSlot scalingFactorSlot;
    //::megamol::core::param::ParamSlot scaleFullscreenSlot;
    //::megamol::core::param::ParamSlot projectionMatrixSlot;
    //::megamol::core::param::ParamSlot viewMatrixSlot;
    //::megamol::core::param::ParamSlot useCustomMatricesSlot;

    //::megamol::core::param::ParamSlot storeCamSlot;
    // bool storeCamSlotCallback(::megamol::core::param::ParamSlot & caller);

    ::megamol::core::param::ParamSlot glDepthTestSlot;
    ::megamol::core::param::ParamSlot glLineSmoothSlot;
    ::megamol::core::param::ParamSlot glLineWidthSlot;
    ::megamol::core::param::ParamSlot sqrtDensitySlot;

    ::megamol::core::param::ParamSlot resetFlagsSlot;
    ::megamol::core::param::ParamSlot resetFiltersSlot;

    float marginX, marginY;
    float axisDistance;
    float axisHeight;
    GLuint numTicks;
    float fontSize;
    float windowAspect;
    int windowWidth;
    int windowHeight;
    float backgroundColor[4];
    vislib::math::Rectangle<float> bounds;

    GLuint columnCount;
    GLuint itemCount;
    GLfloat modelViewMatrix_column[16];
    GLfloat projMatrix_column[16];

    vislib::graphics::gl::GLSLShader drawAxesProgram;
    vislib::graphics::gl::GLSLShader drawScalesProgram;
    vislib::graphics::gl::GLSLShader drawFilterIndicatorsProgram;
    vislib::graphics::gl::GLSLShader drawItemsDiscreteProgram;
    vislib::graphics::gl::GLSLTesselationShader drawItemsDiscreteTessProgram;
    vislib::graphics::gl::GLSLShader drawPickIndicatorProgram;
    vislib::graphics::gl::GLSLShader drawStrokeIndicatorProgram;

    vislib::graphics::gl::GLSLShader drawItemContinuousProgram;
    vislib::graphics::gl::GLSLShader drawItemsHistogramProgram;
    vislib::graphics::gl::GLSLShader traceItemsDiscreteProgram;

    vislib::graphics::gl::GLSLComputeShader filterProgram;
    vislib::graphics::gl::GLSLComputeShader minMaxProgram;

    vislib::graphics::gl::GLSLComputeShader pickProgram;
    vislib::graphics::gl::GLSLComputeShader strokeProgram;

    GLuint dataBuffer, flagsBuffer, minimumsBuffer, maximumsBuffer, axisIndirectionBuffer, filtersBuffer, minmaxBuffer;
    GLuint counterBuffer;

    std::vector<GLuint> axisIndirection;
    std::vector<GLfloat> minimums;
    std::vector<GLfloat> maximums;
    std::vector<DimensionFilter> filters;
    std::vector<GLuint> fragmentMinMax;
    std::vector<std::string> names;

    int pickedAxis;
    bool dragging;
    bool filtering;
    int pickedIndicatorAxis;
    // int lastPickedIndicatorAxis;
    int pickedIndicatorIndex;
    GLint maxAxes;
    GLint isoLinesPerInvocation;

    GLint filterWorkgroupSize[3];
    GLint counterWorkgroupSize[3];
    GLint pickWorkgroupSize[3];
    GLint strokeWorkgroupSize[3];
    GLint maxWorkgroupCount[3];

    megamol::core::utility::SDFFont font;
};

} /* end namespace infovis */
} /* end namespace megamol */

#endif /* MEGAMOL_INFOVIS_PARCORENDERER2D_H_INCLUDED */