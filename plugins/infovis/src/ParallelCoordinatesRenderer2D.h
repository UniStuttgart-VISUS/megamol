#ifndef MEGAMOL_INFOVIS_PARALLELCOORDINATESRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_PARALLELCOORDINATESRENDERER2D_H_INCLUDED

#include "json.hpp"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagStorage.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include "vislib/graphics/gl/FramebufferObject.h"

#include "Renderer2D.h"

#include <map>

namespace megamol {
namespace infovis {

class ParallelCoordinatesRenderer2D : public Renderer2D {
public:
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
    static inline const char* Description(void) {
        return "Parallel coordinates renderer for generic tables.\n"
               "Left-Click to pick/stroke\npress [Shift] to filter axis using the two delimiters (hats)\n"
               "press [Alt] to re-order axes";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) { return true; }

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

    // virtual bool MouseEvent(float x, float y, core::view::MouseFlags flags);

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

    bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    bool selectedItemsColorSlotCallback(core::param::ParamSlot& caller);
    bool otherItemsColorSlotCallback(core::param::ParamSlot& caller);
    bool axesColorSlotCallback(core::param::ParamSlot& caller);
    bool filterIndicatorColorSlotCallback(core::param::ParamSlot& caller);
    bool selectionIndicatorColorSlotCallback(core::param::ParamSlot& caller);
    bool scalingChangedCallback(core::param::ParamSlot& caller);
    bool resetFlagsSlotCallback(core::param::ParamSlot& caller);
    bool resetFiltersSlotCallback(core::param::ParamSlot& caller);

private:
    enum DrawMode { DRAW_DISCRETE = 0, DRAW_CONTINUOUS, DRAW_HISTOGRAM };

    enum SelectionMode { SELECT_PICK = 0, SELECT_STROKE };

    enum InteractionState { NONE = 0, INTERACTION_DRAG, INTERACTION_FILTER, INTERACTION_SELECT };

    struct DimensionFilter {
        uint32_t dimension; // useless but good padding
        float lower;
        float upper;
        uint32_t flags;

        static inline void to_json(nlohmann::json& j, const DimensionFilter& d) {
            j = nlohmann::json{{"dim", d.dimension}, {"lower", d.lower}, {"upper", d.upper}, {"flags", d.flags}};
        }

        static inline void from_json(const nlohmann::json& j, DimensionFilter& d) {
            j.at("dim").get_to(d.dimension);
            j.at("lower").get_to(d.lower);
            j.at("upper").get_to(d.upper);
            j.at("flags").get_to(d.flags);
        }
    };

    inline float relToAbsValue(int axis, float val) {
        return (val * (this->maximums[axis] - this->minimums[axis])) + this->minimums[axis];
    }

    void pickIndicator(float x, float y, int& axis, int& index);

    void assertData(core::view::CallRender2D& call);

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
    void store_filters();
    void load_filters();

    int mouseXtoAxis(float x);

    bool enableProgramAndBind(vislib::graphics::gl::GLSLShader& program);

    core::CallerSlot getDataSlot;

    core::CallerSlot getTFSlot;

    core::CallerSlot readFlagsSlot;

    core::CallerSlot writeFlagsSlot;

    size_t currentHash;

    core::FlagStorage::FlagVersionType currentFlagsVersion;

    ::vislib::graphics::gl::FramebufferObject densityFBO;

    core::param::ParamSlot drawModeSlot;

    core::param::ParamSlot drawSelectedItemsSlot;
    core::param::ParamSlot selectedItemsColorSlot;
    core::param::ParamSlot selectedItemsAlphaSlot;
    float selectedItemsColor[4];

    core::param::ParamSlot drawOtherItemsSlot;
    core::param::ParamSlot otherItemsColorSlot;
    core::param::ParamSlot otherItemsAlphaSlot;
    float otherItemsColor[4];

    core::param::ParamSlot drawAxesSlot;
    core::param::ParamSlot axesColorSlot;
    float axesColor[4];

    core::param::ParamSlot filterIndicatorColorSlot;
    float filterIndicatorColor[4];


    core::param::ParamSlot selectionModeSlot;
    core::param::ParamSlot drawSelectionIndicatorSlot;
    core::param::ParamSlot selectionIndicatorColorSlot;
    float selectionIndicatorColor[4];

    core::param::ParamSlot pickRadiusSlot;

    core::param::ParamSlot scaleToFitSlot;
    // core::param::ParamSlot scalingFactorSlot;
    // core::param::ParamSlot scaleFullscreenSlot;
    // core::param::ParamSlot projectionMatrixSlot;
    // core::param::ParamSlot viewMatrixSlot;
    // core::param::ParamSlot useCustomMatricesSlot;

    // core::param::ParamSlot storeCamSlot;
    // bool storeCamSlotCallback(core::param::ParamSlot & caller);

    core::param::ParamSlot glDepthTestSlot;
    core::param::ParamSlot glLineSmoothSlot;
    core::param::ParamSlot glLineWidthSlot;
    core::param::ParamSlot sqrtDensitySlot;

    // core::param::ParamSlot resetFlagsSlot;
    core::param::ParamSlot resetFiltersSlot;

    core::param::ParamSlot filterStateSlot;

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
    unsigned int lastTimeStep;

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

    GLuint dataBuffer, minimumsBuffer, maximumsBuffer, axisIndirectionBuffer, filtersBuffer, minmaxBuffer;
    GLuint counterBuffer;

    std::vector<GLuint> axisIndirection;
    std::vector<GLfloat> minimums;
    std::vector<GLfloat> maximums;
    std::vector<DimensionFilter> filters;
    std::vector<GLuint> fragmentMinMax;
    std::vector<std::string> names;

    float mouseX;
    float mouseY;
    bool ctrlDown = false, altDown = false, shiftDown = false;
    InteractionState interactionState;
    int pickedAxis;
    int pickedIndicatorAxis;
    int pickedIndicatorIndex;
    float strokeStartX;
    float strokeStartY;
    float strokeEndX;
    float strokeEndY;
    bool needSelectionUpdate;
    bool needFlagsUpdate;

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

#endif /* MEGAMOL_INFOVIS_PARALLELCOORDINATESRENDERER2D_H_INCLUDED */