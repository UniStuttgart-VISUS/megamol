#ifndef MEGAMOL_INFOVIS_PARALLELCOORDINATESRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_PARALLELCOORDINATESRENDERER2D_H_INCLUDED

#include <map>

#include <glm/matrix.hpp>
#include <json.hpp>

#define GLOWL_OPENGL_INCLUDE_GLAD
#include <glowl/glowl.h>

#include "Renderer2D.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagStorage.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/view/CallRender2DGL.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmstd_datatools/table/TableDataCall.h"
#include "vislib/graphics/gl/FramebufferObject.h"

namespace megamol {
namespace infovis {

    class ParallelCoordinatesRenderer2D : public Renderer2D {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char* ClassName(void) {
            return "ParallelCoordinatesRenderer2D";
        }

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
        static inline bool IsAvailable(void) {
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
        virtual bool Render(core::view::CallRender2DGL& call);

        virtual bool GetExtents(core::view::CallRender2DGL& call);

        // virtual bool MouseEvent(float x, float y, core::view::MouseFlags flags);

        bool OnMouseButton(
            core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

        bool OnMouseMove(double x, double y) override;

        bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

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

        void assertData(core::view::CallRender2DGL& call);

        void computeScaling(void);

        void drawAxes(glm::mat4 ortho);

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

        bool enableProgramAndBind(std::unique_ptr<glowl::GLSLProgram>& program);

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

        core::param::ParamSlot drawOtherItemsSlot;
        core::param::ParamSlot otherItemsColorSlot;
        core::param::ParamSlot otherItemsAttribSlot;

        core::param::ParamSlot drawAxesSlot;
        core::param::ParamSlot axesColorSlot;

        core::param::ParamSlot filterIndicatorColorSlot;

        core::param::ParamSlot selectionModeSlot;
        core::param::ParamSlot drawSelectionIndicatorSlot;
        core::param::ParamSlot selectionIndicatorColorSlot;

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

        core::param::ParamSlot triangleModeSlot;
        core::param::ParamSlot lineThicknessSlot;
        core::param::ParamSlot axesLineThicknessSlot;

        float marginX, marginY;
        float axisDistance;
        float axisHeight;
        GLuint numTicks;
        float fontSize;
        float windowAspect;
        int windowWidth;
        int windowHeight;
        float backgroundColor[4];
        core::BoundingBoxes_2 bounds;
        unsigned int lastTimeStep;

        GLuint columnCount;
        GLuint itemCount;
        GLfloat modelViewMatrix_column[16];
        GLfloat projMatrix_column[16];

        std::unique_ptr<glowl::GLSLProgram> drawAxesProgram;
        std::unique_ptr<glowl::GLSLProgram> drawScalesProgram;
        std::unique_ptr<glowl::GLSLProgram> drawFilterIndicatorsProgram;
        std::unique_ptr<glowl::GLSLProgram> drawItemsDiscreteProgram;
        std::unique_ptr<glowl::GLSLProgram> drawItemsTriangleProgram;
        std::unique_ptr<glowl::GLSLProgram> drawItemsDiscreteTessProgram;
        std::unique_ptr<glowl::GLSLProgram> drawPickIndicatorProgram;
        std::unique_ptr<glowl::GLSLProgram> drawStrokeIndicatorProgram;

        std::unique_ptr<glowl::GLSLProgram> drawItemContinuousProgram;
        std::unique_ptr<glowl::GLSLProgram> drawItemsHistogramProgram;
        std::unique_ptr<glowl::GLSLProgram> traceItemsDiscreteProgram;

        std::unique_ptr<glowl::GLSLProgram> filterProgram;
        std::unique_ptr<glowl::GLSLProgram> minMaxProgram;

        std::unique_ptr<glowl::GLSLProgram> pickProgram;
        std::unique_ptr<glowl::GLSLProgram> strokeProgram;

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
        std::map<std::string, uint32_t> columnIndex;

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
