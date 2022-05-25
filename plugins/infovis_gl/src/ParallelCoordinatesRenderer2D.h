/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <unordered_map>

#include <glm/glm.hpp>
#include <glowl/glowl.h>

#include "Renderer2D.h"
#include "datatools/table/TableDataCall.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/flags/FlagStorage.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"

namespace megamol::infovis_gl {

class ParallelCoordinatesRenderer2D : public Renderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "ParallelCoordinatesRenderer2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Parallel coordinates renderer for generic tables.\n"
               "Left-Click to pick/stroke\npress [Shift] to filter axis using the two delimiters (hats)\n"
               "press [Alt] to re-order axes";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    ParallelCoordinatesRenderer2D();

    /**
     * Finalises an instance.
     */
    ~ParallelCoordinatesRenderer2D() override;

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
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

protected:
    enum DrawMode {
        DRAW_DISCRETE = 0,
        DRAW_DENSITY,
    };

    enum SelectionMode {
        SELECT_PICK = 0,
        SELECT_STROKE,
    };

    enum InteractionState {
        NONE = 0,
        INTERACTION_DRAG,
        INTERACTION_FILTER,
        INTERACTION_SELECT,
    };

    struct Range {
        float min;
        float max;
    };

    bool assertData(mmstd_gl::CallRender2DGL& call);

    void calcSizes();

    int mouseXtoAxis(float x);

    void mouseToFilterIndicator(float x, float y, int& axis, int& index);

    bool useProgramAndBindCommon(std::unique_ptr<glowl::GLSLProgram> const& program);

    void doPicking(glm::vec2 pos, float pickRadius);

    void doStroking(glm::vec2 start, glm::vec2 end);

    void drawItemLines(uint32_t testMask, uint32_t passMask, bool useTf, glm::vec4 const& color);

    void drawDiscrete(bool useTf, glm::vec4 const& color, glm::vec4 selectedColor);

    void drawDensity(std::shared_ptr<glowl::FramebufferObject> const& fbo);

    void drawAxes(glm::mat4 ortho);

    void drawIndicatorPick(glm::vec2 pos, float pickRadius, glm::vec4 const& color);

    void drawIndicatorStroke(glm::vec2 start, glm::vec2 end, glm::vec4 const& color);

    void storeFilters();

    void loadFilters();

    // Slots
    core::CallerSlot dataSlot_;
    core::CallerSlot tfSlot_;
    core::CallerSlot readFlagsSlot_;
    core::CallerSlot writeFlagsSlot_;

    // Params
    core::param::ParamSlot drawModeParam_;
    core::param::ParamSlot normalizeDensityParam_;
    core::param::ParamSlot sqrtDensityParam_;
    core::param::ParamSlot triangleModeParam_;
    core::param::ParamSlot lineWidthParam_;
    core::param::ParamSlot dimensionNameParam_;
    core::param::ParamSlot useLineWidthInPixelsParam_;
    core::param::ParamSlot drawItemsParam_;
    core::param::ParamSlot drawSelectedItemsParam_;
    core::param::ParamSlot ignoreTransferFunctionParam_;
    core::param::ParamSlot itemsColorParam_;
    core::param::ParamSlot selectedItemsColorParam_;
    core::param::ParamSlot drawAxesParam_;
    core::param::ParamSlot axesLineWidthParam_;
    core::param::ParamSlot axesColorParam_;
    core::param::ParamSlot filterIndicatorColorParam_;
    core::param::ParamSlot smoothFontParam_;
    core::param::ParamSlot selectionModeParam_;
    core::param::ParamSlot pickRadiusParam_;
    core::param::ParamSlot drawSelectionIndicatorParam_;
    core::param::ParamSlot selectionIndicatorColorParam_;
    core::param::ParamSlot scaleToFitParam_;
    core::param::ParamSlot resetFiltersParam_;
    core::param::ParamSlot filterStateParam_;

    // Data Info
    std::size_t currentTableDataHash_;
    unsigned int currentTableFrameId_;

    std::size_t dimensionCount_;
    std::size_t itemCount_;
    std::vector<std::string> names_;
    std::vector<Range> dimensionRanges_;
    std::vector<int> axisIndirection_;
    std::unordered_map<std::string, int> dimensionIndex_;
    std::vector<Range> filters_;
    const std::array<uint32_t, 2> densityMinMaxInit_;

    std::unique_ptr<glowl::BufferObject> dataBuffer_;
    std::unique_ptr<glowl::BufferObject> dimensionRangesBuffer_;
    std::unique_ptr<glowl::BufferObject> axisIndirectionBuffer_;
    std::unique_ptr<glowl::BufferObject> filtersBuffer_;
    std::unique_ptr<glowl::BufferObject> densityMinMaxBuffer_;

    // Layout
    float marginX_;
    float marginY_;
    float axisDistance_;
    float axisHeight_;
    int numTicks_;
    float tickLength_;
    float fontSize_;
    megamol::core::utility::SDFFont font_;
    core::BoundingBoxes_2 bounds_;

    // Interaction state
    float mouseX_;
    float mouseY_;
    InteractionState interactionState_;
    int pickedAxis_;
    int pickedIndicatorAxis_;
    int pickedIndicatorIndex_;
    glm::vec2 strokeStart_;
    glm::vec2 strokeEnd_;
    bool needAxisUpdate_;
    bool needFilterUpdate_;
    bool needSelectionUpdate_;
    bool needFlagsUpdate_;

    // OpenGL
    std::unique_ptr<glowl::GLSLProgram> filterProgram_;
    std::unique_ptr<glowl::GLSLProgram> selectPickProgram_;
    std::unique_ptr<glowl::GLSLProgram> selectStrokeProgram_;
    std::unique_ptr<glowl::GLSLProgram> densityMinMaxProgram_;

    std::unique_ptr<glowl::GLSLProgram> drawItemsLineProgram_;
    std::unique_ptr<glowl::GLSLProgram> drawItemsTriangleProgram_;
    std::unique_ptr<glowl::GLSLProgram> drawItemsDensityProgram_;
    std::unique_ptr<glowl::GLSLProgram> drawAxesProgram_;
    std::unique_ptr<glowl::GLSLProgram> drawIndicatorPickProgram_;
    std::unique_ptr<glowl::GLSLProgram> drawIndicatorStrokeProgram_;

    std::array<GLint, 3> filterWorkgroupSize_;
    std::array<GLint, 3> selectPickWorkgroupSize_;
    std::array<GLint, 3> selectStrokeWorkgroupSize_;
    std::array<GLint, 3> densityMinMaxWorkgroupSize_;

    std::array<GLint, 3> maxWorkgroupCount_;

    std::unique_ptr<glowl::FramebufferObject> densityFbo_;

    // View and camera state
    std::optional<core::view::Camera> cameraCopy_;
    glm::ivec2 viewRes_;
};

} // namespace megamol::infovis_gl
