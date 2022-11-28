/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#include "ParallelCoordinatesRenderer2D.h"

#include <algorithm>
#include <array>

#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>

#include "datatools/table/TableDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/flags/FlagCallsGL.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"

using namespace megamol;
using namespace megamol::infovis_gl;
using namespace megamol::datatools;
using megamol::core::utility::log::Log;

ParallelCoordinatesRenderer2D::ParallelCoordinatesRenderer2D()
        : Renderer2D()
        , dataSlot_("getData", "Float table input")
        , tfSlot_("getTransferFunction", "Transfer function input")
        , readFlagsSlot_("readFlagStorage", "Flag storage read input")
        , writeFlagsSlot_("writeFlagStorage", "Flag storage write input")
        , drawModeParam_("drawMode", "Draw mode")
        , normalizeDensityParam_("normalizeDensity", "Normalize the range of the density values to [0, 1]")
        , sqrtDensityParam_("sqrtDensity", "Map root of density to transfer function (instead of linear mapping)")
        , triangleModeParam_("triangleMode", "Draw items with triangle lines")
        , lineWidthParam_("lineWidth", "Line width of data points")
        , dimensionNameParam_("dimensionName", "Dimension name of the attribute to use for TF lookup and item coloring")
        , useLineWidthInPixelsParam_("ui::useLineWidthInPixels", "Line width is given in pixel size.")
        , drawItemsParam_("ui::drawItems", "Draw (non-selected) items")
        , drawSelectedItemsParam_("ui::drawSelectedItems", "Draw selected items")
        , ignoreTransferFunctionParam_("ui::ignoreTransferFunction", "Use static color instead of TF color lookup")
        , itemsColorParam_("ui::itemsColor", "Color for (non-selected) items")
        , selectedItemsColorParam_("ui::selectedItemsColor", "Color for selected items")
        , drawAxesParam_("ui::drawAxes", "Draw dimension axes")
        , axesLineWidthParam_("ui::axesLineWidth", "Width of the axes and indicators lines")
        , axesColorParam_("ui::axesColor", "Color for axes lines and text")
        , filterIndicatorColorParam_("ui::filterIndicatorColor", "Color for filter indicators")
        , smoothFontParam_("ui::smoothFont", "Font rendering with smooth edges")
        , selectionModeParam_("ui::selectionMode", "Selection mode")
        , pickRadiusParam_("ui::pickRadius", "Picking radius in object-space")
        , drawSelectionIndicatorParam_("ui::drawSelectionIndicator", "Draw selection indicator")
        , selectionIndicatorColorParam_("ui::selectionIndicatorColor", "Color for selection indicator")
        , scaleToFitParam_("ui::scaleToFit", "fit the diagram in the viewport")
        , resetFiltersParam_("filter::resetFilters", "Reset dimension filters to initial state")
        , filterStateParam_("filter::filterState", "stores filter state for serialization")
        , currentTableDataHash_(std::numeric_limits<std::size_t>::max())
        , currentTableFrameId_(std::numeric_limits<unsigned int>::max())
        , dimensionCount_(0)
        , itemCount_(0)
        , densityMinMaxInit_{std::numeric_limits<uint32_t>::max(), 0}
        , marginX_(0.0f)
        , marginY_(0.0f)
        , axisDistance_(0.0f)
        , axisHeight_(0.0f)
        , numTicks_(5)
        , tickLength_(4.0f)
        , fontSize_(1.0f)
        , font_(core::utility::SDFFont::PRESET_EVOLVENTA_SANS, core::utility::SDFFont::RENDERMODE_FILL)
        , mouseX_(0.0f)
        , mouseY_(0.0f)
        , interactionState_(InteractionState::NONE)
        , pickedAxis_(-1)
        , pickedIndicatorAxis_(-1)
        , pickedIndicatorIndex_(-1)
        , strokeStart_(glm::vec2(0.0f))
        , strokeEnd_(glm::vec2(0.0f))
        , needAxisUpdate_(false)
        , needFilterUpdate_(false)
        , needSelectionUpdate_(false)
        , needFlagsUpdate_(false)
        , filterWorkgroupSize_()
        , selectPickWorkgroupSize_()
        , selectStrokeWorkgroupSize_()
        , densityMinMaxWorkgroupSize_()
        , maxWorkgroupCount_()
        , cameraCopy_(std::nullopt)
        , viewRes_(glm::ivec2(1, 1)) {

    dataSlot_.SetCompatibleCall<table::TableDataCallDescription>();
    MakeSlotAvailable(&dataSlot_);

    tfSlot_.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    MakeSlotAvailable(&tfSlot_);

    readFlagsSlot_.SetCompatibleCall<mmstd_gl::FlagCallRead_GLDescription>();
    MakeSlotAvailable(&readFlagsSlot_);

    writeFlagsSlot_.SetCompatibleCall<mmstd_gl::FlagCallWrite_GLDescription>();
    MakeSlotAvailable(&writeFlagsSlot_);

    auto drawModes = new core::param::EnumParam(DRAW_DISCRETE);
    drawModes->SetTypePair(DRAW_DISCRETE, "Kernel Blending");
    drawModes->SetTypePair(DRAW_DENSITY, "Kernel Density Estimation");
    drawModeParam_.SetParameter(drawModes);
    MakeSlotAvailable(&drawModeParam_);

    normalizeDensityParam_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&normalizeDensityParam_);

    sqrtDensityParam_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&sqrtDensityParam_);

    triangleModeParam_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&triangleModeParam_);

    lineWidthParam_ << new core::param::FloatParam(1.5f, 0.0f);
    MakeSlotAvailable(&lineWidthParam_);

    dimensionNameParam_ << new core::param::FlexEnumParam("[none]");
    MakeSlotAvailable(&dimensionNameParam_);

    useLineWidthInPixelsParam_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&useLineWidthInPixelsParam_);

    drawItemsParam_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&drawItemsParam_);

    drawSelectedItemsParam_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&drawSelectedItemsParam_);

    ignoreTransferFunctionParam_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&ignoreTransferFunctionParam_);

    itemsColorParam_ << new core::param::ColorParam("gray");
    MakeSlotAvailable(&itemsColorParam_);

    selectedItemsColorParam_ << new core::param::ColorParam("red");
    MakeSlotAvailable(&selectedItemsColorParam_);

    drawAxesParam_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&drawAxesParam_);

    axesLineWidthParam_ << new core::param::FloatParam(2.0f);
    MakeSlotAvailable(&axesLineWidthParam_);

    axesColorParam_ << new core::param::ColorParam("white");
    MakeSlotAvailable(&axesColorParam_);

    filterIndicatorColorParam_ << new core::param::ColorParam("orange");
    MakeSlotAvailable(&filterIndicatorColorParam_);

    smoothFontParam_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&smoothFontParam_);

    auto pickModes = new core::param::EnumParam(SELECT_STROKE);
    pickModes->SetTypePair(SELECT_PICK, "Pick");
    pickModes->SetTypePair(SELECT_STROKE, "Stroke");
    selectionModeParam_.SetParameter(pickModes);
    MakeSlotAvailable(&selectionModeParam_);

    pickRadiusParam_ << new core::param::FloatParam(1.0f, 0.01f, 10.0f);
    MakeSlotAvailable(&pickRadiusParam_);

    drawSelectionIndicatorParam_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&drawSelectionIndicatorParam_);

    selectionIndicatorColorParam_ << new core::param::ColorParam("MegaMolBlue");
    MakeSlotAvailable(&selectionIndicatorColorParam_);

    scaleToFitParam_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&scaleToFitParam_);

    resetFiltersParam_ << new core::param::ButtonParam();
    MakeSlotAvailable(&resetFiltersParam_);

    filterStateParam_ << new ::core::param::StringParam("");
    MakeSlotAvailable(&filterStateParam_);
}

ParallelCoordinatesRenderer2D::~ParallelCoordinatesRenderer2D() {
    Release();
}

bool ParallelCoordinatesRenderer2D::create() {
    if (!font_.Initialise(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>())) {
        return false;
    }
    font_.SetBatchDrawMode(true);

    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        filterProgram_ =
            core::utility::make_glowl_shader("pc_filter", shader_options, "infovis_gl/pc/filter.comp.glsl");

        selectPickProgram_ =
            core::utility::make_glowl_shader("pc_select_pick", shader_options, "infovis_gl/pc/select.comp.glsl");

        auto shader_options_stroke = shader_options;
        shader_options_stroke.addDefinition("STROKE");
        selectStrokeProgram_ = core::utility::make_glowl_shader(
            "pc_select_stroke", shader_options_stroke, "infovis_gl/pc/select.comp.glsl");

        densityMinMaxProgram_ = core::utility::make_glowl_shader(
            "pc_density_min_max", shader_options, "infovis_gl/pc/density_min_max.comp.glsl");

        drawItemsLineProgram_ = core::utility::make_glowl_shader(
            "pc_items_line", shader_options, "infovis_gl/pc/items.vert.glsl", "infovis_gl/pc/items.frag.glsl");

        auto shader_options_triangles = shader_options;
        shader_options_triangles.addDefinition("TRIANGLES");
        drawItemsTriangleProgram_ = core::utility::make_glowl_shader("pc_items_triangle", shader_options_triangles,
            "infovis_gl/pc/items.vert.glsl", "infovis_gl/pc/items.frag.glsl");

        drawItemsDensityProgram_ = core::utility::make_glowl_shader(
            "pc_items_density", shader_options, "infovis_gl/pc/density.vert.glsl", "infovis_gl/pc/density.frag.glsl");

        drawAxesProgram_ = core::utility::make_glowl_shader(
            "pc_axes", shader_options, "infovis_gl/pc/axes.vert.glsl", "infovis_gl/pc/axes.frag.glsl");

        drawIndicatorPickProgram_ = core::utility::make_glowl_shader("pc_indicator_pick", shader_options,
            "infovis_gl/pc/indicator_pick.vert.glsl", "infovis_gl/pc/indicator_pick.frag.glsl");

        drawIndicatorStrokeProgram_ = core::utility::make_glowl_shader("pc_indicator_stroke", shader_options,
            "infovis_gl/pc/indicator_stroke.vert.glsl", "infovis_gl/pc/indicator_stroke.frag.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("ParallelCoordinatesRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    glGetProgramiv(filterProgram_->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, filterWorkgroupSize_.data());
    glGetProgramiv(selectPickProgram_->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, selectPickWorkgroupSize_.data());
    glGetProgramiv(selectStrokeProgram_->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, selectStrokeWorkgroupSize_.data());
    glGetProgramiv(densityMinMaxProgram_->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, densityMinMaxWorkgroupSize_.data());

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkgroupCount_[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkgroupCount_[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkgroupCount_[2]);

    return true;
}

void ParallelCoordinatesRenderer2D::release() {}

bool ParallelCoordinatesRenderer2D::GetExtents(mmstd_gl::CallRender2DGL& call) {
    if (!assertData(call)) {
        return false;
    }

    call.AccessBoundingBoxes() = bounds_;

    return true;
}

bool ParallelCoordinatesRenderer2D::Render(mmstd_gl::CallRender2DGL& call) {
    // This check must be first. GetExtent does the same check and we need to be sure, that the outside world has seen
    // an extent before we continue. Otherwise, i.e. the view has not initialized the camera.
    if (!assertData(call)) {
        return false;
    }

    if (needAxisUpdate_) {
        needAxisUpdate_ = false;
        axisIndirectionBuffer_->rebuffer(axisIndirection_);
    }

    if (needFilterUpdate_) {
        needFilterUpdate_ = false;
        filtersBuffer_->rebuffer(filters_);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        storeFilters();

        useProgramAndBindCommon(filterProgram_);

        std::array<GLuint, 3> groupCounts{};
        computeDispatchSizes(itemCount_, filterWorkgroupSize_, maxWorkgroupCount_, groupCounts);
        glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        needFlagsUpdate_ = true;
    }

    // Do stroking/picking
    if (needSelectionUpdate_) {
        needSelectionUpdate_ = false;

        switch (selectionModeParam_.Param<core::param::EnumParam>()->Value()) {
        case SELECT_STROKE:
            doStroking(strokeStart_, strokeEnd_);
            break;
        case SELECT_PICK:
            doPicking(strokeEnd_, pickRadiusParam_.Param<megamol::core::param::FloatParam>()->Value());
            break;
        }

        needFlagsUpdate_ = true;
    }

    if (needFlagsUpdate_) {
        needFlagsUpdate_ = false;

        auto readFlagsCall = readFlagsSlot_.CallAs<mmstd_gl::FlagCallRead_GL>();
        auto writeFlagsCall = writeFlagsSlot_.CallAs<mmstd_gl::FlagCallWrite_GL>();
        if (readFlagsCall != nullptr && writeFlagsCall != nullptr) {
            writeFlagsCall->setData(readFlagsCall->getData(), readFlagsCall->version() + 1);
            (*writeFlagsCall)(mmstd_gl::FlagCallWrite_GL::CallGetData);
        }
    }


    // get camera
    core::view::Camera cam = call.GetCamera();
    cameraCopy_ = cam;
    viewRes_ = call.GetViewResolution();

    const auto viewMx = cam.getViewMatrix();
    const auto projMx = cam.getProjectionMatrix();
    const glm::mat4 orthoMx = projMx * viewMx;

    // Draw
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // Draw items
    const auto& drawmode = drawModeParam_.Param<core::param::EnumParam>()->Value();
    switch (drawmode) {
    case DRAW_DISCRETE:
        drawDiscrete(!ignoreTransferFunctionParam_.Param<core::param::BoolParam>()->Value(),
            glm::make_vec4(itemsColorParam_.Param<core::param::ColorParam>()->Value().data()),
            glm::make_vec4(selectedItemsColorParam_.Param<core::param::ColorParam>()->Value().data()));
        break;
    case DRAW_DENSITY:
        drawDensity(call.GetFramebuffer());
        break;
    }

    // Draw stroking/picking indicator
    if (drawSelectionIndicatorParam_.Param<core::param::BoolParam>()->Value()) {
        const glm::vec4 indicatorColor =
            glm::make_vec4<float>(selectionIndicatorColorParam_.Param<core::param::ColorParam>()->Value().data());
        switch (selectionModeParam_.Param<core::param::EnumParam>()->Value()) {
        case SELECT_PICK:
            drawIndicatorPick(glm::vec2(mouseX_, mouseY_),
                pickRadiusParam_.Param<megamol::core::param::FloatParam>()->Value(), indicatorColor);
            break;
        case SELECT_STROKE:
            if (interactionState_ == InteractionState::INTERACTION_SELECT) {
                drawIndicatorStroke(strokeStart_, strokeEnd_, indicatorColor);
            }
            break;
        }
    }

    if (drawAxesParam_.Param<core::param::BoolParam>()->Value()) {
        drawAxes(orthoMx);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    return true;
}

bool ParallelCoordinatesRenderer2D::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    // Ignore everything which is not left mouse button.
    if (button != core::view::MouseButton::BUTTON_LEFT) {
        return false;
    }

    // Any up/down event stops interaction, but only down changes selection.
    interactionState_ = InteractionState::NONE;

    // If control is pressed, event is meant for view. Nevertheless, the interaction state is set to none before this
    // check, to avoid locking into a state in case CTRL is pressed while an interaction state is active.
    if (mods.test(core::view::Modifier::CTRL)) {
        return false;
    }

    if (action == core::view::MouseButtonAction::PRESS) {
        pickedAxis_ = -1;
        pickedIndicatorAxis_ = -1;
        pickedIndicatorIndex_ = -1;

        if (mods.test(core::view::Modifier::ALT)) {
            pickedAxis_ = mouseXtoAxis(mouseX_);
            if (pickedAxis_ != -1) {
                interactionState_ = InteractionState::INTERACTION_DRAG;
            }
            return true;
        }

        if (mods.test(core::view::Modifier::SHIFT)) {
            const auto axis = mouseXtoAxis(mouseX_);
            if (axis != -1) {
                const float base = marginY_ * 0.5f - fontSize_ * 0.5f;
                if ((mouseY_ > base && mouseY_ < base + fontSize_) ||
                    (mouseY_ > base + marginY_ + axisHeight_ && mouseY_ < base + marginY_ + axisHeight_ + fontSize_)) {
                    std::swap(filters_[axis].min, filters_[axis].max);
                    needFilterUpdate_ = true;
                    return true;
                }
            }

            mouseToFilterIndicator(mouseX_, mouseY_, pickedIndicatorAxis_, pickedIndicatorIndex_);
            if (pickedIndicatorAxis_ != -1) {
                interactionState_ = InteractionState::INTERACTION_FILTER;
            }
            return true;
        }

        interactionState_ = InteractionState::INTERACTION_SELECT;
        strokeStart_ = glm::vec2(mouseX_, mouseY_);
        strokeEnd_ = glm::vec2(mouseX_, mouseY_);
        needSelectionUpdate_ = true;
        return true;
    }

    return false;
}

bool ParallelCoordinatesRenderer2D::OnMouseMove(double x, double y) {
    // mouseCoordsToWorld requires a valid camera and cameraCopy_ is initialized on first render. Before anything is
    // draw, interaction probably is not needed anyway, but this event could be triggered independently.
    if (!cameraCopy_.has_value()) {
        return false;
    }

    auto const& [world_x, world_y] = mouseCoordsToWorld(x, y, cameraCopy_.value(), viewRes_.x, viewRes_.y);

    mouseX_ = static_cast<float>(world_x);
    mouseY_ = static_cast<float>(world_y);

    if (interactionState_ == InteractionState::INTERACTION_DRAG) {
        int currAxis = mouseXtoAxis(mouseX_);
        if (currAxis != pickedAxis_ && currAxis >= 0 && currAxis < static_cast<int>(dimensionCount_)) {
            auto pickedAxisIt = std::find(axisIndirection_.begin(), axisIndirection_.end(), pickedAxis_);
            auto pickedIdx = std::distance(axisIndirection_.begin(), pickedAxisIt);
            axisIndirection_.erase(pickedAxisIt);

            auto currAxisIt = std::find(axisIndirection_.begin(), axisIndirection_.end(), currAxis);
            auto currIdx = std::distance(axisIndirection_.begin(), currAxisIt);
            if (pickedIdx <= currIdx) {
                currAxisIt++;
            }
            axisIndirection_.insert(currAxisIt, pickedAxis_);

            needAxisUpdate_ = true;
        }

        return true;
    }

    if (interactionState_ == InteractionState::INTERACTION_FILTER) {
        const auto& range = dimensionRanges_[pickedIndicatorAxis_];
        float val = ((mouseY_ - marginY_) / axisHeight_) * (range.max - range.min) + range.min;
        val = std::clamp(val, range.min, range.max);
        if (pickedIndicatorIndex_ == 0) {
            filters_[pickedIndicatorAxis_].min = val;
        } else {
            filters_[pickedIndicatorAxis_].max = val;
        }
        needFilterUpdate_ = true;

        return true;
    }

    if (interactionState_ == InteractionState::INTERACTION_SELECT) {
        if (mouseX_ != strokeEnd_.x || mouseY_ != strokeEnd_.y) {
            strokeEnd_.x = mouseX_;
            strokeEnd_.y = mouseY_;
            needSelectionUpdate_ = true;
        }

        return true;
    }

    return false;
}

bool ParallelCoordinatesRenderer2D::assertData(mmstd_gl::CallRender2DGL& call) {
    auto floatTableCall = dataSlot_.CallAs<megamol::datatools::table::TableDataCall>();
    if (floatTableCall == nullptr) {
        return false;
    }
    auto tfCall = tfSlot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    if (tfCall == nullptr) {
        Log::DefaultLog.WriteError("ParallelCoordinatesRenderer2D requires a transfer function!");
        return false;
    }
    auto readFlagsCall = readFlagsSlot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    if (readFlagsCall == nullptr) {
        Log::DefaultLog.WriteError("ParallelCoordinatesRenderer2D requires a read flag storage!");
        return false;
    }
    auto writeFlagsCall = writeFlagsSlot_.CallAs<mmstd_gl::FlagCallWrite_GL>();
    if (writeFlagsCall == nullptr) {
        Log::DefaultLog.WriteError("ParallelCoordinatesRenderer2D requires a write flag storage!");
        return false;
    }

    floatTableCall->SetFrameID(static_cast<unsigned int>(call.Time()));
    (*floatTableCall)(1);
    (*floatTableCall)(0);
    call.SetTimeFramesCount(floatTableCall->GetFrameCount());
    const auto hash = floatTableCall->DataHash();
    const auto frameId = floatTableCall->GetFrameID();
    const bool dataChanged = currentTableDataHash_ != hash || currentTableFrameId_ != frameId;

    (*tfCall)(0);

    (*readFlagsCall)(mmstd_gl::FlagCallRead_GL::CallGetData);

    if (dataChanged) {
        dimensionCount_ = floatTableCall->GetColumnsCount();
        itemCount_ = floatTableCall->GetRowsCount();
        names_.resize(dimensionCount_);
        dimensionRanges_.resize(dimensionCount_);
        axisIndirection_.resize(dimensionCount_);
        filters_.resize(dimensionCount_);

        dimensionIndex_.clear();
        auto* dimensionNameParam = dimensionNameParam_.Param<core::param::FlexEnumParam>();
        dimensionNameParam->ClearValues();
        dimensionNameParam->AddValue("[none]");
        for (int i = 0; i < dimensionCount_; i++) {
            const auto& colInfo = floatTableCall->GetColumnsInfos()[i];
            names_[i] = colInfo.Name();
            dimensionRanges_[i].min = colInfo.MinimumValue();
            dimensionRanges_[i].max = colInfo.MaximumValue();
            if (dimensionRanges_[i].max - dimensionRanges_[i].min < 0.001f) {
                dimensionRanges_[i].max = dimensionRanges_[i].min + 0.001f;
            }
            axisIndirection_[i] = i;
            dimensionIndex_[colInfo.Name()] = i;
            dimensionNameParam->AddValue(colInfo.Name());
            filters_[i] = dimensionRanges_[i];
        }

        dataBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, floatTableCall->GetData(),
            dimensionCount_ * itemCount_ * sizeof(float), GL_DYNAMIC_DRAW);
        dimensionRangesBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, dimensionRanges_);
        axisIndirectionBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, axisIndirection_);
        filtersBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, filters_);
        densityMinMaxBuffer_ =
            std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, densityMinMaxInit_, GL_STATIC_COPY);

        currentTableDataHash_ = hash;
        currentTableFrameId_ = frameId;

        needFlagsUpdate_ = true;
    }

    if (dataChanged || scaleToFitParam_.IsDirty()) {
        scaleToFitParam_.ResetDirty();
        calcSizes();
    }

    if (dataChanged || dimensionNameParam_.IsDirty() || drawModeParam_.IsDirty()) {
        drawModeParam_.ResetDirty();
        dimensionNameParam_.ResetDirty();

        if (drawModeParam_.Param<core::param::EnumParam>()->Value() == DrawMode::DRAW_DISCRETE) {
            const auto& dimensionName = dimensionNameParam_.Param<core::param::FlexEnumParam>()->Value();
            try {
                const auto dimIdx = dimensionIndex_.at(dimensionName);
                tfCall->SetRange({dimensionRanges_[dimIdx].min, dimensionRanges_[dimIdx].max});
            } catch (std::out_of_range& ex) {
                Log::DefaultLog.WriteWarn(
                    "ParallelCoordinatesRenderer2D: tried to color lines by non-existing dimension '%s'",
                    dimensionName.c_str());
                tfCall->SetRange({0.0f, 1.0f});
            }
        } else {
            tfCall->SetRange({0.0f, 1.0f});
        }
    }

    if (filterStateParam_.IsDirty()) {
        filterStateParam_.ResetDirty();

        loadFilters();
        needFilterUpdate_ = true;
    }

    if (resetFiltersParam_.IsDirty()) {
        resetFiltersParam_.ResetDirty();

        for (int i = 0; i < dimensionCount_; i++) {
            filters_[i] = dimensionRanges_[i];
        }
        needFilterUpdate_ = true;
    }

    return true;
}

void ParallelCoordinatesRenderer2D::calcSizes() {
    marginX_ = 20.0f;
    marginY_ = 20.0f;
    axisDistance_ = 40.0f;
    fontSize_ = axisDistance_ / 10.0f;
    auto left = 0.0f;
    auto right = 2.0f * marginX_ + axisDistance_ * static_cast<float>(dimensionCount_ - 1);
    auto width = right - left;

    if (scaleToFitParam_.Param<core::param::BoolParam>()->Value() && cameraCopy_.has_value()) {
        float requiredHeight =
            width / cameraCopy_.value().get<core::view::Camera::OrthographicParameters>().aspect.value();
        axisHeight_ = requiredHeight - 3.0f * marginY_;
    } else {
        axisHeight_ = 80.0f;
    }
    auto bottom = 0.0f;
    auto top = 3.0f * marginY_ + axisHeight_;

    bounds_.SetBoundingBox(left, bottom, 0, right, top, 0);
}

int ParallelCoordinatesRenderer2D::mouseXtoAxis(float x) {
    const float f = (x - marginX_) / axisDistance_;
    const float frac = f - std::floor(f);
    const int integral = static_cast<int>(std::round(f));
    if (integral >= 0 && integral < static_cast<int>(dimensionCount_) && (frac < 0.3f || frac > 0.7f)) {
        return axisIndirection_[integral];
    }
    return -1;
}

void ParallelCoordinatesRenderer2D::mouseToFilterIndicator(float x, float y, int& axis, int& index) {
    axis = mouseXtoAxis(x);
    index = -1;
    if (axis != -1) {
        // calculate position of click and filters in [0, 1] range of axis height
        const float pickPos = (y - marginY_) / axisHeight_;
        const auto& range = dimensionRanges_[axis];
        float upperPos = (filters_[axis].max - range.min) / (range.max - range.min);
        float lowerPos = (filters_[axis].min - range.min) / (range.max - range.min);

        // Add small epsilon for better UI feeling because indicator is drawn only to one side.
        // This also handles intuitive selection if upper and lower filter are set to the same value.
        upperPos += 0.01;
        lowerPos -= 0.01;

        const float distUpper = std::fabs(upperPos - pickPos);
        const float distLower = std::fabs(lowerPos - pickPos);

        const float thresh = 0.1f;
        if (distUpper < thresh && distUpper < distLower) {
            index = 1;
        } else if (distLower < thresh) {
            index = 0;
        }
    }
    if (index == -1) {
        axis = -1;
    }
}

bool ParallelCoordinatesRenderer2D::useProgramAndBindCommon(std::unique_ptr<glowl::GLSLProgram> const& program) {
    program->use();

    dataBuffer_->bind(0);
    auto readFlagsCall = readFlagsSlot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    readFlagsCall->getData()->validateFlagCount(static_cast<int32_t>(itemCount_));
    readFlagsCall->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
    dimensionRangesBuffer_->bind(2);
    axisIndirectionBuffer_->bind(3);
    filtersBuffer_->bind(4);
    densityMinMaxBuffer_->bind(5);

    if (cameraCopy_.has_value()) {
        program->setUniform("projMx", cameraCopy_.value().getProjectionMatrix());
        program->setUniform("viewMx", cameraCopy_.value().getViewMatrix());
    } else {
        static const glm::mat4 one(1.0f);
        program->setUniform("projMx", one);
        program->setUniform("viewMx", one);
    }

    program->setUniform("dimensionCount", static_cast<GLuint>(dimensionCount_));
    program->setUniform("itemCount", static_cast<GLuint>(itemCount_));

    program->setUniform("margin", marginX_, marginY_);
    program->setUniform("axisDistance", axisDistance_);
    program->setUniform("axisHeight", axisHeight_);

    return true;
}

void ParallelCoordinatesRenderer2D::doPicking(glm::vec2 pos, float pickRadius) {
    useProgramAndBindCommon(selectPickProgram_);
    selectPickProgram_->setUniform("mouse", pos);
    selectPickProgram_->setUniform("pickRadius", pickRadius);

    std::array<GLuint, 3> groupCounts{};
    computeDispatchSizes(itemCount_, selectPickWorkgroupSize_, maxWorkgroupCount_, groupCounts);
    glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
}

void ParallelCoordinatesRenderer2D::doStroking(glm::vec2 start, glm::vec2 end) {
    useProgramAndBindCommon(selectStrokeProgram_);
    selectStrokeProgram_->setUniform("strokeStart", start);
    selectStrokeProgram_->setUniform("strokeEnd", end);

    std::array<GLuint, 3> groupCounts{};
    computeDispatchSizes(itemCount_, selectStrokeWorkgroupSize_, maxWorkgroupCount_, groupCounts);
    glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
}

void ParallelCoordinatesRenderer2D::drawItemLines(
    uint32_t testMask, uint32_t passMask, bool useTf, glm::vec4 const& color) {
    auto tfCall = tfSlot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    const bool triangleMode = triangleModeParam_.Param<core::param::BoolParam>()->Value();

    int colorDimension = -1;
    try {
        colorDimension = dimensionIndex_.at(dimensionNameParam_.Param<core::param::FlexEnumParam>()->Value());
    } catch (std::out_of_range& ex) {}

    auto& prog = triangleMode ? drawItemsTriangleProgram_ : drawItemsLineProgram_;

    useProgramAndBindCommon(prog);
    tfCall->BindConvenience(prog, GL_TEXTURE5, 5);
    prog->setUniform("useTransferFunction", static_cast<int>(useTf));
    prog->setUniform("itemColor", color);
    prog->setUniform("colorDimensionIdx", colorDimension);
    prog->setUniform("itemTestMask", testMask);
    prog->setUniform("itemPassMask", passMask);
    prog->setUniform(
        "useLineWidthInPixels", static_cast<int>(useLineWidthInPixelsParam_.Param<core::param::BoolParam>()->Value()));
    prog->setUniform("lineWidth", lineWidthParam_.Param<core::param::FloatParam>()->Value());
    prog->setUniform("viewSize", viewRes_);

    glEnable(GL_CLIP_DISTANCE0);
    if (triangleMode) {
        glDrawArraysInstanced(
            GL_TRIANGLE_STRIP, 0, 2 * static_cast<int>(dimensionCount_), static_cast<int>(itemCount_));
    } else {
        glDrawArraysInstanced(GL_LINE_STRIP, 0, static_cast<int>(dimensionCount_), static_cast<int>(itemCount_));
    }
    glDisable(GL_CLIP_DISTANCE0);
    glUseProgram(0);
}

void ParallelCoordinatesRenderer2D::drawDiscrete(bool useTf, glm::vec4 const& color, glm::vec4 selectedColor) {
    using fst = core::FlagStorageTypes;
    using bits = fst::flag_bits;

    constexpr auto testMask = fst::to_integral(bits::ENABLED | bits::FILTERED | bits::SELECTED);

    if (drawItemsParam_.Param<core::param::BoolParam>()->Value()) {
        constexpr auto passMask = fst::to_integral(bits::ENABLED);
        drawItemLines(testMask, passMask, useTf, color);
    }
    if (drawSelectedItemsParam_.Param<core::param::BoolParam>()->Value()) {
        constexpr auto passMask = fst::to_integral(bits::ENABLED | bits::SELECTED);
        drawItemLines(testMask, passMask, false, selectedColor);
    }
}

void ParallelCoordinatesRenderer2D::drawDensity(std::shared_ptr<glowl::FramebufferObject> const& fbo) {
    const int fboWidth = fbo->getWidth();
    const int fboHeight = fbo->getHeight();

    if (densityFbo_ == nullptr || densityFbo_->getWidth() != fboWidth || densityFbo_->getHeight() != fboHeight) {
        densityFbo_ = std::make_unique<glowl::FramebufferObject>(
            "densityFbo", fboWidth, fboHeight, glowl::FramebufferObject::NONE);
        densityFbo_->createColorAttachment(GL_R32F, GL_RED, GL_FLOAT);
        densityFbo_->createColorAttachment(GL_R8, GL_RED, GL_UNSIGNED_BYTE);
    }
    densityFbo_->bind();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBlendFunc(GL_ONE, GL_ONE);
    static const glm::vec4 red(1.0f, 0.0f, 0.0f, 1.0);
    static const glm::vec4 red_green(1.0f, 1.0f, 0.0f, 1.0);
    drawDiscrete(false, red, red_green);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    fbo->bind();

    const bool normalizeDensity = normalizeDensityParam_.Param<core::param::BoolParam>()->Value();
    if (normalizeDensity) {
        densityMinMaxBuffer_->rebuffer(densityMinMaxInit_);

        static const GLuint blockSize = 16; // Each compute shader invocation will check blockSize * blockSize pixels.
        glm::uvec2 invocations((fboWidth + blockSize - 1) / blockSize, (fboHeight + blockSize - 1) / blockSize);
        std::array<GLuint, 3> groupCounts{
            (invocations.x + densityMinMaxWorkgroupSize_[0] - 1) / densityMinMaxWorkgroupSize_[0],
            (invocations.y + densityMinMaxWorkgroupSize_[1] - 1) / densityMinMaxWorkgroupSize_[1], 1};

        useProgramAndBindCommon(densityMinMaxProgram_);
        glActiveTexture(GL_TEXTURE1);
        densityFbo_->bindColorbuffer(0);
        densityMinMaxProgram_->setUniform("fragmentCountTex", 1);
        densityMinMaxProgram_->setUniform("resolution", fboWidth, fboHeight);
        densityMinMaxProgram_->setUniform("blockSize", blockSize);
        glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
        glUseProgram(0);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    auto tfCall = tfSlot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();

    useProgramAndBindCommon(drawItemsDensityProgram_);
    glActiveTexture(GL_TEXTURE1);
    densityFbo_->bindColorbuffer(0);
    drawItemsDensityProgram_->setUniform("fragmentCountTex", 1);
    glActiveTexture(GL_TEXTURE2);
    densityFbo_->bindColorbuffer(1);
    drawItemsDensityProgram_->setUniform("selectionFlagTex", 2);
    tfCall->BindConvenience(drawItemsDensityProgram_, GL_TEXTURE5, 5);
    drawItemsDensityProgram_->setUniform("normalizeDensity", normalizeDensity ? 1 : 0);
    drawItemsDensityProgram_->setUniform(
        "sqrtDensity", sqrtDensityParam_.Param<core::param::BoolParam>()->Value() ? 1 : 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glUseProgram(0);
}

void ParallelCoordinatesRenderer2D::drawAxes(glm::mat4 ortho) {
    if (dimensionCount_ > 0) {
        useProgramAndBindCommon(drawAxesProgram_);
        drawAxesProgram_->setUniform("lineWidth", axesLineWidthParam_.Param<core::param::FloatParam>()->Value());
        drawAxesProgram_->setUniform("viewSize", viewRes_);
        drawAxesProgram_->setUniform("numTicks", static_cast<GLuint>(numTicks_));
        drawAxesProgram_->setUniform("tickLength", tickLength_);
        drawAxesProgram_->setUniform("pickedAxis", pickedAxis_);
        drawAxesProgram_->setUniform("pickedFilter", pickedIndicatorAxis_, pickedIndicatorIndex_);
        const glm::vec4 axesColor =
            glm::make_vec4<float>(axesColorParam_.Param<core::param::ColorParam>()->Value().data());
        drawAxesProgram_->setUniform("axesColor", axesColor);
        const glm::vec4 filterColor =
            glm::make_vec4<float>(filterIndicatorColorParam_.Param<core::param::ColorParam>()->Value().data());
        drawAxesProgram_->setUniform("filterColor", filterColor);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<int>(dimensionCount_) * (1 + 4 + numTicks_));
        glUseProgram(0);

        const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        const float* color;

        glActiveTexture(GL_TEXTURE0);
        font_.ClearBatchDrawCache();
        font_.SetSmoothMode(smoothFontParam_.Param<core::param::BoolParam>()->Value());
        for (int axisIdx = 0; axisIdx < dimensionCount_; axisIdx++) {
            const int dimIdx = axisIndirection_[axisIdx];
            if (pickedAxis_ == dimIdx) {
                color = red;
            } else {
                color = axesColorParam_.Param<core::param::ColorParam>()->Value().data();
            }

            const float posX = marginX_ + axisDistance_ * static_cast<float>(axisIdx);
            const float posY_bottom = marginY_ * 0.5f;
            const float posY_top = marginY_ * 1.5f + axisHeight_;
            const float posY_label = marginY_ * (2.0f + static_cast<float>(axisIdx % 2) * 0.5f) + axisHeight_;

            const std::string bottom = std::to_string(filters_[dimIdx].min);
            const std::string top = std::to_string(filters_[dimIdx].max);
            font_.DrawString(ortho, color, posX, posY_bottom, fontSize_, false, bottom.c_str(),
                core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
            font_.DrawString(ortho, color, posX, posY_top, fontSize_, false, top.c_str(),
                core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
            font_.DrawString(ortho, color, posX, posY_label, fontSize_ * 2.0f, false, names_[dimIdx].c_str(),
                core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
        }
        font_.BatchDrawString(ortho);
    }
}

void ParallelCoordinatesRenderer2D::drawIndicatorPick(glm::vec2 pos, float pickRadius, glm::vec4 const& color) {
    useProgramAndBindCommon(drawIndicatorPickProgram_);
    drawIndicatorPickProgram_->setUniform("mouse", pos);
    drawIndicatorPickProgram_->setUniform("pickRadius", pickRadius);
    drawIndicatorPickProgram_->setUniform("indicatorColor", color);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glUseProgram(0);
}

void ParallelCoordinatesRenderer2D::drawIndicatorStroke(glm::vec2 start, glm::vec2 end, glm::vec4 const& color) {
    useProgramAndBindCommon(drawIndicatorStrokeProgram_);
    drawIndicatorStrokeProgram_->setUniform("strokeStart", start);
    drawIndicatorStrokeProgram_->setUniform("strokeEnd", end);
    drawIndicatorStrokeProgram_->setUniform("viewSize", viewRes_);
    drawIndicatorStrokeProgram_->setUniform("lineWidth", axesLineWidthParam_.Param<core::param::FloatParam>()->Value());
    drawIndicatorStrokeProgram_->setUniform("indicatorColor", color);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glUseProgram(0);
}

void ParallelCoordinatesRenderer2D::storeFilters() {
    nlohmann::json jf_array;
    for (auto& f : filters_) {
        jf_array.push_back(nlohmann::json{{"lower", f.min}, {"upper", f.max}});
    }
    const auto js = jf_array.dump();
    filterStateParam_.Param<core::param::StringParam>()->SetValue(js.c_str());
}

void ParallelCoordinatesRenderer2D::loadFilters() {
    try {
        auto j = nlohmann::json::parse(filterStateParam_.Param<core::param::StringParam>()->Value());
        int i = 0;
        for (auto& f : j) {
            if (i < filters_.size()) {
                f.at("lower").get_to(filters_[i].min);
                f.at("upper").get_to(filters_[i].max);
                filters_[i].min = std::clamp(filters_[i].min, dimensionRanges_[i].min, dimensionRanges_[i].max);
                filters_[i].max = std::clamp(filters_[i].max, dimensionRanges_[i].min, dimensionRanges_[i].max);
            } else {
                break;
            }
            i++;
        }
    } catch (nlohmann::json::exception const& e) {
        Log::DefaultLog.WriteError(
            "ParallelCoordinatesRenderer2D: could not parse serialized filters (exception %i)!", e.id);
        return;
    }
}
