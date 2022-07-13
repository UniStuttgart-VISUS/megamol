/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#include "ScatterplotMatrixRenderer2D.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "vislib/math/ShallowMatrix.h"

#include "delaunator.hpp"
#include "mmcore_gl/flags/FlagCallsGL.h"
#include <sstream>

using namespace megamol;
using namespace megamol::infovis_gl;
using namespace megamol::datatools;

using megamol::core::utility::log::Log;

const GLuint PlotSSBOBindingPoint = 2;
const GLuint ValueSSBOBindingPoint = 3;
const GLuint FlagsBindingPoint = 4;

inline std::string to_string(float x, int precision = 2) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << x;
    return stream.str();
}

inline float lerp(float x, float y, float a) {
    return x * (1.0f - a) + y * a;
}

inline float rangeToSmallStep(double min, double max) {
    double countBigSteps = 4.0;
    double countMidSteps = countBigSteps * 5.0;
    double countSmallSteps = countMidSteps * 5.0;

    double delta = fabs(max - min);

    // Fit to decimal system: (whole number) * 10^(whole number)
    // Note: should be without -1.0. Could be floating point weirdness, i.e.,
    // 1.00001 being rounded to 2.0 instead to 1.0 with base 10.
    double exponent = ceil(log2(delta / countSmallSteps) / log2(10.0)) - 1.0;
    double power = pow(10.0, exponent);
    double mantissa = (delta / countSmallSteps) / power;
    mantissa = round(mantissa * 2.0);
    mantissa = mantissa / 2.0;

    return mantissa * power;
}

std::optional<size_t> nameToIndex(datatools::table::TableDataCall* tableDataCall, const std::string& name) {
    auto columnInfos = tableDataCall->GetColumnsInfos();
    const size_t colCount = tableDataCall->GetColumnsCount();

    for (size_t i = 0; i < colCount; i++) {
        if (columnInfos[i].Name().compare(name) == 0) {
            return i;
        }
    }
    return std::nullopt;
}

ScatterplotMatrixRenderer2D::ScatterplotMatrixRenderer2D()
        : Renderer2D()
        , floatTableInSlot("ftIn", "Float table input")
        , transferFunctionInSlot("tfIn", "Transfer function input")
        , readFlagStorageSlot("readFlags", "Flag storage input")
        , writeFlagStorageSlot("writeFlags", "Flag storage output")
        , valueMappingParam("valueMappingMode", "Value mapping")
        , valueSelectorParam("valueSelector", "Sets a value column to as additional domain")
        , labelSelectorParam("labelSelector", "Sets a label column (text mode)")
        , labelSizeParam("labelSize", "Sets the fontsize for labels (text mode)")
        , geometryTypeParam("geometryType", "Geometry type to map data to")
        , kernelWidthParam("kernelWidth", "Kernel width of the geometry, i.e., point size or line width")
        , kernelTypeParam("kernelType", "Kernel function, i.e., box or gaussian kernel")
        , splitLinesByValueParam("splitLinesByValue", "Draw lines only between points with same id value")
        , lineConnectedValueSelectorParam("lineConnectedValueSelector", "Select id value column for line connection")
        , pickRadiusParam("pickRadius", "Picking radius")
        , pickColorParam("pickColor", "Picking color")
        , resetSelectionParam("resetSelection", "Reset selection")
        , drawPickIndicatorParam("drawPickIndicator", "Draw picking indicator")
        , drawMouseLabelsParam("drawMouseLabels", "Draw labels on cells on mouse hover")
        , triangulationSmoothnessParam("triangulationSmoothness", "Number of iterations to smooth the triangulation")
        , axisModeParam("axisMode", "Axis drawing mode")
        , axisColorParam("axisColor", "Color of axis")
        , axisWidthParam("axisWidth", "Line width for the axis")
        , axisTicksParam("axisTicks", "Number of ticks on the axis")
        , axisTicksRedundantParam("axisTicksRedundant", "Enable redundant (inner) ticks")
        , axisTickLengthParam("axisTickLength", "Line length for the ticks")
        , axisTickMarginParam("axisTickMargin", "Gap between tick line and font")
        , axisTickSizeParam("axisTickSize", "Sets the fontsize for the ticks")
        , axisTickPrecisionX("axisTickPrecisionX", "Sets the float precision for the x ticks")
        , axisTickPrecisionY("axisTickPrecisionY", "Sets the float precision for the y ticks")
        , drawOuterLabelsParam("drawOuterLabels", "Draw labels outside of the matrix")
        , drawDiagonalLabelsParam("drawDiagonalLabels", "Draw labels at matrix diagonal")
        , cellInvertYParam("cellInvertY", "Draw diagonal top left to bottom right")
        , cellSizeParam("cellSize", "Aspect ratio scaling x axis length")
        , cellMarginParam("cellMargin", "Set the scaling of y axis")
        , cellNameSizeParam("cellNameSize", "Sets the fontsize for cell names, i.e., column names")
        , outerXLabelMarginParam("outerXLabelMarginParam", "Margin between tick labels and name labels on outer x axis")
        , outerYLabelMarginParam("outerYLabelMarginParam", "Margin between tick labels and name labels on outer y axis")
        , alphaScalingParam("alphaScaling", "Scaling factor for overall alpha")
        , alphaAttenuateSubpixelParam("alphaAttenuateSubpixel", "Attenuate alpha of points that have subpixel size")
        , smoothFontParam("smoothFont", "Font rendering with smooth edges")
        , forceRedrawDebugParam("forceRedrawDebug", "Force redraw every frame (for benchmarking and debugging).")
        , mouse({0, 0, BrushState::NOP})
        , plotSSBO("Plots")
        , valueSSBO("Values")
        , triangleVBO(0)
        , triangleIBO(0)
        , triangleVertexCount(0)
        , trianglesValid(false)
        , currentViewRes(glm::ivec2(0, 0))
        , screenFBO(nullptr)
        , screenValid(false)
        , axisFont(core::utility::SDFFont::PRESET_EVOLVENTA_SANS, core::utility::SDFFont::RENDERMODE_FILL)
        , textFont(core::utility::SDFFont::PRESET_EVOLVENTA_SANS, core::utility::SDFFont::RENDERMODE_FILL)
        , textValid(false)
        , dataTime((std::numeric_limits<unsigned int>::max)())
        , flagsBufferVersion(0) {
    this->floatTableInSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->floatTableInSlot);

    this->transferFunctionInSlot.SetCompatibleCall<core_gl::view::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->transferFunctionInSlot);

    this->readFlagStorageSlot.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->readFlagStorageSlot);

    this->writeFlagStorageSlot.SetCompatibleCall<core_gl::FlagCallWrite_GLDescription>();
    this->MakeSlotAvailable(&this->writeFlagStorageSlot);


    auto* valueMappings = new core::param::EnumParam(0);
    valueMappings->SetTypePair(VALUE_MAPPING_KERNEL_BLEND, "Kernel Blending");
    valueMappings->SetTypePair(VALUE_MAPPING_KERNEL_DENSITY, "Kernel Density Estimation");
    valueMappings->SetTypePair(VALUE_MAPPING_WEIGHTED_KERNEL_DENSITY, "Weighted Kernel Density Estimation");
    this->valueMappingParam << valueMappings;
    this->MakeSlotAvailable(&this->valueMappingParam);

    this->valueSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->valueSelectorParam);

    this->labelSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->labelSelectorParam);

    this->labelSizeParam << new core::param::FloatParam(0.1f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->labelSizeParam);

    this->triangulationSmoothnessParam << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->triangulationSmoothnessParam);

    auto* geometryTypes = new core::param::EnumParam(0);
    geometryTypes->SetTypePair(GEOMETRY_TYPE_POINT, "Point");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_POINT_TRIANGLE_SPRITE, "Point(TriangleSprite)");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_LINE, "Line");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_TEXT, "Text");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_TRIANGULATION, "Delaunay Triangulation");
    this->geometryTypeParam << geometryTypes;
    this->MakeSlotAvailable(&this->geometryTypeParam);

    this->kernelWidthParam << new core::param::FloatParam(
        0.1f, std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::max(), 0.1f);
    this->MakeSlotAvailable(&this->kernelWidthParam);

    auto* kernelTypes = new core::param::EnumParam(0);
    kernelTypes->SetTypePair(KERNEL_TYPE_BOX, "Box");
    kernelTypes->SetTypePair(KERNEL_TYPE_GAUSSIAN, "Gaussian");
    this->kernelTypeParam << kernelTypes;
    this->MakeSlotAvailable(&this->kernelTypeParam);

    this->splitLinesByValueParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->splitLinesByValueParam);

    this->lineConnectedValueSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->lineConnectedValueSelectorParam);

    this->pickRadiusParam << new core::param::FloatParam(1.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->pickRadiusParam);

    this->pickColorParam << new core::param::ColorParam("red");
    this->MakeSlotAvailable(&this->pickColorParam);

    this->resetSelectionParam << new core::param::ButtonParam();
    this->resetSelectionParam.SetUpdateCallback(this, &ScatterplotMatrixRenderer2D::resetSelectionCallback);
    this->MakeSlotAvailable(&this->resetSelectionParam);

    this->drawPickIndicatorParam << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->drawPickIndicatorParam);

    this->drawMouseLabelsParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->drawMouseLabelsParam);

    auto* axisModes = new core::param::EnumParam(1);
    axisModes->SetTypePair(AXIS_MODE_NONE, "None");
    axisModes->SetTypePair(AXIS_MODE_MINIMALISTIC, "Minimalistic");
    axisModes->SetTypePair(AXIS_MODE_SCIENTIFIC, "Scientific");
    this->axisModeParam << axisModes;
    this->MakeSlotAvailable(&this->axisModeParam);

    this->axisColorParam << new core::param::ColorParam("white");
    this->MakeSlotAvailable(&this->axisColorParam);

    this->axisWidthParam << new core::param::FloatParam(1.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->axisWidthParam);

    this->axisTicksParam << new core::param::IntParam(5, 2, 100);
    this->MakeSlotAvailable(&this->axisTicksParam);

    this->axisTicksRedundantParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->axisTicksRedundantParam);

    this->axisTickLengthParam << new core::param::FloatParam(0.25f, 0.0f, std::numeric_limits<float>::max(), 0.25f);
    this->MakeSlotAvailable(&this->axisTickLengthParam);

    this->axisTickMarginParam << new core::param::FloatParam(0.1f, 0.0f, std::numeric_limits<float>::max(), 0.1f);
    this->MakeSlotAvailable(&this->axisTickMarginParam);

    this->axisTickSizeParam << new core::param::FloatParam(0.5f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->axisTickSizeParam);

    this->axisTickPrecisionX << new core::param::IntParam(2, 0);
    this->MakeSlotAvailable(&this->axisTickPrecisionX);

    this->axisTickPrecisionY << new core::param::IntParam(2, 0);
    this->MakeSlotAvailable(&this->axisTickPrecisionY);

    this->drawOuterLabelsParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->drawOuterLabelsParam);

    this->drawDiagonalLabelsParam << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->drawDiagonalLabelsParam);

    this->cellInvertYParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->cellInvertYParam);

    this->cellSizeParam << new core::param::FloatParam(10.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->cellSizeParam);

    this->cellMarginParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->cellMarginParam);

    this->cellNameSizeParam << new core::param::FloatParam(2.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->cellNameSizeParam);

    this->outerXLabelMarginParam << new core::param::FloatParam(0.0f, 0.0f);
    this->MakeSlotAvailable(&this->outerXLabelMarginParam);

    this->outerYLabelMarginParam << new core::param::FloatParam(0.5f, 0.0f);
    this->MakeSlotAvailable(&this->outerYLabelMarginParam);

    this->alphaScalingParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->alphaScalingParam);

    this->alphaAttenuateSubpixelParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->alphaAttenuateSubpixelParam);

    this->smoothFontParam << new core::param::BoolParam(true);
    MakeSlotAvailable(&this->smoothFontParam);

    this->forceRedrawDebugParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->forceRedrawDebugParam);

    // Create list of data-sensitive parameters.
    dataParams.push_back(&this->valueSelectorParam);
    dataParams.push_back(&this->labelSelectorParam);
    dataParams.push_back(&this->labelSizeParam);
    dataParams.push_back(&this->triangulationSmoothnessParam);
    dataParams.push_back(&this->cellInvertYParam);
    dataParams.push_back(&this->cellSizeParam);
    dataParams.push_back(&this->cellMarginParam);

    // Create list of screen-sensitive parameters.
    screenParams.push_back(&this->valueMappingParam);
    screenParams.push_back(&this->geometryTypeParam);
    screenParams.push_back(&this->kernelWidthParam);
    screenParams.push_back(&this->kernelTypeParam);
    screenParams.push_back(&this->splitLinesByValueParam);
    screenParams.push_back(&this->lineConnectedValueSelectorParam);
    screenParams.push_back(&this->pickRadiusParam);
    screenParams.push_back(&this->pickColorParam);
    screenParams.push_back(&this->axisModeParam);
    screenParams.push_back(&this->axisColorParam);
    screenParams.push_back(&this->axisWidthParam);
    screenParams.push_back(&this->axisTicksParam);
    screenParams.push_back(&this->axisTicksRedundantParam);
    screenParams.push_back(&this->axisTickLengthParam);
    screenParams.push_back(&this->axisTickSizeParam);
    screenParams.push_back(&this->cellNameSizeParam);
    screenParams.push_back(&this->alphaScalingParam);
    screenParams.push_back(&this->alphaAttenuateSubpixelParam);
}

ScatterplotMatrixRenderer2D::~ScatterplotMatrixRenderer2D() {
    this->Release();
}

bool ScatterplotMatrixRenderer2D::create() {
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

    try {
        minimalisticAxisShader = core::utility::make_glowl_shader("splom_axis_minimalistic", shader_options,
            "infovis_gl/splom/axis_minimalistic.vert.glsl", "infovis_gl/splom/axis_minimalistic.frag.glsl");
        scientificAxisShader = core::utility::make_glowl_shader("splom_axis_scientific", shader_options,
            "infovis_gl/splom/axis_scientific.vert.glsl", "infovis_gl/splom/axis_scientific.frag.glsl");
        pointShader = core::utility::make_glowl_shader(
            "splom_point", shader_options, "infovis_gl/splom/splom.vert.glsl", "infovis_gl/splom/point.frag.glsl");
        pointTriangleSpriteShader = core::utility::make_glowl_shader("splom_trianlge_point_sprite", shader_options,
            "infovis_gl/splom/triangle_point_sprite.vert.glsl", "infovis_gl/splom/triangle_point_sprite.frag.glsl");
        lineShader = core::utility::make_glowl_shader("splom_line", shader_options, "infovis_gl/splom/splom.vert.glsl",
            "infovis_gl/splom/line.geom.glsl", "infovis_gl/splom/line.frag.glsl");
        triangleShader = core::utility::make_glowl_shader("splom_triangle", shader_options,
            "infovis_gl/splom/triangle.vert.glsl", "infovis_gl/splom/triangle.frag.glsl");
        pickIndicatorShader = core::utility::make_glowl_shader("splom_pick_indicator", shader_options,
            "infovis_gl/splom/pick_indicator.vert.glsl", "infovis_gl/splom/pick_indicator.frag.glsl");
        screenShader = core::utility::make_glowl_shader(
            "splom_screen", shader_options, "infovis_gl/splom/screen.vert.glsl", "infovis_gl/splom/screen.frag.glsl");
        pickProgram = core::utility::make_glowl_shader("splom_pick", shader_options, "infovis_gl/splom/pick.comp.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            ("ScatterplotMatrixRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    glGetProgramiv(pickProgram->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, pickWorkgroupSize);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkgroupCount[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkgroupCount[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkgroupCount[2]);

    if (!this->axisFont.Initialise(this->GetCoreInstance()))
        return false;
    if (!this->textFont.Initialise(this->GetCoreInstance()))
        return false;
    this->axisFont.SetBatchDrawMode(true);
    this->textFont.SetBatchDrawMode(true);

    return true;
}

void ScatterplotMatrixRenderer2D::release() {}

bool ScatterplotMatrixRenderer2D::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    if (mods.test(core::view::Modifier::CTRL)) {
        // These clicks go to the view.
        return false;
    }

    if (button == core::view::MouseButton::BUTTON_LEFT && action == core::view::MouseButtonAction::PRESS) {
        this->mouse.selector = BrushState::ADD;
        this->selectionNeedsUpdate = true;
        return true;
    } else if (button == core::view::MouseButton::BUTTON_RIGHT && action == core::view::MouseButtonAction::PRESS) {
        this->mouse.selector = BrushState::REMOVE;
        this->selectionNeedsUpdate = true;
        return true;
    }

    this->mouse.selector = BrushState::NOP;

    return false;
}

bool ScatterplotMatrixRenderer2D::OnMouseMove(double x, double y) {
    // mouseCoordsToWorld requires a valid camera and currentCamera is initialized on first render. Before anything is
    // draw, interaction probably is not needed anyway, but this event could be triggered independently.
    if (!currentCamera.has_value()) {
        return false;
    }

    auto const& [world_x, world_y] =
        mouseCoordsToWorld(x, y, currentCamera.value(), currentViewRes.x, currentViewRes.y);
    this->mouse.x = world_x;
    this->mouse.y = world_y;

    if (this->mouse.selector != BrushState::NOP) {
        this->selectionNeedsUpdate = true;
        return true;
    }

    return false;
}

bool ScatterplotMatrixRenderer2D::Render(core_gl::view::CallRender2DGL& call) {
    try {

        // get camera
        currentCamera = call.GetCamera();
        auto view = currentCamera.value().getViewMatrix();
        auto proj = currentCamera.value().getProjectionMatrix();
        glm::mat4 ortho = proj * view;
        currentViewRes = call.GetViewResolution();

        if (!this->validate(call, false))
            return false;

        if (this->selectionNeedsUpdate) {
            this->updateSelection();
            this->selectionNeedsUpdate = false;
        }

        auto axisMode = this->axisModeParam.Param<core::param::EnumParam>()->Value();
        switch (axisMode) {
        case AXIS_MODE_NONE:
            // NOP.
            break;
        case AXIS_MODE_MINIMALISTIC:
            this->drawMinimalisticAxis(ortho);
            break;
        case AXIS_MODE_SCIENTIFIC:
            this->drawScientificAxis(ortho);
            break;
        }

        auto geometryType = this->geometryTypeParam.Param<core::param::EnumParam>()->Value();
        switch (geometryType) {
        case GEOMETRY_TYPE_POINT:
            glEnable(GL_CLIP_DISTANCE0);
            this->drawPoints();
            glDisable(GL_CLIP_DISTANCE0);
            break;
        case GEOMETRY_TYPE_POINT_TRIANGLE_SPRITE:
            glEnable(GL_CLIP_DISTANCE0);
            this->drawPointTriangleSprites();
            glDisable(GL_CLIP_DISTANCE0);
            break;
        case GEOMETRY_TYPE_LINE:
            glEnable(GL_CLIP_DISTANCE0);
            this->drawLines();
            glDisable(GL_CLIP_DISTANCE0);
            break;
        case GEOMETRY_TYPE_TRIANGULATION:
            this->drawTriangulation();
            break;
        case GEOMETRY_TYPE_TEXT:
            this->drawText(ortho);
            break;
        }

        if (this->drawPickIndicatorParam.Param<core::param::BoolParam>()->Value()) {
            this->drawPickIndicator();
        }

        if (this->drawMouseLabelsParam.Param<core::param::BoolParam>()->Value()) {
            this->drawMouseLabels(ortho);
        }

        this->drawScreen();

    } catch (...) { return false; }

    return true;
}

bool ScatterplotMatrixRenderer2D::GetExtents(core_gl::view::CallRender2DGL& call) {
    this->validate(call, true);
    call.AccessBoundingBoxes() = this->bounds;
    return true;
}

bool ScatterplotMatrixRenderer2D::hasDirtyData() const {
    for (auto* param : this->dataParams) {
        if (param->IsDirty())
            return true;
    }
    return false;
}

void ScatterplotMatrixRenderer2D::resetDirtyData() {
    for (auto* param : this->dataParams) {
        param->ResetDirty();
    }
}

bool ScatterplotMatrixRenderer2D::hasDirtyScreen() const {
    for (auto* param : this->screenParams) {
        if (param->IsDirty())
            return true;
    }
    return false;
}

void ScatterplotMatrixRenderer2D::resetDirtyScreen() {
    for (auto* param : this->screenParams) {
        param->ResetDirty();
    }
}

bool ScatterplotMatrixRenderer2D::validate(core_gl::view::CallRender2DGL& call, bool ignoreMVP) {
    this->floatTable = this->floatTableInSlot.CallAs<table::TableDataCall>();

    this->transferFunction = this->transferFunctionInSlot.CallAs<megamol::core_gl::view::CallGetTransferFunctionGL>();
    if ((this->transferFunction == nullptr) || !(*(this->transferFunction))(0))
        return false;

    if (this->floatTable == nullptr || !(*this->floatTable)(1))
        return false;
    const auto cntFrames = this->floatTable->GetFrameCount();
    call.SetTimeFramesCount(cntFrames); // Tell view about the data set size.

    const auto now = static_cast<unsigned int>(call.Time());
    this->floatTable->SetFrameID(now);

    if (this->floatTable == nullptr || !(*(this->floatTable))(0))
        return false;
    if (this->floatTable->GetColumnsCount() == 0)
        return false;

    this->readFlags = this->readFlagStorageSlot.CallAs<core_gl::FlagCallRead_GL>();
    if (this->readFlags == nullptr)
        return false;
    (*this->readFlags)(core_gl::FlagCallRead_GL::CallGetData);

    auto columnInfos = this->floatTable->GetColumnsInfos();
    const size_t colCount = this->floatTable->GetColumnsCount();

    // get camera
    core::view::Camera cam = call.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto mvp = proj * view;
    // mvp is unstable across GetExtents and Render, so we just do these checks when rendering
    if (hasDirtyScreen() || hasDirtyData() || (!ignoreMVP && (screenLastMVP != mvp || this->readFlags->hasUpdate())) ||
        this->transferFunction->IsDirty() || this->forceRedrawDebugParam.Param<core::param::BoolParam>()->Value()) {
        this->screenValid = false;
        resetDirtyScreen();
        screenLastMVP = mvp;
        this->transferFunction->ResetDirty();
    }
    if (hasDirtyData()) {
        // Update transfer function range
        map.valueIdx =
            nameToIndex(this->floatTable, this->valueSelectorParam.Param<core::param::FlexEnumParam>()->Value());
        map.labelIdx =
            nameToIndex(this->floatTable, this->labelSelectorParam.Param<core::param::FlexEnumParam>()->Value())
                .value_or(0);
        if (map.valueIdx.has_value() &&
            this->valueMappingParam.Param<core::param::EnumParam>()->Value() == VALUE_MAPPING_KERNEL_BLEND) {
            this->transferFunction->SetRange(
                {columnInfos[map.valueIdx.value()].MinimumValue(), columnInfos[map.valueIdx.value()].MaximumValue()});
        } else {
            this->transferFunction->SetRange({0.0f, 1.0f});
        }
    }

    if (this->dataHash == this->floatTable->DataHash() && now == this->dataTime && !hasDirtyData())
        return true;

    if (this->dataHash != this->floatTable->DataHash()) {
        // Update dynamic parameters.
        this->valueSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->labelSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->lineConnectedValueSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        for (size_t i = 0; i < colCount; i++) {
            this->valueSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
            this->labelSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
            this->lineConnectedValueSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
        }
    }

    this->screenValid = false;
    this->trianglesValid = false;
    this->textValid = false;
    this->updateColumns();

    this->dataHash = this->floatTable->DataHash();
    this->dataTime = now;
    this->resetDirtyData();

    return true;
}

void ScatterplotMatrixRenderer2D::updateColumns() {
    const auto columnCount = this->floatTable->GetColumnsCount();
    const auto columnInfos = this->floatTable->GetColumnsInfos();
    const float size = this->cellSizeParam.Param<core::param::FloatParam>()->Value();
    const float margin = this->cellMarginParam.Param<core::param::FloatParam>()->Value();
    const bool invertY = this->cellInvertYParam.Param<core::param::BoolParam>()->Value();

    plots.clear();
    for (GLuint y = 0; y < columnCount; ++y) {
        GLfloat offsetY = (invertY ? (columnCount - y - 1) : y) * (size + margin);
        for (GLuint x = 0; x < y; ++x) {
            auto minXValue = columnInfos[x].MinimumValue();
            auto minYValue = columnInfos[y].MinimumValue();
            auto maxXValue = columnInfos[x].MaximumValue();
            auto maxYValue = columnInfos[y].MaximumValue();
            if (maxXValue - minXValue < 0.00001f) {
                maxXValue = minXValue + 0.00001f;
            }
            if (maxYValue - minYValue < 0.00001f) {
                maxYValue = minYValue + 0.00001f;
            }
            plots.push_back({x, y, x * (size + margin), offsetY, size, size, minXValue, minYValue, maxXValue, maxYValue,
                rangeToSmallStep(minXValue, maxXValue), rangeToSmallStep(minYValue, maxYValue)});
        }
    }

    this->bounds.SetBoundingBox(
        0, 0, 0, columnCount * (size + margin) - margin, columnCount * (size + margin) - margin, 0);

    this->plotSSBO.SetData(plots.data(), sizeof(PlotInfo), sizeof(PlotInfo), plots.size());
}

void ScatterplotMatrixRenderer2D::drawMinimalisticAxis(glm::mat4 ortho) {
    debugPush(1, "drawMinimalisticAxis");

    const auto axisColor = this->axisColorParam.Param<core::param::ColorParam>()->Value();
    const auto columnCount = this->floatTable->GetColumnsCount();
    const auto columnInfos = this->floatTable->GetColumnsInfos();

    const float size = this->cellSizeParam.Param<core::param::FloatParam>()->Value();
    const float margin = this->cellMarginParam.Param<core::param::FloatParam>()->Value();
    const float nameSize = this->cellNameSizeParam.Param<core::param::FloatParam>()->Value();
    const GLsizei numTicks = this->axisTicksParam.Param<core::param::IntParam>()->Value();
    const GLfloat tickLength = this->axisTickLengthParam.Param<core::param::FloatParam>()->Value();
    const float tickMargin = this->axisTickMarginParam.Param<core::param::FloatParam>()->Value();
    const float tickSize = this->axisTickSizeParam.Param<core::param::FloatParam>()->Value();
    const int tickPrecisionX = this->axisTickPrecisionX.Param<core::param::IntParam>()->Value();
    const int tickPrecisionY = this->axisTickPrecisionY.Param<core::param::IntParam>()->Value();
    const bool drawOuter = this->drawOuterLabelsParam.Param<core::param::BoolParam>()->Value();
    const bool drawDiagonal = this->drawDiagonalLabelsParam.Param<core::param::BoolParam>()->Value();
    const bool invertY = this->cellInvertYParam.Param<core::param::BoolParam>()->Value();
    const float xLabelMargin = this->outerXLabelMarginParam.Param<core::param::FloatParam>()->Value();
    const float yLabelMargin = this->outerYLabelMarginParam.Param<core::param::FloatParam>()->Value();
    const float totalSize = columnCount * (size + margin) - margin;
    // Line width.
    auto axisWidth = this->axisWidthParam.Param<core::param::FloatParam>()->Value();

    this->minimalisticAxisShader->use();

    // Transformation uniform.
    this->minimalisticAxisShader->setUniform("modelViewProjection", screenLastMVP);

    // Other uniforms.
    glUniform4fv(this->minimalisticAxisShader->getUniformLocation("axisColor"), 1,
        this->axisColorParam.Param<core::param::ColorParam>()->Value().data());
    glUniform1ui(this->minimalisticAxisShader->getUniformLocation("numTicks"), numTicks);
    glUniform1f(this->minimalisticAxisShader->getUniformLocation("tickLength"), tickLength);
    glUniform1i(this->minimalisticAxisShader->getUniformLocation("redundantTicks"),
        this->axisTicksRedundantParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);
    glUniform1i(this->minimalisticAxisShader->getUniformLocation("drawOuter"),
        this->drawOuterLabelsParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);
    glUniform1i(this->minimalisticAxisShader->getUniformLocation("drawDiagonal"),
        this->drawDiagonalLabelsParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);
    glUniform1i(this->minimalisticAxisShader->getUniformLocation("invertY"),
        this->cellInvertYParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);
    glUniform1i(this->minimalisticAxisShader->getUniformLocation("columnCount"), columnCount);
    glUniform1f(this->minimalisticAxisShader->getUniformLocation("axisWidth"), axisWidth);
    this->minimalisticAxisShader->setUniform("viewSize", currentViewRes);

    // Render all plots at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(0));
    const GLsizei numVerticesPerLine = 2;
    const GLsizei numBorderVertices = numVerticesPerLine * 4;
    const GLsizei numTickVertices = numVerticesPerLine * numTicks * 4;
    const GLsizei numItems = numBorderVertices + numTickVertices;
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, 4 * this->plots.size() * (1 + numTicks));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    this->axisFont.ClearBatchDrawCache();
    this->axisFont.SetSmoothMode(smoothFontParam.Param<core::param::BoolParam>()->Value());

    for (size_t i = 0; i < columnCount; ++i) {
        // this will be bottom left of diagonal cell
        const float offsetX = i * (size + margin);
        const float offsetY = (invertY ? (columnCount - i - 1) : i) * (size + margin);

        std::string label = columnInfos[i].Name();

        // draw labels
        // horizontal
        if (drawOuter && i < columnCount - 1) {
            if (invertY) {
                this->axisFont.DrawString(ortho, axisColor.data(), offsetX,
                    -tickLength - tickMargin - tickSize - xLabelMargin, size, size, nameSize, false, label.c_str(),
                    core::utility::SDFFont::ALIGN_CENTER_TOP);
            } else {
                this->axisFont.DrawString(ortho, axisColor.data(), offsetX,
                    totalSize + tickLength + tickMargin + tickSize + xLabelMargin + size, size, size, nameSize, false,
                    label.c_str(), core::utility::SDFFont::ALIGN_CENTER_BOTTOM);
            }
        }
        // vertical
        if (drawOuter && i > 0) {
            this->axisFont.SetRotation(90.0, 0.0, 0.0, 1.0);
            this->axisFont.DrawString(ortho, axisColor.data(), offsetY,
                tickLength + tickMargin + tickSize + yLabelMargin + size, size, size, nameSize, false, label.c_str(),
                core::utility::SDFFont::ALIGN_CENTER_BOTTOM);
            this->axisFont.ResetRotation();
        }
        // diagonal
        if (drawDiagonal) {
            this->axisFont.DrawString(ortho, axisColor.data(), offsetX, offsetY + size, size, size, nameSize, false,
                label.c_str(), core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
        }
        glm::vec2 resOffset = (screenLastMVP * glm::vec4(currentViewRes, 0.0, 1.0));

        // draw tick labels
        float horizontalY =
            offsetY + (invertY ? -margin + tickLength + tickMargin : size + margin - tickLength - tickMargin);
        for (size_t tick = 0; tick < numTicks; ++tick) {
            const float t = static_cast<float>(tick) / (numTicks - 1);
            const float px = lerp(offsetX, offsetX + size, t);
            const float py = lerp(offsetY, offsetY + size, t);
            const float pValue = lerp(columnInfos[i].MinimumValue(), columnInfos[i].MaximumValue(), t);
            const std::string pLabelX = to_string(pValue, tickPrecisionX);
            const std::string pLabelY = to_string(pValue, tickPrecisionY);
            if (drawOuter && i < columnCount - 1) {
                if (invertY) {
                    this->axisFont.DrawString(ortho, axisColor.data(), px, -tickLength - tickMargin, tickSize, false,
                        pLabelX.c_str(), core::utility::SDFFont::ALIGN_CENTER_TOP);
                } else {
                    this->axisFont.DrawString(ortho, axisColor.data(), px, totalSize + tickLength + tickMargin,
                        tickSize, false, pLabelX.c_str(), core::utility::SDFFont::ALIGN_CENTER_BOTTOM);
                }
            }
            if (drawDiagonal && i < columnCount - 1) {
                this->axisFont.DrawString(ortho, axisColor.data(), px, horizontalY, tickSize, false, pLabelX.c_str(),
                    invertY ? core::utility::SDFFont::ALIGN_CENTER_BOTTOM : core::utility::SDFFont::ALIGN_CENTER_TOP);
            }

            if (drawOuter && i > 0) {
                this->axisFont.DrawString(ortho, axisColor.data(), -tickLength - tickMargin, py, tickSize, false,
                    pLabelY.c_str(), core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
            }
            if (drawDiagonal && i > 0) {
                this->axisFont.DrawString(ortho, axisColor.data(), offsetX - margin + tickLength + tickMargin, py,
                    tickSize, false, pLabelY.c_str(), core::utility::SDFFont::ALIGN_LEFT_MIDDLE);
            }
        }
    }

    this->axisFont.BatchDrawString(ortho);

    debugPop();
}

void ScatterplotMatrixRenderer2D::drawScientificAxis(glm::mat4 ortho) {
    debugPush(2, "drawScientificAxis");

    const auto axisColor = this->axisColorParam.Param<core::param::ColorParam>()->Value();
    const auto columnCount = this->floatTable->GetColumnsCount();
    const auto columnInfos = this->floatTable->GetColumnsInfos();
    const float size = this->cellSizeParam.Param<core::param::FloatParam>()->Value();
    const float margin = this->cellMarginParam.Param<core::param::FloatParam>()->Value();
    const float nameSize = this->cellNameSizeParam.Param<core::param::FloatParam>()->Value();
    const float tickLabelSize = this->axisTickSizeParam.Param<core::param::FloatParam>()->Value();
    const GLfloat tickLength = this->axisTickLengthParam.Param<core::param::FloatParam>()->Value();
    const float tickMargin = this->axisTickMarginParam.Param<core::param::FloatParam>()->Value();
    const bool invertY = this->cellInvertYParam.Param<core::param::BoolParam>()->Value();
    const float axisWidth = this->axisWidthParam.Param<core::param::FloatParam>()->Value();

    // Compute cell size in viewport space.
    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    auto ndcSpaceSize = screenLastMVP * glm::vec4(size, size, 0.0f, 0.0f);
    auto screenSpaceSize = vislib::math::Vector<float, 2>(
        currentViewRes.x / 2.0 * ndcSpaceSize.x, currentViewRes.y / 2.0 * ndcSpaceSize.y);
    float approximateLineWidth = screenSpaceSize.X() * axisWidth * 0.005;
    // 0: no grid <-> 3: big,mid,small grid
    GLint recursiveDepth = 0;
    if (approximateLineWidth / 25.0 > 1.0) {
        recursiveDepth = 3;
    } else if (approximateLineWidth / 5.0 > 1.0) {
        recursiveDepth = 2;
    } else if (approximateLineWidth > 1.0) {
        recursiveDepth = 1;
    }

    this->scientificAxisShader->use();

    // Blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    // Transformation uniform.
    glUniformMatrix4fv(this->scientificAxisShader->getUniformLocation("modelViewProjection"), 1, GL_FALSE,
        glm::value_ptr(screenLastMVP));

    // Other uniforms.
    glUniform1ui(this->scientificAxisShader->getUniformLocation("depth"), recursiveDepth);
    glUniform4fv(this->scientificAxisShader->getUniformLocation("axisColor"), 1,
        this->axisColorParam.Param<core::param::ColorParam>()->Value().data());
    this->scientificAxisShader->setUniform("tickLength", tickLength);
    this->scientificAxisShader->setUniform("axisWidth", axisWidth);

    // Render all plots at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(0));

    glDrawArraysInstanced(GL_QUADS, 0, 4, this->plots.size());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    glDisable(GL_TEXTURE_1D);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    this->axisFont.ClearBatchDrawCache();
    this->axisFont.SetSmoothMode(smoothFontParam.Param<core::param::BoolParam>()->Value());

    for (size_t i = 0; i < columnCount; ++i) {
        // this will be bottom left of diagonal cell
        const float offsetX = i * (size + margin);
        const float offsetY = (invertY ? (columnCount - i - 1) : i) * (size + margin);

        // Labels
        std::string label = columnInfos[i].Name();
        this->axisFont.DrawString(ortho, axisColor.data(), offsetX, offsetY + size, size, size, nameSize, false,
            label.c_str(), core::utility::SDFFont::ALIGN_CENTER_MIDDLE);

        float delta = columnInfos[i].MaximumValue() - columnInfos[i].MinimumValue();
        // Tick sizes: big *25, mid *5, small *1
        float tickSize = rangeToSmallStep(columnInfos[i].MaximumValue(), columnInfos[i].MinimumValue()) * 25;
        float firstTick = ceil(columnInfos[i].MinimumValue() / tickSize) * tickSize;

        float horizontalY = offsetY + (invertY ? -margin + tickMargin : size + margin - tickMargin);

        for (float tickPos = firstTick; tickPos <= columnInfos[i].MaximumValue(); tickPos += tickSize) {
            float normalized = (tickPos - columnInfos[i].MinimumValue()) / delta;
            float offset = normalized * size;

            const std::string pLabel = to_string(tickPos);

            // Tick labels for x axis
            if (i < columnCount - 1) {
                this->axisFont.DrawString(ortho, axisColor.data(), offsetX + offset, horizontalY, tickLabelSize, false,
                    pLabel.c_str(),
                    invertY ? core::utility::SDFFont::ALIGN_CENTER_BOTTOM : core::utility::SDFFont::ALIGN_CENTER_TOP);
            }
            // Tick labels for y axis
            if (i > 0) {
                this->axisFont.DrawString(ortho, axisColor.data(), offsetX - margin + tickMargin, offsetY + offset,
                    tickLabelSize, false, pLabel.c_str(), core::utility::SDFFont::ALIGN_LEFT_MIDDLE);
            }
        }
    }

    this->axisFont.BatchDrawString(ortho);

    debugPop();
}

void ScatterplotMatrixRenderer2D::bindMappingUniforms(std::unique_ptr<glowl::GLSLProgram>& shader) {
    auto valueMapping = this->valueMappingParam.Param<core::param::EnumParam>()->Value();
    glUniform1i(shader->getUniformLocation("valueMapping"), valueMapping);

    auto columnInfos = this->floatTable->GetColumnsInfos();
    if (map.valueIdx.has_value()) {
        GLfloat valueColumnMinMax[] = {
            columnInfos[map.valueIdx.value()].MinimumValue(), columnInfos[map.valueIdx.value()].MaximumValue()};
        glUniform1i(shader->getUniformLocation("valueColumn"), map.valueIdx.value());
        glUniform2fv(shader->getUniformLocation("valueColumnMinMax"), 1, valueColumnMinMax);
    } else {
        glUniform1i(shader->getUniformLocation("valueColumn"), -1);
        glUniform2f(shader->getUniformLocation("valueColumnMinMax"), 0.f, 1.f);
    }
    glUniform1f(
        shader->getUniformLocation("alphaScaling"), this->alphaScalingParam.Param<core::param::FloatParam>()->Value());
    glUniform4fv(shader->getUniformLocation("pickColor"), 1,
        this->pickColorParam.Param<core::param::ColorParam>()->Value().data());

    this->transferFunction->BindConvenience(shader, GL_TEXTURE0, 0);
}

void ScatterplotMatrixRenderer2D::bindFlagsAttribute() {
    if (this->readFlags->hasUpdate()) {
        this->flagsBufferVersion = this->readFlags->version();
    }
    auto count = this->floatTable->GetRowsCount();
    this->readFlags->getData()->validateFlagCount(count);
    this->readFlags->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, FlagsBindingPoint);
}

void ScatterplotMatrixRenderer2D::drawPoints() {
    if (this->screenValid) {
        return;
    }

    debugPush(11, "drawPoints");

    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    // Point sprites.
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glPointSize(std::max(viewport[2], viewport[3]));

    this->pointShader->use();
    this->bindAndClearScreen();
    this->bindMappingUniforms(this->pointShader);

    // Transformation uniforms.
    glUniform4fv(this->pointShader->getUniformLocation("viewport"), 1, viewport);
    glUniformMatrix4fv(
        this->pointShader->getUniformLocation("modelViewProjection"), 1, GL_FALSE, glm::value_ptr(screenLastMVP));

    // Other uniforms.
    const auto columnCount = this->floatTable->GetColumnsCount();
    glUniform1i(this->pointShader->getUniformLocation("rowStride"), columnCount);
    glUniform1f(this->pointShader->getUniformLocation("kernelWidth"),
        this->kernelWidthParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->pointShader->getUniformLocation("kernelType"),
        this->kernelTypeParam.Param<core::param::EnumParam>()->Value());
    glUniform1i(this->pointShader->getUniformLocation("attenuateSubpixel"),
        this->alphaAttenuateSubpixelParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);

    this->bindFlagsAttribute();

    // Setup streaming.
    // const GLuint numBuffers = 3;
    // const GLuint bufferSize = 32 * 1024 * 1024;
    const float* data = this->floatTable->GetData();
    const GLuint dataStride = columnCount * sizeof(float);
    const GLuint dataItems = this->floatTable->GetRowsCount();
    this->valueSSBO.SetData(data, dataStride, dataStride, dataItems);

    // For each chunk of values, render all points in the lower half of the scatterplot matrix at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(0));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle(0));
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glDrawArraysInstanced(GL_POINTS, 0, static_cast<GLsizei>(dataItems), this->plots.size());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_1D, 0);
    this->unbindScreen();
    glUseProgram(0);

    glPointSize(1);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    debugPop();
}

void ScatterplotMatrixRenderer2D::drawPointTriangleSprites() {
    if (this->screenValid || !currentCamera.has_value()) {
        return;
    }

    debugPush(11, "drawPointTriangleSprites");

    //TODO use different
    this->pointTriangleSpriteShader->use();
    this->bindAndClearScreen();
    this->bindMappingUniforms(this->pointTriangleSpriteShader);

    // Transformation uniforms.
    auto ortho_params = currentCamera.value().get<core::view::Camera::OrthographicParameters>();
    glm::vec2 frustrumSize =
        glm::vec2(ortho_params.frustrum_height * ortho_params.aspect, ortho_params.frustrum_height);
    glUniform2fv(this->pointTriangleSpriteShader->getUniformLocation("frustrumSize"), 1, glm::value_ptr(frustrumSize));
    glUniform2iv(this->pointTriangleSpriteShader->getUniformLocation("viewRes"), 1, glm::value_ptr(currentViewRes));
    glUniformMatrix4fv(this->pointTriangleSpriteShader->getUniformLocation("modelViewProjection"), 1, GL_FALSE,
        glm::value_ptr(screenLastMVP));

    // Other uniforms.
    const auto columnCount = this->floatTable->GetColumnsCount();
    glUniform1i(this->pointTriangleSpriteShader->getUniformLocation("rowStride"), columnCount);
    glUniform1f(this->pointTriangleSpriteShader->getUniformLocation("kernelWidth"),
        this->kernelWidthParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->pointTriangleSpriteShader->getUniformLocation("kernelType"),
        this->kernelTypeParam.Param<core::param::EnumParam>()->Value());
    glUniform1i(this->pointTriangleSpriteShader->getUniformLocation("attenuateSubpixel"),
        this->alphaAttenuateSubpixelParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);

    this->bindFlagsAttribute();

    // Setup streaming.
    // const GLuint numBuffers = 3;
    // const GLuint bufferSize = 32 * 1024 * 1024;
    const float* data = this->floatTable->GetData();
    const GLuint dataStride = columnCount * sizeof(float);
    const GLuint dataItems = this->floatTable->GetRowsCount();
    this->valueSSBO.SetData(data, dataStride, dataStride, dataItems);

    // For each chunk of values, render all points in the lower half of the scatterplot matrix at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(0));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle(0));
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glDrawArraysInstanced(GL_TRIANGLES, 0, static_cast<GLsizei>(dataItems) * 3, this->plots.size());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_1D, 0);
    this->unbindScreen();
    glUseProgram(0);

    debugPop();
}

void ScatterplotMatrixRenderer2D::drawLines() {
    if (this->screenValid) {
        return;
    }

    debugPush(12, "drawLines");

    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    this->lineShader->use();
    this->bindAndClearScreen();
    this->bindMappingUniforms(this->lineShader);

    // Transformation uniforms.
    glUniform4fv(this->lineShader->getUniformLocation("viewport"), 1, viewport);
    glUniformMatrix4fv(
        this->lineShader->getUniformLocation("modelViewProjection"), 1, GL_FALSE, glm::value_ptr(screenLastMVP));

    // Other uniforms.
    const auto columnCount = this->floatTable->GetColumnsCount();
    glUniform1i(this->lineShader->getUniformLocation("rowStride"), columnCount);
    glUniform1f(this->lineShader->getUniformLocation("kernelWidth"),
        this->kernelWidthParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->lineShader->getUniformLocation("kernelType"),
        this->kernelTypeParam.Param<core::param::EnumParam>()->Value());
    glUniform1i(this->lineShader->getUniformLocation("attenuateSubpixel"),
        this->alphaAttenuateSubpixelParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);

    this->bindFlagsAttribute();

    // Setup streaming.
    const GLuint numBuffers = 3;
    const GLuint bufferSize = 32 * 1024 * 1024;
    const float* data = this->floatTable->GetData();
    const GLuint dataStride = columnCount * sizeof(float);
    const GLuint dataItems = this->floatTable->GetRowsCount();
    this->valueSSBO.SetData(data, dataStride, dataStride, dataItems);

    // For each chunk of values, render all points in the lower half of the scatterplot matrix at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(0));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle(0));
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if (!this->splitLinesByValueParam.Param<core::param::BoolParam>()->Value()) {
        glDrawArraysInstanced(GL_LINE_STRIP, 0, static_cast<GLsizei>(dataItems), this->plots.size());
    } else {
        auto col_idx = nameToIndex(
            this->floatTable, this->lineConnectedValueSelectorParam.Param<core::param::FlexEnumParam>()->Value());
        if (col_idx.has_value()) {
            // Connect neighboring dataItems with the same value in the idx column.
            std::size_t row_id = 0;
            while (row_id < dataItems) {
                float current_value = floatTable->GetData(col_idx.value(), row_id);
                const GLint line_start = static_cast<GLint>(row_id);
                row_id++;
                while (row_id < dataItems && floatTable->GetData(col_idx.value(), row_id) == current_value) {
                    row_id++;
                }
                glDrawArraysInstanced(
                    GL_LINE_STRIP, line_start, static_cast<GLsizei>(row_id - line_start), this->plots.size());
            }
        }
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_1D, 0);
    this->unbindScreen();
    glUseProgram(0);

    glDisable(GL_TEXTURE_1D);

    debugPop();
}

struct TriangulationVertex {
    float x;
    float y;
    float value;
};

void ScatterplotMatrixRenderer2D::validateTriangulation() {
    if (this->trianglesValid) {
        return;
    }
    auto rowCount = this->floatTable->GetRowsCount();
    std::function valueCallback = [this](size_t index) -> float { return 1; };
    if (this->map.valueIdx.has_value())
        valueCallback = [this](size_t index) -> float {
            auto columnInfos = this->floatTable->GetColumnsInfos()[this->map.valueIdx.value()];
            auto minValue = columnInfos.MinimumValue();
            auto maxValue = columnInfos.MaximumValue();
            float value = this->floatTable->GetData(this->map.valueIdx.value(), index);
            return (value - minValue) / (maxValue - minValue);
        };


    std::vector<TriangulationVertex> vertices;
    std::vector<GLuint> indices;
    for (const auto& plot : this->plots) {
        std::vector<double> coords;
        std::vector<double> values;

        // Copy coordinates for Delaunator.
        for (size_t i = 0; i < rowCount; ++i) {
            const float xValue = this->floatTable->GetData(plot.indexX, i);
            const float yValue = this->floatTable->GetData(plot.indexY, i);
            const float xPos = (xValue - plot.minX) / (plot.maxX - plot.minX);
            const float yPos = (yValue - plot.minY) / (plot.maxY - plot.minY);
            coords.push_back(plot.offsetX + xPos * plot.sizeX);
            coords.push_back(plot.offsetY + yPos * plot.sizeY);
            values.push_back(valueCallback(i));
        }

        // Compute initial Delauney triangulation.
        delaunator::Delaunator d(coords);

        // Smooth triangulation by adding new vertices.
        auto smoothIterations = this->triangulationSmoothnessParam.Param<core::param::IntParam>()->Value();
        for (size_t i = 0; i < smoothIterations; i++) {
            for (size_t triangleIndex = 0; triangleIndex < d.triangles.size(); triangleIndex += 3) {
                size_t aIndex = d.triangles[triangleIndex];
                size_t bIndex = d.triangles[triangleIndex + 1];
                size_t cIndex = d.triangles[triangleIndex + 2];

                // Insert centroid.
                double sumX = d.coords[2 * aIndex] + d.coords[2 * bIndex] + d.coords[2 * cIndex];
                double sumY = d.coords[2 * aIndex + 1] + d.coords[2 * bIndex + 1] + d.coords[2 * cIndex + 1];
                double sumValue = values[aIndex] + values[bIndex] + values[cIndex];
                coords.push_back(sumX / 3.0);
                coords.push_back(sumY / 3.0);
                values.push_back(sumValue / 3.0);
            }

            // Recompute Delauney triangulation.
            d.~Delaunator();
            new (&d) delaunator::Delaunator(coords);
        }

        // We need to offset indices, thus rember one before adding vertices.
        const auto indexOffset = static_cast<GLuint>(vertices.size());

        // Copy vertices to vertex buffer.
        for (size_t vertexIndex = 0; vertexIndex < values.size(); vertexIndex++) {
            TriangulationVertex vertex = {coords[vertexIndex * 2], coords[vertexIndex * 2 + 1], values[vertexIndex]};
            vertices.push_back(vertex);
        }

        // Copy indices to index buffer.
        for (auto triangle : d.triangles) {
            indices.push_back(indexOffset + triangle);
        }
    }

    // Delete old buffers, if present.
    if (triangleVBO != 0 || triangleIBO != 0) {
        glDeleteBuffers(1, &triangleVBO);
        glDeleteBuffers(1, &triangleIBO);
        triangleVBO = 0;
        triangleIBO = 0;
        triangleVertexCount = 0;
    }

    // Create vertex buffer and index buffer (streaming is not possible due to triangulation, anyway)
    glGenBuffers(1, &triangleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(TriangulationVertex), &vertices[0], GL_STATIC_DRAW);
    glGenBuffers(1, &triangleIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangleIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);
    triangleVertexCount = indices.size();

    this->trianglesValid = true;
}

void ScatterplotMatrixRenderer2D::drawTriangulation() {
    if (this->screenValid) {
        return;
    }

    debugPush(13, "drawTriangulation");

    this->validateTriangulation();

    this->triangleShader->use();
    this->bindAndClearScreen();
    this->bindMappingUniforms(this->triangleShader);

    // Bind buffers.
    glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TriangulationVertex), reinterpret_cast<GLvoid**>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 1, GL_FLOAT, GL_FALSE, sizeof(TriangulationVertex), reinterpret_cast<GLvoid**>(sizeof(float) * 2));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangleIBO);

    // Set uniforms.
    glUniformMatrix4fv(
        this->triangleShader->getUniformLocation("modelViewProjection"), 1, GL_FALSE, glm::value_ptr(screenLastMVP));

    // Emit draw call.
    glDrawElements(GL_TRIANGLES, triangleVertexCount, GL_UNSIGNED_INT, nullptr);
    this->unbindScreen();
    glUseProgram(0);

    debugPop();
}

void ScatterplotMatrixRenderer2D::validateText(glm::mat4 ortho) {
    if (this->textValid) {
        return;
    }

    this->textFont.ClearBatchDrawCache();

    const auto columnInfos = this->floatTable->GetColumnsInfos();
    const auto rowCount = this->floatTable->GetRowsCount();

    const float labelSize = this->labelSizeParam.Param<core::param::FloatParam>()->Value();
    for (size_t i = 0; i < rowCount; ++i) {
        for (const auto& plot : this->plots) {
            const float xValue = this->floatTable->GetData(plot.indexX, i);
            const float yValue = this->floatTable->GetData(plot.indexY, i);
            const float xPos = (xValue - plot.minX) / (plot.maxX - plot.minX);
            const float yPos = (yValue - plot.minY) / (plot.maxY - plot.minY);

            // const size_t colorIndex = this->floatTable->GetData(this->map.valueIdx, i);
            float labelColor[4] = {0, 0, 0, 1}; // TODO: param please!

            // XXX: this will be a lot more useful when have support for string-columns!
            std::string label = to_string(this->floatTable->GetData(map.labelIdx, i));

            this->textFont.DrawString(ortho, labelColor, plot.offsetX + xPos * plot.sizeX,
                plot.offsetY + yPos * plot.sizeY, labelSize, false, label.c_str(),
                core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
        }
    }

    this->textValid = true;
}

void ScatterplotMatrixRenderer2D::drawText(glm::mat4 ortho) {
    debugPush(14, "drawText");

    validateText(ortho);

    this->textFont.BatchDrawString(ortho);

    debugPop();
}

void ScatterplotMatrixRenderer2D::drawPickIndicator() {
    debugPush(15, "drawPickIndicator");

    this->pickIndicatorShader->use();

    float color[] = {0.0, 1.0, 1.0, 1.0};
    glUniformMatrix4fv(this->pickIndicatorShader->getUniformLocation("modelViewProjection"), 1, GL_FALSE,
        glm::value_ptr(screenLastMVP));
    glUniform2f(this->pickIndicatorShader->getUniformLocation("mouse"), this->mouse.x, this->mouse.y);
    glUniform1f(this->pickIndicatorShader->getUniformLocation("pickRadius"),
        this->pickRadiusParam.Param<core::param::FloatParam>()->Value());
    glUniform4fv(this->pickIndicatorShader->getUniformLocation("indicatorColor"), 1, color);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(0);

    debugPop();
}

void ScatterplotMatrixRenderer2D::drawMouseLabels(glm::mat4 ortho) {
    const float cellSize = this->cellSizeParam.Param<core::param::FloatParam>()->Value();
    const float cellMargin = this->cellMarginParam.Param<core::param::FloatParam>()->Value();
    const auto axisColor = this->axisColorParam.Param<core::param::ColorParam>()->Value();
    const auto columnCount = this->floatTable->GetColumnsCount();
    const auto columnInfos = this->floatTable->GetColumnsInfos();
    const float nameSize = this->cellNameSizeParam.Param<core::param::FloatParam>()->Value();
    const GLsizei numTicks = this->axisTicksParam.Param<core::param::IntParam>()->Value();
    const GLfloat tickLength = this->axisTickLengthParam.Param<core::param::FloatParam>()->Value();
    const float tickMargin = this->axisTickMarginParam.Param<core::param::FloatParam>()->Value();
    const float tickSize = this->axisTickSizeParam.Param<core::param::FloatParam>()->Value();
    const int tickPrecisionX = this->axisTickPrecisionX.Param<core::param::IntParam>()->Value();
    const int tickPrecisionY = this->axisTickPrecisionY.Param<core::param::IntParam>()->Value();
    const bool invertY = this->cellInvertYParam.Param<core::param::BoolParam>()->Value();
    const float xLabelMargin = this->outerXLabelMarginParam.Param<core::param::FloatParam>()->Value();
    const float yLabelMargin = this->outerYLabelMarginParam.Param<core::param::FloatParam>()->Value();

    if (this->mouse.x < 0 || this->mouse.y < 0) {
        return;
    }

    // cell ids as global grid coords
    const int cellPosIdX = static_cast<int>(this->mouse.x / (cellSize + cellMargin));
    const int cellPosIdY = static_cast<int>(this->mouse.y / (cellSize + cellMargin));
    if (this->mouse.x - static_cast<float>(cellPosIdX) * (cellSize + cellMargin) > cellSize ||
        this->mouse.y - static_cast<float>(cellPosIdY) * (cellSize + cellMargin) > cellSize) {
        return;
    }

    // map to actual column ids
    const int cellColIdX = cellPosIdX;
    const int cellColIdY = invertY ? (columnCount - cellPosIdY - 1) : cellPosIdY;

    if (cellColIdX >= columnCount || cellColIdY >= columnCount || cellColIdX >= cellColIdY) {
        return;
    }

    this->axisFont.ClearBatchDrawCache();
    this->axisFont.SetSmoothMode(smoothFontParam.Param<core::param::BoolParam>()->Value());

    // bottom left of cell
    const float offsetX = static_cast<float>(cellPosIdX) * (cellSize + cellMargin);
    const float offsetY = static_cast<float>(cellPosIdY) * (cellSize + cellMargin);

    // Labels
    std::string labelX = columnInfos[cellColIdX].Name();
    std::string labelY = columnInfos[cellColIdY].Name();

    this->axisFont.DrawString(ortho, axisColor.data(), offsetX,
        offsetY - tickLength - tickMargin - tickSize - xLabelMargin, cellSize, cellSize, nameSize, false,
        labelX.c_str(), core::utility::SDFFont::ALIGN_CENTER_TOP);

    this->axisFont.SetRotation(90.0, 0.0, 0.0, 1.0);
    this->axisFont.DrawString(ortho, axisColor.data(), offsetY,
        -offsetX + cellSize + tickLength + tickMargin + tickSize + yLabelMargin, cellSize, cellSize, nameSize, false,
        labelY.c_str(), core::utility::SDFFont::ALIGN_CENTER_BOTTOM);
    this->axisFont.ResetRotation();

    // draw tick labels
    // float horizontalY = offsetY + (invertY ? -margin + tickLength + tickMargin : size + margin - tickLength - tickMargin);
    for (size_t tick = 0; tick < numTicks; ++tick) {
        const float t = static_cast<float>(tick) / static_cast<float>(numTicks - 1);
        const float px = lerp(offsetX, offsetX + cellSize, t);
        const float py = lerp(offsetY, offsetY + cellSize, t);
        const float pValueX = lerp(columnInfos[cellColIdX].MinimumValue(), columnInfos[cellColIdX].MaximumValue(), t);
        const float pValueY = lerp(columnInfos[cellColIdY].MinimumValue(), columnInfos[cellColIdY].MaximumValue(), t);
        const std::string pLabelX = to_string(pValueX, tickPrecisionX);
        const std::string pLabelY = to_string(pValueY, tickPrecisionY);

        this->axisFont.DrawString(ortho, axisColor.data(), px, offsetY - tickLength - tickMargin, tickSize, false,
            pLabelX.c_str(), core::utility::SDFFont::ALIGN_CENTER_TOP);
        this->axisFont.DrawString(ortho, axisColor.data(), offsetX - tickLength - tickMargin, py, tickSize, false,
            pLabelY.c_str(), core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
    }

    this->axisFont.BatchDrawString(ortho);
}

void ScatterplotMatrixRenderer2D::bindAndClearScreen() {
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &screenRestoreFBO);

    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    if (!this->screenFBO || this->screenFBO->getWidth() != static_cast<int>(viewport[2]) ||
        this->screenFBO->getHeight() != static_cast<int>(viewport[3])) {
        this->screenFBO = std::make_unique<glowl::FramebufferObject>(viewport[2], viewport[3]);
        this->screenFBO->createColorAttachment(GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }

    this->screenFBO->bind();

    // Blending and clear color.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    switch (this->valueMappingParam.Param<core::param::EnumParam>()->Value()) {
    case VALUE_MAPPING_KERNEL_BLEND:
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        break;
    case VALUE_MAPPING_KERNEL_DENSITY:
    case VALUE_MAPPING_WEIGHTED_KERNEL_DENSITY:
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        break;
    default:
        assert(false && "Unexpected value");
    }
    glDisable(GL_DEPTH_TEST);

    // Clear FBO.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void ScatterplotMatrixRenderer2D::unbindScreen() {
    glBindFramebuffer(GL_FRAMEBUFFER, screenRestoreFBO);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    this->screenValid = true;
}

void ScatterplotMatrixRenderer2D::drawScreen() {
    debugPush(20, "drawScreen");

    // Enable shader.
    this->screenShader->use();
    this->bindMappingUniforms(this->screenShader);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    // Screen texture.
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE1);
    glUniform1i(this->screenShader->getUniformLocation("screenTexture"), 1);
    this->screenFBO->bindColorbuffer(0);

    // Other uniforms.
    const float contourColor[] = {0.0, 1.0, 0.0, 1.0};                                   // TODO: param
    const float contourSize = 0.5;                                                       // TODO: param
    const float contourIsoValues[] = {0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // TODO: param
    const int contourIsoValueCount = 1;                                                  // TODO: infer
    glUniform4fv(this->screenShader->getUniformLocation("contourColor"), 1, contourColor);
    glUniform1f(this->screenShader->getUniformLocation("contourSize"), contourSize);
    glUniform1fv(this->screenShader->getUniformLocation("contourIsoValues"), 10, contourIsoValues);
    glUniform1i(this->screenShader->getUniformLocation("contourIsoValueCount"), contourIsoValueCount);

    // Emit draw call.
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindTexture(GL_TEXTURE_1D, 0);
    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    // glEnable(GL_DEPTH_TEST);

    this->transferFunction->UnbindConvenience(); // bound in bindMappingUniforms()
    glUseProgram(0);

    debugPop();
}

void ScatterplotMatrixRenderer2D::updateSelection() {
    this->debugPush(42, "splom::picking");

    this->pickProgram->use();

    glUniform2f(pickProgram->getUniformLocation("mouse"), this->mouse.x, this->mouse.y);
    glUniform1i(pickProgram->getUniformLocation("numPlots"), this->plots.size());
    glUniform1ui(pickProgram->getUniformLocation("itemCount"), this->floatTable->GetRowsCount());
    glUniform1i(pickProgram->getUniformLocation("rowStride"), this->floatTable->GetColumnsCount());
    glUniform1f(pickProgram->getUniformLocation("kernelWidth"),
        this->kernelWidthParam.Param<core::param::FloatParam>()->Value());
    glUniform1f(
        pickProgram->getUniformLocation("pickRadius"), this->pickRadiusParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(pickProgram->getUniformLocation("selector"), static_cast<int>(this->mouse.selector));
    glUniform1i(pickProgram->getUniformLocation("reset"), static_cast<int>(this->selectionNeedsReset));
    this->selectionNeedsReset = false;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(0));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle(0));
    this->readFlags->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, FlagsBindingPoint);

    GLuint groupCounts[3];
    computeDispatchSizes(this->floatTable->GetRowsCount(), pickWorkgroupSize, maxWorkgroupCount, groupCounts);

    glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);

    this->flagsBufferVersion++;

    auto writeFlags = writeFlagStorageSlot.CallAs<core_gl::FlagCallWrite_GL>();
    if (this->readFlags != nullptr && writeFlags != nullptr) {
        writeFlags->setData(this->readFlags->getData(), this->flagsBufferVersion);
        (*writeFlags)(core_gl::FlagCallWrite_GL::CallGetData);
    }
    this->debugPop();
    this->screenValid = false;
}

bool ScatterplotMatrixRenderer2D::resetSelectionCallback(core::param::ParamSlot& caller) {
    this->selectionNeedsUpdate = true;
    this->selectionNeedsReset = true;
    return true;
}
