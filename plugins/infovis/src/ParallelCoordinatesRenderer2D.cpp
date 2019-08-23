#include "stdafx.h"
#include "ParallelCoordinatesRenderer2D.h"

#include <algorithm>
#include "mmcore/CoreInstance.h"
#include "mmcore/FlagCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmstd_datatools/table/TableDataCall.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include <array>
#include <iostream>
#include "vislib/graphics/InputModifiers.h"

//#define FUCK_THE_PIPELINE
//#define USE_TESSELLATION
//#define REMOVE_TEXT

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

ParallelCoordinatesRenderer2D::ParallelCoordinatesRenderer2D(void)
    : Renderer2D()
    , getDataSlot("getdata", "Float table input")
    , getTFSlot("getTF", "connects to the transfer function")
    , getFlagsSlot("getFlags", "connects to the flag storage")
    , currentHash(0xFFFFFFFF)
    , currentFlagsVersion(0xFFFFFFFF)
    , densityFBO()
    , drawModeSlot("drawMode", "Draw mode")
    , drawSelectedItemsSlot("drawSelectedItems", "Draw selected items")
    , selectedItemsColorSlot("selectedItemsColor", "Color for selected items")
    , selectedItemsAlphaSlot("selectedItemsAlpha", "Alpha for selected items")
    , selectedItemsColor()
    , drawOtherItemsSlot("drawOtherItems", "Draw other (e.g., non-selected) items")
    , otherItemsColorSlot("otherItemsColor", "Color for other items (e.g., non-selected)")
    , otherItemsAlphaSlot("otherItemsAlpha", "Alpha for other items (e.g., non-selected)")
    , otherItemsColor()
    , drawAxesSlot("drawAxes", "Draw dimension axes")
    , axesColorSlot("axesColor", "Color for dimension axes")
    , axesColor()
    , filterIndicatorColorSlot("filterIndicatorCol", "Color for filter indicators")
    , filterIndicatorColor()
    , selectionModeSlot("selectionMode", "Selection mode")
    , drawSelectionIndicatorSlot("drawSelectionIndicator", "Draw selection indicator")
    , selectionIndicatorColorSlot("selectionIndicatorColor", "Color for selection indicator")
    , selectionIndicatorColor()
    , pickRadiusSlot("pickRadius", "Picking radius in object-space")
    , scaleToFitSlot("scaleToFit", "fit the diagram in the viewport")
    , glDepthTestSlot("glEnableDepthTest", "Toggle GLDEPTHTEST")
    , glLineSmoothSlot("glEnableLineSmooth", "Toggle GLLINESMOOTH")
    , glLineWidthSlot("glLineWidth", "Value for glLineWidth")
    , sqrtDensitySlot("sqrtDensity", "map root of density to transfer function (instead of linear mapping)")
    //, resetFlagsSlot("resetFlags", "Reset item flags to initial state")
    , resetFiltersSlot("resetFilters", "Reset dimension filters to initial state")
    , numTicks(5)
    , columnCount(0)
    , itemCount(0)
    , dataBuffer(0)
    , flagsBuffer(0)
    , minimumsBuffer(0)
    , maximumsBuffer(0)
    , axisIndirectionBuffer(0)
    , filtersBuffer(0)
    , minmaxBuffer(0)
    , interactionState(InteractionState::NONE)
    , pickedAxis(-1)
    , pickedIndicatorAxis(-1)
    , pickedIndicatorIndex(-1)
    , strokeStartX(0)
    , strokeStartY(0)
    , strokeEndX(0)
    , strokeEndY(0)
    , needSelectionUpdate(false)
    , needFlagsUpdate(false)
    , lastTimeStep(0)
    , font("Evolventa-SansSerif", core::utility::SDFFont::RenderType::RENDERTYPE_FILL) {

    this->getDataSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getFlagsSlot.SetCompatibleCall<core::FlagCallDescription>();
    this->MakeSlotAvailable(&this->getFlagsSlot);

    auto drawModes = new core::param::EnumParam(DRAW_DISCRETE);
    drawModes->SetTypePair(DRAW_DISCRETE, "Discrete");
    drawModes->SetTypePair(DRAW_CONTINUOUS, "Continuous");
    drawModes->SetTypePair(DRAW_HISTOGRAM, "Histogram");
    drawModeSlot.SetParameter(drawModes);
    this->MakeSlotAvailable(&drawModeSlot);

    drawSelectedItemsSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&drawSelectedItemsSlot);

    selectedItemsColorSlot << new core::param::StringParam("red");
    selectedItemsColorSlot.SetUpdateCallback(&ParallelCoordinatesRenderer2D::selectedItemsColorSlotCallback);
    this->MakeSlotAvailable(&selectedItemsColorSlot);
    selectedItemsAlphaSlot << new core::param::FloatParam(1.0f, 0.0f, 1.0f);
    selectedItemsAlphaSlot.SetUpdateCallback(&ParallelCoordinatesRenderer2D::selectedItemsColorSlotCallback);
    this->MakeSlotAvailable(&selectedItemsAlphaSlot);
    selectedItemsColorSlotCallback(selectedItemsColorSlot);

    drawOtherItemsSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&drawOtherItemsSlot);

    otherItemsColorSlot << new core::param::StringParam("gray");
    otherItemsColorSlot.SetUpdateCallback(&ParallelCoordinatesRenderer2D::otherItemsColorSlotCallback);
    this->MakeSlotAvailable(&otherItemsColorSlot);
    otherItemsAlphaSlot << new core::param::FloatParam(1.0f, 0.0f, 1.0f);
    otherItemsAlphaSlot.SetUpdateCallback(&ParallelCoordinatesRenderer2D::otherItemsColorSlotCallback);
    this->MakeSlotAvailable(&otherItemsAlphaSlot);
    otherItemsColorSlotCallback(otherItemsColorSlot);

    drawAxesSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&drawAxesSlot);

    axesColorSlot << new core::param::StringParam("white");
    axesColorSlot.SetUpdateCallback(&ParallelCoordinatesRenderer2D::axesColorSlotCallback);
    this->MakeSlotAvailable(&axesColorSlot);
    axesColorSlotCallback(axesColorSlot);

    filterIndicatorColorSlot << new core::param::StringParam("orange");
    filterIndicatorColorSlot.SetUpdateCallback(&ParallelCoordinatesRenderer2D::filterIndicatorColorSlotCallback);
    this->MakeSlotAvailable(&filterIndicatorColorSlot);
    filterIndicatorColorSlotCallback(filterIndicatorColorSlot);

    drawSelectionIndicatorSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&drawSelectionIndicatorSlot);

    selectionIndicatorColorSlot << new core::param::StringParam("MegaMolBlue");
    selectionIndicatorColorSlot.SetUpdateCallback(&ParallelCoordinatesRenderer2D::selectionIndicatorColorSlotCallback);
    this->MakeSlotAvailable(&selectionIndicatorColorSlot);
    selectionIndicatorColorSlotCallback(selectionIndicatorColorSlot);

    auto pickModes = new core::param::EnumParam(SELECT_PICK);
    pickModes->SetTypePair(SELECT_PICK, "Pick");
    pickModes->SetTypePair(SELECT_STROKE, "Stroke");
    selectionModeSlot.SetParameter(pickModes);
    this->MakeSlotAvailable(&selectionModeSlot);

    pickRadiusSlot << new core::param::FloatParam(0.1f, 0.01f, 10.0f);
    this->MakeSlotAvailable(&pickRadiusSlot);

    scaleToFitSlot << new core::param::BoolParam(false);
    scaleToFitSlot.SetUpdateCallback(this, &ParallelCoordinatesRenderer2D::scalingChangedCallback);
    this->MakeSlotAvailable(&scaleToFitSlot);

    glDepthTestSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&glDepthTestSlot);

    glLineSmoothSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&glLineSmoothSlot);

    glLineWidthSlot << new core::param::FloatParam(1.0f, 0.1f);
    this->MakeSlotAvailable(&glLineWidthSlot);

    sqrtDensitySlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&sqrtDensitySlot);

    // resetFlagsSlot << new core::param::ButtonParam();
    // resetFlagsSlot.SetUpdateCallback(this, &ParallelCoordinatesRenderer2D::resetFlagsSlotCallback);
    // this->MakeSlotAvailable(&resetFlagsSlot);

    resetFiltersSlot << new core::param::ButtonParam();
    resetFiltersSlot.SetUpdateCallback(this, &ParallelCoordinatesRenderer2D::resetFiltersSlotCallback);
    this->MakeSlotAvailable(&resetFiltersSlot);

    fragmentMinMax.resize(2);
}

ParallelCoordinatesRenderer2D::~ParallelCoordinatesRenderer2D(void) { this->Release(); }

bool ParallelCoordinatesRenderer2D::enableProgramAndBind(vislib::graphics::gl::GLSLShader& program) {
    program.Enable();
    // bindbuffer?
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dataBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, flagsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, minimumsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, maximumsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, axisIndirectionBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, filtersBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, minmaxBuffer);

    glUniform2f(program.ParameterLocation("scaling"), 1.0f, 1.0f); // scaling, whatever
    glUniformMatrix4fv(program.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(program.ParameterLocation("projection"), 1, GL_FALSE, projMatrix_column);
    glUniform1ui(program.ParameterLocation("dimensionCount"), this->columnCount);
    glUniform1ui(program.ParameterLocation("itemCount"), this->itemCount);

    glUniform2f(program.ParameterLocation("margin"), this->marginX, this->marginY);
    glUniform1f(program.ParameterLocation("axisDistance"), this->axisDistance);
    glUniform1f(program.ParameterLocation("axisHeight"), this->axisHeight);

    return true;
}

bool ParallelCoordinatesRenderer2D::create(void) {
    glGenBuffers(1, &dataBuffer);
    glGenBuffers(1, &flagsBuffer);
    glGenBuffers(1, &minimumsBuffer);
    glGenBuffers(1, &maximumsBuffer);
    glGenBuffers(1, &axisIndirectionBuffer);
    glGenBuffers(1, &filtersBuffer);
    glGenBuffers(1, &minmaxBuffer);
    glGenBuffers(1, &counterBuffer);

#ifndef REMOVE_TEXT
    if (!font.Initialise(this->GetCoreInstance())) return false;
#endif

    if (!makeProgram("::pc_axes_draw::axes", this->drawAxesProgram)) return false;
    if (!makeProgram("::pc_axes_draw::scales", this->drawScalesProgram)) return false;
    if (!makeProgram("::pc_axes_draw::filterindicators", this->drawFilterIndicatorsProgram)) return false;

    if (!makeProgram("::pc_item_stroke::indicator", this->drawStrokeIndicatorProgram)) return false;
    if (!makeProgram("::pc_item_pick::indicator", this->drawPickIndicatorProgram)) return false;

    if (!makeProgram("::pc_item_draw::discrete", this->drawItemsDiscreteProgram)) return false;
    if (!makeProgram("::pc_item_draw::muhaha", this->traceItemsDiscreteProgram)) return false;

    if (!makeProgram("::pc_item_draw::discTess", drawItemsDiscreteTessProgram)) return false;
    glGetIntegerv(GL_MAX_TESS_GEN_LEVEL, &this->maxAxes); // TODO we should reject data with more axes!
    this->isoLinesPerInvocation = maxAxes; // warning: for tesslevel n there are JUST n lines!!! not n+1 !!

    if (!makeProgram("::fragment_count", this->drawItemContinuousProgram)) return false;
    if (!makeProgram("::fragment_count", this->minMaxProgram)) return false;

    if (!makeProgram("::pc_item_draw::histogram", this->drawItemsHistogramProgram)) return false;

    if (!makeProgram("::pc_item_filter", this->filterProgram)) return false;
    if (!makeProgram("::pc_item_pick", this->pickProgram)) return false;
    if (!makeProgram("::pc_item_stroke", this->strokeProgram)) return false;

    glGetProgramiv(this->filterProgram, GL_COMPUTE_WORK_GROUP_SIZE, filterWorkgroupSize);
    glGetProgramiv(this->minMaxProgram, GL_COMPUTE_WORK_GROUP_SIZE, counterWorkgroupSize);
    glGetProgramiv(this->pickProgram, GL_COMPUTE_WORK_GROUP_SIZE, pickWorkgroupSize);
    glGetProgramiv(this->strokeProgram, GL_COMPUTE_WORK_GROUP_SIZE, strokeWorkgroupSize);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkgroupCount[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkgroupCount[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkgroupCount[2]);

    return true;
}

void ParallelCoordinatesRenderer2D::release(void) {
    glDeleteBuffers(1, &dataBuffer);
    glDeleteBuffers(1, &flagsBuffer);
    glDeleteBuffers(1, &minimumsBuffer);
    glDeleteBuffers(1, &maximumsBuffer);
    glDeleteBuffers(1, &axisIndirectionBuffer);
    glDeleteBuffers(1, &filtersBuffer);
    glDeleteBuffers(1, &minmaxBuffer);
    glDeleteBuffers(1, &counterBuffer);

    this->drawAxesProgram.Release();
}

int ParallelCoordinatesRenderer2D::mouseXtoAxis(float x) {
    float f = (x - this->marginX) / this->axisDistance;
    float frac = f - static_cast<long>(f);
    int integral = static_cast<int>(std::round(f));
    if (integral >= static_cast<int>(this->columnCount) || integral < 0) return -1;
    if (frac > 0.7 || frac < 0.3) {
        // vislib::sys::Log::DefaultLog.WriteInfo("picking axis %i at mouse position of axis %i",
        // axisIndirection[integral], integral);
        return axisIndirection[integral];
    } else {
        return -1;
    }
}

void ParallelCoordinatesRenderer2D::pickIndicator(float x, float y, int& axis, int& index) {
    axis = mouseXtoAxis(x);
    index = -1;
    if (axis != -1) {
        float val = (y - this->marginY) / this->axisHeight;
        float thresh = 0.1f;

        float middle = (this->filters[axis].upper + this->filters[axis].lower) * 0.5f;

        if (fabs(this->filters[axis].upper - val) < thresh && val > middle) {
            index = 1;
        } else if (fabs(this->filters[axis].lower - val) < thresh) {
            index = 0;
        }
    }
    if (index == -1) {
        axis = -1;
    }
}

bool ParallelCoordinatesRenderer2D::selectedItemsColorSlotCallback(core::param::ParamSlot& caller) {
    core::utility::ColourParser::FromString(
        this->selectedItemsColorSlot.Param<core::param::StringParam>()->Value(), 4, selectedItemsColor);
    selectedItemsColor[3] = this->selectedItemsAlphaSlot.Param<core::param::FloatParam>()->Value();
    return true;
}

bool ParallelCoordinatesRenderer2D::otherItemsColorSlotCallback(core::param::ParamSlot& caller) {
    core::utility::ColourParser::FromString(
        this->otherItemsColorSlot.Param<core::param::StringParam>()->Value(), 4, otherItemsColor);
    otherItemsColor[3] = this->otherItemsAlphaSlot.Param<core::param::FloatParam>()->Value();
    return true;
}
bool ParallelCoordinatesRenderer2D::axesColorSlotCallback(core::param::ParamSlot& caller) {
    core::utility::ColourParser::FromString(
        this->axesColorSlot.Param<core::param::StringParam>()->Value(), 4, axesColor);
    return true;
}
bool ParallelCoordinatesRenderer2D::filterIndicatorColorSlotCallback(core::param::ParamSlot& caller) {
    core::utility::ColourParser::FromString(
        this->filterIndicatorColorSlot.Param<core::param::StringParam>()->Value(), 4, filterIndicatorColor);
    return true;
}
bool ParallelCoordinatesRenderer2D::selectionIndicatorColorSlotCallback(core::param::ParamSlot& caller) {
    core::utility::ColourParser::FromString(
        this->selectionIndicatorColorSlot.Param<core::param::StringParam>()->Value(), 4, selectionIndicatorColor);
    return true;
}

bool ParallelCoordinatesRenderer2D::scalingChangedCallback(core::param::ParamSlot& caller) {
    this->computeScaling();
    return true;
}

// bool ParallelCoordinatesRenderer2D::resetFlagsSlotCallback(core::param::ParamSlot& caller) { return true; }

bool ParallelCoordinatesRenderer2D::resetFiltersSlotCallback(core::param::ParamSlot& caller) {
    for (GLuint i = 0; i < this->columnCount; i++) {
        this->filters[i].lower = 0.0f;
        this->filters[i].upper = 1.0f;
    }
    this->needFlagsUpdate = true;
    return true;
}

void ParallelCoordinatesRenderer2D::assertData(core::view::CallRender2D& call) {
    auto floats = getDataSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    if (floats == nullptr) return;
    auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
    if (tc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "ParallelCoordinatesRenderer2D requires a transfer function!");
        return;
    }
    auto flagsc = getFlagsSlot.CallAs<core::FlagCall>();
    if (flagsc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "ParallelCoordinatesRenderer2D requires a flag storage!");
        return;
    }

    floats->SetFrameID(static_cast<unsigned int>(call.Time()));
    (*floats)(1);
    (*floats)(0);
    call.SetTimeFramesCount(floats->GetFrameCount());
    auto hash = floats->DataHash();
    (*tc)(0);
    (*flagsc)(core::FlagCall::CallMapFlags);
    auto version = flagsc->GetVersion();

    if (hash != this->currentHash || this->lastTimeStep != static_cast<unsigned int>(call.Time())) {

        this->computeScaling();

        this->columnCount = static_cast<GLuint>(floats->GetColumnsCount());
        this->itemCount = static_cast<GLuint>(floats->GetRowsCount());
        this->axisIndirection.resize(columnCount);
        this->filters.resize(columnCount);
        this->minimums.resize(columnCount);
        this->maximums.resize(columnCount);
        this->names.resize(columnCount);
        for (GLuint x = 0; x < columnCount; x++) {
            axisIndirection[x] = x;
            filters[x].dimension = 0;
            filters[x].flags = 0;
            minimums[x] = floats->GetColumnsInfos()[x].MinimumValue();
            maximums[x] = floats->GetColumnsInfos()[x].MaximumValue();
            names[x] = floats->GetColumnsInfos()[x].Name();
            // TODO this is shit the user needs his real values DAMMIT!
            // hopefully fixed through proper axis labels.
            filters[x].lower = 0.0f; // minimums[x];
            filters[x].upper = 1.0f; // maximums[x];
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dataBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->columnCount * this->itemCount * sizeof(float), floats->GetData(),
            GL_STATIC_DRAW); // TODO: huh.
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, minimumsBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(GLfloat), this->minimums.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maximumsBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(GLfloat), this->maximums.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, axisIndirectionBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(GLuint), axisIndirection.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, filtersBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->columnCount * sizeof(DimensionFilter), this->filters.data(),
            GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, minmaxBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLfloat), fragmentMinMax.data(), GL_DYNAMIC_READ); // TODO: huh.
        this->currentHash = hash;
        this->lastTimeStep = static_cast<unsigned int>(call.Time());
    }

    if (version != this->currentFlagsVersion || version == 0) {
        flagsc->validateFlagsCount(itemCount);
        auto flagsvector = flagsc->GetFlags();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, flagsBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->itemCount * sizeof(core::FlagStorage::FlagItemType),
            flagsvector.get()->data(), GL_STATIC_DRAW);

        // give the data back
        flagsc->SetFlags(flagsvector);
        this->currentFlagsVersion = flagsc->GetVersion();
    }
    (*flagsc)(core::FlagCall::CallUnmapFlags);

    makeDebugLabel(GL_BUFFER, DEBUG_NAME(dataBuffer));
    makeDebugLabel(GL_BUFFER, DEBUG_NAME(flagsBuffer));
    makeDebugLabel(GL_BUFFER, DEBUG_NAME(minimumsBuffer));
    makeDebugLabel(GL_BUFFER, DEBUG_NAME(maximumsBuffer));
    makeDebugLabel(GL_BUFFER, DEBUG_NAME(axisIndirectionBuffer));
    makeDebugLabel(GL_BUFFER, DEBUG_NAME(filtersBuffer));
    makeDebugLabel(GL_BUFFER, DEBUG_NAME(minmaxBuffer));
}

void ParallelCoordinatesRenderer2D::computeScaling(void) {
    auto fc = getDataSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    if (fc == nullptr) return;

    this->marginX = 20.f;
    this->marginY = 20.f;
    this->axisDistance = 40.0f;
    this->bounds.SetLeft(0.0f);
    this->bounds.SetRight(2.0f * marginX + this->axisDistance * (fc->GetColumnsCount() - 1));

    if (this->scaleToFitSlot.Param<core::param::BoolParam>()->Value()) {
        // scale to fit
        float requiredHeight = this->bounds.Width() / windowAspect;
        this->axisHeight = requiredHeight - 3.0f * marginY;
    } else {
        this->axisHeight = 80.0f;
    }
    this->bounds.SetBottom(0.0f);
    this->bounds.SetTop(3.0f * marginY + this->axisHeight);
}

bool ParallelCoordinatesRenderer2D::GetExtents(core::view::CallRender2D& call) {
    windowAspect = static_cast<float>(call.GetViewport().AspectRatio());

    this->assertData(call);

    call.SetBoundingBox(this->bounds);

    return true;
}

bool ParallelCoordinatesRenderer2D::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    if (this->ctrlDown) {
        // these clicks go to the view
        return false;
    }

    // TODO could values from mouse move be updated from other threads? otherwise this copy is useless
    float mousePressedX = this->mouseX;
    float mousePressedY = this->mouseY;

    if (button != core::view::MouseButton::BUTTON_LEFT) {
        // ignore everything which is not left mouse button
        return false;
    }

    // any up/down event stops interaction, but only down changes selection
    this->interactionState = InteractionState::NONE;
    if (action == core::view::MouseButtonAction::PRESS) {
        this->pickedAxis = -1;
        this->pickedIndicatorAxis = -1;
        this->pickedIndicatorIndex = -1;

        if (this->altDown) {
            this->pickedAxis = mouseXtoAxis(mousePressedX);
            if (this->pickedAxis != -1) {
                this->interactionState = InteractionState::INTERACTION_DRAG;
            }
            return true;
        }

        if (this->shiftDown) {
            pickIndicator(mousePressedX, mousePressedY, this->pickedIndicatorAxis, this->pickedIndicatorIndex);
            if (this->pickedIndicatorAxis != -1) {
                this->interactionState = InteractionState::INTERACTION_FILTER;
            }
            return true;
        }

        this->interactionState = InteractionState::INTERACTION_SELECT;
        this->strokeStartX = mousePressedX;
        this->strokeStartY = mousePressedY;
        this->strokeEndX = mousePressedX;
        this->strokeEndY = mousePressedY;
        this->needSelectionUpdate = true;
        return true;
    }

    return true;
}

bool ParallelCoordinatesRenderer2D::OnMouseMove(double x, double y) {
    this->mouseX = x;
    this->mouseY = y;

    if (this->ctrlDown) {
        // these clicks go to the view
        return false;
    }

    if (this->interactionState == InteractionState::INTERACTION_DRAG) {
        int currAxis = mouseXtoAxis(this->mouseX);
        if (currAxis != this->pickedAxis && currAxis >= 0 && currAxis < static_cast<int>(this->columnCount)) {
            auto pickedAxisIt = std::find(this->axisIndirection.begin(), this->axisIndirection.end(), this->pickedAxis);
            int pickedIdx = std::distance(this->axisIndirection.begin(), pickedAxisIt);
            this->axisIndirection.erase(pickedAxisIt);

            auto currAxisIt = std::find(this->axisIndirection.begin(), this->axisIndirection.end(), currAxis);
            int currIdx = std::distance(this->axisIndirection.begin(), currAxisIt);
            if (pickedIdx <= currIdx) {
                currAxisIt++;
            }
            this->axisIndirection.insert(currAxisIt, this->pickedAxis);

            this->needFlagsUpdate = true;
        }

        return true;
    }

    if (this->interactionState == InteractionState::INTERACTION_FILTER) {
        float val = (this->mouseY - this->marginY) / this->axisHeight;
        if (this->pickedIndicatorIndex == 0) {
            val = std::clamp(val, 0.0f, this->filters[pickedIndicatorAxis].upper);
            this->filters[this->pickedIndicatorAxis].lower = val;
        } else {
            val = std::clamp(val, this->filters[pickedIndicatorAxis].lower, 1.0f);
            this->filters[this->pickedIndicatorAxis].upper = val;
        }
        this->needFlagsUpdate = true;

        return true;
    }

    if (this->interactionState == InteractionState::INTERACTION_SELECT) {
        if (this->mouseX != this->strokeEndX || this->mouseY != this->strokeEndY) {
            this->strokeEndX = this->mouseX;
            this->strokeEndY = this->mouseY;
            this->needSelectionUpdate = true;
        }

        return true;
    }

    return false;
}

bool ParallelCoordinatesRenderer2D::OnKey(
    core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    ctrlDown = mods.test(core::view::Modifier::CTRL);
    altDown = mods.test(core::view::Modifier::ALT);
    shiftDown = mods.test(core::view::Modifier::SHIFT);
    return false;
}

void ParallelCoordinatesRenderer2D::drawAxes(void) {
    debugPush(1, "drawAxes");
    if (this->columnCount > 0) {
        this->enableProgramAndBind(this->drawAxesProgram);
        glUniform4fv(this->drawAxesProgram.ParameterLocation("color"), 1, this->axesColor);
        glUniform1i(this->drawAxesProgram.ParameterLocation("pickedAxis"), pickedAxis);
        glDrawArraysInstanced(GL_LINES, 0, 2, this->columnCount);
        this->drawAxesProgram.Disable();

        this->enableProgramAndBind(this->drawScalesProgram);
        glUniform4fv(this->drawScalesProgram.ParameterLocation("color"), 1, this->axesColor);
        glUniform1ui(this->drawScalesProgram.ParameterLocation("numTicks"), this->numTicks);
        glUniform1f(this->drawScalesProgram.ParameterLocation("axisHalfTick"), 2.0f);
        glUniform1i(this->drawScalesProgram.ParameterLocation("pickedAxis"), pickedAxis);
        glDrawArraysInstanced(GL_LINES, 0, 2, this->columnCount * this->numTicks);
        this->drawScalesProgram.Disable();

        this->enableProgramAndBind(this->drawFilterIndicatorsProgram);
        glUniform4fv(this->drawFilterIndicatorsProgram.ParameterLocation("color"), 1, this->filterIndicatorColor);
        glUniform1f(this->drawFilterIndicatorsProgram.ParameterLocation("axisHalfTick"), 2.0f);
        glUniform2i(this->drawFilterIndicatorsProgram.ParameterLocation("pickedIndicator"), pickedIndicatorAxis,
            pickedIndicatorIndex);
        glDrawArraysInstanced(GL_LINE_STRIP, 0, 3, this->columnCount * 2);
        this->drawScalesProgram.Disable();
        float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float* color;
#ifndef REMOVE_TEXT
        glActiveTexture(GL_TEXTURE0);
        for (unsigned int c = 0; c < this->columnCount; c++) {
            unsigned int realCol = this->axisIndirection[c];
            if (this->pickedAxis == realCol) {
                color = red;
            } else {
                color = this->axesColor;
            }
            float x = this->marginX + this->axisDistance * c;
            float fontsize = this->axisDistance / 10.0f;
#    if 0
            this->font.DrawString(color, x, this->marginY * 0.5f                   , fontsize, false, std::to_string(minimums[realCol]).c_str(), vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
            this->font.DrawString(color, x, this->marginY * 1.5f + this->axisHeight, fontsize, false, std::to_string(maximums[realCol]).c_str(), vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
#    else
            float bottom = filters[realCol].lower;
            bottom *= (maximums[realCol] - minimums[realCol]);
            bottom += minimums[realCol];
            float top = filters[realCol].upper;
            top *= (maximums[realCol] - minimums[realCol]);
            top += minimums[realCol];
            this->font.DrawString(color, x, this->marginY * 0.5f, fontsize, false, std::to_string(bottom).c_str(),
                core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);
            this->font.DrawString(color, x, this->marginY * 1.5f + this->axisHeight, fontsize, false,
                std::to_string(top).c_str(), core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);
#    endif
            this->font.DrawString(color, x, this->marginY * 2.5f + this->axisHeight, fontsize * 2.0f, false,
                names[realCol].c_str(), core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);
        }
#endif
    }
    debugPop();
}

void ParallelCoordinatesRenderer2D::drawDiscrete(
    const float otherColor[4], const float selectedColor[4], float tfColorFactor) {
    if (this->drawOtherItemsSlot.Param<core::param::BoolParam>()->Value()) {
        this->drawItemsDiscrete(core::FlagStorage::ENABLED | core::FlagStorage::SELECTED | core::FlagStorage::FILTERED,
            core::FlagStorage::ENABLED, otherColor, tfColorFactor);
    }
    if (this->drawSelectedItemsSlot.Param<core::param::BoolParam>()->Value()) {
        this->drawItemsDiscrete(core::FlagStorage::ENABLED | core::FlagStorage::SELECTED | core::FlagStorage::FILTERED,
            core::FlagStorage::ENABLED | core::FlagStorage::SELECTED, selectedColor, tfColorFactor);
    }
}

void ParallelCoordinatesRenderer2D::drawItemsDiscrete(
    uint32_t testMask, uint32_t passMask, const float color[4], float tfColorFactor) {
    auto tf = this->getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
    if (tf == nullptr) return;

    debugPush(2, "drawItemsDiscrete");

#ifdef FUCK_THE_PIPELINE
    vislib::graphics::gl::GLSLShader& prog = this->traceItemsDiscreteProgram;
#else
#    ifdef USE_TESSELLATION
    vislib::graphics::gl::GLSLShader& prog = this->drawItemsDiscreteTessProgram;
#    else
    vislib::graphics::gl::GLSLShader& prog = this->drawItemsDiscreteProgram;
#    endif
#endif

    this->enableProgramAndBind(prog);
    tf->BindConvenience(prog, GL_TEXTURE5, 5);

    glUniform4fv(prog.ParameterLocation("color"), 1, color);
    glUniform1f(prog.ParameterLocation("tfColorFactor"), tfColorFactor);
    glUniform1ui(prog.ParameterLocation("fragmentTestMask"), testMask);
    glUniform1ui(prog.ParameterLocation("fragmentPassMask"), passMask);

#ifdef FUCK_THE_PIPELINE
    glDrawArrays(GL_TRIANGLES, 0, 6 * ((this->itemCount / 128) + 1));
#else
#    ifdef USE_TESSELLATION
    glUniform1i(prog.ParameterLocation("isoLinesPerInvocation"), isoLinesPerInvocation);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    glDrawArrays(GL_PATCHES, 0, (this->itemCount / isoLinesPerInvocation) + 1);
#    else
    // glDrawArraysInstanced(GL_LINE_STRIP, 0, this->columnCount, this->itemCount);
    // glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, this->columnCount * 2, this->itemCount);
    glDrawArrays(GL_LINES, 0, (this->columnCount - 1) * 2 * this->itemCount);
    // glDrawArrays(GL_TRIANGLES, 0, (this->columnCount - 1) * 6 * this->itemCount);
#    endif
#endif
    prog.Disable();
    debugPop();
}

void ParallelCoordinatesRenderer2D::drawPickIndicator(float x, float y, float pickRadius, const float color[4]) {
    auto& program = this->drawPickIndicatorProgram;

    this->enableProgramAndBind(program);

    glUniform2f(program.ParameterLocation("mouse"), x, y);
    glUniform1f(program.ParameterLocation("pickRadius"), pickRadius);

    glUniform4fv(program.ParameterLocation("indicatorColor"), 1, color);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glEnable(GL_DEPTH_TEST);
    program.Disable();
}

void ParallelCoordinatesRenderer2D::drawStrokeIndicator(float x0, float y0, float x1, float y1, const float color[4]) {
    auto& prog = this->drawStrokeIndicatorProgram;

    this->enableProgramAndBind(prog);

    glUniform2f(prog.ParameterLocation("mousePressed"), x0, y0);
    glUniform2f(prog.ParameterLocation("mouseReleased"), x1, y1);

    glUniform4fv(prog.ParameterLocation("indicatorColor"), 1, color);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_LINES, 0, 2);
    glEnable(GL_DEPTH_TEST);
    prog.Disable();
}

void computeDispatchSizes(
    uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]) {
    const auto localSize = localSizes[0] * localSizes[1] * localSizes[2];
    const uint64_t needed_groups = (numItems + localSize - 1) / localSize; // round up int div
    dispatchCounts[0] = std::clamp<GLint>(needed_groups, 1, maxCounts[0]);
    dispatchCounts[1] = std::clamp<GLint>((needed_groups + dispatchCounts[0] - 1) / dispatchCounts[0], 1, maxCounts[1]);
    const auto tmp = dispatchCounts[0] * dispatchCounts[1];
    dispatchCounts[2] = std::clamp<GLint>((needed_groups + tmp - 1) / tmp, 1, maxCounts[2]);
    const uint64_t totalCounts = dispatchCounts[0] * dispatchCounts[1] * dispatchCounts[2];
    ASSERT(totalCounts * localSize >= numItems);
    ASSERT(totalCounts * localSize - numItems < localSize);
}

void ParallelCoordinatesRenderer2D::doPicking(float x, float y, float pickRadius) {
    debugPush(3, "doPicking");

    this->enableProgramAndBind(pickProgram);

    glUniform2f(pickProgram.ParameterLocation("mouse"), x, y);
    glUniform1f(pickProgram.ParameterLocation("pickRadius"), pickRadius);

    GLuint groupCounts[3];
    computeDispatchSizes(itemCount, pickWorkgroupSize, maxWorkgroupCount, groupCounts);

    pickProgram.Dispatch(groupCounts[0], groupCounts[1], groupCounts[2]);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    pickProgram.Disable();
    debugPop();
}

void ParallelCoordinatesRenderer2D::doStroking(float x0, float y0, float x1, float y1) {
    debugPush(3, "doStroking");

    this->enableProgramAndBind(strokeProgram);

    glUniform2f(strokeProgram.ParameterLocation("mousePressed"), x0, y0);
    glUniform2f(strokeProgram.ParameterLocation("mouseReleased"), x1, y1);

    GLuint groupCounts[3];
    computeDispatchSizes(itemCount, strokeWorkgroupSize, maxWorkgroupCount, groupCounts);

    strokeProgram.Dispatch(groupCounts[0], groupCounts[1], groupCounts[2]);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    strokeProgram.Disable();
    debugPop();
}


void ParallelCoordinatesRenderer2D::doFragmentCount(void) {
    debugPush(4, "doFragmentCount");
    int invocations[] = {static_cast<int>(std::ceil(windowWidth / 16)), static_cast<int>(std::ceil(windowHeight / 16))};
    GLuint invocationCount = invocations[0] * invocations[1];

    size_t bytes = sizeof(uint32_t) * 2 * invocationCount;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_STATIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, counterBuffer);

    makeDebugLabel(GL_BUFFER, DEBUG_NAME(counterBuffer));

    glActiveTexture(GL_TEXTURE1);
    densityFBO.BindColourTexture();

    GLuint groupCounts[3] = {
        static_cast<GLuint>((std::max)(1.0f, std::ceil(float(invocations[0]) / counterWorkgroupSize[0]))),
        static_cast<GLuint>((std::max)(1.0f, std::ceil(float(invocations[1]) / counterWorkgroupSize[1]))), 1};

    this->enableProgramAndBind(minMaxProgram);

    // uniforms invocationcount etc.
    glUniform1ui(minMaxProgram.ParameterLocation("invocationCount"), invocationCount);
    glUniform4fv(minMaxProgram.ParameterLocation("clearColor"), 1, backgroundColor);
    glUniform2ui(minMaxProgram.ParameterLocation("resolution"), windowWidth, windowHeight);
    glUniform2ui(minMaxProgram.ParameterLocation("fragmentCountStepSize"), invocations[0], invocations[1]);


    minMaxProgram.Dispatch(groupCounts[0], groupCounts[1], groupCounts[2]);

    minMaxProgram.Disable();

    // todo read back minmax and check for plausibility!
    debugPop();
}

void ParallelCoordinatesRenderer2D::drawItemsContinuous(void) {
    auto tf = this->getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
    if (tf == nullptr) return;
    debugPush(6, "drawItemsContinuous");
    doFragmentCount();
    this->enableProgramAndBind(drawItemContinuousProgram);
    // glUniform2f(drawItemContinuousProgram.ParameterLocation("bottomLeft"), 0.0f, 0.0f);
    // glUniform2f(drawItemContinuousProgram.ParameterLocation("topRight"), windowWidth, windowHeight);
    densityFBO.BindColourTexture();
    tf->BindConvenience(drawItemContinuousProgram, GL_TEXTURE5, 5);
    glUniform1i(this->drawItemContinuousProgram.ParameterLocation("fragmentCount"), 1);
    glUniform4fv(this->drawItemContinuousProgram.ParameterLocation("clearColor"), 1, backgroundColor);
    glUniform1i(this->drawItemContinuousProgram.ParameterLocation("sqrtDensity"),
        this->sqrtDensitySlot.Param<core::param::BoolParam>()->Value() ? 1 : 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    drawItemContinuousProgram.Disable();
    debugPop();
}

void ParallelCoordinatesRenderer2D::drawItemsHistogram(void) {
    debugPush(7, "drawItemsHistogram");
    doFragmentCount();
    this->enableProgramAndBind(drawItemsHistogramProgram);
    glActiveTexture(GL_TEXTURE1);
    densityFBO.BindColourTexture();
    glUniform4fv(this->drawItemContinuousProgram.ParameterLocation("clearColor"), 1, backgroundColor);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    drawItemContinuousProgram.Disable();
    debugPop();
}

void ParallelCoordinatesRenderer2D::drawParcos(void) {

    // TODO only when filters changed!
    GLuint groups = this->itemCount / (filterWorkgroupSize[0] * filterWorkgroupSize[1] * filterWorkgroupSize[2]);
    GLuint groupCounts[3] = {(groups % maxWorkgroupCount[0]) + 1, (groups / maxWorkgroupCount[0]) + 1, 1};
    this->enableProgramAndBind(this->filterProgram);
    filterProgram.Dispatch(groupCounts[0], groupCounts[1], groupCounts[2]);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    const float red[] = {1.0f, 0.0f, 0.0f, 1.0};
    const float moreRed[] = {10.0f, 0.0f, 0.0f, 1.0};

    auto drawmode = this->drawModeSlot.Param<core::param::EnumParam>()->Value();

    switch (drawmode) {
    case DRAW_DISCRETE:
        this->drawDiscrete(this->otherItemsColor, this->selectedItemsColor, 1.0f);
        break;
    case DRAW_CONTINUOUS:
    case DRAW_HISTOGRAM:
        bool ok = true;
        if (!this->densityFBO.IsValid() || this->densityFBO.GetWidth() != windowWidth ||
            this->densityFBO.GetHeight() != windowHeight) {
            densityFBO.Release();
            ok = densityFBO.Create(windowWidth, windowHeight, GL_R32F, GL_RED, GL_FLOAT);
            makeDebugLabel(GL_TEXTURE, densityFBO.GetColourTextureID(), "densityFBO");
        }
        if (ok) {
            densityFBO.Enable();
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            //::glDisable(GL_ALPHA_TEST);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glBlendEquation(GL_FUNC_ADD);
            this->drawDiscrete(red, moreRed, 0.0f);
            densityFBO.Disable();
            glDisable(GL_BLEND);

            if (drawmode == DRAW_CONTINUOUS) {
                this->drawItemsContinuous();
            } else if (drawmode == DRAW_HISTOGRAM) {
                this->drawItemsHistogram();
            }

        } else {
            vislib::sys::Log::DefaultLog.WriteError("could not create FBO");
        }
        break;
    }
}

bool ParallelCoordinatesRenderer2D::Render(core::view::CallRender2D& call) {
    windowAspect = static_cast<float>(call.GetViewport().AspectRatio());

    // this is the apex of suck and must die
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck
    windowWidth = call.GetViewport().Width();
    windowHeight = call.GetViewport().Height();
    auto bg = call.GetBackgroundColour();
    backgroundColor[0] = bg[0] / 255.0f;
    backgroundColor[1] = bg[1] / 255.0f;
    backgroundColor[2] = bg[2] / 255.0f;
    backgroundColor[3] = bg[3] / 255.0f;

    this->assertData(call);

    auto fc = getDataSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    if (fc == nullptr) return false;
    auto tc = getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
    if (tc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "%s cannot draw without a transfer function!", ParallelCoordinatesRenderer2D::ClassName());
        return false;
    }

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, axisIndirectionBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, this->columnCount * sizeof(GLuint), axisIndirection.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, filtersBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, this->columnCount * sizeof(DimensionFilter), this->filters.data());

    // Do stroking/picking
    if (this->needSelectionUpdate) {
        this->needSelectionUpdate = false;

        switch (selectionModeSlot.Param<core::param::EnumParam>()->Value()) {
        case SELECT_STROKE:
            this->doStroking(this->strokeStartX, this->strokeStartY, this->strokeEndX, this->strokeEndY);
            break;
        case SELECT_PICK:
            this->doPicking(this->strokeEndX, this->strokeEndY,
                this->pickRadiusSlot.Param<megamol::core::param::FloatParam>()->Value());
            break;
        }

        this->needFlagsUpdate = true;
    }

    drawParcos();

    // Draw stroking/picking indicator
    if (this->drawSelectionIndicatorSlot.Param<core::param::BoolParam>()->Value()) {
        switch (selectionModeSlot.Param<core::param::EnumParam>()->Value()) {
        case SELECT_STROKE:
            if (this->interactionState == InteractionState::INTERACTION_SELECT) {
                this->drawStrokeIndicator(this->strokeStartX, this->strokeStartY, this->strokeEndX, this->strokeEndY,
                    this->selectionIndicatorColor);
            }
            break;
        case SELECT_PICK:
            this->drawPickIndicator(this->mouseX, this->mouseY,
                this->pickRadiusSlot.Param<megamol::core::param::FloatParam>()->Value(), this->selectionIndicatorColor);
            break;
        }
    }

    if (needFlagsUpdate) {
        needFlagsUpdate = false;
        // HAZARD: downloading everything over and over is slowish
        auto flagsc = getFlagsSlot.CallAs<core::FlagCall>();
        if (flagsc != nullptr) {
            (*flagsc)(core::FlagCall::CallMapFlags);
            auto version = flagsc->GetVersion();
            auto flags = flagsc->GetFlags();
            auto f = flags.get();
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, flagsBuffer);
            glGetBufferSubData(
                GL_SHADER_STORAGE_BUFFER, 0, f->size() * sizeof(core::FlagStorage::FlagItemType), f->data());
#if 0

            core::FlagStorage::FlagVectorType::size_type numFiltered = 0, numEnabled = 0, numSelected = 0,
                                                         numSoftSelected = 0;
            for (unsigned int& i : *f) {
                if ((i & core::FlagStorage::FILTERED) > 0) numFiltered++;
                if ((i & core::FlagStorage::ENABLED) > 0) numEnabled++;
                if ((i & core::FlagStorage::SELECTED) > 0) numSelected++;
                if ((i & core::FlagStorage::SOFTSELECTED) > 0) numSoftSelected++;
            }
            vislib::sys::Log::DefaultLog.WriteInfo(
                "ParallelCoordinateRenderer2D: %lu items: %lu enabled, %lu filtered, %lu selected, %lu "
                "soft-selected.",
                f->size(), numEnabled, numFiltered, numSelected, numSoftSelected);
#endif
            this->currentFlagsVersion = version + 1;
            flagsc->SetFlags(flags, this->currentFlagsVersion);
            (*flagsc)(core::FlagCall::CallUnmapFlags);
        }
    }

    if (this->drawAxesSlot.Param<core::param::BoolParam>()->Value()) {
        drawAxes();
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDepthMask(GL_TRUE);

    return true;
}
