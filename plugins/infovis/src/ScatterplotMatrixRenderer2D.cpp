#include "stdafx.h"
#include "ScatterplotMatrixRenderer2D.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "vislib/math/ShallowMatrix.h"

#include <sstream>
#include "delaunator.hpp"

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

using vislib::sys::Log;

const GLuint PlotSSBOBindingPoint = 2;
const GLuint ValueSSBOBindingPoint = 3;

vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> getModelViewProjection() {
    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // end suck
    return projMatrix * modelViewMatrix;
}

inline std::string to_string(float x) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << x;
    return stream.str();
}

inline float lerp(float x, float y, float a) { return x * (1.0f - a) + y * a; }

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

size_t nameToIndex(
    stdplugin::datatools::table::TableDataCall* tableDataCall, const std::string& name, size_t defaultIdx) {
    auto columnInfos = tableDataCall->GetColumnsInfos();
    const size_t colCount = tableDataCall->GetColumnsCount();

    for (size_t i = 0; i < colCount; i++) {
        if (columnInfos[i].Name().compare(name) == 0) {
            return i;
        }
    }
    return defaultIdx;
}

ScatterplotMatrixRenderer2D::ScatterplotMatrixRenderer2D()
    : Renderer2D()
    , floatTableInSlot("ftIn", "Float table input")
    , transferFunctionInSlot("tfIn", "Transfer function input")
    , flagStorageInSlot("fsIn", "Flag storage input")
    , colorSelectorParam("colorSelector", "Sets a color column")
    , labelSelectorParam("labelSelector", "Sets a label column (text mode)")
    , labelSizeParam("labelSize", "Sets the fontsize for labels (text mode)")
    , geometryTypeParam("geometryType", "Geometry type to map data to")
    , kernelWidthParam("kernelWidth", "Kernel width of the geometry, i.e., point size or line width")
    , kernelTypeParam("kernelType", "Kernel function, i.e., box or gaussian kernel")
    , triangulationSelectorParam("triangulationSelector",
          "Column to compute the Delaunay triangulation for (linear interpolated scalar field)")
    , triangulationSmoothnessParam("triangulationSmoothness", "Number of iterations to smooth the triangulation")
    , axisModeParam("axisMode", "Axis drawing mode")
    , axisColorParam("axisColor", "Color of axis")
    , axisWidthParam("axisWidth", "Line width for the axis")
    , axisTicksParam("axisTicks", "Number of ticks on the axis")
    , axisTicksRedundantParam("axisTicksRedundant", "Enable redundant (inner) ticks")
    , axisTickLengthParam("axisTickLength", "Line length for the ticks")
    , axisTickSizeParam("axisTickSize", "Sets the fontsize for the ticks")
    , cellSizeParam("cellSize", "Aspect ratio scaling x axis length")
    , cellMarginParam("cellMargin", "Set the scaling of y axis")
    , cellNameSizeParam("cellNameSize", "Sets the fontsize for cell names, i.e., column names")
    , alphaScalingParam("alphaScaling", "Scaling factor for overall alpha")
    , alphaAttenuateSubpixelParam("alphaAttenuateSubpixel", "Attenuate alpha of points that have subpixel size")
    , mouse({0, 0, false, false})
    , axisFont("Evolventa-SansSerif", core::utility::SDFFont::RenderType::RENDERTYPE_FILL)
    , labelFont("Evolventa-SansSerif", core::utility::SDFFont::RenderType::RENDERTYPE_FILL)
    , valueSSBO("Values")
    , plotSSBO("Plots")
    , triangleVBO(0)
    , triangleIBO(0)
    , triangleVertexCount(0)
    , trianglesValid(false)
    , textValid(false) {
    this->floatTableInSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->floatTableInSlot);

    this->transferFunctionInSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferFunctionInSlot);

    this->flagStorageInSlot.SetCompatibleCall<FlagCallDescription>();
    this->MakeSlotAvailable(&this->flagStorageInSlot);

    this->colorSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->colorSelectorParam);

    this->labelSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->labelSelectorParam);

    this->labelSizeParam << new core::param::FloatParam(0.1f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->labelSizeParam);

    this->triangulationSelectorParam << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->triangulationSelectorParam);

    this->triangulationSmoothnessParam << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->triangulationSmoothnessParam);

    core::param::EnumParam* geometryTypes = new core::param::EnumParam(0);
    geometryTypes->SetTypePair(GEOMETRY_TYPE_POINT, "Point");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_LINE, "Line");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_TEXT, "Text");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_TRIANGULATION, "Delaunay Triangulation");
    this->geometryTypeParam << geometryTypes;
    this->MakeSlotAvailable(&this->geometryTypeParam);

    this->kernelWidthParam << new core::param::FloatParam(1.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->kernelWidthParam);

    core::param::EnumParam* kernelTypes = new core::param::EnumParam(0);
    kernelTypes->SetTypePair(KERNEL_TYPE_BOX, "Box");
    kernelTypes->SetTypePair(KERNEL_TYPE_GAUSSIAN, "Gaussian");
    this->kernelTypeParam << kernelTypes;
    this->MakeSlotAvailable(&this->kernelTypeParam);

    core::param::EnumParam* axisModes = new core::param::EnumParam(1);
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

    this->axisTickLengthParam << new core::param::FloatParam(0.5f, 0.5f);
    this->MakeSlotAvailable(&this->axisTickLengthParam);

    this->axisTickSizeParam << new core::param::FloatParam(0.5f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->axisTickSizeParam);

    this->cellSizeParam << new core::param::FloatParam(10.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->cellSizeParam);

    this->cellMarginParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->cellMarginParam);

    this->cellNameSizeParam << new core::param::FloatParam(2.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->cellNameSizeParam);

    this->alphaScalingParam << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->alphaScalingParam);

    this->alphaAttenuateSubpixelParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->alphaAttenuateSubpixelParam);
}

ScatterplotMatrixRenderer2D::~ScatterplotMatrixRenderer2D() { this->Release(); }

bool ScatterplotMatrixRenderer2D::create(void) {
    if (!this->axisFont.Initialise(this->GetCoreInstance())) return false;
    if (!this->labelFont.Initialise(this->GetCoreInstance())) return false;
    if (!makeProgram("::splom::minimalisticAxis", this->minimalisticAxisShader)) return false;
    if (!makeProgram("::splom::scientificAxis", this->scientificAxisShader)) return false;
    if (!makeProgram("::splom::point", this->pointShader)) return false;
    if (!makeProgram("::splom::line", this->lineShader)) return false;
    if (!makeProgram("::splom::triangle", this->triangleShader)) return false;

    this->axisFont.SetBatchDrawMode(true);
    this->labelFont.SetBatchDrawMode(true);

    return true;
}

void ScatterplotMatrixRenderer2D::release(void) {}

bool ScatterplotMatrixRenderer2D::MouseEvent(float x, float y, core::view::MouseFlags flags) {
    bool leftDown = (flags & core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) != 0;
    bool rightDown = (flags & core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN) != 0;
    bool rightChanged = (flags & core::view::MOUSEFLAG_BUTTON_RIGHT_CHANGED) != 0;

    this->mouse.x = x;
    this->mouse.y = y;
    this->mouse.selects = leftDown;
    this->mouse.inspects = rightDown && rightChanged;

    if (this->mouse.selects || this->mouse.inspects) {
        // TODO: Some hit testing might be nice here when clicking transparent areas.
        // return itemAt(mouse.x, mouse.y) != -1;
    }

    return false;
}

bool ScatterplotMatrixRenderer2D::Render(core::view::CallRender2D& call) {
    try {
        if (!this->validate()) return false;

        auto axisMode = this->axisModeParam.Param<core::param::EnumParam>()->Value();
        switch (axisMode) {
        case AXIS_MODE_NONE:
            // NOP.
            break;
        case AXIS_MODE_MINIMALISTIC:
            this->drawMinimalisticAxis();
            break;
        case AXIS_MODE_SCIENTIFIC:
            this->drawScientificAxis();
            break;
        }

        auto geometryType = this->geometryTypeParam.Param<core::param::EnumParam>()->Value();
        switch (geometryType) {
        case GEOMETRY_TYPE_POINT:
            this->drawPoints();
            break;
        case GEOMETRY_TYPE_LINE:
            this->drawLines();
            break;
        case GEOMETRY_TYPE_TRIANGULATION:
            this->drawTriangulation();
            break;
        case GEOMETRY_TYPE_TEXT:
            this->drawText();
            break;
        }
    } catch (...) {
        return false;
    }

    return true;
}

bool ScatterplotMatrixRenderer2D::GetExtents(core::view::CallRender2D& call) {
    this->validate();
    call.SetBoundingBox(this->bounds);
    return true;
}

bool ScatterplotMatrixRenderer2D::isDirty(void) const {
    return this->colorSelectorParam.IsDirty() || this->labelSelectorParam.IsDirty() || this->labelSizeParam.IsDirty() ||
           this->triangulationSelectorParam.IsDirty() || this->triangulationSmoothnessParam.IsDirty() ||
           this->cellSizeParam.IsDirty() || this->cellMarginParam.IsDirty();
}

void ScatterplotMatrixRenderer2D::resetDirty(void) {
    this->colorSelectorParam.ResetDirty();
    this->labelSelectorParam.ResetDirty();
    this->labelSizeParam.ResetDirty();
    this->triangulationSelectorParam.ResetDirty();
    this->triangulationSmoothnessParam.ResetDirty();
    this->cellSizeParam.ResetDirty();
    this->cellMarginParam.ResetDirty();
}

bool ScatterplotMatrixRenderer2D::validate(void) {
    this->floatTable = this->floatTableInSlot.CallAs<table::TableDataCall>();
    if (this->floatTable == nullptr || !(*(this->floatTable))(0)) return false;
    if (this->floatTable->GetColumnsCount() == 0) return false;

    this->flagStorage = this->flagStorageInSlot.CallAs<FlagCall>();
    if (this->flagStorage != nullptr && !(*(this->flagStorage))()) return false;

    this->transferFunction = this->transferFunctionInSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
    if (this->transferFunction == nullptr || !(*(this->transferFunction))()) return false;

    if (this->dataHash == this->floatTable->DataHash() && !isDirty()) return true;

    auto columnInfos = this->floatTable->GetColumnsInfos();
    const size_t colCount = this->floatTable->GetColumnsCount();

    if (this->dataHash != this->floatTable->DataHash()) {
        // Update dynamic parameters.
        this->colorSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->labelSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->triangulationSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        for (size_t i = 0; i < colCount; i++) {
            this->colorSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
            this->labelSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
            this->triangulationSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
        }
    }

    // Resolve selectors.
    map.colorIdx =
        nameToIndex(this->floatTable, this->colorSelectorParam.Param<core::param::FlexEnumParam>()->Value(), 0);
    map.labelIdx =
        nameToIndex(this->floatTable, this->labelSelectorParam.Param<core::param::FlexEnumParam>()->Value(), 0);

    this->trianglesValid = false;
    this->textValid = false;
    this->updateColumns();

    this->dataHash = this->floatTable->DataHash();
    this->resetDirty();

    return true;
}

void ScatterplotMatrixRenderer2D::updateColumns(void) {
    const auto columnCount = this->floatTable->GetColumnsCount();
    const auto columnInfos = this->floatTable->GetColumnsInfos();
    const float size = this->cellSizeParam.Param<core::param::FloatParam>()->Value();
    const float margin = this->cellMarginParam.Param<core::param::FloatParam>()->Value();

    plots.clear();
    for (GLuint y = 0; y < columnCount; ++y) {
        for (GLuint x = 0; x < y; ++x) {
            plots.push_back({x, y, x * (size + margin), y * (size + margin), size, size, columnInfos[x].MinimumValue(),
                columnInfos[y].MinimumValue(), columnInfos[x].MaximumValue(), columnInfos[y].MaximumValue(),
                rangeToSmallStep(columnInfos[x].MinimumValue(), columnInfos[x].MaximumValue()),
                rangeToSmallStep(columnInfos[y].MinimumValue(), columnInfos[y].MaximumValue())});
        }
    }

    this->bounds.Set(0, 0, columnCount * (size + margin) - margin, columnCount * (size + margin) - margin);

    const GLuint plotItems = plotSSBO.GetNumItemsPerChunkAligned(plots.size(), true);
    const GLuint bufferSize = plotItems * sizeof(PlotInfo);
    const GLuint numChunks =
        this->plotSSBO.SetDataWithSize(plots.data(), sizeof(PlotInfo), sizeof(PlotInfo), plots.size(), 1, bufferSize);
    assert(numChunks == 1 && "Number of chunks should be one");

    GLuint numItems, sync;
    plotSSBO.UploadChunk(0, numItems, sync, this->plotDstOffset, this->plotDstLength);
    plotSSBO.SignalCompletion(sync);
}

void ScatterplotMatrixRenderer2D::drawMinimalisticAxis(void) {
    this->minimalisticAxisShader.Enable();

    // Transformation uniform.
    glUniformMatrix4fv(this->minimalisticAxisShader.ParameterLocation("modelViewProjection"), 1, GL_FALSE,
        getModelViewProjection().PeekComponents());

    // Other uniforms.
    const GLfloat tickLength = this->axisTickLengthParam.Param<core::param::FloatParam>()->Value();
    const GLsizei numTicks = this->axisTicksParam.Param<core::param::IntParam>()->Value();
    glUniform4fv(this->minimalisticAxisShader.ParameterLocation("axisColor"), 1,
        this->axisColorParam.Param<core::param::ColorParam>()->Value().data());
    glUniform1ui(this->minimalisticAxisShader.ParameterLocation("numTicks"), numTicks);
    glUniform1f(this->minimalisticAxisShader.ParameterLocation("tickLength"), tickLength);
    glUniform1i(this->minimalisticAxisShader.ParameterLocation("redundantTicks"),
        this->axisTicksRedundantParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);

    // Line width.
    auto axisWidth = this->axisWidthParam.Param<core::param::FloatParam>()->Value();
    glLineWidth(axisWidth);

    // Render all plots at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle());
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(), this->plotDstOffset,
        this->plotDstLength);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    const GLsizei numVerticesPerLine = 2;
    const GLsizei numBorderVertices = numVerticesPerLine * 4;
    const GLsizei numTickVertices = numVerticesPerLine * numTicks * 2;
    const GLsizei numItems = numBorderVertices + numTickVertices;
    glDrawArraysInstanced(GL_LINES, 0, numItems, this->plots.size());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    this->minimalisticAxisShader.Disable();

    this->axisFont.ClearBatchDrawCache();

    const auto axisColor = this->axisColorParam.Param<core::param::ColorParam>()->Value();
    const auto columnCount = this->floatTable->GetColumnsCount();
    const auto columnInfos = this->floatTable->GetColumnsInfos();

    const float size = this->cellSizeParam.Param<core::param::FloatParam>()->Value();
    const float margin = this->cellMarginParam.Param<core::param::FloatParam>()->Value();
    const float nameSize = this->cellNameSizeParam.Param<core::param::FloatParam>()->Value();
    const float tickSize = this->axisTickSizeParam.Param<core::param::FloatParam>()->Value();

    for (size_t i = 0; i < columnCount; ++i) {
        const float xyBL = i * (size + margin);
        const float xyTL = i * (size + margin) + size;
        std::string label = columnInfos[i].Name();
        this->axisFont.DrawString(axisColor.data(), xyBL, xyTL, size, size, nameSize, false, label.c_str(),
            core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);

        const float tickStart = i * (size + margin);
        const float tickEnd = (i + 1) * (size + margin) - margin;

        for (size_t tick = 0; tick < numTicks; ++tick) {
            const float t = static_cast<float>(tick) / (numTicks - 1);
            const float p = lerp(tickStart, tickEnd, t);
            const float pValue = lerp(columnInfos[i].MinimumValue(), columnInfos[i].MaximumValue(), t);
            const std::string pLabel = to_string(pValue);
            if (i < columnCount - 1) {
                this->axisFont.DrawString(axisColor.data(), p, xyTL + tickLength, tickSize, false, pLabel.c_str(),
                    core::utility::AbstractFont::ALIGN_CENTER_TOP);
            }
            if (i > 0) {
                this->axisFont.DrawString(axisColor.data(), xyBL - margin + tickLength, p, tickSize, false,
                    pLabel.c_str(), core::utility::AbstractFont::ALIGN_LEFT_MIDDLE);
            }
        }
    }

    this->axisFont.BatchDrawString();
}

void ScatterplotMatrixRenderer2D::drawScientificAxis(void) {
    const auto axisColor = this->axisColorParam.Param<core::param::ColorParam>()->Value();
    const auto columnCount = this->floatTable->GetColumnsCount();
    const auto columnInfos = this->floatTable->GetColumnsInfos();
    const float size = this->cellSizeParam.Param<core::param::FloatParam>()->Value();
    const float margin = this->cellMarginParam.Param<core::param::FloatParam>()->Value();
    const float nameSize = this->cellNameSizeParam.Param<core::param::FloatParam>()->Value();
    const float tickLabelSize = this->axisTickSizeParam.Param<core::param::FloatParam>()->Value();
    const GLfloat tickLength = this->axisTickLengthParam.Param<core::param::FloatParam>()->Value();

    // Compute cell size in viewport space.
    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    auto mvpMatrix = getModelViewProjection();
    auto ndcSpaceSize = mvpMatrix * vislib::math::Vector<float, 4>(size, size, 0.0f, 0.0f);
    auto screenSpaceSize =
        vislib::math::Vector<float, 2>(viewport[2] / 2.0 * ndcSpaceSize.X(), viewport[3] / 2.0 * ndcSpaceSize.Y());

    // 0: no grid <-> 3: big,mid,small grid
    GLint recursiveDepth = 0;
    if (screenSpaceSize.X() > 900) {
        recursiveDepth = 3;
    } else if (screenSpaceSize.X() > 300) {
        recursiveDepth = 2;
    } else if (screenSpaceSize.X() > 75) {
        recursiveDepth = 1;
    }

    this->scientificAxisShader.Enable();

    // Blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    // Transformation uniform.
    glUniformMatrix4fv(
        this->scientificAxisShader.ParameterLocation("modelViewProjection"), 1, GL_FALSE, mvpMatrix.PeekComponents());

    // Other uniforms.
    glUniform1ui(this->scientificAxisShader.ParameterLocation("depth"), recursiveDepth);
    glUniform4fv(this->scientificAxisShader.ParameterLocation("axisColor"), 1,
        this->axisColorParam.Param<core::param::ColorParam>()->Value().data());

    // Render all plots at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle());
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(), this->plotDstOffset,
        this->plotDstLength);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDrawArraysInstanced(GL_QUADS, 0, 4, this->plots.size());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    this->scientificAxisShader.Disable();

    glDisable(GL_TEXTURE_1D);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    this->axisFont.ClearBatchDrawCache();

    for (size_t i = 0; i < columnCount; ++i) {
        const float cellStart = i * (size + margin);
        const float cellEnd = (i + 1) * (size + margin) - margin;

        // Labels
        std::string label = columnInfos[i].Name();
        this->axisFont.DrawString(axisColor.data(), cellStart, cellEnd, size, size, nameSize, false, label.c_str(),
            core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);

        float delta = columnInfos[i].MaximumValue() - columnInfos[i].MinimumValue();
        // Tick sizes: big *25, mid *5, small *1
        float tickSize = rangeToSmallStep(columnInfos[i].MaximumValue(), columnInfos[i].MinimumValue()) * 25;
        float firstTick = ceil(columnInfos[i].MinimumValue() / tickSize) * tickSize;

        for (float tickPos = firstTick; tickPos <= columnInfos[i].MaximumValue(); tickPos += tickSize) {
            float normalized = (tickPos - columnInfos[i].MinimumValue()) / delta;
            float offset = normalized * size;

            float pos = cellStart + offset;

            const std::string pLabel = to_string(tickPos);

            // Tick labels for x axis
            if (i < columnCount - 1) {
                this->axisFont.DrawString(axisColor.data(), pos, cellEnd + tickLength, tickLabelSize, false,
                    pLabel.c_str(), core::utility::AbstractFont::ALIGN_CENTER_TOP);
            }
            // Tick labels for y axis
            if (i > 0) {
                this->axisFont.DrawString(axisColor.data(), cellStart - margin + tickLength, pos, tickLabelSize, false,
                    pLabel.c_str(), core::utility::AbstractFont::ALIGN_LEFT_MIDDLE);
            }
        }
    }

    this->axisFont.BatchDrawString();
}

void ScatterplotMatrixRenderer2D::drawPoints(void) {
    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    // Blending.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    // Point sprites.
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glPointSize(std::max(viewport[2], viewport[3]));

    this->pointShader.Enable();

    // Transformation uniforms.
    glUniform4fv(this->pointShader.ParameterLocation("viewport"), 1, viewport);
    glUniformMatrix4fv(this->pointShader.ParameterLocation("modelViewProjection"), 1, GL_FALSE,
        getModelViewProjection().PeekComponents());

    // Color map uniforms.
    auto columnInfos = this->floatTable->GetColumnsInfos();
    GLfloat colorColumnMinMax[] = {columnInfos[map.colorIdx].MinimumValue(), columnInfos[map.colorIdx].MaximumValue()};
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, this->transferFunction->OpenGLTexture());
    glUniform1i(this->pointShader.ParameterLocation("colorTable"), 0);
    glUniform1i(this->pointShader.ParameterLocation("colorCount"), this->transferFunction->TextureSize());
    glUniform1i(this->pointShader.ParameterLocation("colorColumn"), map.colorIdx);
    glUniform2fv(this->pointShader.ParameterLocation("colorColumnMinMax"), 1, colorColumnMinMax);

    // Other uniforms.
    const auto columnCount = this->floatTable->GetColumnsCount();
    glUniform1i(this->pointShader.ParameterLocation("rowStride"), columnCount);
    glUniform1f(this->pointShader.ParameterLocation("kernelWidth"),
        this->kernelWidthParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->pointShader.ParameterLocation("kernelType"),
        this->kernelTypeParam.Param<core::param::EnumParam>()->Value());
    glUniform1f(this->pointShader.ParameterLocation("alphaScaling"),
        this->alphaScalingParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->pointShader.ParameterLocation("attenuateSubpixel"),
        this->alphaAttenuateSubpixelParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);

    // Setup streaming.
    const GLuint numBuffers = 3;
    const GLuint bufferSize = 32 * 1024 * 1024;
    const float* data = this->floatTable->GetData();
    const GLuint dataStride = columnCount * sizeof(float);
    const GLuint dataItems = this->floatTable->GetRowsCount();
    const GLuint numChunks =
        this->valueSSBO.SetDataWithSize(data, dataStride, dataStride, dataItems, numBuffers, bufferSize);

    // For each chunk of values, render all points in the lower half of the scatterplot matrix at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle());
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(), this->plotDstOffset,
        this->plotDstLength);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle());
    for (GLuint chunk = 0; chunk < numChunks; ++chunk) {
        GLuint numItems, sync;
        GLsizeiptr dstOffset, dstLength;
        valueSSBO.UploadChunk(chunk, numItems, sync, dstOffset, dstLength);
        glBindBufferRange(
            GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle(), dstOffset, dstLength);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glDrawArraysInstanced(GL_POINTS, 0, static_cast<GLsizei>(numItems), this->plots.size());
        valueSSBO.SignalCompletion(sync);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_1D, 0);

    this->pointShader.Disable();

    glDisable(GL_TEXTURE_1D);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void ScatterplotMatrixRenderer2D::drawLines(void) {
    GLfloat viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);

    // Blending.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    this->lineShader.Enable();

    // Transformation uniforms.
    glUniform4fv(this->lineShader.ParameterLocation("viewport"), 1, viewport);
    glUniformMatrix4fv(this->lineShader.ParameterLocation("modelViewProjection"), 1, GL_FALSE,
        getModelViewProjection().PeekComponents());

    // Color map uniforms.
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, this->transferFunction->OpenGLTexture());
    glUniform1i(this->lineShader.ParameterLocation("colorTable"), 0);
    glUniform1i(this->lineShader.ParameterLocation("colorCount"), this->transferFunction->TextureSize());
    glUniform1i(this->lineShader.ParameterLocation("colorColumn"), map.colorIdx);

    // Other uniforms.
    const auto columnCount = this->floatTable->GetColumnsCount();
    glUniform1i(this->lineShader.ParameterLocation("rowStride"), columnCount);
    glUniform1f(this->lineShader.ParameterLocation("kernelWidth"),
        this->kernelWidthParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->lineShader.ParameterLocation("kernelType"),
        this->kernelTypeParam.Param<core::param::EnumParam>()->Value());
    glUniform1f(this->lineShader.ParameterLocation("alphaScaling"),
        this->alphaScalingParam.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->lineShader.ParameterLocation("attenuateSubpixel"),
        this->alphaAttenuateSubpixelParam.Param<core::param::BoolParam>()->Value() ? 1 : 0);

    // Setup streaming.
    const GLuint numBuffers = 3;
    const GLuint bufferSize = 32 * 1024 * 1024;
    const float* data = this->floatTable->GetData();
    const GLuint dataStride = columnCount * sizeof(float);
    const GLuint dataItems = this->floatTable->GetRowsCount();
    const GLuint numChunks =
        this->valueSSBO.SetDataWithSize(data, dataStride, dataStride, dataItems, numBuffers, bufferSize);

    // For each chunk of values, render all points in the lower half of the scatterplot matrix at once.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle());
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, PlotSSBOBindingPoint, this->plotSSBO.GetHandle(), this->plotDstOffset,
        this->plotDstLength);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle());
    for (GLuint chunk = 0; chunk < numChunks; ++chunk) {
        GLuint numItems, sync;
        GLsizeiptr dstOffset, dstLength;
        valueSSBO.UploadChunk(chunk, numItems, sync, dstOffset, dstLength);
        glBindBufferRange(
            GL_SHADER_STORAGE_BUFFER, ValueSSBOBindingPoint, this->valueSSBO.GetHandle(), dstOffset, dstLength);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glDrawArraysInstanced(GL_LINE_STRIP, 0, static_cast<GLsizei>(numItems), this->plots.size());
        valueSSBO.SignalCompletion(sync);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_1D, 0);

    this->lineShader.Disable();

    glDisable(GL_TEXTURE_1D);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
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
    auto columnIndex =
        nameToIndex(this->floatTable, this->triangulationSelectorParam.Param<core::param::FlexEnumParam>()->Value(), 0);
    auto columnInfos = this->floatTable->GetColumnsInfos()[columnIndex];
    auto minValue = columnInfos.MinimumValue();
    auto maxValue = columnInfos.MaximumValue();

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
            // Normalize value.
            float value = this->floatTable->GetData(columnIndex, i);
            values.push_back((value - minValue) / (maxValue - minValue));
        }

        // Compute initial Delauney triangulation.
        delaunator::Delaunator d(coords);

        // Smooth triangulation by adding new vertices.
        auto smoothIterations = this->triangulationSmoothnessParam.Param<core::param::IntParam>()->Value();
        for (size_t i = 0; i < smoothIterations; i++) {
            for (int triangleIndex = 0; triangleIndex < d.triangles.size(); triangleIndex += 3) {
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
        const GLuint indexOffset = static_cast<GLuint>(vertices.size());

        // Copy vertices to vertex buffer.
        for (int vertexIndex = 0; vertexIndex < values.size(); vertexIndex++) {
            TriangulationVertex vertex = {coords[vertexIndex * 2], coords[vertexIndex * 2 + 1], values[vertexIndex]};
            vertices.push_back(vertex);
        }

        // Copy indices to index buffer.
        for (int triangleIndex = 0; triangleIndex < d.triangles.size(); triangleIndex++) {
            indices.push_back(indexOffset + d.triangles[triangleIndex]);
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
    this->validateTriangulation();

    triangleShader.Enable();
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Bind buffers.
    glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TriangulationVertex), reinterpret_cast<GLvoid**>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 1, GL_FLOAT, GL_FALSE, sizeof(TriangulationVertex), reinterpret_cast<GLvoid**>(sizeof(float) * 2));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangleIBO);

    // Set uniforms.
    const int contourLevels = 3;                       // 3; // TODO: param!
    const float contourColor[] = {1.0, 0.0, 0.0, 1.0}; // 3; // TODO: param!
    auto mvpMatrix = getModelViewProjection();
    glUniform1i(triangleShader.ParameterLocation("contourLevels"), contourLevels);
    glUniform4fv(triangleShader.ParameterLocation("contourColor"), 1, contourColor);
    glUniformMatrix4fv(
        this->triangleShader.ParameterLocation("modelViewProjection"), 1, GL_FALSE, mvpMatrix.PeekComponents());

    // Emit draw call.
    glDrawElements(GL_TRIANGLES, triangleVertexCount, GL_UNSIGNED_INT, 0);
    triangleShader.Disable();
}

void ScatterplotMatrixRenderer2D::validateText() {
    if (this->textValid) {
        return;
    }

    this->labelFont.ClearBatchDrawCache();

    const auto columnInfos = this->floatTable->GetColumnsInfos();
    const auto rowCount = this->floatTable->GetRowsCount();

    const float labelSize = this->labelSizeParam.Param<core::param::FloatParam>()->Value();
    for (size_t i = 0; i < rowCount; ++i) {
        for (const auto& plot : this->plots) {
            const float xValue = this->floatTable->GetData(plot.indexX, i);
            const float yValue = this->floatTable->GetData(plot.indexY, i);
            const float xPos = (xValue - plot.minX) / (plot.maxX - plot.minX);
            const float yPos = (yValue - plot.minY) / (plot.maxY - plot.minY);

            const size_t colorIndex = this->floatTable->GetData(this->map.colorIdx, i);
            float labelColor[4] = {0, 0, 0, 1}; // TODO: param please!

            // XXX: this will be a lot more useful when have support for string-columns!
            std::string label = to_string(this->floatTable->GetData(map.labelIdx, i));

            this->labelFont.DrawString(labelColor, plot.offsetX + xPos * plot.sizeX, plot.offsetY + yPos * plot.sizeY,
                labelSize, false, label.c_str(), core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);
        }
    }

    this->textValid = true;
}


void ScatterplotMatrixRenderer2D::drawText() {
    validateText();

    this->labelFont.BatchDrawString();
}

int ScatterplotMatrixRenderer2D::itemAt(const float x, const float y) {
    /*
    if (y >= 0.0f) { //< within scatterplot
        auto trans = this->oglTrans;
        trans.Invert();
        auto query_p = trans*vislib::math::Vector<float, 4>(x, y, 0.0f, 1.0f);
        float qp[2] = {query_p.X(), query_p.Y()};
        // search with nanoflann tree
        size_t idx[1] = {0};
        float dis[1] = {0.0f};
        this->tree->index->knnSearch(qp, 1, idx, dis);

        idx[0] = *reinterpret_cast<unsigned int *>(&this->series[0][idx[0] * 4 + 3]); //< toxic, which is the
    correct series?

        auto ssp = this->nvgTrans*vislib::math::Vector<float, 3>(x, y, 1.0f);
        TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
        if (tic == nullptr) {
            // show tool tip
            this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Call"));
        } else {
            tic->SetRequest(TraceInfoCall::RequestType::GetSymbolString, idx[0]);
            if (!(*tic)(0)) {
                // show tool tip
                this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Found"));
            } else {
                auto st = tic->GetInfo();
                this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, st);
            }
        }

        return idx[0];
    } else { //< within callstack
        // calculate depth
        // search for fitting range in chosen depth
        float boxHeight = std::get<1>(this->viewport) / 40.0f;
        float yCoord = std::fabsf(y);
        unsigned int depth = std::floorf(yCoord / boxHeight);
        auto ssp = this->nvgTrans*vislib::math::Vector<float, 3>(x, y, 1.0f);
        float aspect = this->cellSize.Param<core::param::FloatParam>()->Value();
        for (auto &r : this->callStack[depth]) {
            // rb / norm*dw
            float rb = std::get<0>(r);
            float re = std::get<1>(r);
            if ((rb / this->abcissa.size()*std::get<0>(this->viewport)*aspect) <= x && x <= (re /
    this->abcissa.size()*std::get<0>(this->viewport)*aspect)) { //< abcissa missing size_t symbolIdx =
    std::get<2>(r); TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>(); if (tic == nullptr) {
                    // show tool tip
                    this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Call"));
                } else {
                    tic->SetRequest(TraceInfoCall::RequestType::GetSymbolString, symbolIdx);
                    if (!(*tic)(0)) {
                        // show tool tip
                        this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, std::string("No Info Found"));
                    } else {
                        auto st = tic->GetInfo();
                        st += std::string(" ") + std::to_string((unsigned int)rb) + std::string(" ") +
    std::to_string((unsigned int)re); this->drawToolTip(ssp.X() + 10, ssp.Y() + 10, st);
                    }
                }
                return symbolIdx;
            }
        }
    }
    */
    return -1;
}
