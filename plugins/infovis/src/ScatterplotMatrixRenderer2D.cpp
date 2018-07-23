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

inline float lerp(float x, float y, float a) { return x * (1.0f - a) + y * a; }

inline std::string to_string(float x) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << x;
    return stream.str();
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
    , labelsValid(false) {
    this->floatTableInSlot.SetCompatibleCall<floattable::CallFloatTableDataDescription>();
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

    core::param::EnumParam* geometryTypes = new core::param::EnumParam(0);
    geometryTypes->SetTypePair(GEOMETRY_TYPE_POINT, "Point");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_LINE, "Line");
    geometryTypes->SetTypePair(GEOMETRY_TYPE_TEXT, "Text");
    this->geometryTypeParam << geometryTypes;
    this->MakeSlotAvailable(&this->geometryTypeParam);

    this->kernelWidthParam << new core::param::FloatParam(1.0f, std::numeric_limits<float>::epsilon());
    this->MakeSlotAvailable(&this->kernelWidthParam);

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
    if (!makeProgram("::splom::axis", this->axisShader)) return false;
    if (!makeProgram("::splom::point", this->pointShader)) return false;
    if (!makeProgram("::splom::line", this->lineShader)) return false;

    this->axisFont.EnableBatchDraw();
    this->labelFont.EnableBatchDraw();

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
        if (!this->validateData()) return false;

        this->drawAxes();

        auto geometryType = this->geometryTypeParam.Param<core::param::EnumParam>()->Value();
        switch (geometryType) {
        case GEOMETRY_TYPE_POINT:
            this->drawPoints();
            break;
        case GEOMETRY_TYPE_LINE:
            this->drawLines();
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
    this->validateData();
    call.SetBoundingBox(this->bounds);
    return true;
}

bool ScatterplotMatrixRenderer2D::isDirty(void) const {
    return this->colorSelectorParam.IsDirty() || this->labelSelectorParam.IsDirty() || this->labelSizeParam.IsDirty() ||
           this->cellSizeParam.IsDirty() || this->cellMarginParam.IsDirty();
}

void ScatterplotMatrixRenderer2D::resetDirty(void) {
    this->colorSelectorParam.ResetDirty();
    this->labelSelectorParam.ResetDirty();
    this->labelSizeParam.ResetDirty();
    this->cellSizeParam.ResetDirty();
    this->cellMarginParam.ResetDirty();
}

bool ScatterplotMatrixRenderer2D::validateData(void) {
    this->floatTable = this->floatTableInSlot.CallAs<floattable::CallFloatTableData>();
    if (this->floatTable == nullptr || !(*(this->floatTable))(0)) return false;

    this->transferFunction = this->transferFunctionInSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
    if (this->transferFunction == nullptr || !(*(this->transferFunction))()) return false;

    // TODO: store selection inside flag storage.
    this->flagStorage = this->flagStorageInSlot.CallAs<FlagCall>();
    if (this->flagStorage == nullptr || !(*(this->flagStorage))()) return false;

    if (this->dataHash == this->floatTable->DataHash() && !isDirty()) return true;

    auto columnInfos = this->floatTable->GetColumnsInfos();
    const size_t colCount = this->floatTable->GetColumnsCount();

    if (this->dataHash != this->floatTable->DataHash()) {
        // Update dynamic parameters.
        this->colorSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        this->labelSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();
        for (size_t i = 0; i < colCount; i++) {
            this->colorSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
            this->labelSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(columnInfos[i].Name());
        }
    }

    // Resolve selectors.
    auto nameToIndex = [&](const std::string& name, size_t defaultIdx) -> size_t {
        for (size_t i = 0; i < colCount; i++) {
            if (columnInfos[i].Name().compare(name) == 0) {
                return i;
            }
        }
        return defaultIdx;
    };
    map.colorIdx = nameToIndex(this->colorSelectorParam.Param<core::param::FlexEnumParam>()->Value(), 0);
    map.labelIdx = nameToIndex(this->labelSelectorParam.Param<core::param::FlexEnumParam>()->Value(), 0);

    updateColumns();

    this->labelsValid = false;

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
                columnInfos[y].MinimumValue(), columnInfos[x].MaximumValue(), columnInfos[y].MaximumValue()});
        }
    }

    this->bounds.Set(0, 0, columnCount * (size + margin) - margin, columnCount * (size + margin) - margin);

    const GLuint plotItems = core::utility::SSBOStreamer::GetNumItemsPerChunkAligned(plots.size(), true);
    const GLuint bufferSize = plotItems * sizeof(PlotInfo);
    const GLuint numChunks =
        this->plotSSBO.SetDataWithSize(plots.data(), sizeof(PlotInfo), sizeof(PlotInfo), plots.size(), 1, bufferSize);
    assert(numChunks == 1 && "Number of chunks should be one");

    GLuint numItems, sync;
    plotSSBO.UploadChunk(0, numItems, sync, this->plotDstOffset, this->plotDstLength);
    plotSSBO.SignalCompletion(sync);
}

void ScatterplotMatrixRenderer2D::drawAxes(void) {
    this->axisShader.Enable();

    // Transformation uniform.
    glUniformMatrix4fv(this->axisShader.ParameterLocation("modelViewProjection"), 1, GL_FALSE,
        getModelViewProjection().PeekComponents());

    // Other uniforms.
    const GLfloat tickLength = this->axisTickLengthParam.Param<core::param::FloatParam>()->Value();
    const GLsizei numTicks = this->axisTicksParam.Param<core::param::IntParam>()->Value();
    glUniform4fv(this->axisShader.ParameterLocation("axisColor"), 1,
        this->axisColorParam.Param<core::param::ColorParam>()->Value());
    glUniform1ui(this->axisShader.ParameterLocation("numTicks"), numTicks);
    glUniform1f(this->axisShader.ParameterLocation("tickLength"), tickLength);
    glUniform1i(this->axisShader.ParameterLocation("redundantTicks"),
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
    this->axisShader.Disable();


    this->axisFont.ClearBatchCache();

    const auto* axisColor = this->axisColorParam.Param<core::param::ColorParam>()->Value();
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
        this->axisFont.DrawString(axisColor, xyBL, xyTL, size, size, nameSize, false, label.c_str(),
            core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);

        const float tickStart = i * (size + margin);
        const float tickEnd = (i + 1) * (size + margin) - margin;

        for (size_t tick = 0; tick < numTicks; ++tick) {
            const float t = static_cast<float>(tick) / (numTicks - 1);
            const float p = lerp(tickStart, tickEnd, t);
            const float pValue = lerp(columnInfos[i].MinimumValue(), columnInfos[i].MaximumValue(), t);
            const std::string pLabel = to_string(pValue);
            if (i < columnCount - 1) {
                this->axisFont.DrawString(axisColor, p, xyTL + tickLength, tickSize, false, pLabel.c_str(),
                    core::utility::AbstractFont::ALIGN_CENTER_TOP);
            }
            if (i > 0) {
                this->axisFont.DrawString(axisColor, xyBL - margin + tickLength, p, tickSize, false, pLabel.c_str(),
                    core::utility::AbstractFont::ALIGN_LEFT_MIDDLE);
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
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_1D, this->transferFunction->OpenGLTexture());
    glUniform1i(this->pointShader.ParameterLocation("colorTable"), 0);
    glUniform1i(this->pointShader.ParameterLocation("colorCount"), this->transferFunction->TextureSize());
    glUniform1i(this->pointShader.ParameterLocation("colorColumn"), map.colorIdx);

    // Other uniforms.
    const auto columnCount = this->floatTable->GetColumnsCount();
    glUniform1i(this->pointShader.ParameterLocation("rowStride"), columnCount);
    glUniform1f(this->pointShader.ParameterLocation("kernelWidth"),
        this->kernelWidthParam.Param<core::param::FloatParam>()->Value());
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


void ScatterplotMatrixRenderer2D::drawText(void) {
    if (this->labelsValid) {
        this->labelFont.BatchDrawString();
        return;
    }

    this->labelFont.ClearBatchCache();

    const auto columnCount = this->floatTable->GetColumnsCount();
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
            float labelColor[4];
            this->transferFunction->CopyColor(colorIndex, labelColor, sizeof(labelColor));

            // XXX: this will be a lot more useful when have support for string-columns!
            std::string label = to_string(this->floatTable->GetData(map.labelIdx, i));

            this->labelFont.DrawString(labelColor, plot.offsetX + xPos * plot.sizeX, plot.offsetY + yPos * plot.sizeY,
                labelSize, false, label.c_str(), core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);
        }
    }

    this->labelFont.BatchDrawString();
    this->labelsValid = true;
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

        idx[0] = *reinterpret_cast<unsigned int *>(&this->series[0][idx[0] * 4 + 3]); //< toxic, which is the correct
    series?

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
    this->abcissa.size()*std::get<0>(this->viewport)*aspect)) { //< abcissa missing size_t symbolIdx = std::get<2>(r);
                TraceInfoCall *tic = this->getPointInfoSlot.CallAs<TraceInfoCall>();
                if (tic == nullptr) {
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
