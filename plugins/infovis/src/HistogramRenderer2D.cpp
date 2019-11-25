#include "stdafx.h"
#include "HistogramRenderer2D.h"

#include "mmcore/param/IntParam.h"

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

using vislib::sys::Log;

HistogramRenderer2D::HistogramRenderer2D()
    : Renderer2D()
    , tableDataCallerSlot("getData", "Float table input")
    , transferFunctionCallerSlot("getTransferFunction", "Transfer function input")
    , flagStorageCallerSlot("getFlagStorage", "Flag storage input")
    , numberOfBinsSlot("numberOfBins", "Number of bins")
    , currentTableDataHash(std::numeric_limits<std::size_t>::max())
    , currentTableFrameId(std::numeric_limits<unsigned int>::max())
    , currentFlagStorageVersion(std::numeric_limits<core::FlagStorage::FlagVersionType>::max())
    , bins(10)
    , colCount(0)
    , maxBinValue(0)
    , font("Evolventa-SansSerif", core::utility::SDFFont::RenderType::RENDERTYPE_FILL)
{
    this->tableDataCallerSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableDataCallerSlot);

    this->transferFunctionCallerSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferFunctionCallerSlot);

    this->flagStorageCallerSlot.SetCompatibleCall<core::FlagCallDescription>();
    this->MakeSlotAvailable(&this->flagStorageCallerSlot);

    this->numberOfBinsSlot << new core::param::IntParam(this->bins, 1);
    this->MakeSlotAvailable(&this->numberOfBinsSlot);
}

HistogramRenderer2D::~HistogramRenderer2D() {
    this->Release();
}

bool HistogramRenderer2D::create() {
    if (!font.Initialise(this->GetCoreInstance())) return false;

    if (!makeProgram("::histo::draw", this->histogramProgram)) return false;

    static const GLfloat quadVertices[] = {
            0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
    };

    static const GLuint quadIndices[] = {
            0, 1, 2,
            1, 2, 3,
    };

    glGenVertexArrays(1, &quadVertexArray);
    glGenBuffers(1, &quadVertexBuffer);
    glGenBuffers(1, &quadIndexBuffer);

    glBindVertexArray(quadVertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, quadVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 3 * 4, quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * 3 * sizeof(GLuint), quadIndices, GL_STATIC_DRAW);
    glBindVertexArray(0);

    return true;
}

void HistogramRenderer2D::release() {
}

bool HistogramRenderer2D::GetExtents(core::view::CallRender2D &call) {
    if (!handleCall(call)) {
        return false;
    }

    // Draw histogram within 10.0 x 10.0 quads, left + right margin 1.0, top and bottom 2.0 for title and axes
    float sizeX = static_cast<float>(std::max<size_t>(1, this->colCount)) * 12.0f;
    call.SetBoundingBox(0.0f, 0.0f, sizeX, 14.0f);
    return true;
}

bool HistogramRenderer2D::Render(core::view::CallRender2D &call) {
    if (!handleCall(call)) {
        return false;
    }

    auto tfCall = this->transferFunctionCallerSlot.CallAs<core::view::CallGetTransferFunction>();
    if (tfCall == nullptr) { return false; }

    // this is the apex of suck and must die
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck

    glUseProgram(histogramProgram);
    glUniformMatrix4fv(histogramProgram.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(histogramProgram.ParameterLocation("projection"), 1, GL_FALSE, projMatrix_column);

    glBindVertexArray(quadVertexArray);
    tfCall->BindConvenience(histogramProgram, GL_TEXTURE0, 0);

    // TODO use something better like instanced rendering
    for (size_t c = 0; c < this->colCount; ++c) {
        for (size_t b = 0; b < this->bins; ++b) {
            float width = 10.0f / this->bins;
            float height = 10.0f * this->histogram[b * this->colCount + c] / this->maxBinValue;
            float posX = 12.0f * c + 1.0f + b * width;
            float posY = 2.0f;
            glUniform1f(histogramProgram.ParameterLocation("binColor"), static_cast<float>(b) / static_cast<float>((this->bins - 1)));
            glUniform1f(histogramProgram.ParameterLocation("posX"), posX);
            glUniform1f(histogramProgram.ParameterLocation("posY"), posY);
            glUniform1f(histogramProgram.ParameterLocation("width"), width);
            glUniform1f(histogramProgram.ParameterLocation("height"), height);
            glUniform1i(histogramProgram.ParameterLocation("selected"), 0);
            glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, nullptr);

            height = 10.0f * this->selectedHistogram[b * this->colCount + c] / this->maxBinValue;
            glUniform1f(histogramProgram.ParameterLocation("height"), height);
            glUniform1i(histogramProgram.ParameterLocation("selected"), 1);
            glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, nullptr);
        }
    }

    tfCall->UnbindConvenience();
    glBindVertexArray(0);
    glUseProgram(0);

    float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    for (size_t c = 0; c < this->colCount; ++c) {
        float posX = 12.0f * c + 6.0f;
        this->font.DrawString(white, posX, 13.0f, 1.0f, false, this->colNames[c].c_str(), core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);
    }

    return true;
}

bool HistogramRenderer2D::handleCall(core::view::CallRender2D &call) {
    auto floatTableCall = this->tableDataCallerSlot.CallAs<table::TableDataCall>();
    if (floatTableCall == nullptr) {
        return false;
    }
    auto tfCall = this->transferFunctionCallerSlot.CallAs<core::view::CallGetTransferFunction>();
    if (tfCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "HistogramRenderer2D requires a transfer function!");
        return false;
    }
    auto flagsCall = this->flagStorageCallerSlot.CallAs<core::FlagCall>();
    if (flagsCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "HistogramRenderer2D requires a flag storage!");
        return false;
    }

    floatTableCall->SetFrameID(static_cast<unsigned int>(call.Time()));
    (*floatTableCall)(1);
    (*floatTableCall)(0);
    call.SetTimeFramesCount(floatTableCall->GetFrameCount());
    auto hash = floatTableCall->DataHash();
    auto frameId = floatTableCall->GetFrameID();
    (*tfCall)(0);
    (*flagsCall)(core::FlagCall::CallMapFlags);
    auto version = flagsCall->GetVersion();

    auto binsParam = static_cast<size_t>(this->numberOfBinsSlot.Param<core::param::IntParam>()->Value());
    if (this->currentTableDataHash != hash || this->currentTableFrameId != frameId || this->currentFlagStorageVersion != version || this->bins != binsParam) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Calculate Histogram");

        this->bins = binsParam;

        this->colCount = floatTableCall->GetColumnsCount();
        size_t rowCount = floatTableCall->GetRowsCount();

        flagsCall->validateFlagsCount(rowCount);

        auto flags = flagsCall->GetFlags();
        auto data = floatTableCall->GetData(); // use data pointer for direct access without assert checks

        this->colMinimums.resize(this->colCount);
        this->colMaximums.resize(this->colCount);
        this->colNames.resize(this->colCount);
        this->histogram.resize(this->colCount * bins);
        std::fill(this->histogram.begin(), this->histogram.end(), 0.0);
        this->selectedHistogram.resize(this->colCount * bins);
        std::fill(this->selectedHistogram.begin(), this->selectedHistogram.end(), 0.0);

        for (size_t i = 0; i < this->colCount; ++i) {
            auto colInfo = floatTableCall->GetColumnsInfos()[i];
            this->colMinimums[i] = colInfo.MinimumValue();
            this->colMaximums[i] = colInfo.MaximumValue();
            this->colNames[i] = colInfo.Name();
        }

        static const core::FlagStorage::FlagItemType filteredTestMask = core::FlagStorage::ENABLED | core::FlagStorage::FILTERED;
        static const core::FlagStorage::FlagItemType filteredPassMask = core::FlagStorage::ENABLED;
        static const core::FlagStorage::FlagItemType selectedTestMask = core::FlagStorage::ENABLED | core::FlagStorage::SELECTED | core::FlagStorage::FILTERED;
        static const core::FlagStorage::FlagItemType selectedPassMask = core::FlagStorage::ENABLED | core::FlagStorage::SELECTED;

        // TODO parallelize
        for (size_t r = 0; r < rowCount; ++r) {
            auto f = flags->operator[](r);
            if ((f & filteredTestMask) == filteredPassMask) {
                bool isSelected = (f & selectedTestMask) == selectedPassMask;
                for (size_t c = 0; c < this->colCount; ++c) {
                    float val = (data[r * this->colCount + c] - this->colMinimums[c]) / (this->colMaximums[c] - this->colMinimums[c]);
                    int bin_idx = std::clamp(static_cast<int>(val * this->bins), 0, static_cast<int>(this->bins) - 1);
                    this->histogram[bin_idx * this->colCount + c] += 1.0;
                    if (isSelected) {
                        this->selectedHistogram[bin_idx * this->colCount + c] += 1.0;
                    }
                }
            }
        }

        this->maxBinValue = *std::max_element(this->histogram.begin(), this->histogram.end());

        // we do read only, therefore version does not change
        flagsCall->SetFlags(flags, version);

        this->currentTableDataHash = hash;
        this->currentTableFrameId = frameId;
        this->currentFlagStorageVersion = version;
    }

    (*flagsCall)(core::FlagCall::CallUnmapFlags);

    return true;
}

bool HistogramRenderer2D::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    return false;
}

bool HistogramRenderer2D::OnMouseMove(double x, double y) {
    return false;
}
