#include "stdafx.h"
#include "HistogramRenderer2D.h"

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

using vislib::sys::Log;

HistogramRenderer2D::HistogramRenderer2D()
    : Renderer2D()
    , tableDataCallerSlot("getData", "Float table input")
    , transferFunctionCallerSlot("getTransferFunction", "Transfer function input")
    , flagStorageCallerSlot("getFlagStorage", "Flag storage input")
    , currentTableDataHash(std::numeric_limits<std::size_t>::max())
    , currentTableFrameId(std::numeric_limits<unsigned int>::max())
    , currentFlagStorageVersion(std::numeric_limits<core::FlagStorage::FlagVersionType>::max())
    , bins(10) // TODO
    , font("Evolventa-SansSerif", core::utility::SDFFont::RenderType::RENDERTYPE_FILL)
{
    this->tableDataCallerSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableDataCallerSlot);

    this->transferFunctionCallerSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferFunctionCallerSlot);

    this->flagStorageCallerSlot.SetCompatibleCall<core::FlagCallDescription>();
    this->MakeSlotAvailable(&this->flagStorageCallerSlot);
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

    call.SetBoundingBox(0.0f, 0.0f, 10.0f, 10.0f);
    return true;
}

bool HistogramRenderer2D::Render(core::view::CallRender2D &call) {
    if (!handleCall(call)) {
        return false;
    }

    // this is the apex of suck and must die
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck

    glUseProgram(histogramProgram);
    glUniformMatrix4fv(histogramProgram.ParameterLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(histogramProgram.ParameterLocation("projection"), 1, GL_FALSE, projMatrix_column);

    glBindVertexArray(quadVertexArray);
    glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
    glUseProgram(0);

    float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    this->font.DrawString(red, 5.0, 5.0, 0.5f, false, "hello", core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);

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

    if (this->currentTableDataHash != hash || this->currentTableFrameId != frameId || this->currentFlagStorageVersion != version) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Calculate Histogram");

        size_t colCount = floatTableCall->GetColumnsCount();
        size_t rowCount = floatTableCall->GetRowsCount();

        flagsCall->validateFlagsCount(rowCount);

        auto data = floatTableCall->GetData(); // use data pointer for direct access without assert checks

        this->colMinimums.resize(colCount);
        this->colMaximums.resize(colCount);
        this->colNames.resize(colCount);
        this->histogram.resize(colCount * bins);
        std::fill(this->histogram.begin(), this->histogram.end(), 0.0);

        for (size_t i = 0; i < colCount; ++i) {
            auto colInfo = floatTableCall->GetColumnsInfos()[i];
            this->colMinimums[i] = colInfo.MinimumValue();
            this->colMaximums[i] = colInfo.MaximumValue();
            this->colNames[i] = colInfo.Name();
        }

        // TODO parallelize
        for (size_t r = 0; r < rowCount; ++r) {
            for (size_t c = 0; c < colCount; ++c) {
                float val = (data[r * colCount + c] - colMinimums[c]) / (colMaximums[c] - colMinimums[c]);
                int bin_idx = std::clamp(static_cast<int>(val * bins), 0, static_cast<int>(bins) - 1);
                this->histogram[bin_idx * colCount + c] += 1.0;
            }
        }

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
