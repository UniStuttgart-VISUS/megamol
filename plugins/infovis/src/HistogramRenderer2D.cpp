#include "stdafx.h"
#include "HistogramRenderer2D.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/ShaderFactory.h"

using namespace megamol;
using namespace megamol::infovis;
using namespace megamol::stdplugin::datatools;

using megamol::core::utility::log::Log;

HistogramRenderer2D::HistogramRenderer2D()
        : Renderer2D()
        , tableDataCallerSlot("getData", "Float table input")
        , transferFunctionCallerSlot("getTransferFunction", "Transfer function input")
        , flagStorageReadCallerSlot("readFlagStorage", "Flag storage read input")
        , flagStorageWriteCallerSlot("writeFlagStorage", "Flag storage write input")
        , numberOfBinsParam("numberOfBins", "Number of bins")
        , logPlotParam("logPlot", "Logarithmic scale")
        , selectionColorParam("selectionColorParam", "Color of selection")
        , currentTableDataHash(std::numeric_limits<std::size_t>::max())
        , currentTableFrameId(std::numeric_limits<unsigned int>::max())
        , bins(10)
        , colCount(0)
        , rowCount(0)
        , maxBinValue(0)
        , font(core::utility::SDFFont::PRESET_EVOLVENTA_SANS, core::utility::SDFFont::RENDERMODE_FILL)
        , mouseX(0.0f)
        , mouseY(0.0f)
        , needSelectionUpdate(false)
        , selectionMode(0)
        , selectedCol(-1)
        , selectedBin(-1) {
    this->tableDataCallerSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableDataCallerSlot);

    this->transferFunctionCallerSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferFunctionCallerSlot);

    this->flagStorageReadCallerSlot.SetCompatibleCall<core::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->flagStorageReadCallerSlot);

    this->flagStorageWriteCallerSlot.SetCompatibleCall<core::FlagCallWrite_GLDescription>();
    this->MakeSlotAvailable(&this->flagStorageWriteCallerSlot);

    this->numberOfBinsParam << new core::param::IntParam(this->bins, 1);
    this->MakeSlotAvailable(&this->numberOfBinsParam);

    this->logPlotParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->logPlotParam);

    this->selectionColorParam << new core::param::ColorParam("red");
    this->MakeSlotAvailable(&this->selectionColorParam);
}

HistogramRenderer2D::~HistogramRenderer2D() {
    this->Release();
}

bool HistogramRenderer2D::create() {
    if (!this->font.Initialise(this->GetCoreInstance()))
        return false;
    this->font.SetBatchDrawMode(true);

    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

    try {
        calcHistogramProgram =
            core::utility::make_glowl_shader("histo_calc", shader_options, "infovis/histo_calc.comp.glsl");
        selectionProgram =
            core::utility::make_glowl_shader("histo_select", shader_options, "infovis/histo_select.comp.glsl");
        histogramProgram = core::utility::make_glowl_shader(
            "histo_draw", shader_options, "infovis/histo_draw.vert.glsl", "infovis/histo_draw.frag.glsl");
        axesProgram = core::utility::make_glowl_shader(
            "histo_axes", shader_options, "infovis/histo_axes.vert.glsl", "infovis/histo_axes.frag.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, ("HistogramRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    glGenBuffers(1, &this->floatDataBuffer);
    glGenBuffers(1, &this->minBuffer);
    glGenBuffers(1, &this->maxBuffer);
    glGenBuffers(1, &this->histogramBuffer);
    glGenBuffers(1, &this->selectedHistogramBuffer);
    glGenBuffers(1, &this->maxBinValueBuffer);

    glGetProgramiv(selectionProgram->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, selectionWorkgroupSize);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkgroupCount[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkgroupCount[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkgroupCount[2]);

    return true;
}

void HistogramRenderer2D::release() {
    glDeleteBuffers(1, &this->floatDataBuffer);
    glDeleteBuffers(1, &this->minBuffer);
    glDeleteBuffers(1, &this->maxBuffer);
    glDeleteBuffers(1, &this->histogramBuffer);
    glDeleteBuffers(1, &this->selectedHistogramBuffer);
    glDeleteBuffers(1, &this->maxBinValueBuffer);
}

bool HistogramRenderer2D::GetExtents(core::view::CallRender2DGL& call) {
    if (!handleCall(call)) {
        return false;
    }

    // Draw histogram within 10.0 x 10.0 quads, left + right margin 1.0, top and bottom 2.0 for title and axes
    float sizeX = static_cast<float>(std::max<size_t>(1, this->colCount)) * 12.0f;
    call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, 0, sizeX, 14.0f, 0);
    return true;
}

bool HistogramRenderer2D::Render(core::view::CallRender2DGL& call) {
    if (!handleCall(call)) {
        return false;
    }

    auto tfCall = this->transferFunctionCallerSlot.CallAs<core::view::CallGetTransferFunction>();
    if (tfCall == nullptr) {
        return false;
    }

    // Update selection
    if (needSelectionUpdate) {
        needSelectionUpdate = false;
        auto readFlagsCall = flagStorageReadCallerSlot.CallAs<core::FlagCallRead_GL>();
        auto writeFlagsCall = flagStorageWriteCallerSlot.CallAs<core::FlagCallWrite_GL>();
        if (readFlagsCall != nullptr && writeFlagsCall != nullptr) {
            selectionProgram->use();
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->floatDataBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->minBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->maxBuffer);
            readFlagsCall->getData()->flags->bind(3);

            selectionProgram->setUniform("binCount", static_cast<GLuint>(this->bins));
            selectionProgram->setUniform("colCount", static_cast<GLuint>(this->colCount));
            selectionProgram->setUniform("rowCount", static_cast<GLuint>(this->rowCount));
            selectionProgram->setUniform("selectionMode", selectionMode);
            selectionProgram->setUniform("selectedCol", selectedCol);
            selectionProgram->setUniform("selectedBin", selectedBin);

            GLuint groupCounts[3];
            computeDispatchSizes(rowCount, selectionWorkgroupSize, maxWorkgroupCount, groupCounts);

            glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            glUseProgram(0);

            writeFlagsCall->setData(readFlagsCall->getData(), readFlagsCall->version() + 1);
            (*writeFlagsCall)(core::FlagCallWrite_GL::CallGetData);
        }
    }

    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck

    histogramProgram->use();
    glUniformMatrix4fv(histogramProgram->getUniformLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(histogramProgram->getUniformLocation("projection"), 1, GL_FALSE, projMatrix_column);

    tfCall->BindConvenience(histogramProgram, GL_TEXTURE0, 0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->histogramBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->selectedHistogramBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->maxBinValueBuffer);

    histogramProgram->setUniform("binCount", static_cast<GLuint>(this->bins));
    histogramProgram->setUniform("colCount", static_cast<GLuint>(this->colCount));
    histogramProgram->setUniform(
        "logPlot", static_cast<int>(this->logPlotParam.Param<core::param::BoolParam>()->Value()));
    glUniform4fv(histogramProgram->getUniformLocation("selectionColor"), 1,
        selectionColorParam.Param<core::param::ColorParam>()->Value().data());

    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, this->bins * this->colCount);

    tfCall->UnbindConvenience();
    glBindVertexArray(0);
    glUseProgram(0);

    axesProgram->use();
    glUniformMatrix4fv(axesProgram->getUniformLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(axesProgram->getUniformLocation("projection"), 1, GL_FALSE, projMatrix_column);
    axesProgram->setUniform("colTotalSize", 12.0f, 14.0f);
    axesProgram->setUniform("colDrawSize", 10.0f, 10.0f);
    axesProgram->setUniform("colDrawOffset", 1.0f, 2.0f);

    axesProgram->setUniform("mode", 0);
    glDrawArraysInstanced(GL_LINES, 0, 2, this->colCount);

    axesProgram->setUniform("mode", 1);
    glDrawArrays(GL_LINES, 0, 2);

    glUseProgram(0);

    this->font.ClearBatchDrawCache();

    float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    glm::mat4 ortho = glm::make_mat4(projMatrix_column) * glm::make_mat4(modelViewMatrix_column);

    for (size_t c = 0; c < this->colCount; ++c) {
        float posX = 12.0f * c + 6.0f;
        this->font.DrawString(ortho, white, posX, 13.0f, 1.0f, false, this->colNames[c].c_str(),
            core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
        this->font.DrawString(ortho, white, posX - 5.0f, 2.0f, 1.0f, false,
            std::to_string(this->colMinimums[c]).c_str(), core::utility::SDFFont::ALIGN_LEFT_TOP);
        this->font.DrawString(ortho, white, posX + 5.0f, 2.0f, 1.0f, false,
            std::to_string(this->colMaximums[c]).c_str(), core::utility::SDFFont::ALIGN_RIGHT_TOP);
    }
    this->font.DrawString(ortho, white, 1.0f, 12.0f, 1.0f, false, std::to_string(this->maxBinValue).c_str(),
        core::utility::SDFFont::ALIGN_RIGHT_TOP);
    this->font.DrawString(ortho, white, 1.0f, 2.0f, 1.0f, false, "0", core::utility::SDFFont::ALIGN_RIGHT_BOTTOM);

    this->font.BatchDrawString(ortho);

    return true;
}

bool HistogramRenderer2D::handleCall(core::view::CallRender2DGL& call) {
    auto floatTableCall = this->tableDataCallerSlot.CallAs<table::TableDataCall>();
    if (floatTableCall == nullptr) {
        return false;
    }
    auto tfCall = this->transferFunctionCallerSlot.CallAs<core::view::CallGetTransferFunction>();
    if (tfCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "HistogramRenderer2D requires a transfer function!");
        return false;
    }
    auto readFlagsCall = this->flagStorageReadCallerSlot.CallAs<core::FlagCallRead_GL>();
    if (readFlagsCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "HistogramRenderer2D requires a read flag storage!");
        return false;
    }
    auto writeFlagsCall = this->flagStorageWriteCallerSlot.CallAs<core::FlagCallWrite_GL>();
    if (writeFlagsCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "HistogramRenderer2D requires a write flag storage!");
        return false;
    }

    floatTableCall->SetFrameID(static_cast<unsigned int>(call.Time()));
    (*floatTableCall)(1);
    (*floatTableCall)(0);
    call.SetTimeFramesCount(floatTableCall->GetFrameCount());
    auto hash = floatTableCall->DataHash();
    auto frameId = floatTableCall->GetFrameID();
    (*tfCall)(0);
    (*readFlagsCall)(core::FlagCallRead_GL::CallGetData);

    bool dataChanged = this->currentTableDataHash != hash || this->currentTableFrameId != frameId;
    if (dataChanged) {
        this->colCount = floatTableCall->GetColumnsCount();
        this->rowCount = floatTableCall->GetRowsCount();

        this->colMinimums.resize(this->colCount);
        this->colMaximums.resize(this->colCount);
        this->colNames.resize(this->colCount);

        for (size_t i = 0; i < this->colCount; ++i) {
            auto colInfo = floatTableCall->GetColumnsInfos()[i];
            this->colMinimums[i] = colInfo.MinimumValue();
            this->colMaximums[i] = colInfo.MaximumValue();
            this->colNames[i] = colInfo.Name();
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->floatDataBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->colCount * this->rowCount * sizeof(float),
            floatTableCall->GetData(), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->minBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, this->colCount * sizeof(float), this->colMinimums.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->maxBuffer);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, this->colCount * sizeof(float), this->colMaximums.data(), GL_STATIC_DRAW);
    }

    auto binsParam = static_cast<size_t>(this->numberOfBinsParam.Param<core::param::IntParam>()->Value());
    if (dataChanged || readFlagsCall->hasUpdate() || this->bins != binsParam) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_INFO, "Calculate Histogram");

        this->bins = binsParam;

        GLint zero = 0.0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->histogramBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->colCount * this->bins * sizeof(GLint), nullptr, GL_STATIC_COPY);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->selectedHistogramBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, this->colCount * this->bins * sizeof(GLint), nullptr, GL_STATIC_COPY);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->maxBinValueBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLint), nullptr, GL_STATIC_COPY);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);

        readFlagsCall->getData()->validateFlagCount(floatTableCall->GetRowsCount());

        calcHistogramProgram->use();

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->floatDataBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->minBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->maxBuffer);
        readFlagsCall->getData()->flags->bind(3);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, this->histogramBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, this->selectedHistogramBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, this->maxBinValueBuffer);

        calcHistogramProgram->setUniform("binCount", static_cast<GLuint>(this->bins));
        calcHistogramProgram->setUniform("colCount", static_cast<GLuint>(this->colCount));
        calcHistogramProgram->setUniform("rowCount", static_cast<GLuint>(this->rowCount));

        glDispatchCompute(1, 1, 1);

        glUseProgram(0);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Download max bin value for text label.
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->maxBinValueBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLint), &this->maxBinValue);

        this->currentTableDataHash = hash;
        this->currentTableFrameId = frameId;
    }

    return true;
}

bool HistogramRenderer2D::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    // Ctrl goes to the view and ignore everything than press event.
    if (mods.test(core::view::Modifier::CTRL) || action != core::view::MouseButtonAction::PRESS) {
        return false;
    }

    bool left = button == core::view::MouseButton::BUTTON_LEFT;
    bool right = button == core::view::MouseButton::BUTTON_RIGHT;
    bool shift = mods.test(core::view::Modifier::SHIFT);

    if (left && !shift) {
        selectionMode = 0;
    } else if (left && shift) {
        selectionMode = 1;
    } else if (right && shift) {
        selectionMode = 2;
    } else {
        return false;
    }

    needSelectionUpdate = true;
    selectedCol = -1;
    selectedBin = -1;

    if (mouseY < 2.0f || mouseY > 12.0f) {
        return true;
    }

    selectedCol = static_cast<int>(std::floor(mouseX / 12.0f));
    if (selectedCol < 0 || selectedCol >= colCount) {
        selectedCol = -1;
        return true;
    }

    float posX = (std::fmod(mouseX, 12.0f) - 1.0f) / 10.0f;
    if (posX < 0.0f || posX >= 1.0f) {
        return true;
    }
    selectedBin = static_cast<int>(posX * bins);
    if (selectedBin < 0 || selectedBin >= bins) {
        selectedBin = -1;
    }

    return true;
}

bool HistogramRenderer2D::OnMouseMove(double x, double y) {
    this->mouseX = static_cast<float>(x);
    this->mouseY = static_cast<float>(y);
    return false;
}
