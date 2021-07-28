/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "BaseHistogramRenderer2D.h"

#include <algorithm>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/IntParam.h"

using namespace megamol;
using namespace megamol::infovis;

using megamol::core::utility::log::Log;

BaseHistogramRenderer2D::BaseHistogramRenderer2D()
        : Renderer2D()
        , transferFunctionCallerSlot_("getTransferFunction", "Transfer function input")
        , binsParam_("numberOfBins", "Number of bins")
        , logPlotParam_("logPlot", "Logarithmic scale")
        , selectionColorParam_("selectionColorParam", "Color of selection")
        , numBins_(10)
        , numCols_(0)
        , maxBinValue_(0)
        , needMaxBinValueUpdate_(true)
        , font_(core::utility::SDFFont::PRESET_EVOLVENTA_SANS, core::utility::SDFFont::RENDERMODE_FILL)
        , mouseX_(0.0f)
        , mouseY_(0.0f)
        , needSelectionUpdate_(false)
        , selectionMode_(0)
        , selectedCol_(-1)
        , selectedBin_(-1) {
    transferFunctionCallerSlot_.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&transferFunctionCallerSlot_);

    binsParam_ << new core::param::IntParam(static_cast<int>(numBins_), 1);
    MakeSlotAvailable(&binsParam_);

    logPlotParam_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&logPlotParam_);

    selectionColorParam_ << new core::param::ColorParam("red");
    MakeSlotAvailable(&selectionColorParam_);
}

bool BaseHistogramRenderer2D::create() {
    if (!font_.Initialise(GetCoreInstance())) {
        return false;
    }
    font_.SetBatchDrawMode(true);

    auto const shaderOptions = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        drawHistogramProgram_ = core::utility::make_glowl_shader(
            "histo_base_draw", shaderOptions, "infovis/histo/base_draw.vert.glsl", "infovis/histo/base_draw.frag.glsl");
        drawAxesProgram_ = core::utility::make_glowl_shader(
            "histo_base_axes", shaderOptions, "infovis/histo/base_axes.vert.glsl", "infovis/histo/base_axes.frag.glsl");
        maxBinProgram_ =
            core::utility::make_glowl_shader("histo_base_axes", shaderOptions, "infovis/histo/base_max_bin.comp.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("HistogramRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    glGenBuffers(1, &histogramBuffer_);
    glGenBuffers(1, &selectedHistogramBuffer_);
    glGenBuffers(1, &minBuffer_);
    glGenBuffers(1, &maxBuffer_);

    createHistoImpl(shaderOptions);

    return true;
}

void BaseHistogramRenderer2D::release() {
    glDeleteBuffers(1, &histogramBuffer_);
    glDeleteBuffers(1, &selectedHistogramBuffer_);
    glDeleteBuffers(1, &minBuffer_);
    glDeleteBuffers(1, &maxBuffer_);

    releaseHistoImpl();
}

bool BaseHistogramRenderer2D::GetExtents(core::view::CallRender2DGL& call) {
    if (!handleCall(call)) {
        return false;
    }

    // Draw histogram within 10.0 x 10.0 quads, left + right margin 1.0, top and bottom 2.0 for title and axes
    float sizeX = static_cast<float>(std::max<std::size_t>(1, numCols_)) * 12.0f;
    call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, 0, sizeX, 14.0f, 0);
    return true;
}

bool BaseHistogramRenderer2D::Render(core::view::CallRender2DGL& call) {
    if (!handleCall(call)) {
        return false;
    }

    auto tfCall = transferFunctionCallerSlot_.CallAs<core::view::CallGetTransferFunction>();
    if (tfCall == nullptr) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "HistogramRenderer2D requires a transfer function!");
        return false;
    }
    (*tfCall)(0);

    if (needMaxBinValueUpdate_) {
        needMaxBinValueUpdate_ = false;

        GLuint maxBinValueBuffer;
        glGenBuffers(1, &maxBinValueBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxBinValueBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLint), nullptr, GL_STATIC_COPY);
        GLint zero = 0;
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);

        maxBinProgram_->use();
        bindCommon(maxBinProgram_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, maxBinValueBuffer);
        glDispatchCompute(1, 1, 1);
        glUseProgram(0);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Download max bin value for text label.
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxBinValueBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLint), &maxBinValue_);

        glDeleteBuffers(1, &maxBinValueBuffer);
    }

    // Update selection
    if (needSelectionUpdate_) {
        needSelectionUpdate_ = false;
        updateSelection(selectionMode_, selectedCol_, selectedBin_);
    }

    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck

    drawHistogramProgram_->use();
    glUniformMatrix4fv(drawHistogramProgram_->getUniformLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(drawHistogramProgram_->getUniformLocation("projection"), 1, GL_FALSE, projMatrix_column);

    bindCommon(drawHistogramProgram_);
    tfCall->BindConvenience(drawHistogramProgram_, GL_TEXTURE0, 0);

    drawHistogramProgram_->setUniform("maxBinValue", static_cast<GLuint>(maxBinValue_));
    drawHistogramProgram_->setUniform(
        "logPlot", static_cast<int>(logPlotParam_.Param<core::param::BoolParam>()->Value()));
    glUniform4fv(drawHistogramProgram_->getUniformLocation("selectionColor"), 1,
        selectionColorParam_.Param<core::param::ColorParam>()->Value().data());

    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, numBins_ * numCols_);

    tfCall->UnbindConvenience();
    glBindVertexArray(0);
    glUseProgram(0);

    drawAxesProgram_->use();
    glUniformMatrix4fv(drawAxesProgram_->getUniformLocation("modelView"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(drawAxesProgram_->getUniformLocation("projection"), 1, GL_FALSE, projMatrix_column);
    drawAxesProgram_->setUniform("colTotalSize", 12.0f, 14.0f);
    drawAxesProgram_->setUniform("colDrawSize", 10.0f, 10.0f);
    drawAxesProgram_->setUniform("colDrawOffset", 1.0f, 2.0f);

    drawAxesProgram_->setUniform("mode", 0);
    glDrawArraysInstanced(GL_LINES, 0, 2, numCols_);

    drawAxesProgram_->setUniform("mode", 1);
    glDrawArrays(GL_LINES, 0, 2);

    glUseProgram(0);

    font_.ClearBatchDrawCache();

    float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    glm::mat4 ortho = glm::make_mat4(projMatrix_column) * glm::make_mat4(modelViewMatrix_column);

    for (std::size_t c = 0; c < numCols_; ++c) {
        float posX = 12.0f * static_cast<float>(c) + 6.0f;
        font_.DrawString(
            ortho, white, posX, 13.0f, 1.0f, false, colNames_[c].c_str(), core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
        font_.DrawString(ortho, white, posX - 5.0f, 2.0f, 1.0f, false, std::to_string(colMinimums_[c]).c_str(),
            core::utility::SDFFont::ALIGN_LEFT_TOP);
        font_.DrawString(ortho, white, posX + 5.0f, 2.0f, 1.0f, false, std::to_string(colMaximums_[c]).c_str(),
            core::utility::SDFFont::ALIGN_RIGHT_TOP);
    }
    font_.DrawString(ortho, white, 1.0f, 12.0f, 1.0f, false, std::to_string(maxBinValue_).c_str(),
        core::utility::SDFFont::ALIGN_RIGHT_TOP);
    font_.DrawString(ortho, white, 1.0f, 2.0f, 1.0f, false, "0", core::utility::SDFFont::ALIGN_RIGHT_BOTTOM);

    font_.BatchDrawString(ortho);

    return true;
}

bool BaseHistogramRenderer2D::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    // Ctrl goes to the view and ignore everything than press event.
    if (mods.test(core::view::Modifier::CTRL) || action != core::view::MouseButtonAction::PRESS) {
        return false;
    }

    bool left = button == core::view::MouseButton::BUTTON_LEFT;
    bool right = button == core::view::MouseButton::BUTTON_RIGHT;
    bool shift = mods.test(core::view::Modifier::SHIFT);

    if (left && !shift) {
        selectionMode_ = 0;
    } else if (left && shift) {
        selectionMode_ = 1;
    } else if (right && shift) {
        selectionMode_ = 2;
    } else {
        return false;
    }

    needSelectionUpdate_ = true;
    selectedCol_ = -1;
    selectedBin_ = -1;

    if (mouseY_ < 2.0f || mouseY_ > 12.0f) {
        return true;
    }

    selectedCol_ = static_cast<int>(std::floor(mouseX_ / 12.0f));
    if (selectedCol_ < 0 || selectedCol_ >= numCols_) {
        selectedCol_ = -1;
        return true;
    }

    float posX = (std::fmod(mouseX_, 12.0f) - 1.0f) / 10.0f;
    if (posX < 0.0f || posX >= 1.0f) {
        return true;
    }
    selectedBin_ = static_cast<int>(posX * numBins_);
    if (selectedBin_ < 0 || selectedBin_ >= numBins_) {
        selectedBin_ = -1;
    }

    return true;
}

bool BaseHistogramRenderer2D::OnMouseMove(double x, double y) {
    mouseX_ = static_cast<float>(x);
    mouseY_ = static_cast<float>(y);
    return false;
}

void BaseHistogramRenderer2D::bindCommon(std::unique_ptr<glowl::GLSLProgram>& program) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogramBuffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, selectedHistogramBuffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, minBuffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, maxBuffer_);

    program->setUniform("numBins", static_cast<GLuint>(numBins_));
    program->setUniform("numCols", static_cast<GLuint>(numCols_));
}

void BaseHistogramRenderer2D::setColHeaders(
    std::vector<std::string> colNames, std::vector<float> colMinimums, std::vector<float> colMaximums) {
    auto numCols = colNames.size();
    if (colMinimums.size() != numCols || colMaximums.size() != numCols) {
        throw std::invalid_argument("Name, minimum and maximum lists must have the same size!");
    }
    numCols_ = numCols;
    colNames_ = std::move(colNames);
    colMinimums_ = std::move(colMinimums);
    colMaximums_ = std::move(colMaximums);

    auto bufSize = static_cast<GLsizeiptr>(numCols_ * sizeof(float));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, minBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, colMinimums_.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, colMaximums_.data(), GL_STATIC_DRAW);
}

void BaseHistogramRenderer2D::resizeAndClearHistoBuffers() {
    numBins_ = static_cast<std::size_t>(binsParam_.Param<core::param::IntParam>()->Value());
    GLint zero = 0;
    auto bufSize = static_cast<GLsizeiptr>(numCols_ * numBins_ * sizeof(GLint));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogramBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, nullptr, GL_STATIC_COPY);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, selectedHistogramBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, nullptr, GL_STATIC_COPY);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);
    needMaxBinValueUpdate_ = true;
}
