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
using namespace megamol::infovis_gl;

using megamol::core::utility::log::Log;

BaseHistogramRenderer2D::BaseHistogramRenderer2D()
        : Renderer2D()
        , transferFunctionCallerSlot_("getTransferFunction", "Transfer function input")
        , binsParam_("numberOfBins", "Number of bins")
        , logPlotParam_("logPlot", "Logarithmic scale")
        , selectionColorParam_("selectionColorParam", "Color of selection")
        , numBins_(10)
        , numComponents_(0)
        , maxBinValue_(0)
        , needMaxBinValueUpdate_(true)
        , font_(core::utility::SDFFont::PRESET_EVOLVENTA_SANS, core::utility::SDFFont::RENDERMODE_FILL)
        , mouseX_(0.0f)
        , mouseY_(0.0f)
        , viewRes_(0, 0)
        , needSelectionUpdate_(false)
        , selectionMode_(SelectionMode::PICK)
        , selectedComponent_(-1)
        , selectedBin_(-1) {
    transferFunctionCallerSlot_.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
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
        drawHistogramProgram_ = core::utility::make_glowl_shader("histo_base_draw", shaderOptions,
            "infovis_gl/histo/base_draw.vert.glsl", "infovis_gl/histo/base_draw.frag.glsl");
        drawAxesProgram_ = core::utility::make_glowl_shader("histo_base_axes", shaderOptions,
            "infovis_gl/histo/base_axes.vert.glsl", "infovis_gl/histo/base_axes.frag.glsl");
        calcMaxBinProgram_ = core::utility::make_glowl_shader(
            "histo_base_axes", shaderOptions, "infovis_gl/histo/base_max_bin.comp.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("BaseHistogramRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    glGenBuffers(1, &histogramBuffer_);
    glGenBuffers(1, &selectedHistogramBuffer_);
    glGenBuffers(1, &componentMinBuffer_);
    glGenBuffers(1, &componentMaxBuffer_);

    createImpl(shaderOptions);

    return true;
}

void BaseHistogramRenderer2D::release() {
    glDeleteBuffers(1, &histogramBuffer_);
    glDeleteBuffers(1, &selectedHistogramBuffer_);
    glDeleteBuffers(1, &componentMinBuffer_);
    glDeleteBuffers(1, &componentMaxBuffer_);

    releaseImpl();
}

bool BaseHistogramRenderer2D::GetExtents(mmstd_gl::CallRender2DGL& call) {
    if (!handleCall(call)) {
        return false;
    }

    // Draw histogram within 10.0 x 10.0 quads, left + right margin 1.0, top and bottom 2.0 for title and axes
    float sizeX = static_cast<float>(std::max<std::size_t>(1, numComponents_)) * 12.0f;
    call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, 0, sizeX, 14.0f, 0);
    return true;
}

bool BaseHistogramRenderer2D::Render(mmstd_gl::CallRender2DGL& call) {
    // store cam and view info for input transformation
    camera_ = call.GetCamera();
    viewRes_ = call.GetViewResolution();

    if (!handleCall(call)) {
        return false;
    }

    auto tfCall = transferFunctionCallerSlot_.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    if (tfCall == nullptr) {
        Log::DefaultLog.WriteError("BaseHistogramRenderer2D requires a transfer function!");
        return false;
    }
    (*tfCall)(0);

    if (needMaxBinValueUpdate_) {
        needMaxBinValueUpdate_ = false;

        GLuint maxBinValueBuffer;
        glCreateBuffers(1, &maxBinValueBuffer);
        GLint zero = 0;
        glNamedBufferStorage(maxBinValueBuffer, sizeof(GLint), &zero, GL_DYNAMIC_STORAGE_BIT | GL_CLIENT_STORAGE_BIT);

        calcMaxBinProgram_->use();
        bindCommon(calcMaxBinProgram_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, maxBinValueBuffer);
        glDispatchCompute(1, 1, 1);
        glUseProgram(0);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Download max bin value for text label.
        glGetNamedBufferSubData(maxBinValueBuffer, 0, sizeof(GLint), &maxBinValue_);

        glDeleteBuffers(1, &maxBinValueBuffer);
    }

    // Update selection
    if (needSelectionUpdate_) {
        needSelectionUpdate_ = false;
        updateSelection(selectionMode_, selectedComponent_, selectedBin_);
    }

    // get camera
    core::view::Camera cam = call.GetCamera();
    const auto viewMx = cam.getViewMatrix();
    const auto projMx = cam.getProjectionMatrix();

    drawHistogramProgram_->use();
    drawHistogramProgram_->setUniform("modelView", viewMx);
    drawHistogramProgram_->setUniform("projection", projMx);

    bindCommon(drawHistogramProgram_);
    tfCall->BindConvenience(drawHistogramProgram_, GL_TEXTURE0, 0);

    drawHistogramProgram_->setUniform("maxBinValue", static_cast<GLuint>(maxBinValue_));
    drawHistogramProgram_->setUniform(
        "logPlot", static_cast<int>(logPlotParam_.Param<core::param::BoolParam>()->Value()));
    glUniform4fv(drawHistogramProgram_->getUniformLocation("selectionColor"), 1,
        selectionColorParam_.Param<core::param::ColorParam>()->Value().data());

    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(numBins_ * numComponents_));

    tfCall->UnbindConvenience();
    glBindVertexArray(0);
    glUseProgram(0);

    drawAxesProgram_->use();
    drawAxesProgram_->setUniform("modelView", viewMx);
    drawAxesProgram_->setUniform("projection", projMx);
    drawAxesProgram_->setUniform("componentTotalSize", 12.0f, 14.0f);
    drawAxesProgram_->setUniform("componentDrawSize", 10.0f, 10.0f);
    drawAxesProgram_->setUniform("componentDrawOffset", 1.0f, 2.0f);

    drawAxesProgram_->setUniform("mode", 0);
    glDrawArraysInstanced(GL_LINES, 0, 2, static_cast<GLsizei>(numComponents_));

    drawAxesProgram_->setUniform("mode", 1);
    glDrawArrays(GL_LINES, 0, 2);

    glUseProgram(0);

    font_.ClearBatchDrawCache();

    float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    glm::mat4 ortho = projMx * viewMx;

    for (std::size_t c = 0; c < numComponents_; ++c) {
        float posX = 12.0f * static_cast<float>(c) + 6.0f;
        font_.DrawString(ortho, white, posX, 13.0f, 1.0f, false, componentNames_[c].c_str(),
            core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
        font_.DrawString(ortho, white, posX - 5.0f, 2.0f, 1.0f, false, std::to_string(componentMinimums_[c]).c_str(),
            core::utility::SDFFont::ALIGN_LEFT_TOP);
        font_.DrawString(ortho, white, posX + 5.0f, 2.0f, 1.0f, false, std::to_string(componentMaximums_[c]).c_str(),
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
        selectionMode_ = SelectionMode::PICK;
    } else if (left && shift) {
        selectionMode_ = SelectionMode::APPEND;
    } else if (right && shift) {
        selectionMode_ = SelectionMode::REMOVE;
    } else {
        return false;
    }

    needSelectionUpdate_ = true;
    selectedComponent_ = -1;
    selectedBin_ = -1;

    if (mouseY_ < 2.0f || mouseY_ > 12.0f) {
        return true;
    }

    selectedComponent_ = static_cast<int>(std::floor(mouseX_ / 12.0f));
    if (selectedComponent_ < 0 || selectedComponent_ >= numComponents_) {
        selectedComponent_ = -1;
        return true;
    }

    float posX = (std::fmod(mouseX_, 12.0f) - 1.0f) / 10.0f;
    if (posX < 0.0f || posX >= 1.0f) {
        return true;
    }
    selectedBin_ = static_cast<int>(posX * static_cast<float>(numBins_));
    if (selectedBin_ < 0 || selectedBin_ >= numBins_) {
        selectedBin_ = -1;
    }

    return true;
}

bool BaseHistogramRenderer2D::OnMouseMove(double x, double y) {
    auto const& [world_x, world_y] = mouseCoordsToWorld(x, y, camera_, viewRes_.x, viewRes_.y);
    mouseX_ = world_x;
    mouseY_ = world_y;
    return false;
}

void BaseHistogramRenderer2D::bindCommon(std::unique_ptr<glowl::GLSLProgram>& program) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, histogramBuffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, selectedHistogramBuffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, componentMinBuffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, componentMaxBuffer_);

    program->setUniform("numBins", static_cast<GLuint>(numBins_));
    program->setUniform("numComponents", static_cast<GLuint>(numComponents_));
}

void BaseHistogramRenderer2D::setComponentHeaders(
    std::vector<std::string> names, std::vector<float> minimums, std::vector<float> maximums) {
    auto numComponents = names.size();
    if (minimums.size() != numComponents || maximums.size() != numComponents) {
        throw std::invalid_argument("Name, minimum and maximum lists must have the same size!");
    }
    numComponents_ = numComponents;
    componentNames_ = std::move(names);
    componentMinimums_ = std::move(minimums);
    componentMaximums_ = std::move(maximums);

    auto bufSize = static_cast<GLsizeiptr>(numComponents_ * sizeof(float));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, componentMinBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, componentMinimums_.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, componentMaxBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, componentMaximums_.data(), GL_STATIC_DRAW);
}

void BaseHistogramRenderer2D::resetHistogramBuffers() {
    numBins_ = static_cast<std::size_t>(binsParam_.Param<core::param::IntParam>()->Value());
    GLint zero = 0;
    auto bufSize = static_cast<GLsizeiptr>(numComponents_ * numBins_ * sizeof(GLint));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogramBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, nullptr, GL_STATIC_COPY);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, selectedHistogramBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, nullptr, GL_STATIC_COPY);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED, GL_INT, &zero);
    needMaxBinValueUpdate_ = true;
}
