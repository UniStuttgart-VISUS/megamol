/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "TextureHistogramRenderer2D.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/flags/FlagCallsGL.h"

using namespace megamol::infovis_gl;

using megamol::core::utility::log::Log;

TextureHistogramRenderer2D::TextureHistogramRenderer2D()
        : BaseHistogramRenderer2D()
        , textureDataCallerSlot_("getData", "Texture input")
        , flagStorageReadCallerSlot_("readFlagStorage", "Flag storage read input")
        , flagStorageWriteCallerSlot_("writeFlagStorage", "Flag storage write input") {

    textureDataCallerSlot_.SetCompatibleCall<compositing::CallTexture2DDescription>();
    MakeSlotAvailable(&textureDataCallerSlot_);

    flagStorageReadCallerSlot_.SetCompatibleCall<mmstd_gl::FlagCallRead_GLDescription>();
    MakeSlotAvailable(&flagStorageReadCallerSlot_);

    flagStorageWriteCallerSlot_.SetCompatibleCall<mmstd_gl::FlagCallWrite_GLDescription>();
    MakeSlotAvailable(&flagStorageWriteCallerSlot_);
}

TextureHistogramRenderer2D::~TextureHistogramRenderer2D() {
    this->Release();
}

bool TextureHistogramRenderer2D::createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) {
    try {
        calcMinMaxLinesProgram_ = core::utility::make_glowl_shader(
            "histo_tex_minmax_lines", shaderOptions, "infovis_gl/histo/tex_minmax_lines.comp.glsl");
        calcMinMaxAllProgram_ = core::utility::make_glowl_shader(
            "histo_tex_minmax_all", shaderOptions, "infovis_gl/histo/tex_minmax_all.comp.glsl");
        calcHistogramProgram_ =
            core::utility::make_glowl_shader("histo_tex_calc", shaderOptions, "infovis_gl/histo/tex_calc.comp.glsl");
        selectionProgram_ = core::utility::make_glowl_shader(
            "histo_tex_select", shaderOptions, "infovis_gl/histo/tex_select.comp.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("TextureHistogramRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    glGetProgramiv(selectionProgram_->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, selectionWorkgroupSize_);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkgroupCount_[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkgroupCount_[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkgroupCount_[2]);

    glGenBuffers(1, &minValueBuffer);
    glGenBuffers(1, &maxValueBuffer);

    return true;
}

void TextureHistogramRenderer2D::releaseImpl() {
    glDeleteBuffers(1, &minValueBuffer);
    glDeleteBuffers(1, &maxValueBuffer);
}

bool TextureHistogramRenderer2D::handleCall(mmstd_gl::CallRender2DGL& call) {
    auto textureCall = textureDataCallerSlot_.CallAs<compositing::CallTexture2D>();
    if (textureCall == nullptr) {
        return false;
    }

    auto readFlagsCall = flagStorageReadCallerSlot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    if (readFlagsCall == nullptr) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "TextureHistogramRenderer2D requires a read flag storage!");
        return false;
    }
    auto writeFlagsCall = flagStorageWriteCallerSlot_.CallAs<mmstd_gl::FlagCallWrite_GL>();
    if (writeFlagsCall == nullptr) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "TextureHistogramRenderer2D requires a write flag storage!");
        return false;
    }

    (*readFlagsCall)(mmstd_gl::FlagCallRead_GL::CallGetData);

    // stupid cheat, but is that really worth it?
    // static uint32_t lastFrame = -1;
    // auto currFrame = this->GetCoreInstance()->GetFrameID();
    // if (lastFrame == currFrame)
    //    return true;
    // lastFrame = currFrame;
    (*textureCall)(compositing::CallTexture2D::CallGetData);
    data_ = textureCall->getData();

    std::size_t numComponents = 0;
    std::vector<std::string> names;
    const auto int_form = data_->getFormat();

    switch (int_form) {
    case GL_RGBA:
        numComponents = 4;
        names = {"R", "G", "B", "A"};
        break;
    case GL_RGB:
        numComponents = 3;
        names = {"R", "G", "B"};
        break;
    case GL_RG:
        numComponents = 2;
        names = {"R", "G"};
        break;
    case GL_RED:
        numComponents = 1;
        names = {"R"};
        break;
    case GL_DEPTH_COMPONENT:
        numComponents = 1;
        names = {"Depth"};
        break;
    default:
        Log::DefaultLog.WriteError("TextureHistogramRenderer2D: unknown internal format: 0x%x", int_form);
        break;
    }

    // Local shader blocks are 256 x 4
    auto h = data_->getHeight();
    GLuint numGroups = (h > 0 ? h - 1 : h) / 256 + 1;

    // Get min and max value of texture

    // Use two buffers for min and max.
    // First numComponents values are for global values, after this numComponents values for each row.

    const GLsizeiptr bufSize = (numComponents + numComponents * h) * sizeof(float);

    if (lastTexSize != glm::ivec2(data_->getWidth(), data_->getHeight()) || readFlagsCall->hasUpdate()) {
        readFlagsCall->getData()->validateFlagCount(data_->getWidth() * data_->getHeight());
        lastTexSize = glm::ivec2(data_->getWidth(), data_->getHeight());

        glNamedBufferData(minValueBuffer, bufSize, nullptr, GL_STATIC_COPY);
        glNamedBufferData(maxValueBuffer, bufSize, nullptr, GL_STATIC_COPY);
    }
    readFlagsCall->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, 5);

    applySelections();

    const float maxVal = std::numeric_limits<float>::max();
    glClearNamedBufferData(minValueBuffer, GL_R32F, GL_RED, GL_FLOAT, &maxVal);

    const float minVal = std::numeric_limits<float>::lowest();
    glClearNamedBufferData(maxValueBuffer, GL_R32F, GL_RED, GL_FLOAT, &minVal);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, minValueBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, maxValueBuffer);

    calcMinMaxLinesProgram_->use();
    calcMinMaxLinesProgram_->setUniform("numComponents", static_cast<GLuint>(numComponents));

    glActiveTexture(GL_TEXTURE0);
    data_->bindTexture();
    calcMinMaxLinesProgram_->setUniform("tex", 0);

    glDispatchCompute(numGroups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    calcMinMaxAllProgram_->use();
    calcMinMaxAllProgram_->setUniform("numComponents", static_cast<GLuint>(numComponents));
    calcMinMaxAllProgram_->setUniform("texHeight", static_cast<GLuint>(h));

    glDispatchCompute(1, 1, 1);

    glUseProgram(0);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::vector<float> minimums(numComponents);
    glGetNamedBufferSubData(minValueBuffer, 0, numComponents * sizeof(float), minimums.data());

    std::vector<float> maximums(numComponents);
    glGetNamedBufferSubData(maxValueBuffer, 0, numComponents * sizeof(float), maximums.data());

    setComponentHeaders(std::move(names), std::move(minimums), std::move(maximums));

    resetHistogramBuffers();

    calcHistogramProgram_->use();
    bindCommon(calcHistogramProgram_);

    glActiveTexture(GL_TEXTURE0);
    data_->bindTexture();
    calcHistogramProgram_->setUniform("tex", 0);

    glDispatchCompute(numGroups, 1, 1);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    return true;
}

void TextureHistogramRenderer2D::updateSelection(SelectionMode selectionMode, int selectedComponent, int selectedBin) {

    auto binComp = std::make_pair(selectedBin, selectedComponent);
    switch (selectionMode) {
    case SelectionMode::PICK:
        selectedBinComps_.clear();
        selectedBinComps_.push_back(binComp);
        break;
    case SelectionMode::APPEND:
        selectedBinComps_.push_back(binComp);
        break;
    case SelectionMode::REMOVE:
        selectedBinComps_.erase(
            std::remove(selectedBinComps_.begin(), selectedBinComps_.end(), binComp), selectedBinComps_.end());
        break;
    }
}

void TextureHistogramRenderer2D::applySelections() {
    auto readFlagsCall = flagStorageReadCallerSlot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    auto writeFlagsCall = flagStorageWriteCallerSlot_.CallAs<mmstd_gl::FlagCallWrite_GL>();
    if (readFlagsCall != nullptr && writeFlagsCall != nullptr) {
        selectionProgram_->use();

        bindCommon(selectionProgram_);
        glActiveTexture(GL_TEXTURE0);
        data_->bindTexture();
        selectionProgram_->setUniform("tex", 0);
        readFlagsCall->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, 5);

        auto numRows = data_->getWidth() * data_->getHeight();
        selectionProgram_->setUniform("numRows", static_cast<GLuint>(numRows));
        GLuint groupCounts[3];
        computeDispatchSizes(numRows, selectionWorkgroupSize_, maxWorkgroupCount_, groupCounts);

        // deselect all
        selectionProgram_->setUniform(
            "selectionMode", static_cast<std::underlying_type_t<SelectionMode>>(SelectionMode::PICK));
        selectionProgram_->setUniform("selectedComponent", -1);
        selectionProgram_->setUniform("selectedBin", -1);
        glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);

        // select relevant
        for (auto& binComp : selectedBinComps_) {
            selectionProgram_->setUniform(
                "selectionMode", static_cast<std::underlying_type_t<SelectionMode>>(SelectionMode::APPEND));
            selectionProgram_->setUniform("selectedComponent", binComp.second);
            selectionProgram_->setUniform("selectedBin", binComp.first);
            glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
        }

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glUseProgram(0);

        writeFlagsCall->setData(readFlagsCall->getData(), readFlagsCall->version() + 1);
        (*writeFlagsCall)(mmstd_gl::FlagCallWrite_GL::CallGetData);
    }
}
