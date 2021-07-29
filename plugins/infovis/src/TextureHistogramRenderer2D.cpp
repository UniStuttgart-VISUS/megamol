/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "TextureHistogramRenderer2D.h"

#include "compositing/CompositingCalls.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/ShaderFactory.h"

using namespace megamol::infovis;

using megamol::core::utility::log::Log;

TextureHistogramRenderer2D::TextureHistogramRenderer2D()
        : BaseHistogramRenderer2D(), textureDataCallerSlot_("getData", "Texture input") {
    textureDataCallerSlot_.SetCompatibleCall<compositing::CallTexture2DDescription>();
    MakeSlotAvailable(&textureDataCallerSlot_);
}

TextureHistogramRenderer2D::~TextureHistogramRenderer2D() {
    this->Release();
}

bool TextureHistogramRenderer2D::createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) {
    try {
        calcMinMaxLinesProgram_ = core::utility::make_glowl_shader(
            "histo_tex_minmax_lines", shaderOptions, "infovis/histo/tex_minmax_lines.comp.glsl");
        calcMinMaxAllProgram_ = core::utility::make_glowl_shader(
            "histo_tex_minmax_all", shaderOptions, "infovis/histo/tex_minmax_all.comp.glsl");
        calcHistogramProgram_ =
            core::utility::make_glowl_shader("histo_tex_calc", shaderOptions, "infovis/histo/tex_calc.comp.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("TextureHistogramRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}

void TextureHistogramRenderer2D::releaseImpl() {}

bool TextureHistogramRenderer2D::handleCall(core::view::CallRender2DGL& call) {
    auto textureCall = textureDataCallerSlot_.CallAs<compositing::CallTexture2D>();
    if (textureCall == nullptr) {
        return false;
    }

    // stupid cheat, but is that really worth it?
    // static uint32_t lastFrame = -1;
    // auto currFrame = this->GetCoreInstance()->GetFrameID();
    // if (lastFrame == currFrame)
    //    return true;
    // lastFrame = currFrame;
    (*textureCall)(compositing::CallTexture2D::CallGetData);
    const auto data = textureCall->getData();

    std::size_t numComponents = 0;
    std::vector<std::string> names;
    const auto int_form = data->getFormat();

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
    auto h = data->getHeight();
    GLuint numGroups = (h > 0 ? h - 1 : h) / 256 + 1;

    // Get min and max value of texture

    // Use two buffers for min and max.
    // First numComponents values are for global values, after this numComponents values for each row.

    const GLsizeiptr bufSize = (numComponents + numComponents * h) * sizeof(float);

    GLuint minValueBuffer;
    glGenBuffers(1, &minValueBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, minValueBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, nullptr, GL_STATIC_COPY);
    const float maxVal = std::numeric_limits<float>::max();
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &maxVal);

    GLuint maxValueBuffer;
    glGenBuffers(1, &maxValueBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxValueBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, nullptr, GL_STATIC_COPY);
    const float minVal = std::numeric_limits<float>::lowest();
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &minVal);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, minValueBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, maxValueBuffer);

    calcMinMaxLinesProgram_->use();
    calcMinMaxLinesProgram_->setUniform("numComponents", static_cast<GLuint>(numComponents));

    glActiveTexture(GL_TEXTURE0);
    data->bindTexture();
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, minValueBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numComponents * sizeof(float), minimums.data());
    glDeleteBuffers(1, &minValueBuffer);

    std::vector<float> maximums(numComponents);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxValueBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numComponents * sizeof(float), maximums.data());
    glDeleteBuffers(1, &maxValueBuffer);

    setComponentHeaders(std::move(names), std::move(minimums), std::move(maximums));

    resetHistogramBuffers();

    calcHistogramProgram_->use();
    bindCommon(calcHistogramProgram_);

    glActiveTexture(GL_TEXTURE0);
    data->bindTexture();
    calcHistogramProgram_->setUniform("tex", 0);

    glDispatchCompute(numGroups, 1, 1);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    return true;
}

void TextureHistogramRenderer2D::updateSelection(SelectionMode selectionMode, int selectedComponent, int selectedBin) {
    // Selection not implemented.
}
