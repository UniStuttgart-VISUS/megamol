#include "OpenEXRWriter.h"
/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "OpenEXRWriter.h"
#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include <vector>

#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfStringAttribute.h>

using namespace megamol;
using namespace megamol::compositing_gl;
using namespace Imf;
using namespace Imath;

OpenEXRWriter::OpenEXRWriter()
        : mmstd_gl::ModuleGL()
        , m_filename_slot("Filename", "Filename to read from")
        , m_button_slot("Screenshot Button", "Button triggering writing of input texture to file")
        , m_input_tex_slot("Color", "Texture to be written to file")
        , m_texture_pipe_out("Passthrough", "slot to pass texture through to calling module")
        , m_version(0) {
    this->m_filename_slot << new core::param::FilePathParam(
        "OPENEXRTESTOUTPUTFILE.exr", core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"exr"});
    this->MakeSlotAvailable(&this->m_filename_slot);

    this->m_button_slot << new core::param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::ALT);
    this->m_button_slot.SetUpdateCallback(&OpenEXRWriter::triggerButtonClicked);
    this->MakeSlotAvailable(&this->m_button_slot);

    this->m_texture_pipe_out.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetData", &OpenEXRWriter::getDataCallback);
    this->m_texture_pipe_out.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetMetaData", &OpenEXRWriter::getMetaDataCallback);
    this->MakeSlotAvailable(&m_texture_pipe_out);

    this->m_input_tex_slot.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);
}


OpenEXRWriter::~OpenEXRWriter() {
    this->Release();
}

bool OpenEXRWriter::create() {
    m_output_layout = glowl::TextureLayout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);
    return true;
}

void OpenEXRWriter::release() {}

bool OpenEXRWriter::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing_gl::CallTexture2D*>(&caller);
    auto rhs_call_input = m_input_tex_slot.CallAs<compositing_gl::CallTexture2D>();

    if (rhs_call_input == NULL)
        return false;
    if (!(*rhs_call_input)(0))
        return false;

    if (saveRequested) {
        m_input_tex_slot.CallAs<compositing_gl::CallTexture2D>();
        int width = m_input_tex_slot.CallAs<compositing_gl::CallTexture2D>()->getData()->getWidth();
        int height = m_input_tex_slot.CallAs<compositing_gl::CallTexture2D>()->getData()->getHeight();
        RgbaOutputFile file(
            m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str(), width, height, WRITE_RGBA);
        std::vector<float> rawPixels(width * height * 4);
        glGetTextureImage(m_input_tex_slot.CallAs<compositing_gl::CallTexture2D>()->getData()->getName(), 0,
            m_input_tex_slot.CallAs<compositing_gl::CallTexture2D>()->getData()->getFormat(), GL_FLOAT, 1,
            &rawPixels[0]);
        Array2D<Rgba> pixels(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Rgba temp(rawPixels[4 * (i * width + j)], rawPixels[4 * (i * width + j) + 1],
                    rawPixels[4 * (i * width + j) + 2], rawPixels[4 * (i * width + j) + 3]);
                pixels[i][j] = temp;
            }
        }
        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.writePixels(height);
        saveRequested = false;
    }

    lhs_tc->setData(rhs_call_input->getData(), rhs_call_input->version());
    return true;
}

bool OpenEXRWriter::triggerButtonClicked(core::param::ParamSlot& button) {
    saveRequested = true;
    return true;
}

bool OpenEXRWriter::getMetaDataCallback(core::Call& caller) {
    return true;
}

bool OpenEXRWriter::textureFormatUpdate() {

    m_output_texture = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);

    return true;
}
