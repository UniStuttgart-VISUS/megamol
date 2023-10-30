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
        , version_(0)
        , m_filename_slot("Filename", "Filename to read from")
        , m_button_slot("Screenshot Button", "Button triggering writing of input texture to file")
        , m_input_tex_slot("ColorTexture", "Texture to be written to file")
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
    if (rhs_call_input == nullptr)
        return false;
    if (!(*rhs_call_input)(CallTexture2D::CallGetData))
        return false;
    auto interm = rhs_call_input->getData();
    ++version_;
    lhs_tc->setData(rhs_call_input->getData(), version_);

    if (saveRequested) {
        try {

            int width = rhs_call_input->getData()->getWidth();
            int height = rhs_call_input->getData()->getHeight();
            std::cout << "w:" << width << " h:" << height << std::endl;
            RgbaOutputFile file(
                m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str(), width, height, WRITE_RGBA);
            std::vector<float> rawPixels(width * height * 4);
            glGetError();
            interm->bindTexture();
            glGetTextureImage(interm->getName(), 0, interm->getFormat(), GL_FLOAT, width * height * 4*4,
                &rawPixels[0]);
            printf("\n%i", glGetError());
            Array2D<Rgba> pixels(width, height);
            //rawPixels[4 * (i * width + j) + 2]
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    Rgba temp(0.25, 0.25, 1.0, 1.0);
                    pixels[j][i] = temp;
                }
            }
            file.setFrameBuffer(&pixels[0][0], 1, width);
            file.writePixels(height);
            saveRequested = false;
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
        }
    }
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
