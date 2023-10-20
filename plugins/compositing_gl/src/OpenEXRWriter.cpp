#include "OpenEXRWriter.h"
/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "OpenEXRWriter.h"
#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

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
        , m_image_width_slot("ImageWidth", "Width of the loaded image")
        , m_image_height_slot("ImageHeight", "Height of the loaded image")
        , m_input_tex_slot("Color", "Texture to be written to file")
        , m_version(0) {
    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);

    this->m_image_width_slot << new core::param::IntParam(0);
    this->m_image_width_slot.Param<megamol::core::param::IntParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->m_image_width_slot);

    this->m_image_height_slot << new core::param::IntParam(0);
    this->m_image_height_slot.Param<megamol::core::param::IntParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->m_image_height_slot);

    this->m_input_tex_slot.SetCompatibleCall<compositing_gl::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);
}


megamol::compositing_gl::OpenEXRWriter::~OpenEXRWriter() {
    this->Release();
}

bool megamol::compositing_gl::OpenEXRWriter::create() {
    m_output_layout = glowl::TextureLayout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);
    return true;
}

void megamol::compositing_gl::OpenEXRWriter::release() {}

bool megamol::compositing_gl::OpenEXRWriter::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing_gl::CallTexture2D*>(&caller);

    if (lhs_tc == NULL)
        return false;

    if (m_filename_slot.IsDirty()) {
        RgbaInputFile file(m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str()); //(filename)
        Box2i dw = file.dataWindow();

        int width = dw.max.x - dw.min.x+1;
        int height = dw.max.y - dw.min.y + 1;
        Imf::Array2D<Rgba> pixels;
        pixels.resizeErase(height, width);

        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.readPixels(dw.min.y, dw.max.y);

        m_output_layout = glowl::TextureLayout(GL_RGBA16F, width, height, 1, GL_RGBA, GL_HALF_FLOAT, 1);
        m_output_texture->reload(m_output_layout, &pixels[0][0]);
    }

    //if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_output_texture, m_version);
    //}

    return true;
}

bool megamol::compositing_gl::OpenEXRWriter::getMetaDataCallback(core::Call& caller) {
    return true;
}

bool OpenEXRWriter::textureFormatUpdate() {

    m_output_texture = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);

    return true;
}
