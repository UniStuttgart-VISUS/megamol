/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "PNGDataSource.h"

#include <png.h>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

using namespace megamol;
using namespace megamol::compositing_gl;

PNGDataSource::PNGDataSource()
        : mmstd_gl::ModuleGL()
        , m_filename_slot("Filename", "Filename to read from")
        , m_image_width_slot("ImageWidth", "Width of the loaded image")
        , m_image_height_slot("ImageHeight", "Height of the loaded image")
        , m_output_tex_slot("Color", "Slot providing the data as Texture2D (RGBA16F)")
        , m_version(0)
        , out_format_handler_("OUTFORMAT", {GL_RGBA8_SNORM, GL_RGBA16F, GL_RGBA32F},
              std::function<bool()>(std::bind(&PNGDataSource::textureFormatUpdate, this))) {
    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);

    this->m_image_width_slot << new core::param::IntParam(0);
    this->m_image_width_slot.Param<megamol::core::param::IntParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->m_image_width_slot);

    this->m_image_height_slot << new core::param::IntParam(0);
    this->m_image_height_slot.Param<megamol::core::param::IntParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->m_image_height_slot);

    this->m_output_tex_slot.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetData", &PNGDataSource::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetMetaData", &PNGDataSource::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);
}

PNGDataSource::~PNGDataSource() {
    this->Release();
}

bool PNGDataSource::create() {
    return textureFormatUpdate();
}

void PNGDataSource::release() {}

bool PNGDataSource::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing_gl::CallTexture2D*>(&caller);

    if (lhs_tc == NULL)
        return false;

    if (m_filename_slot.IsDirty()) {

        png_image image; /* The control structure used by libpng */

        /* Initialize the 'png_image' structure. */
        memset(&image, 0, (sizeof image));
        image.version = PNG_IMAGE_VERSION;

        std::filesystem::path file_path = m_filename_slot.Param<core::param::FilePathParam>()->Value();

        /* The first argument is the file to read: */
        if (png_image_begin_read_from_file(&image, file_path.string().c_str()) != 0) {
            size_t buffer_size = 4ULL * image.width * image.height;
            std::vector<unsigned char> image_buffer(buffer_size);

            m_image_width_slot.Param<core::param::IntParam>()->SetValue(image.width, false);
            m_image_height_slot.Param<core::param::IntParam>()->SetValue(image.height, false);

            //image.format = PNG_FORMAT_LINEAR_RGB_ALPHA;
            image.format = PNG_FORMAT_RGBA;


            if (buffer_size != NULL && png_image_finish_read(&image, NULL /*background*/, image_buffer.data(),
                                           0 /*row_stride*/, NULL /*colormap*/) != 0) {
                std::vector<unsigned char> tx2D_buffer(buffer_size);

                // need to flip image around horizontal axis
                for (size_t y = 0; y < image.height; ++y) {
                    for (size_t x = 0; x < image.width; ++x) {
                        size_t id = x + y * image.width;

                        size_t flip_id = x + (image.height - 1 - y) * image.width;

                        tx2D_buffer[4 * id + 0] = image_buffer[4 * flip_id + 0]; // R
                        tx2D_buffer[4 * id + 1] = image_buffer[4 * flip_id + 1]; // G
                        tx2D_buffer[4 * id + 2] = image_buffer[4 * flip_id + 2]; // B
                        tx2D_buffer[4 * id + 3] = image_buffer[4 * flip_id + 3]; // A
                    }
                }

                m_output_layout.width = image.width;
                m_output_layout.height = image.height;
                m_output_texture->reload(m_output_layout, tx2D_buffer.data());

                ++m_version;
                m_filename_slot.ResetDirty();
            }
        } else {
            png_image_free(&image);
            core::utility::log::Log::DefaultLog.WriteError("Failed to read .png image in PNGDataSource");
            return false;
        }

        png_image_free(&image);
    }

    if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_output_texture, m_version);
    }

    return true;
}

bool PNGDataSource::getMetaDataCallback(core::Call& caller) {
    return true;
}

bool PNGDataSource::textureFormatUpdate() {
    m_output_layout = glowl::TextureLayout(out_format_handler_.getInternalFormat(), 1, 1, 1,
        out_format_handler_.getFormat(), out_format_handler_.getType(), 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("png_tx2D", m_output_layout, nullptr);

    return true;
}
