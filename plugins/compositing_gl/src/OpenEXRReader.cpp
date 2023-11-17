#include "OpenEXRReader.h"
/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "OpenEXRReader.h"
#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/IntParam.h"

#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfChannelListAttribute.h>
#include <OpenEXR/ImfGenericInputFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfStringAttribute.h>


using namespace megamol;
using namespace megamol::compositing_gl;
using namespace Imf;
using namespace Imath;

OpenEXRReader::OpenEXRReader()
        : mmstd_gl::ModuleGL()
        , m_filename_slot("Filename", "Filename to read from")
        , m_image_width_slot("ImageWidth", "Width of the loaded image")
        , m_image_height_slot("ImageHeight", "Height of the loaded image")
        , red_mapping_slot("Red Out Channel", "Channel of input file mapped to red output channel.")
        , green_mapping_slot("Green Out Channel", "Channel of input file mapped to green output channel.")
        , blue_mapping_slot("Blue Out Channel", "Channel of input file mapped to blue output channel.")
        , alpha_mapping_slot("Alpha Out Channel", "Channel of input file mapped to alpha output channel.")
        , m_output_tex_slot("Color", "Slot providing the data as Texture2D (RGBA16F)")
        , m_version(0) {
    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);

    this->m_image_width_slot << new core::param::IntParam(0);
    this->m_image_width_slot.Param<megamol::core::param::IntParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->m_image_width_slot);

    this->m_image_height_slot << new core::param::IntParam(0);
    this->m_image_height_slot.Param<megamol::core::param::IntParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->m_image_height_slot);

    this->red_mapping_slot << new core::param::FlexEnumParam("-");
    this->green_mapping_slot << new core::param::FlexEnumParam("-");
    this->blue_mapping_slot << new core::param::FlexEnumParam("-");
    this->alpha_mapping_slot << new core::param::FlexEnumParam("-");
    this->MakeSlotAvailable(&this->red_mapping_slot);
    this->MakeSlotAvailable(&this->green_mapping_slot);
    this->MakeSlotAvailable(&this->blue_mapping_slot);
    this->MakeSlotAvailable(&this->alpha_mapping_slot);

    this->m_output_tex_slot.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetData", &OpenEXRReader::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetMetaData", &OpenEXRReader::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);
}


megamol::compositing_gl::OpenEXRReader::~OpenEXRReader() {
    this->Release();
}

bool megamol::compositing_gl::OpenEXRReader::create() {
    m_output_layout = glowl::TextureLayout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);
    return true;
}

void megamol::compositing_gl::OpenEXRReader::release() {}

bool megamol::compositing_gl::OpenEXRReader::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing_gl::CallTexture2D*>(&caller);

    if (lhs_tc == NULL)
        return false;

    /**
     *  When input file changes EnumParams need to be updated.
     *  Updates in actual mapping and output texture are handled later.
     */
    if (m_filename_slot.IsDirty()) {
        m_filename_slot.ResetDirty();
        InputFile file(m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str()); //(filename)

        red_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();
        green_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();

        try {
            const ChannelList& channels = file.header().channels();
            for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
                const Channel& c = i.channel();
                red_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(i.name());
                green_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(i.name());
                blue_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(i.name());
                alpha_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(i.name());
            }
        } catch (std::exception const& ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("OpenEXR Reader Exception: %s", ex.what());
        }
        /*
        red_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        green_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        */
    }

    /***
     * When mapping slots change, file needs to be reread and output texture needs to be updated.
     */
    if (red_mapping_slot.IsDirty() || green_mapping_slot.IsDirty() || blue_mapping_slot.IsDirty() ||
        alpha_mapping_slot.IsDirty()) {

        red_mapping_slot.ResetDirty();
        green_mapping_slot.ResetDirty();
        blue_mapping_slot.ResetDirty();
        alpha_mapping_slot.ResetDirty();

        InputFile file(m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str()); //(filename)
        // TODO : double check if correct window is used
        Box2i dw = file.header().dataWindow();
        int width = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;

        //TODO correct struct depending on channel names/types
        Imf::Array2D<Rgba> pixels;
        pixels.resizeErase(height, width);
        FrameBuffer fb;

        //RED
        std::string currentChannelName = red_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
        if (currentChannelName != "-") {

            //file.header().channels().find(currentChannelName).channel().type;

            fb.insert(currentChannelName, Slice(HALF, (char*)&pixels[-dw.min.y][-dw.min.x].r, sizeof(pixels[0][0]) * 1,
                                              sizeof(pixels[0][0]) * width));
        }

        //GREEN
        currentChannelName = green_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
        if (currentChannelName != "-")
            fb.insert(currentChannelName, Slice(HALF, (char*)&pixels[-dw.min.y][-dw.min.x].g, sizeof(pixels[0][0]) * 1,
                                              sizeof(pixels[0][0]) * width));

        //BLUE
        currentChannelName = blue_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
        if (currentChannelName != "-")
            fb.insert(currentChannelName, Slice(HALF, (char*)&pixels[-dw.min.y][-dw.min.x].b, sizeof(pixels[0][0]) * 1,
                                              sizeof(pixels[0][0]) * width));

        //ALPHA
        currentChannelName = alpha_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
        if (currentChannelName != "-")
            fb.insert(currentChannelName, Slice(HALF, (char*)&pixels[-dw.min.y][-dw.min.x].a, sizeof(pixels[0][0]) * 1,
                                              sizeof(pixels[0][0]) * width));

        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);

        m_output_layout = glowl::TextureLayout(GL_RGBA16F, width, height, 1, GL_RGBA, GL_HALF_FLOAT, 1);
        Imf::Array2D<Rgba> flippedPixels; // flipped Y
        flippedPixels.resizeErase(height, width);

        //TODO easier way?
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                flippedPixels[y][x] = pixels[height - 1 - y][x];
            }
        }

        m_output_texture->reload(m_output_layout, &flippedPixels[0][0]);
    }

    //if (lhs_tc->version() < m_version) {
    lhs_tc->setData(m_output_texture, m_version);
    //}

    return true;
}

bool megamol::compositing_gl::OpenEXRReader::getMetaDataCallback(core::Call& caller) {
    return true;
}

bool OpenEXRReader::textureFormatUpdate() {

    m_output_texture = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);

    return true;
}
