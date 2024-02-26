/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "OpenEXRReader.h"

#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfChannelListAttribute.h>
#include <OpenEXR/ImfGenericInputFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfStringAttribute.h>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/IntParam.h"

using namespace megamol;
using namespace megamol::compositing_gl;
using namespace Imf;
using namespace Imath;

OpenEXRReader::OpenEXRReader()
        : mmstd_gl::ModuleGL()
        , m_filename_slot("Filename", "Filename to read from")
        , m_image_width_slot("ImageWidth", "Width of the loaded image")
        , m_image_height_slot("ImageHeight", "Height of the loaded image")
        , type_mapping_slot("Type Mapping", "Chooses output format of GPU texture.")
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

    this->type_mapping_slot << new core::param::FlexEnumParam("UINT");
    this->type_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("HALF");
    this->type_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("FLOAT");
    this->type_mapping_slot.Param<core::param::FlexEnumParam>()->SetValue("HALF");
    this->MakeSlotAvailable(&this->type_mapping_slot);

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
    if (m_filename_slot.Param<core::param::FilePathParam>()->Value() == "")
        return false;
    /**
     *  When input file changes EnumParams need to be updated.
     *  Updates in actual mapping and output texture are handled later.
     */
    if (m_filename_slot.IsDirty() || type_mapping_slot.IsDirty()) {
        m_filename_slot.ResetDirty();

        red_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();
        green_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->ClearValues();

        red_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        green_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();

        red_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("-");
        red_mapping_slot.Param<core::param::FlexEnumParam>()->SetValue("-");
        green_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("-");
        green_mapping_slot.Param<core::param::FlexEnumParam>()->SetValue("-");
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("-");
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->SetValue("-");
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("-");
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->SetValue("-");

        red_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("0");
        green_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("0");
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("0");
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("0");

        red_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("1");
        green_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("1");
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("1");
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue("1");

        std::vector<std::string> nochannel{"-"};
        // 0 = uint, 1=half, 2=float
        channelNamesByType.push_back(nochannel);
        channelNamesByType.push_back(nochannel);
        channelNamesByType.push_back(nochannel);

        try {
            InputFile file(m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str()); //(filename)
            const ChannelList& channels = file.header().channels();
            for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
                channelNamesByType[i.channel().type].push_back(i.name());
                std::cout << i.name() << std::endl;
            }
        } catch (std::exception const& ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("OpenEXR Reader Exception: %s", ex.what());
            return false;
        }

        int typeIdx = typeStringToIndex(type_mapping_slot.Param<core::param::FlexEnumParam>()->Value());
        for (int i = 0; i < channelNamesByType[typeIdx].size(); i++) {
            red_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
            green_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
            blue_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
            alpha_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
        }
        type_mapping_slot.ResetDirty();
        red_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        green_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
    }

    /**
     * When type mapping changes, all input channel need to be selected again.
     */
    if (type_mapping_slot.IsDirty()) {
        type_mapping_slot.ResetDirty();
        int typeIdx = typeStringToIndex(type_mapping_slot.Param<core::param::FlexEnumParam>()->Value());

        for (int i = 0; i < channelNamesByType[typeIdx].size(); i++) {
            red_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
            green_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
            blue_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
            alpha_mapping_slot.Param<core::param::FlexEnumParam>()->AddValue(channelNamesByType[typeIdx][i]);
        }
        //also sets slots dirty
        red_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        green_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->setDirty();
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

        setRelevantParamState();


        m_output_texture = readToTex2D<float>();
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

int OpenEXRReader::typeStringToIndex(const std::string str) {
    if (str == "UINT") {
        return 0;
    } else if (str == "HALF") {
        return 1;
    } else if (str == "FLOAT") {
        return 2;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "OpenEXR Reader: Custom channel data types not supported");
        return 3;
    }
}

template<typename T>
std::shared_ptr<glowl::Texture2D> OpenEXRReader::readToTex2D() {
    InputFile file(m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str()); //(filename)
    // TODO : double check if correct window is used
    Box2i dw = file.header().dataWindow();
    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    //TODO correct struct depending on channel names/types
    Imf::Array2D<T> rPixels;
    Imf::Array2D<T> gPixels;
    Imf::Array2D<T> bPixels;
    Imf::Array2D<T> aPixels;

    PixelType sliceType;
    auto outTexType = GL_FLOAT;
    if (typeid(T) == typeid(float)) {
        sliceType = PixelType::FLOAT;
        outTexType = GL_FLOAT;
    } else if (typeid(T) == typeid(half)) {
        sliceType = PixelType::HALF;
        outTexType = GL_HALF_FLOAT;
    } else if (typeid(T) == typeid(unsigned int)) {
        sliceType = PixelType::UINT;
        outTexType = GL_UNSIGNED_INT;
    } else {
        sliceType = PixelType::NUM_PIXELTYPES;
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "OpenEXR Reader Exception: Custom data types not supported.");
        return nullptr;
    }
    FrameBuffer fb;
    int numOutChannels = 0;
    //RED
    std::string currentChannelName = red_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
    if (currentChannelName != "-") {
        numOutChannels++;
        rPixels.resizeErase(height, width);
        //file.header().channels().find(currentChannelName).channel().type;
        if (currentChannelName == "0" || currentChannelName == "1") {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    rPixels[y][x] = static_cast<T>(stoi(currentChannelName));
        } else {
            fb.insert(currentChannelName, Slice(sliceType, (char*) &rPixels[-dw.min.y][-dw.min.x],
                                              sizeof(rPixels[0][0]) * 1, sizeof(rPixels[0][0]) * width));
        }
    }

    //GREEN
    currentChannelName = green_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
    if (currentChannelName != "-") {
        numOutChannels++;
        gPixels.resizeErase(height, width);
        if (currentChannelName == "0" || currentChannelName == "1") {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    gPixels[y][x] = static_cast<T>(stoi(currentChannelName));
        } else {
            fb.insert(currentChannelName, Slice(sliceType, (char*) &gPixels[-dw.min.y][-dw.min.x],
                                              sizeof(gPixels[0][0]) * 1, sizeof(gPixels[0][0]) * width));
        }
    }

    //BLUE
    currentChannelName = blue_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
    if (currentChannelName != "-") {
        numOutChannels++;
        bPixels.resizeErase(height, width);
        if (currentChannelName == "0" || currentChannelName == "1") {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    bPixels[y][x] = static_cast<T>(stoi(currentChannelName));
        } else {
            fb.insert(currentChannelName, Slice(sliceType, (char*) &bPixels[-dw.min.y][-dw.min.x],
                                              sizeof(bPixels[0][0]) * 1, sizeof(bPixels[0][0]) * width));
        }
    }
    //ALPHA
    currentChannelName = alpha_mapping_slot.Param<core::param::FlexEnumParam>()->Value();
    if (currentChannelName != "-") {
        numOutChannels++;
        aPixels.resizeErase(height, width);
        if (currentChannelName == "0" || currentChannelName == "1") {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    aPixels[y][x] = static_cast<T>(stoi(currentChannelName));
        } else {
            fb.insert(currentChannelName, Slice(sliceType, (char*) &aPixels[-dw.min.y][-dw.min.x],
                                              sizeof(aPixels[0][0]) * 1, sizeof(aPixels[0][0]) * width));
        }
    }
    auto outTexFormat = GL_RGBA;

    switch (numOutChannels) {
    case 1:
        outTexFormat = GL_RED;
        break;
    case 2:
        outTexFormat = GL_RG;
        break;
    case 3:
        outTexFormat = GL_RGB;
        break;
    case 4:
        outTexFormat = GL_RGBA;
        break;
    case 0:
        megamol::core::utility::log::Log::DefaultLog.WriteError("OpenEXR Reader Exception: No channels defined.");
        return nullptr;
        break;
    }

    try {
        file.setFrameBuffer(fb);
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("OpenEXR Reader Exception: %s", ex.what());
    }
    try {
        file.readPixels(dw.min.y, dw.max.y);
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("OpenEXR Reader Exception: %s", ex.what());
    }
    //TODO out tex format and type
    m_output_layout = glowl::TextureLayout(
        internalFromTypeFormat(outTexFormat, outTexType), width, height, 1, outTexFormat, outTexType, 1);
    std::vector<T> flippedPixels; // flipped Y
    flippedPixels.resize(height * width * numOutChannels);

    //TODO easier way?
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //std::cout << (height - y-1) * width + x << std::endl;
            //std::cout << y << " " << x << std::endl;
            switch (numOutChannels) {
            case 4:
                flippedPixels[numOutChannels * ((height - y - 1) * width + x) + 3] = aPixels[y][x];
            case 3:
                flippedPixels[numOutChannels * ((height - y - 1) * width + x) + 2] = bPixels[y][x];
            case 2:
                flippedPixels[numOutChannels * ((height - y - 1) * width + x) + 1] = gPixels[y][x];
            case 1:
                flippedPixels[numOutChannels * ((height - y - 1) * width + x)] = rPixels[y][x];
                break;
            }
        }
    }
    m_output_texture = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);
    m_output_texture->reload(m_output_layout, &flippedPixels[0]);
    // TODO: void return type?
    return m_output_texture;
}

GLenum OpenEXRReader::internalFromTypeFormat(GLenum format, GLenum type) {
    if (type == GL_FLOAT) {
        if (format == GL_RGBA) {
            return GL_RGBA32F;
        } else if (format == GL_RGB) {
            return GL_RGB32F;
        } else if (format == GL_RG) {
            return GL_RG32F;
        } else if (format == GL_RED) {
            return GL_R32F;
        }
    } else if (type == GL_HALF_FLOAT) {
        if (format == GL_RGBA) {
            return GL_RGBA16F;
        } else if (format == GL_RGB) {
            return GL_RGB16F;
        } else if (format == GL_RG) {
            return GL_RG16F;
        } else if (format == GL_RED) {
            return GL_R16F;
        }
    } else if (type == GL_INT) {
        if (format == GL_RGBA) {
            return GL_RGBA32UI;
        } else if (format == GL_RGB) {
            return GL_RGB32UI;
        } else if (format == GL_RG) {
            return GL_RG32UI;
        } else if (format == GL_RED) {
            return GL_R32UI;
        }
    }
}

void OpenEXRReader::setRelevantParamState() {
    if (alpha_mapping_slot.Param<core::param::FlexEnumParam>()->Value() != "-") {
        red_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        green_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
    } else if (blue_mapping_slot.Param<core::param::FlexEnumParam>()->Value() != "-") {
        red_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        green_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
    } else if (green_mapping_slot.Param<core::param::FlexEnumParam>()->Value() != "-") {
        red_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        green_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
    } else if (red_mapping_slot.Param<core::param::FlexEnumParam>()->Value() != "-") {
        red_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
        green_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
    } else {
        red_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(false);
        green_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        blue_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
        alpha_mapping_slot.Param<core::param::FlexEnumParam>()->SetGUIReadOnly(true);
    }
}
