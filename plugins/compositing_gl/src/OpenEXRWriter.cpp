/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "OpenEXRWriter.h"

#include <vector>
#include "limits.h"

#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfImageChannel.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfStringAttribute.h>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

using namespace megamol;
using namespace megamol::compositing_gl;
using namespace Imf;
using namespace Imath;

OpenEXRWriter::OpenEXRWriter()
        : mmstd_gl::ModuleGL()
        , version_(0)
        , m_filename_slot("Filename", "Filename to read from")
        , m_bitmode_slot("8bit integer as color",
              "8 bit integer formats will be treated as colors and converted to half float color format before saving.")
        , m_channel_name_red("First Channel Name", "Name of first Channel in exr file. The red component of the input "
                                                   "texture will be written to this channel.")
        , m_channel_name_green("Second Channel Name", "Name of second Channel in exr file. The green component of the "
                                                      "input texture will be written to this channel.")
        , m_channel_name_blue("Third Channel Name", "Name of third Channel in exr file. The blue component of the "
                                                    "input texture will be written to this channel.")
        , m_channel_name_alpha("Forth Channel Name", "Name of forth Channel in exr file. The alpha component of the "
                                                     "input texture will be written to this channel.")
        , m_button_slot("Screenshot Button", "Button triggering writing of input texture to file")
        , m_input_tex_slot("ColorTexture", "Texture to be written to file")
        , m_texture_pipe_out("Passthrough", "slot to pass texture through to calling module")
        , m_version(0) {
    this->m_filename_slot << new core::param::FilePathParam(
        "OPENEXRTESTOUTPUTFILE.exr", core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"exr"});
    this->MakeSlotAvailable(&this->m_filename_slot);
    this->m_bitmode_slot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->m_bitmode_slot);

    this->m_button_slot << new core::param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::ALT);
    this->m_button_slot.SetUpdateCallback(&OpenEXRWriter::triggerButtonClicked);
    this->MakeSlotAvailable(&this->m_button_slot);

    this->m_texture_pipe_out.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetData", &OpenEXRWriter::getDataCallback);
    this->m_texture_pipe_out.SetCallback(
        compositing_gl::CallTexture2D::ClassName(), "GetMetaData", &OpenEXRWriter::getMetaDataCallback);
    this->MakeSlotAvailable(&m_texture_pipe_out);

    this->m_channel_name_red << new core::param::StringParam("R");
    this->m_channel_name_green << new core::param::StringParam("G");
    this->m_channel_name_blue << new core::param::StringParam("B");
    this->m_channel_name_alpha << new core::param::StringParam("A");

    this->MakeSlotAvailable(&this->m_channel_name_red);
    this->MakeSlotAvailable(&this->m_channel_name_green);
    this->MakeSlotAvailable(&this->m_channel_name_blue);
    this->MakeSlotAvailable(&this->m_channel_name_alpha);

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
    if (interm == nullptr) {
        //errorstate
        auto output_layout = glowl::TextureLayout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1);
        interm = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Texture layout from right hand side module is empty. Please check texture layout. Default texture layout "
            "assumed.");
    } else if (!interm->getInternalFormat()) {
        //errorstate
        auto output_layout = glowl::TextureLayout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1);
        interm = std::make_shared<glowl::Texture2D>("exr_tx2D", m_output_layout, nullptr);
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Texture layout from right hand side module contains empty entry. Please check texture layout. Default "
            "texture layout assumed.");
    }
    ++version_;
    lhs_tc->setData(rhs_call_input->getData(), version_);

    setRelevantParamState();

    if (saveRequested) {
        saveRequested = false;
        try {
            int width = rhs_call_input->getData()->getWidth();
            int height = rhs_call_input->getData()->getHeight();
            int inputTextureChNum = formatToChannelNumber(interm->getFormat());

            megamol::core::utility::log::Log::DefaultLog.WriteInfo("%x, %x, %x, %s \n", interm->getFormat(),
                interm->getType(), interm->getInternalFormat(), interm->getType() == GL_FLOAT ? "true" : "false");

            Header header(width, height);
            FrameBuffer fb;
            PixelType headerType = PixelType::FLOAT;

            m_channel_name_red.Param<core::param::StringParam>()->SetGUIVisible(false);
            m_channel_name_green.Param<core::param::StringParam>()->SetGUIVisible(false);
            m_channel_name_blue.Param<core::param::StringParam>()->SetGUIVisible(false);
            m_channel_name_alpha.Param<core::param::StringParam>()->SetGUIVisible(false);

            switch (inputTextureChNum) {
            case 4:
                m_channel_name_alpha.Param<core::param::StringParam>()->SetGUIVisible(true);
            case 3:
                m_channel_name_blue.Param<core::param::StringParam>()->SetGUIVisible(true);
            case 2:
                m_channel_name_green.Param<core::param::StringParam>()->SetGUIVisible(true);
            case 1:
                m_channel_name_red.Param<core::param::StringParam>()->SetGUIVisible(true);
                break;
            default:
                // errorstate
                break;
            }

            auto copyTextureData = [&]<typename PixelFormat>() {
                std::vector<PixelFormat> rawPixels(width * height * inputTextureChNum);
                std::vector<PixelFormat> rPixels(0);
                std::vector<PixelFormat> gPixels(0);
                std::vector<PixelFormat> bPixels(0);
                std::vector<PixelFormat> aPixels(0);
                //only Init used channels
                switch (inputTextureChNum) {
                case 4:
                    aPixels.resize(width * height);
                case 3:
                    bPixels.resize(width * height);
                case 2:
                    gPixels.resize(width * height);
                case 1:
                    rPixels.resize(width * height);
                    break;
                }

                interm->bindTexture();
                glGetTextureImage(interm->getName(), 0, interm->getFormat(), interm->getType(),
                    width * height * inputTextureChNum * sizeof(PixelFormat), rawPixels.data());
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        // the number of channels in the texture determines the stride in raw pixels and how many single-channel vectors can be filed with pixel data.
                        switch (inputTextureChNum) {
                        case 4:
                            aPixels[x + y * width] = static_cast<PixelFormat>(
                                rawPixels[(x + (height - y - 1) * width) * inputTextureChNum + 3]);
                        case 3:
                            bPixels[x + y * width] = static_cast<PixelFormat>(
                                rawPixels[(x + (height - y - 1) * width) * inputTextureChNum + 2]);
                        case 2:
                            gPixels[x + y * width] = static_cast<PixelFormat>(
                                rawPixels[(x + (height - y - 1) * width) * inputTextureChNum + 1]);
                        case 1:
                            rPixels[x + y * width] =
                                static_cast<PixelFormat>(rawPixels[(x + (height - y - 1) * width) * inputTextureChNum]);
                            break;
                        }
                    }
                }

                if (m_channel_name_red.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum >= 1) {
                    header.channels().insert(
                        m_channel_name_red.Param<core::param::StringParam>()->Value(), Channel(headerType));
                    fb.insert(m_channel_name_red.Param<core::param::StringParam>()->Value(),
                        Slice(headerType, (char*) &rPixels[0], sizeof(rPixels[0]), sizeof(rPixels[0]) * width));
                }
                if (m_channel_name_green.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum >= 2) {
                    header.channels().insert(
                        m_channel_name_green.Param<core::param::StringParam>()->Value(), Channel(headerType));
                    fb.insert(m_channel_name_green.Param<core::param::StringParam>()->Value(),
                        Slice(headerType, (char*) &gPixels[0], sizeof(gPixels[0]) * 1, sizeof(gPixels[0]) * width));
                }
                if (m_channel_name_blue.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum >= 3) {
                    header.channels().insert(
                        m_channel_name_blue.Param<core::param::StringParam>()->Value(), Channel(headerType));
                    fb.insert(m_channel_name_blue.Param<core::param::StringParam>()->Value(),
                        Slice(headerType, (char*) &bPixels[0], sizeof(bPixels[0]) * 1, sizeof(bPixels[0]) * width));
                }
                if (m_channel_name_alpha.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum == 4) {
                    header.channels().insert(
                        m_channel_name_alpha.Param<core::param::StringParam>()->Value(), Channel(headerType));
                    fb.insert(m_channel_name_alpha.Param<core::param::StringParam>()->Value(),
                        Slice(headerType, (char*) &aPixels[0], sizeof(aPixels[0]) * 1, sizeof(aPixels[0]) * width));
                }

                OutputFile file(m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str(), header);

                file.setFrameBuffer(fb);
                file.writePixels(height);
            };

            //TODO remove header type from outside template
            if (interm->getType() == GL_FLOAT) {
                headerType = PixelType::FLOAT;
                copyTextureData.template operator()<float>();
            } else if (interm->getType() == GL_HALF_FLOAT) { //TODO: Snorm conversion?
                headerType = PixelType::HALF;
                copyTextureData.template operator()<half>();
            } else if (interm->getType() == GL_INT ||
                       interm->getType() == GL_UNSIGNED_INT) { //TODO: unsigned int not really supported
                if (m_bitmode_slot.Param<core::param::BoolParam>()->Value() &&
                    checkNeedsConversion(interm->getInternalFormat())) {
                    copyAndConvertTextureData(interm, header, fb);
                } else {
                    headerType = PixelType::UINT;
                    copyTextureData.template operator()<int>();
                }
            }

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

int OpenEXRWriter::formatToChannelNumber(GLenum format) {
    switch (format) {
    case GL_RED:
    case GL_STENCIL_INDEX:
    case GL_DEPTH_COMPONENT:
    case GL_DEPTH_COMPONENT16:
    case GL_DEPTH_COMPONENT24:
    case GL_DEPTH_COMPONENT32:
    case GL_DEPTH_COMPONENT32F:
        return 1;
        break;
    case GL_RG:
    case GL_DEPTH_STENCIL:
    case GL_DEPTH24_STENCIL8:
    case GL_DEPTH32F_STENCIL8:
        return 2;
        break;
    case GL_RGB:
        return 3;
        break;
    case GL_RGBA:
        return 4;
        break;
    }
    return 0;
}

void OpenEXRWriter::setRelevantParamState() {
    if (m_channel_name_red.Param<core::param::StringParam>()->Value() == "") {
        m_channel_name_red.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_green.Param<core::param::StringParam>()->SetGUIReadOnly(true);
        m_channel_name_blue.Param<core::param::StringParam>()->SetGUIReadOnly(true);
        m_channel_name_alpha.Param<core::param::StringParam>()->SetGUIReadOnly(true);
        m_channel_name_green.Param<core::param::StringParam>()->SetValue("");
        m_channel_name_blue.Param<core::param::StringParam>()->SetValue("");
        m_channel_name_alpha.Param<core::param::StringParam>()->SetValue("");
    } else if (m_channel_name_green.Param<core::param::StringParam>()->Value() == "") {
        m_channel_name_red.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_green.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_blue.Param<core::param::StringParam>()->SetGUIReadOnly(true);
        m_channel_name_alpha.Param<core::param::StringParam>()->SetGUIReadOnly(true);
        m_channel_name_blue.Param<core::param::StringParam>()->SetValue("");
        m_channel_name_alpha.Param<core::param::StringParam>()->SetValue("");
    } else if (m_channel_name_blue.Param<core::param::StringParam>()->Value() == "") {
        m_channel_name_red.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_green.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_blue.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_alpha.Param<core::param::StringParam>()->SetGUIReadOnly(true);
        m_channel_name_alpha.Param<core::param::StringParam>()->SetValue("");
    } else {
        m_channel_name_red.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_green.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_blue.Param<core::param::StringParam>()->SetGUIReadOnly(false);
        m_channel_name_alpha.Param<core::param::StringParam>()->SetGUIReadOnly(false);
    }
}

int megamol::compositing_gl::OpenEXRWriter::checkNeedsConversion(GLenum internalFormat) {
    //TODO: more color formats?
    switch (internalFormat) {
    case GL_RGBA8:
    case GL_RGB8:
    case GL_RG8:
    case GL_R8:
        return 1;
        break;
    case GL_RGBA8_SNORM:
    case GL_RGB8_SNORM:
    case GL_RG8_SNORM:
    case GL_R8_SNORM:
        return 2;
        break;
    default:
        return 0;
    }
    return 0;
}

void megamol::compositing_gl::OpenEXRWriter::copyAndConvertTextureData(
    std::shared_ptr<glowl::Texture2D> interm, Header header, FrameBuffer fb) {
    //TODO: this is redundant and should be removed somehow
    int inputTextureChNum = formatToChannelNumber(interm->getFormat());
    int width = interm->getWidth();
    int height = interm->getHeight();
    Imf_3_1::PixelType headerType = Imf_3_1::HALF;
    std::vector<half> rawPixels(width * height * inputTextureChNum);
    std::vector<half> rPixels(0);
    std::vector<half> gPixels(0);
    std::vector<half> bPixels(0);
    std::vector<half> aPixels(0);
    //only Init used channels
    switch (inputTextureChNum) {
    case 4:
        aPixels.resize(width * height);
    case 3:
        bPixels.resize(width * height);
    case 2:
        gPixels.resize(width * height);
    case 1:
        rPixels.resize(width * height);
        break;
    }

    interm->bindTexture();
    glGetTextureImage(interm->getName(), 0, interm->getFormat(), GL_HALF_FLOAT,
        width * height * inputTextureChNum * sizeof(half), rawPixels.data());
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // the number of channels in the texture determines the stride in raw pixels and how many single-channel vectors can be filed with pixel data.
            switch (inputTextureChNum) {
            case 4:
                aPixels[x + y * width] =
                    static_cast<half>(rawPixels[(x + (height - y - 1) * width) * inputTextureChNum + 3]);
            case 3:
                bPixels[x + y * width] =
                    static_cast<half>(rawPixels[(x + (height - y - 1) * width) * inputTextureChNum + 2]);
            case 2:
                gPixels[x + y * width] =
                    static_cast<half>(rawPixels[(x + (height - y - 1) * width) * inputTextureChNum + 1]);
            case 1:
                rPixels[x + y * width] =
                    static_cast<half>(rawPixels[(x + (height - y - 1) * width) * inputTextureChNum]);
                break;
            }
        }
    }

    if (m_channel_name_red.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum >= 1) {
        header.channels().insert(m_channel_name_red.Param<core::param::StringParam>()->Value(), Channel(headerType));
        fb.insert(m_channel_name_red.Param<core::param::StringParam>()->Value(),
            Slice(headerType, (char*) &rPixels[0], sizeof(rPixels[0]), sizeof(rPixels[0]) * width));
    }
    if (m_channel_name_green.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum >= 2) {
        header.channels().insert(m_channel_name_green.Param<core::param::StringParam>()->Value(), Channel(headerType));
        fb.insert(m_channel_name_green.Param<core::param::StringParam>()->Value(),
            Slice(headerType, (char*) &gPixels[0], sizeof(gPixels[0]) * 1, sizeof(gPixels[0]) * width));
    }
    if (m_channel_name_blue.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum >= 3) {
        header.channels().insert(m_channel_name_blue.Param<core::param::StringParam>()->Value(), Channel(headerType));
        fb.insert(m_channel_name_blue.Param<core::param::StringParam>()->Value(),
            Slice(headerType, (char*) &bPixels[0], sizeof(bPixels[0]) * 1, sizeof(bPixels[0]) * width));
    }
    if (m_channel_name_alpha.Param<core::param::StringParam>()->Value() != "" && inputTextureChNum == 4) {
        header.channels().insert(m_channel_name_alpha.Param<core::param::StringParam>()->Value(), Channel(headerType));
        fb.insert(m_channel_name_alpha.Param<core::param::StringParam>()->Value(),
            Slice(headerType, (char*) &aPixels[0], sizeof(aPixels[0]) * 1, sizeof(aPixels[0]) * width));
    }

    OutputFile file(m_filename_slot.Param<core::param::FilePathParam>()->ValueString().c_str(), header);

    file.setFrameBuffer(fb);
    file.writePixels(height);
}
