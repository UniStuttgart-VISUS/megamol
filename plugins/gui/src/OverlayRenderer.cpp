/*
 * OverlayRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OverlayRenderer.h"


using namespace megamol::core;
using namespace megamol::gui;


/*
drawScreenSpaceBillboard(vec2 rel_pos, float rel_width, enum anchor, OpenGLTexture2D &tex) {
    lrtb = getScreenSpaceRect(...);
    // upload lrtb
    // in shader, transform to [-1,1]
    // upload NO pos, NO tex

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

}

drawScreenSpaceText(vec2 rel_pos, float rel_width, enum anchor, OpenGLTexture2D &tex) {
    lrtb = ...

        drawText(...) ? ?
}
*/


OverlayRenderer::OverlayRenderer(void)
    : view::Renderer3DModule_2()
    , texture()
    , shader()
    , font(nullptr)
    , media_buttons()
    , paramMode("mode", "Overlay mode.")
    , paramAnchor("anchor", "Anchor of overlay.")
    // Custom position
    , paramCustomPositionSwitch(
          "enable_custom_position", "Enable custom relative position depending on selected anchor.")
    , paramCustomPosition("relative_position", "Custom relative position.")
    // Texture Mode
    , paramFileName("texture::file_name", "The file name of the texture.")
    // Media Buttons Mode
    /// nothing so far ....
    // Parameter Mode
    , paramPrefix("parameter::prefix", "The parmaeter value prefix.")
    , paramSufix("parameter::sufix", "The parameter value sufix.")
    , paramParameterName("parameter::name", "The full megamol parameter name.")
    // Label
    , paramText("label::text", "The displayed text.")
    , paramFont("label::font", "Choose predefined font type.")
    , paramFontSize("label::font_size", "The font size.") {

    core::param::EnumParam* mep = new core::param::EnumParam(Mode::TEXTURE);
    mep->SetTypePair(Mode::TEXTURE, "Texture");
    mep->SetTypePair(Mode::MEDIA_BUTTONS, "Media Buttons");
    mep->SetTypePair(Mode::PARAMETER, "Parameter");
    mep->SetTypePair(Mode::LABEL, "Label");
    this->paramMode << mep;
    this->MakeSlotAvailable(&this->paramMode);

    core::param::EnumParam* aep = new core::param::EnumParam(Anchor::ALIGN_LEFT_TOP);
    aep->SetTypePair(Anchor::ALIGN_LEFT_TOP, "Left Top");
    aep->SetTypePair(Anchor::ALIGN_LEFT_MIDDLE, "Left Middle");
    aep->SetTypePair(Anchor::ALIGN_LEFT_BOTTOM, "Left Bottom");
    aep->SetTypePair(Anchor::ALIGN_CENTER_TOP, "Center Top");
    aep->SetTypePair(Anchor::ALIGN_CENTER_MIDDLE, "Center Middle");
    aep->SetTypePair(Anchor::ALIGN_CENTER_BOTTOM, "Center Bottom");
    aep->SetTypePair(Anchor::ALIGN_RIGHT_TOP, "Right Top");
    aep->SetTypePair(Anchor::ALIGN_RIGHT_MIDDLE, "Right Middle");
    aep->SetTypePair(Anchor::ALIGN_RIGHT_BOTTOM, "Right Bottom");
    this->paramAnchor << aep;
    this->MakeSlotAvailable(&this->paramAnchor);

    // Custom overlay position
    this->paramCustomPositionSwitch << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramCustomPositionSwitch);

    this->paramCustomPosition << new core::param::Vector2fParam(vislib::math::Vector<float, 2>(0.0f, 0.0f));
    this->MakeSlotAvailable(&this->paramCustomPosition);

    // Texture Mode
    this->paramFileName << new core::param::FilePathParam("");
    this->paramFileName.SetUpdateCallback(this, &OverlayRenderer::onTextureFileName);
    this->MakeSlotAvailable(&this->paramFileName);

    // Parameter Mode
    this->paramPrefix << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    this->paramSufix << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    this->paramParameterName << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    // Label Mode
    this->paramText << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    core::param::EnumParam* fep = new core::param::EnumParam(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS);
    fep->SetTypePair(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS, "Roboto Sans");
    fep->SetTypePair(megamol::core::utility::SDFFont::FontName::EVOLVENTA_SANS, "Evolventa");
    fep->SetTypePair(megamol::core::utility::SDFFont::FontName::UBUNTU_MONO, "Ubuntu Mono");
    fep->SetTypePair(megamol::core::utility::SDFFont::FontName::VOLLKORN_SERIF, "Vollkorn Serif");
    this->paramFont << fep;
    this->paramFont.SetUpdateCallback(this, &OverlayRenderer::onFontName);
    this->MakeSlotAvailable(&this->paramFont);

    this->paramFontSize << new core::param::FloatParam(20.0f, 0.0f);
    this->MakeSlotAvailable(&this->paramFileName);
}


OverlayRenderer::~OverlayRenderer(void) { this->Release(); }


void OverlayRenderer::release(void) {

    this->font.reset();
    this->texture.tex.Release();
    this->shader.Release();
    for (size_t i = 0; i < this->media_buttons.size(); i++) {
        this->media_buttons[i].tex.Release();
    }
}


bool OverlayRenderer::create(void) {

    if (!this->loadShader(this->shader)) return false;

    return true;
}


bool OverlayRenderer::onToggleMode(megamol::core::param::ParamSlot& slot) {

    this->release();

    auto mode = this->paramMode.Param<core::param::EnumParam>()->Value();
    switch (mode) {
    case (Mode::TEXTURE): {

        this->onTextureFileName(slot);

    } break;
    case (Mode::MEDIA_BUTTONS): {

        // Load media button texutres from hard coded texture file names.
        std::string filename;
        for (size_t i = 0; i < this->media_buttons.size(); i++) {
            switch (static_cast<MediaButton>(i)) {
            case (MediaButton::PLAY):
                filename = "media_button_play.png";
                break;
            case (MediaButton::PAUSE):
                filename = "media_button_pause.png";
                break;
            case (MediaButton::STOP):
                filename = "media_button_stop.png";
                break;
            case (MediaButton::REWIND):
                filename = "media_button_rewind.png";
                break;
            case (MediaButton::FAST_FORWARD):
                filename = "media_button_fast-forward.png";
                break;
            }
            if (!this->loadTexture(filename, this->media_buttons[i])) return false;
        }

    } break;
    case (Mode::PARAMETER): {

    } break;
    case (Mode::LABEL): {

        this->onFontName(slot);

    } break;
    default:
        break;
    }

    return true;
}


bool OverlayRenderer::onTextureFileName(megamol::core::param::ParamSlot& slot) {

    this->texture.tex.Release();
    std::string filename = std::string(this->paramFileName.Param<core::param::FilePathParam>()->Value().PeekBuffer());
    if (!this->loadTexture(filename, this->texture)) return false;

    return true;
}


bool OverlayRenderer::onFontName(megamol::core::param::ParamSlot& slot) {

    this->font.reset();
    auto font_name = static_cast<megamol::core::utility::SDFFont::FontName>(
        this->paramFont.Param<core::param::EnumParam>()->Value());
    this->font = std::make_unique<megamol::core::utility::SDFFont>(font_name);
    if (!this->font->Initialise(this->GetCoreInstance())) return false;

    return true;
}


void megamol::gui::OverlayRenderer::setParameterGUIVisibility(void) {

    Mode mode = static_cast<Mode>(this->paramMode.Param<core::param::EnumParam>()->Value());

    // Custom position
    bool custom_position = this->paramCustomPositionSwitch.Param<core::param::BoolParam>()->Value();
    this->paramCustomPosition.Param<core::param::Vector2fParam>()->SetGUIVisible(custom_position);

    // Texture Mode
    bool texture_mode = (mode == Mode::TEXTURE);
    this->paramFileName.Param<core::param::FilePathParam>()->SetGUIVisible(texture_mode);

    // Parameter Mode
    bool parameter_mode = (mode == Mode::PARAMETER);
    this->paramPrefix.Param<core::param::StringParam>()->SetGUIVisible(parameter_mode);
    this->paramSufix.Param<core::param::StringParam>()->SetGUIVisible(parameter_mode);
    this->paramParameterName.Param<core::param::StringParam>()->SetGUIVisible(parameter_mode);

    // Label Mode
    bool label_mode = (mode == Mode::LABEL);
    this->paramText.Param<core::param::StringParam>()->SetGUIVisible(label_mode);
    this->paramFont.Param<core::param::EnumParam>()->SetGUIVisible(label_mode);
    this->paramFontSize.Param<core::param::FloatParam>()->SetGUIVisible(label_mode);
}


bool OverlayRenderer::GetExtents(megamol::core::view::CallRender3D_2& call) {

    auto chainedCall = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (chainedCall != nullptr) {
        *chainedCall = call;
        bool retVal = (*chainedCall)(view::AbstractCallRender::FnGetExtents);
        call = *chainedCall;
        return retVal;
    }

    return true;
}


bool OverlayRenderer::Render(megamol::core::view::CallRender3D_2& call) {

    auto cr3d = &call;
    if (cr3d == nullptr) return false;

    view::Camera_2 cam;
    cr3d->GetCamera(cam);
    glm::vec4 viewport;
    if (!cam.image_tile().empty()) { /// or better auto viewport = cr3d_in->GetViewport().GetSize()?
        viewport = glm::vec4(
            cam.image_tile().left(), cam.image_tile().bottom(), cam.image_tile().width(), cam.image_tile().height());
    } else {
        viewport = glm::vec4(0.0f, 0.0f, cam.resolution_gate().width(), cam.resolution_gate().height());
    }
    glm::vec2 vp = {viewport.z, viewport.w};
    glm::mat4 ortho = glm::ortho(0.0f, vp.x, 0.0f, vp.y, -1.0f, 1.0f);


    // this->paramMode.Param<core::param::EnumParam>()->Value;
    // this->paramAnchor.Param<core::param::EnumParam>()->Value;
    // this->paramCustomPositionSwitch.Param<core::param::BoolParam>()->Value;
    // this->paramCustomPosition.Param<core::param::Vector2fParam>()->Value;
    // this->paramFileName.Param<core::param::FilePathParam>()->Value;
    // this->paramPrefix.Param<core::param::StringParam>()->Value;
    // this->paramSufix.Param<core::param::StringParam>()->Value;
    // this->paramParameterName.Param<core::param::StringParam>()->Value;
    // this->paramText.Param<core::param::StringParam>()->Value;
    // this->paramFont.Param<core::param::EnumParam>()->Value;
    // this->paramFontSize.Param<core::param::FloatParam>()->Value;


    // Initialise new mode
    auto mode = this->paramMode.Param<core::param::EnumParam>()->Value();
    switch (mode) {
    case (Mode::TEXTURE): {


    } break;
    case (Mode::MEDIA_BUTTONS): {


    } break;
    case (Mode::PARAMETER): {


    } break;
    case (Mode::LABEL): {


    } break;
    default:
        break;
    }


    glm::vec4 lrtb;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    this->texture.tex.Bind();

    this->shader.Enable();

    glUniformMatrix4fv(this->shader.ParameterLocation("ortho"), 1, GL_FALSE, glm::value_ptr(ortho));
    glUniform4fv(this->shader.ParameterLocation("lrtb"), 1, glm::value_ptr(lrtb));
    glUniform1i(this->shader.ParameterLocation("tex"), 0);

    glDrawArrays(GL_POINTS, 0, 1); /// Vertex position is implicitly set via uniform 'lrtb'.

    this->shader.Disable();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glDisable(GL_BLEND);

    return true;
}


glm::vec4 OverlayRenderer::getScreenSpaceRect(glm::vec2 rel_pos, float rel_width, Anchor anchor, TextureData& io_tex) {

    glm::vec4 ltbr = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);

    // float tex_aspect = tex.
    // rel_height = rel_width * aspect_from_tex;
    // switch (anchor) {
    // case TOP_LEFT:
    //    left = rel_pos.x;
    //    right = left + rel_width;
    //    top = 1.0 - rel_pos.y;
    //    bottom = 1.0 - rel_pos.y - rel_height;
    //    break;
    //    // usw
    //}

    return ltbr;
}


bool OverlayRenderer::loadTexture(const std::string& fn, TextureData& io_tex) {

    if (io_tex.tex.IsValid()) io_tex.tex.Release();

    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    void* buf = nullptr;
    size_t size = 0;

    if ((size = this->loadFile(fn, &buf)) > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
            if (io_tex.tex.Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to create texture: %s [%s, %s, line %d]\n", fn.c_str(), __FILE__, __FUNCTION__, __LINE__);
                ARY_SAFE_DELETE(buf);
                return false;
            }
            io_tex.width = img.Width();
            io_tex.height = img.Height();
            io_tex.tex.Bind();
            glGenerateMipmap(GL_TEXTURE_2D);
            io_tex.tex.SetFilter(GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
            io_tex.tex.SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

            ARY_SAFE_DELETE(buf);
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to read texture: %s [%s, %s, line %d]\n", fn.c_str(), __FILE__, __FUNCTION__, __LINE__);
            ARY_SAFE_DELETE(buf);
            return false;
        }
    } else {
        ARY_SAFE_DELETE(buf);
        return false;
    }

    return true;
}


bool OverlayRenderer::loadShader(vislib::graphics::gl::GLSLShader& io_shader) {

    io_shader.Release();
    vislib::graphics::gl::ShaderSource vert, frag;
    try {
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("overlay::vertex", vert)) {
            return false;
        }
        if (!megamol::core::Module::instance()->ShaderSourceFactory().MakeShaderSource("overlay::fragment", frag)) {
            return false;
        }
        if (!io_shader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile: Unknown error [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile shader (@%s): %s [%s, %s, line %d]\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile shader: %s [%s, %s, line %d]\n", e.GetMsgA(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile shader: Unknown exception [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


size_t OverlayRenderer::loadFile(std::string name, void** outData) {

    *outData = nullptr;

    vislib::StringW filename = static_cast<vislib::StringW>(name.c_str());
    if (filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to load file: No file name given. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    if (!vislib::sys::File::Exists(filename)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load file \"%s\": Not existing. [%s, %s, line %d]\n",
            name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    size_t size = static_cast<size_t>(vislib::sys::File::GetSize(filename));
    if (size < 1) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load file \"%s\": File is empty. [%s, %s, line %d]\n",
            name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    vislib::sys::FastFile f;
    if (!f.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load file \"%s\": Unable to open file. [%s, %s, line %d]\n",
            name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    *outData = new BYTE[size];
    size_t num = static_cast<size_t>(f.Read(*outData, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to load file \"%s\": Unable to read whole file. [%s, %s, line %d]\n", name.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}
