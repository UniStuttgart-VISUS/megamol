/*
 * OverlayRenderer.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OverlayRenderer.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::gui;


OverlayRenderer::OverlayRenderer(void)
    : view::RendererModule<view::CallRender3D_2>()
    , paramMode("mode", "Overlay mode.")
    , paramAnchor("anchor", "Anchor of overlay.")
    , paramCustomPositionSwitch(
          "enable_custom_position", "Enable custom relative position depending on selected anchor.")
    , paramCustomPosition("relative_position", "Custom relative position.")
    , paramFileName("texture::file_name", "The file name of the texture.")
    , paramRelativeWidth("texture::relative_width", "Relative screen space width of texture.")
    , paramPrefix("parameter::prefix", "The parmaeter value prefix.")
    , paramSufix("parameter::sufix", "The parameter value sufix.")
    , paramParameterName("parameter::name", "The full megamol parameter name.")
    , paramText("label::text", "The displayed text.")
    , paramFont("label::font", "Choose predefined font type.")
    , paramFontSize("label::font_size", "The font size.")
    , texture()
    , shader()
    , font(nullptr)
    , media_buttons()
    , parameter_ptr(nullptr) {

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);

    param::EnumParam* mep = new param::EnumParam(Mode::TEXTURE);
    mep->SetTypePair(Mode::TEXTURE, "Texture");
    mep->SetTypePair(Mode::MEDIA_BUTTONS, "Media Buttons");
    mep->SetTypePair(Mode::PARAMETER, "Parameter");
    mep->SetTypePair(Mode::LABEL, "Label");
    this->paramMode << mep;
    this->MakeSlotAvailable(&this->paramMode);

    param::EnumParam* aep = new param::EnumParam(Anchor::ALIGN_LEFT_TOP);
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
    this->paramCustomPositionSwitch << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramCustomPositionSwitch);

    this->paramCustomPosition << new param::Vector2fParam(vislib::math::Vector<float, 2>(0.0f, 0.0f));
    this->MakeSlotAvailable(&this->paramCustomPosition);

    // Texture Mode
    this->paramFileName << new param::FilePathParam("");
    this->paramFileName.SetUpdateCallback(this, &OverlayRenderer::onTextureFileName);
    this->MakeSlotAvailable(&this->paramFileName);

    this->paramRelativeWidth << new param::FloatParam(25.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->paramRelativeWidth);

    // Parameter Mode
    this->paramPrefix << new param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    this->paramSufix << new param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    this->paramParameterName << new param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    // Label Mode
    this->paramText << new param::StringParam("");
    this->MakeSlotAvailable(&this->paramFileName);

    param::EnumParam* fep = new param::EnumParam(utility::SDFFont::FontName::ROBOTO_SANS);
    fep->SetTypePair(utility::SDFFont::FontName::ROBOTO_SANS, "Roboto Sans");
    fep->SetTypePair(utility::SDFFont::FontName::EVOLVENTA_SANS, "Evolventa");
    fep->SetTypePair(utility::SDFFont::FontName::UBUNTU_MONO, "Ubuntu Mono");
    fep->SetTypePair(utility::SDFFont::FontName::VOLLKORN_SERIF, "Vollkorn Serif");
    this->paramFont << fep;
    this->paramFont.SetUpdateCallback(this, &OverlayRenderer::onFontName);
    this->MakeSlotAvailable(&this->paramFont);

    this->paramFontSize << new param::FloatParam(20.0f, 0.0f);
    this->MakeSlotAvailable(&this->paramFileName);
}

OverlayRenderer::~OverlayRenderer(void) { this->Release(); }


void OverlayRenderer::release(void) {

    this->parameter_ptr = nullptr;
    this->font.reset();
    this->texture.tex.Release();
    this->shader.Release();
    for (size_t i = 0; i < this->media_buttons.size(); i++) {
        this->media_buttons[i].tex.Release();
    }
}


bool OverlayRenderer::create(void) {

    this->setParameterGUIVisibility();
    if (!this->loadShader(this->shader, "overlay::vertex", "overlay::fragment")) return false;

    return true;
}


bool OverlayRenderer::onToggleMode(param::ParamSlot& slot) {

    this->release();

    auto mode = this->paramMode.Param<param::EnumParam>()->Value();
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
        // ...
    } break;
    case (Mode::LABEL): {
        this->onFontName(slot);
    } break;
    }

    this->setParameterGUIVisibility();

    return true;
}


bool OverlayRenderer::onTextureFileName(param::ParamSlot& slot) {

    this->texture.tex.Release();
    std::string filename = std::string(this->paramFileName.Param<param::FilePathParam>()->Value().PeekBuffer());
    if (!this->loadTexture(filename, this->texture)) return false;
    return true;
}


bool OverlayRenderer::onFontName(param::ParamSlot& slot) {

    this->font.reset();
    auto font_name = static_cast<utility::SDFFont::FontName>(this->paramFont.Param<param::EnumParam>()->Value());
    this->font = std::make_unique<utility::SDFFont>(font_name);
    if (!this->font->Initialise(this->GetCoreInstance())) return false;
    return true;
}


bool OverlayRenderer::onParameterName(param::ParamSlot& slot) {

    this->parameter_ptr = nullptr;
    auto parameter_name = this->paramParameterName.Param<param::StringParam>()->Value();
    auto parameter = this->GetCoreInstance()->FindParameter(parameter_name, false, false);
    if (parameter.IsNull()) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to find parameter by name. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto* float_param = parameter.DynamicCast<param::FloatParam>();
    if (float_param == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Parameter is no FloatParam. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parameter_ptr = std::make_shared<param::FloatParam>((*float_param));
    return true;
}


void gui::OverlayRenderer::setParameterGUIVisibility(void) {

    // Custom position
    bool custom_position = this->paramCustomPositionSwitch.Param<param::BoolParam>()->Value();
    this->paramCustomPosition.Param<param::Vector2fParam>()->SetGUIVisible(custom_position);

    Mode mode = static_cast<Mode>(this->paramMode.Param<param::EnumParam>()->Value());

    // Texture Mode
    bool texture_mode = (mode == Mode::TEXTURE);
    this->paramFileName.Param<param::FilePathParam>()->SetGUIVisible(texture_mode);
    this->paramRelativeWidth.Param<param::FloatParam>()->SetGUIVisible(texture_mode);

    // Parameter Mode
    bool parameter_mode = (mode == Mode::PARAMETER);
    this->paramPrefix.Param<param::StringParam>()->SetGUIVisible(parameter_mode);
    this->paramSufix.Param<param::StringParam>()->SetGUIVisible(parameter_mode);
    this->paramParameterName.Param<param::StringParam>()->SetGUIVisible(parameter_mode);

    // Label Mode
    bool label_mode = (mode == Mode::LABEL);
    this->paramText.Param<param::StringParam>()->SetGUIVisible(label_mode);
    this->paramFont.Param<param::EnumParam>()->SetGUIVisible(label_mode);
    this->paramFontSize.Param<param::FloatParam>()->SetGUIVisible(label_mode);
}


bool OverlayRenderer::GetExtents(view::CallRender3D_2& call) {

    auto chainedCall = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (chainedCall != nullptr) {
        *chainedCall = call;
        bool retVal = (*chainedCall)(view::AbstractCallRender::FnGetExtents);
        call = *chainedCall;
        return retVal;
    }
    return true;
}


bool OverlayRenderer::Render(view::CallRender3D_2& call) {

    auto leftSlotParent = call.PeekCallerSlot()->Parent();
    std::shared_ptr<const view::AbstractView> viewptr =
        std::dynamic_pointer_cast<const view::AbstractView>(leftSlotParent);
    if (viewptr != nullptr) { // TODO move this behind the fbo magic?
        auto vp = call.GetViewport();
        glViewport(vp.Left(), vp.Bottom(), vp.Width(), vp.Height());
        auto backCol = call.BackgroundColor();
        glClearColor(backCol.x, backCol.y, backCol.z, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    // First call chained renderer
    auto* chainedCall = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (chainedCall != nullptr) {
        *chainedCall = call;
        if (!(*chainedCall)(view::AbstractCallRender::FnRender)) {
            return false;
        }
    }

    view::Camera_2 cam;
    call.GetCamera(cam);
    glm::vec4 viewport;
    if (!cam.image_tile().empty()) { /// or better: auto viewport = cr3d_in->GetViewport().GetSize()?
        viewport = glm::vec4(
            cam.image_tile().left(), cam.image_tile().bottom(), cam.image_tile().width(), cam.image_tile().height());
    } else {
        viewport = glm::vec4(0.0f, 0.0f, cam.resolution_gate().width(), cam.resolution_gate().height());
    }
    glm::vec2 vp = {viewport.z, viewport.w};
    glm::mat4 ortho = glm::ortho(0.0f, vp.x, 0.0f, vp.y, -1.0f, 1.0f);


    // this->paramMode.Param<param::EnumParam>()->Value();
    // this->paramAnchor.Param<param::EnumParam>()->Value();
    // this->paramCustomPositionSwitch.Param<param::BoolParam>()->Value();
    // this->paramCustomPosition.Param<param::Vector2fParam>()->Value();
    // this->paramFileName.Param<param::FilePathParam>()->Value();
    // this->paramRelativeWidth.Param<param::FloatParam>()->Value();
    // this->paramPrefix.Param<param::StringParam>()->Value();
    // this->paramSufix.Param<param::StringParam>()->Value();
    // this->paramParameterName.Param<param::StringParam>()->Value();
    // this->paramText.Param<param::StringParam>()->Value();
    // this->paramFont.Param<param::EnumParam>()->Value();
    // this->paramFontSize.Param<param::FloatParam>()->Value();


    // Initialise new mode
    auto mode = this->paramMode.Param<param::EnumParam>()->Value();
    switch (mode) {
    case (Mode::TEXTURE): {


    } break;
    case (Mode::MEDIA_BUTTONS): {


    } break;
    case (Mode::PARAMETER): {
        this->parameter_ptr->Value();
        this->parameter_ptr->ValueString();

    } break;
    case (Mode::LABEL): {


    } break;
    default:
        break;
    }


    Rectangle rectangle;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    this->texture.tex.Bind();

    this->shader.Enable();

    glUniformMatrix4fv(this->shader.ParameterLocation("ortho"), 1, GL_FALSE, glm::value_ptr(ortho));
    glUniform1f(this->shader.ParameterLocation("left"), rectangle.left);
    glUniform1f(this->shader.ParameterLocation("right"), rectangle.right);
    glUniform1f(this->shader.ParameterLocation("top"), rectangle.top);
    glUniform1f(this->shader.ParameterLocation("bottom"), rectangle.bottom);
    glUniform1i(this->shader.ParameterLocation("tex"), 0);

    glDrawArrays(GL_POINTS, 0, 1); /// Vertex position is implicitly set via uniform 'lrtb'.

    this->shader.Disable();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glDisable(GL_BLEND);

    return true;
}


void OverlayRenderer::drawScreenSpaceBillboard(
    glm::vec2 rel_pos, float rel_width, Anchor anchor, const TextureData& tex) {


    // lrtb = getScreenSpaceRect(...);
    // upload lrtb
    // in shader, transform to [-1,1]
    // upload NO pos, NO tex

    // glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void OverlayRenderer::drawScreenSpaceText(glm::vec2 rel_pos, float rel_width, Anchor anchor, const TextureData& tex) {

    // lrtb = ...

    //    drawText(...) ? ?
}


OverlayRenderer::Rectangle OverlayRenderer::getScreenSpaceRect(
    glm::vec2 rel_pos, float rel_width, Anchor anchor, const TextureData& tex) {

    Rectangle rectangle = {0.0f, 0.0f, 0.0f, 0.0f};

    float tex_aspect = static_cast<float>(tex.width) / static_cast<float>(tex.height);
    auto rel_height = rel_width * tex_aspect;

    switch (anchor) {
    case (Anchor::ALIGN_LEFT_TOP): {
        rectangle.left = rel_pos.x;
        rectangle.right = rectangle.left + rel_width;
        rectangle.top = 1.0 - rel_pos.y;
        rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;

    case (Anchor::ALIGN_LEFT_MIDDLE): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_LEFT_BOTTOM): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_CENTER_TOP): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_CENTER_MIDDLE): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_CENTER_BOTTOM): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_RIGHT_TOP): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_RIGHT_MIDDLE): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_RIGHT_BOTTOM): {
        // rectangle.left = rel_pos.x;
        // rectangle.right = rectangle.left + rel_width;
        // rectangle.top = 1.0 - rel_pos.y;
        // rectangle.bottom = 1.0 - rel_pos.y - rel_height;
    } break;
    }

    return rectangle;
}


bool OverlayRenderer::loadTexture(const std::string& fn, TextureData& io_tex) {

    if (io_tex.tex.IsValid()) io_tex.tex.Release();

    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    void* buf = nullptr;
    size_t size = 0;

    size = utility::ResourceWrapper::LoadResource(
        this->GetCoreInstance()->Configuration(), vislib::StringA(fn.c_str()), (void**)(&buf));
    if (size == 0) {
        size = this->loadRawFile(fn, &buf);
    }

    if (size > 0) {
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


bool OverlayRenderer::loadShader(
    vislib::graphics::gl::GLSLShader& io_shader, const std::string& vert_name, const std::string& frag_name) {

    io_shader.Release();
    vislib::graphics::gl::ShaderSource vert, frag;
    try {
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
                vislib::StringA(vert_name.c_str()), vert)) {
            return false;
        }
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
                vislib::StringA(frag_name.c_str()), frag)) {
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


size_t OverlayRenderer::loadRawFile(std::string name, void** outData) {

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
