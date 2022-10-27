/*
 * OverlayRenderer.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */


#include "OverlayRenderer.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/utility/log/Log.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematic_gl;


OverlayRenderer::OverlayRenderer()
        : view::RendererModule<mmstd_gl::CallRender3DGL, mmstd_gl::ModuleGL>()
        , megamol::core_gl::utility::RenderUtils()
        , paramMode("mode", "Overlay mode.")
        , paramAnchor("anchor", "Anchor of overlay. NOTE: Hide GUI menu to see overlay anchored on the top.")
        , paramCustomPosition("position_offset", "Custom relative position offset in respect to selected anchor.")
        , paramFileName("texture::file_name", "The file name of the texture.")
        , paramRelativeWidth("texture::relative_width", "Relative screen space width of texture.")
        , paramIconColor("transport_ctrl::color", "Color of transpctrl icons.")
        , paramDuration("transport_ctrl::duration",
              "Duration transport ctrl icons are shown after value changes. Value of zero "
              "means showing transport ctrl icons permanently.")
        , paramFastSpeed(
              "transport_ctrl::fast_speed", "Define factor of default speed for fast transport ctrl icon threshold.")
        , paramUltraFastSpeed("transport_ctrl::value_scaling",
              "Define factor of default speed for ultra fast transport ctrl icon threshold.")
        , paramSpeedParameter("transport_ctrl::speed_parameter_name",
              "The full parameter name for the animation speed, e.g. '::Project_1::View3D_21::anim::speed'.")
        , paramTimeParameter("transport_ctrl::time_parameter_name",
              "The full parameter name for the animation time, e.g. '::Project_1::View3D_21::anim::time'.")
        , paramPrefix("parameter::prefix", "The parameter value prefix.")
        , paramSuffix("parameter::suffix", "The parameter value suffix.")
        , paramParameterName("parameter::name",
              "The full parameter name, e.g. '::Project_1::View3D_21::cam::position'. "
              "Supprted parameter types: float, int, Vector2f/3f/4f")
        , paramText("label::text", "The displayed text.")
        , paramFontName("font::name", "The font name.")
        , paramFontSize("font::size", "The font size.")
        , paramFontColor("font::color", "The font color.")
        , m_texture_id(0)
        , m_font_ptr(nullptr)
        , m_viewport()
        , m_current_rectangle({0.0f, 0.0f, 0.0f, 0.0f})
        , m_parameter_ptr(nullptr)
        , m_transpctrl_icons()
        , m_state()
        , m_speed_parameter_ptr(nullptr)
        , m_time_parameter_ptr(nullptr) {

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);

    param::EnumParam* mep = new param::EnumParam(Mode::TEXTURE);
    mep->SetTypePair(Mode::TEXTURE, "Texture");
    mep->SetTypePair(Mode::TRANSPORT_CTRL, "Transport Ctrl Icons");
    mep->SetTypePair(Mode::PARAMETER, "Parameter");
    mep->SetTypePair(Mode::LABEL, "Label");
    this->paramMode << mep;
    this->paramMode.SetUpdateCallback(this, &OverlayRenderer::onToggleMode);
    this->MakeSlotAvailable(&this->paramMode);
    mep = nullptr;

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
    this->paramAnchor.SetUpdateCallback(this, &OverlayRenderer::onTriggerRecalcRectangle);
    this->MakeSlotAvailable(&this->paramAnchor);
    aep = nullptr;

    this->paramCustomPosition << new param::Vector2fParam(vislib::math::Vector<float, 2>(0.0f, 0.0f),
        vislib::math::Vector<float, 2>(0.0f, 0.0f), vislib::math::Vector<float, 2>(100.0f, 100.0f));
    this->paramCustomPosition.SetUpdateCallback(this, &OverlayRenderer::onTriggerRecalcRectangle);
    this->MakeSlotAvailable(&this->paramCustomPosition);

    // Texture Mode
    this->paramFileName << new param::FilePathParam("", param::FilePathParam::Flag_File, {"png"});
    this->paramFileName.SetUpdateCallback(this, &OverlayRenderer::onTextureFileName);
    this->MakeSlotAvailable(&this->paramFileName);

    this->paramRelativeWidth << new param::FloatParam(25.0f, 0.0f, 100.0f);
    this->paramRelativeWidth.SetUpdateCallback(this, &OverlayRenderer::onTriggerRecalcRectangle);
    this->MakeSlotAvailable(&this->paramRelativeWidth);

    // TranspCtrl Icon Mode
    this->paramIconColor << new param::ColorParam(0.5f, 0.75f, 0.75f, 1.0f);
    this->MakeSlotAvailable(&this->paramIconColor);

    this->paramDuration << new param::FloatParam(3.0f, 0.0f);
    this->MakeSlotAvailable(&this->paramDuration);

    this->paramFastSpeed << new param::FloatParam(5.0f, 1.0f);
    this->MakeSlotAvailable(&this->paramFastSpeed);

    this->paramUltraFastSpeed << new param::FloatParam(10.0f, 1.0f);
    this->MakeSlotAvailable(&this->paramUltraFastSpeed);

    this->paramSpeedParameter << new param::StringParam("");
    this->paramSpeedParameter.SetUpdateCallback(this, &OverlayRenderer::onParameterName);
    this->MakeSlotAvailable(&this->paramSpeedParameter);

    this->paramTimeParameter << new param::StringParam("");
    this->paramTimeParameter.SetUpdateCallback(this, &OverlayRenderer::onParameterName);
    this->MakeSlotAvailable(&this->paramTimeParameter);

    // Parameter Mode
    this->paramPrefix << new param::StringParam("");
    this->MakeSlotAvailable(&this->paramPrefix);

    this->paramSuffix << new param::StringParam("");
    this->MakeSlotAvailable(&this->paramSuffix);

    this->paramParameterName << new param::StringParam("");
    this->paramParameterName.SetUpdateCallback(this, &OverlayRenderer::onParameterName);
    this->MakeSlotAvailable(&this->paramParameterName);

    // Label Mode
    this->paramText << new param::StringParam("");
    this->MakeSlotAvailable(&this->paramText);

    // Font Settings
    param::EnumParam* fep = new param::EnumParam(utility::SDFFont::PRESET_ROBOTO_SANS);
    fep->SetTypePair(utility::SDFFont::PRESET_ROBOTO_SANS, "Roboto Sans");
    fep->SetTypePair(utility::SDFFont::PRESET_EVOLVENTA_SANS, "Evolventa");
    fep->SetTypePair(utility::SDFFont::PRESET_UBUNTU_MONO, "Ubuntu Mono");
    fep->SetTypePair(utility::SDFFont::PRESET_VOLLKORN_SERIF, "Vollkorn Serif");
    this->paramFontName << fep;
    this->paramFontName.SetUpdateCallback(this, &OverlayRenderer::onFontName);
    this->MakeSlotAvailable(&this->paramFontName);
    fep = nullptr;

    this->paramFontSize << new param::FloatParam(20.0f, 0.0f);
    this->MakeSlotAvailable(&this->paramFontSize);

    this->paramFontColor << new param::ColorParam(0.5f, 0.5f, 0.5f, 1.0f);
    this->MakeSlotAvailable(&this->paramFontColor);

    // init state
    this->m_state.icon = TranspCtrlIcon::NONE_COUNT;
    this->m_state.current_anim_time = 0.0f;
    this->m_state.start_real_time = std::chrono::system_clock::now();
}


OverlayRenderer::~OverlayRenderer() {
    this->Release();
}


void OverlayRenderer::release() {

    this->m_font_ptr.reset();
    this->m_parameter_ptr = nullptr;
    this->m_speed_parameter_ptr = nullptr;
    this->m_time_parameter_ptr = nullptr;
}


bool OverlayRenderer::create() {

    if (!this->InitPrimitiveRendering(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>())) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Couldn't initialize primitive rendering. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return this->onToggleMode(this->paramMode);
}


bool OverlayRenderer::onToggleMode(param::ParamSlot& slot) {

    slot.ResetDirty();
    this->m_font_ptr.reset();
    this->m_parameter_ptr = nullptr;
    this->m_speed_parameter_ptr = nullptr;
    this->m_time_parameter_ptr = nullptr;
    this->DeleteAllTextures();
    this->m_texture_id = 0;

    this->setParameterGUIVisibility();

    auto mode = static_cast<Mode>(this->paramMode.Param<param::EnumParam>()->Value());
    switch (mode) {
    case (Mode::TEXTURE): {
        this->onTextureFileName(this->paramFileName);
    } break;
    case (Mode::TRANSPORT_CTRL): {
        this->onParameterName(this->paramTimeParameter);
        this->onParameterName(this->paramSpeedParameter);
        std::string filename;
        for (size_t i = 0; i < this->m_transpctrl_icons.size(); i++) {
            switch (static_cast<TranspCtrlIcon>(i)) {
            case (TranspCtrlIcon::PLAY):
                filename = "transport_ctrl_play.png";
                break;
            case (TranspCtrlIcon::PAUSE):
                filename = "transport_ctrl_pause.png";
                break;
            case (TranspCtrlIcon::STOP):
                filename = "transport_ctrl_stop.png";
                break;
            case (TranspCtrlIcon::FAST_REWIND):
                filename = "transport_ctrl_fast-rewind.png";
                break;
            case (TranspCtrlIcon::FAST_FORWARD):
                filename = "transport_ctrl_fast-forward.png";
                break;
            case (TranspCtrlIcon::ULTRA_FAST_FORWARD):
                filename = "transport_ctrl_ultra-fast-forward.png";
                break;
            case (TranspCtrlIcon::ULTRA_FAST_REWIND):
                filename = "transport_ctrl_ultra-fast-forward.png";
                break;
            default:
                break;
            }
            std::wstring texture_filename(megamol::core::utility::ResourceWrapper::getFileName(
                this->GetCoreInstance()->Configuration(), vislib::StringA(filename.c_str()))
                                              .PeekBuffer());
            if (!this->LoadTextureFromFile(
                    this->m_transpctrl_icons[i], megamol::core::utility::WChar2Utf8String(texture_filename))) {
                return false;
            }
        }
    } break;
    case (Mode::PARAMETER): {
        this->onParameterName(this->paramParameterName);
        this->onFontName(this->paramFontName);
    } break;
    case (Mode::LABEL): {
        this->onFontName(this->paramFontName);
    } break;
    }

    this->onTriggerRecalcRectangle(slot);
    return true;
}


bool OverlayRenderer::onTextureFileName(param::ParamSlot& slot) {

    slot.ResetDirty();
    auto filename = this->paramFileName.Param<param::FilePathParam>()->Value();
    if (!this->LoadTextureFromFile(this->m_texture_id, filename, (this->m_texture_id != 0))) {
        return false;
    }
    this->onTriggerRecalcRectangle(slot);
    return true;
}


bool OverlayRenderer::onFontName(param::ParamSlot& slot) {

    slot.ResetDirty();
    this->m_font_ptr.reset();
    auto font_name =
        static_cast<utility::SDFFont::PresetFontName>(this->paramFontName.Param<param::EnumParam>()->Value());
    this->m_font_ptr = std::make_unique<utility::SDFFont>(font_name);
    if (!this->m_font_ptr->Initialise(
            this->GetCoreInstance(), frontend_resources.get<megamol::frontend_resources::RuntimeConfig>())) {
        return false;
    }
    return true;
}


bool OverlayRenderer::onParameterName(param::ParamSlot& slot) {

    slot.ResetDirty();

    auto parameter_name = vislib::StringA(slot.Param<param::StringParam>()->Value().c_str());
    if (parameter_name.IsEmpty()) {
        return false;
    }

    // Check megamol graph for available parameter:
    megamol::core::param::AbstractParam* param_ptr = nullptr;
    auto& megamolgraph = frontend_resources.get<megamol::core::MegaMolGraph>();
    param_ptr = megamolgraph.FindParameter(std::string(parameter_name.PeekBuffer()));
    if (param_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to find parameter by name. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool found_valid_param_type = false;

    if (&slot == &this->paramTimeParameter) {
        if (dynamic_cast<param::FloatParam*>(param_ptr) != nullptr) {
            found_valid_param_type = true;
        }
        this->m_time_parameter_ptr = nullptr;
        if (found_valid_param_type) {
            this->m_time_parameter_ptr = param_ptr;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Parameter type is not supported, only: Float. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    } else if (&slot == &this->paramSpeedParameter) {
        if (dynamic_cast<param::FloatParam*>(param_ptr) != nullptr) {
            found_valid_param_type = true;
        }
        this->m_speed_parameter_ptr = nullptr;
        if (found_valid_param_type) {
            this->m_speed_parameter_ptr = param_ptr;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Parameter type is not supported, only: Float. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    } else if (&slot == &this->paramParameterName) {
        if (dynamic_cast<param::FloatParam*>(param_ptr) != nullptr) {
            found_valid_param_type = true;
        } else if (dynamic_cast<param::IntParam*>(param_ptr) != nullptr) {
            found_valid_param_type = true;
        } else if (dynamic_cast<param::Vector2fParam*>(param_ptr) != nullptr) {
            found_valid_param_type = true;
        } else if (dynamic_cast<param::Vector3fParam*>(param_ptr) != nullptr) {
            found_valid_param_type = true;
        } else if (dynamic_cast<param::Vector4fParam*>(param_ptr) != nullptr) {
            found_valid_param_type = true;
        }
        this->m_parameter_ptr = nullptr;
        if (found_valid_param_type) {
            this->m_parameter_ptr = param_ptr;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Parameter type is not supported, only: Float, Int, Vector2f, Vector3f and Vector4f. [%s, %s, line "
                "%d]\n",
                __FILE__, __FUNCTION__, __LINE__);
        }
    }

    return found_valid_param_type;
}

bool OverlayRenderer::onTriggerRecalcRectangle(core::param::ParamSlot& slot) {

    slot.ResetDirty();

    Mode mode = static_cast<Mode>(this->paramMode.Param<param::EnumParam>()->Value());
    bool transpctrl_mode = (mode == Mode::TRANSPORT_CTRL);

    auto anchor = static_cast<Anchor>(this->paramAnchor.Param<param::EnumParam>()->Value());
    auto param_position = this->paramCustomPosition.Param<param::Vector2fParam>()->Value();
    glm::vec2 rel_pos = glm::vec2(param_position.X() / 100.0f, param_position.Y() / 100.0f);
    auto rel_width = this->paramRelativeWidth.Param<param::FloatParam>()->Value() / 100.0f;

    float width = this->m_viewport.x;
    float height = this->m_viewport.y;
    if (mode == Mode::TRANSPORT_CTRL) {
        if (this->m_state.icon != TranspCtrlIcon::NONE_COUNT) {
            width = this->GetTextureWidth(this->m_transpctrl_icons[this->m_state.icon]);
            height = this->GetTextureHeight(this->m_transpctrl_icons[this->m_state.icon]);
        }
    } else if (mode == Mode::TEXTURE) {
        width = this->GetTextureWidth(this->m_texture_id);
        height = this->GetTextureHeight(this->m_texture_id);
    }
    this->m_current_rectangle = this->getScreenSpaceRect(rel_pos, rel_width, anchor, width, height, this->m_viewport);

    return true;
}


void OverlayRenderer::setParameterGUIVisibility() {

    Mode mode = static_cast<Mode>(this->paramMode.Param<param::EnumParam>()->Value());
    bool texture_mode = (mode == Mode::TEXTURE);
    bool transpctrl_mode = (mode == Mode::TRANSPORT_CTRL);
    bool parameter_mode = (mode == Mode::PARAMETER);
    bool label_mode = (mode == Mode::LABEL);

    // Texture Mode
    this->paramFileName.Parameter()->SetGUIVisible(texture_mode);
    this->paramRelativeWidth.Parameter()->SetGUIVisible(texture_mode || transpctrl_mode);

    // TranspCtrl Icons Mode
    this->paramIconColor.Parameter()->SetGUIVisible(transpctrl_mode);
    this->paramDuration.Parameter()->SetGUIVisible(transpctrl_mode);
    this->paramFastSpeed.Parameter()->SetGUIVisible(transpctrl_mode);
    this->paramUltraFastSpeed.Parameter()->SetGUIVisible(transpctrl_mode);
    this->paramSpeedParameter.Parameter()->SetGUIVisible(transpctrl_mode);
    this->paramTimeParameter.Parameter()->SetGUIVisible(transpctrl_mode);

    // Parameter Mode
    this->paramPrefix.Parameter()->SetGUIVisible(parameter_mode);
    this->paramSuffix.Parameter()->SetGUIVisible(parameter_mode);
    this->paramParameterName.Parameter()->SetGUIVisible(parameter_mode);

    // Label Mode
    this->paramText.Parameter()->SetGUIVisible(label_mode);

    // Font Settings
    this->paramFontName.Parameter()->SetGUIVisible(label_mode || parameter_mode);
    this->paramFontSize.Parameter()->SetGUIVisible(label_mode || parameter_mode);
    this->paramFontColor.Parameter()->SetGUIVisible(label_mode || parameter_mode);
}


bool OverlayRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    auto* chainedCall = this->chainRenderSlot.CallAs<mmstd_gl::CallRender3DGL>();
    if (chainedCall != nullptr) {
        *chainedCall = call;
        bool retVal = (*chainedCall)(view::AbstractCallRender::FnGetExtents);
        call = *chainedCall;
        return retVal;
    }
    return true;
}


bool OverlayRenderer::Render(mmstd_gl::CallRender3DGL& call) {

    // Framebuffer object
    auto const lhsFBO = call.GetFramebuffer();

    auto cr3d_out = this->chainRenderSlot.CallAs<mmstd_gl::CallRender3DGL>();
    if (cr3d_out != nullptr) {
        *cr3d_out = call;
        if (!(*cr3d_out)(view::AbstractCallRender::FnRender)) {
            return false;
        }
    }

    // Get current viewport
    if ((this->m_viewport.x != lhsFBO->getWidth()) || (this->m_viewport.y != lhsFBO->getHeight())) {
        this->m_viewport = {lhsFBO->getWidth(), lhsFBO->getHeight()};
        // Reload rectangle on viewport changes
        this->onTriggerRecalcRectangle(this->paramMode);
    }

    glm::mat4 ortho = glm::ortho(0.0f, this->m_viewport.x, 0.0f, this->m_viewport.y, -1.0f, 1.0f);

    // Draw mode dependent stuff
    auto mode = this->paramMode.Param<param::EnumParam>()->Value();
    switch (mode) {
    case (Mode::TEXTURE): {
        auto overwrite_color = glm::vec4(0.0f); // Ignored when alpha = 0. Using texture color.
        this->drawScreenSpaceBillboard(
            ortho, this->m_viewport, this->m_current_rectangle, this->m_texture_id, overwrite_color);
    } break;
    case (Mode::TRANSPORT_CTRL): {
        auto param_color = this->paramIconColor.Param<param::ColorParam>()->Value();
        glm::vec4 overwrite_color = glm::vec4(param_color[0], param_color[1], param_color[2], param_color[3]);
        float fast_speed = this->paramFastSpeed.Param<param::FloatParam>()->Value();
        float ultra_fast_speed = this->paramUltraFastSpeed.Param<param::FloatParam>()->Value();

        float current_speed = 0.0f;
        if (this->m_speed_parameter_ptr != nullptr) {
            if (auto* float_param = dynamic_cast<param::FloatParam*>(this->m_speed_parameter_ptr)) {
                current_speed = float_param->Value();
            }
        }
        float current_anim_time = 0.0f;
        if (this->m_time_parameter_ptr != nullptr) {
            if (auto* float_param = dynamic_cast<param::FloatParam*>(this->m_time_parameter_ptr)) {
                current_anim_time = float_param->Value();
            }
        }

        /// XXX Since delta_time might not have changed significantly compared to last frame,
        /// this leads to flickering for high framerate (>~1000fps)
        float delta_time = current_anim_time - this->m_state.current_anim_time;
        this->m_state.current_anim_time = current_anim_time;

        TranspCtrlIcon current_icon = TranspCtrlIcon::NONE_COUNT;
        if (current_speed < (-ultra_fast_speed)) {
            current_icon = TranspCtrlIcon::ULTRA_FAST_REWIND;
        } else if (current_speed > ultra_fast_speed) {
            current_icon = TranspCtrlIcon::ULTRA_FAST_FORWARD;
        } else if (current_speed < (-fast_speed)) {
            current_icon = TranspCtrlIcon::FAST_REWIND;
        } else if (current_speed > fast_speed) {
            current_icon = TranspCtrlIcon::FAST_FORWARD;
        } else if (delta_time > 0.0f) {
            current_icon = TranspCtrlIcon::PLAY;
        }
        if (delta_time == 0.0f) {
            current_icon = TranspCtrlIcon::PAUSE;
        }

        if (current_icon != this->m_state.icon) {
            this->m_state.icon = current_icon;
            this->onTriggerRecalcRectangle(this->paramCustomPosition);
            this->m_state.start_real_time = std::chrono::system_clock::now();
        }

        float duration = this->paramDuration.Param<param::FloatParam>()->Value();
        std::chrono::duration<double> diff = (std::chrono::system_clock::now() - this->m_state.start_real_time);
        if ((duration > 0.0f) && (diff.count() > static_cast<double>(duration))) {
            current_icon = TranspCtrlIcon::NONE_COUNT;
        }

        if (current_icon != TranspCtrlIcon::NONE_COUNT) {
            this->drawScreenSpaceBillboard(ortho, this->m_viewport, this->m_current_rectangle,
                this->m_transpctrl_icons[current_icon], overwrite_color);
        }
    } break;
    case (Mode::PARAMETER): {
        auto param_color = this->paramFontColor.Param<param::ColorParam>()->Value();
        glm::vec4 color = glm::vec4(param_color[0], param_color[1], param_color[2], param_color[3]);
        auto font_size = this->paramFontSize.Param<param::FloatParam>()->Value();
        auto anchor = static_cast<Anchor>(this->paramAnchor.Param<param::EnumParam>()->Value());

        auto param_prefix = this->paramPrefix.Param<param::StringParam>()->Value();
        std::string prefix = param_prefix;

        auto param_sufix = this->paramSuffix.Param<param::StringParam>()->Value();
        std::string sufix = param_sufix;

        std::string text("");
        if (this->m_parameter_ptr != nullptr) {
            if (auto* float_param = dynamic_cast<param::FloatParam*>(this->m_parameter_ptr)) {
                auto value = float_param->Value();
                std::stringstream stream;
                stream << std::fixed << std::setprecision(8) << " " << value << " ";
                text = prefix + stream.str() + sufix;
            } else if (auto* int_param = dynamic_cast<param::IntParam*>(this->m_parameter_ptr)) {
                auto value = int_param->Value();
                std::stringstream stream;
                stream << " " << value << " ";
                text = prefix + stream.str() + sufix;
            } else if (auto* vec2_param = dynamic_cast<param::Vector2fParam*>(this->m_parameter_ptr)) {
                auto value = vec2_param->Value();
                std::stringstream stream;
                stream << std::fixed << std::setprecision(8) << " (" << value.X() << ", " << value.Y() << ") ";
                text = prefix + stream.str() + sufix;
            } else if (auto* vec3_param = dynamic_cast<param::Vector3fParam*>(this->m_parameter_ptr)) {
                auto value = vec3_param->Value();
                std::stringstream stream;
                stream << std::fixed << std::setprecision(8) << " (" << value.X() << ", " << value.Y() << ", "
                       << value.Z() << ") ";
                text = prefix + stream.str() + sufix;
            } else if (auto* vec4_param = dynamic_cast<param::Vector4fParam*>(this->m_parameter_ptr)) {
                auto value = vec4_param->Value();
                std::stringstream stream;
                stream << std::fixed << std::setprecision(8) << " (" << value.X() << ", " << value.Y() << ", "
                       << value.Z() << ", " << value.W() << ") ";
                text = prefix + stream.str() + sufix;
            }
        }
        if (text.empty()) {
            // megamol::core::utility::log::Log::DefaultLog.WriteError(
            //    "Unable to read parmeter value [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        this->drawScreenSpaceText(
            ortho, (*this->m_font_ptr), text, color, font_size, anchor, this->m_current_rectangle);
    } break;
    case (Mode::LABEL): {
        auto param_color = this->paramFontColor.Param<param::ColorParam>()->Value();
        glm::vec4 color = glm::vec4(param_color[0], param_color[1], param_color[2], param_color[3]);
        auto font_size = this->paramFontSize.Param<param::FloatParam>()->Value();
        auto anchor = static_cast<Anchor>(this->paramAnchor.Param<param::EnumParam>()->Value());

        auto param_text = this->paramText.Param<param::StringParam>()->Value();
        std::string text = param_text;

        if (this->m_font_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to read texture: %s [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        this->drawScreenSpaceText(
            ortho, (*this->m_font_ptr), text, color, font_size, anchor, this->m_current_rectangle);
    } break;
    }

    return true;
}


void OverlayRenderer::drawScreenSpaceBillboard(
    glm::mat4 ortho, glm::vec2 viewport, Rectangle rectangle, GLuint texture_id, glm::vec4 overwrite_color) {

    glm::vec3 pos_bottom_left = {rectangle.left, rectangle.bottom, 1.0f};
    glm::vec3 pos_upper_left = {rectangle.left, rectangle.top, 1.0f};
    glm::vec3 pos_upper_right = {rectangle.right, rectangle.top, 1.0f};
    glm::vec3 pos_bottom_right = {rectangle.right, rectangle.bottom, 1.0f};
    this->Push2DColorTexture(
        texture_id, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true, overwrite_color);
    this->DrawTextures(ortho, viewport);
}

void OverlayRenderer::drawScreenSpaceText(glm::mat4 ortho, megamol::core::utility::SDFFont& font,
    const std::string& text, glm::vec4 color, float size, Anchor anchor, Rectangle rectangle) const {

    float x = rectangle.left;
    float y = rectangle.top;
    float z = 1.0f;

    switch (anchor) {
    case (Anchor::ALIGN_LEFT_TOP): {
        // y -= size;
    } break;
    case (Anchor::ALIGN_LEFT_MIDDLE): {
        y = rectangle.top + (rectangle.bottom - rectangle.top) / 2.0f;
    } break;
    case (Anchor::ALIGN_LEFT_BOTTOM): {
        y = rectangle.bottom;
    } break;
    case (Anchor::ALIGN_CENTER_TOP): {
        x = rectangle.left + (rectangle.right - rectangle.left) / 2.0f;
    } break;
    case (Anchor::ALIGN_CENTER_MIDDLE): {
        x = rectangle.left + (rectangle.right - rectangle.left) / 2.0f;
        y = rectangle.top + (rectangle.bottom - rectangle.top) / 2.0f;
    } break;
    case (Anchor::ALIGN_CENTER_BOTTOM): {
        x = rectangle.left + (rectangle.right - rectangle.left) / 2.0f;
        y = rectangle.bottom;
    } break;
    case (Anchor::ALIGN_RIGHT_TOP): {
        x = rectangle.right;
    } break;
    case (Anchor::ALIGN_RIGHT_MIDDLE): {
        x = rectangle.right;
        y = rectangle.top + (rectangle.bottom - rectangle.top) / 2.0f;
    } break;
    case (Anchor::ALIGN_RIGHT_BOTTOM): {
        x = rectangle.right;
        y = rectangle.bottom;
    } break;
    }

    font.DrawString(ortho, glm::value_ptr(color), x, y, z, size, false, text.c_str(), anchor);
}


OverlayRenderer::Rectangle OverlayRenderer::getScreenSpaceRect(glm::vec2 rel_pos, float rel_width, Anchor anchor,
    unsigned int texture_width, unsigned int texture_height, glm::vec2 viewport) const {

    Rectangle rectangle = {0.0f, 0.0f, 0.0f, 0.0f};
    if ((texture_width == 0) || (texture_height == 0)) {
        return rectangle;
    }

    float tex_aspect = static_cast<float>(texture_width) / static_cast<float>(texture_height);
    float viewport_aspect = viewport.x / viewport.y;
    float rel_height = rel_width / tex_aspect * viewport_aspect;

    switch (anchor) {
    case (Anchor::ALIGN_LEFT_TOP): {
        rectangle.left = rel_pos.x;
        rectangle.right = rectangle.left + rel_width;
        rectangle.top = 1.0f - rel_pos.y;
        rectangle.bottom = rectangle.top - rel_height;
    } break;

    case (Anchor::ALIGN_LEFT_MIDDLE): {
        rectangle.left = rel_pos.x;
        rectangle.right = rectangle.left + rel_width;
        rectangle.top = 0.5f - (rel_pos.y / 2.0f) + (rel_height / 2.0f);
        rectangle.bottom = rectangle.top - rel_height;
    } break;
    case (Anchor::ALIGN_LEFT_BOTTOM): {
        rectangle.left = rel_pos.x;
        rectangle.right = rectangle.left + rel_width;
        rectangle.top = rel_pos.y + rel_height;
        rectangle.bottom = rel_pos.y;
    } break;
    case (Anchor::ALIGN_CENTER_TOP): {
        rectangle.left = 0.5f + (rel_pos.x / 2.0f) - (rel_width / 2.0f);
        rectangle.right = rectangle.left + rel_width;
        rectangle.top = 1.0f - rel_pos.y;
        rectangle.bottom = 1.0f - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_CENTER_MIDDLE): {
        rectangle.left = 0.5f + (rel_pos.x / 2.0f) - (rel_width / 2.0f);
        rectangle.right = rectangle.left + rel_width;
        rectangle.top = 0.5f - (rel_pos.y / 2.0f) + (rel_height / 2.0f);
        rectangle.bottom = rectangle.top - rel_height;
    } break;
    case (Anchor::ALIGN_CENTER_BOTTOM): {
        rectangle.left = 0.5f + (rel_pos.x / 2.0f) - (rel_width / 2.0f);
        rectangle.right = rectangle.left + rel_width;
        rectangle.top = rel_pos.y + rel_height;
        rectangle.bottom = rel_pos.y;
    } break;
    case (Anchor::ALIGN_RIGHT_TOP): {
        rectangle.left = 1.0f - rel_pos.x - rel_width;
        rectangle.right = 1.0f - rel_pos.x;
        rectangle.top = 1.0f - rel_pos.y;
        rectangle.bottom = 1.0f - rel_pos.y - rel_height;
    } break;
    case (Anchor::ALIGN_RIGHT_MIDDLE): {
        rectangle.left = 1.0f - rel_pos.x - rel_width;
        rectangle.right = 1.0f - rel_pos.x;
        rectangle.top = 0.5f - (rel_pos.y / 2.0f) + (rel_height / 2.0f);
        rectangle.bottom = rectangle.top - rel_height;
    } break;
    case (Anchor::ALIGN_RIGHT_BOTTOM): {
        rectangle.left = 1.0f - rel_pos.x - rel_width;
        rectangle.right = 1.0f - rel_pos.x;
        rectangle.top = rel_pos.y + rel_height;
        rectangle.bottom = rel_pos.y;
    } break;
    }

    rectangle.left *= viewport.x;
    rectangle.right *= viewport.x;
    rectangle.top *= viewport.y;
    rectangle.bottom *= viewport.y;

    return rectangle;
}
