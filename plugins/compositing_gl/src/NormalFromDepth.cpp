#include "NormalFromDepth.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore/param/EnumParam.h"

using megamol::core::utility::log::Log;

megamol::compositing_gl::NormalFromDepth::NormalFromDepth()
        : mmstd_gl::ModuleGL()
        , m_version(0)
        , m_output_texture(nullptr)
        , m_output_tex_slot("NormalTexture", "Gives access to resulting output normal texture")
        , m_input_tex_slot("DepthTexture", "Connects the depth input texture")
        , m_camera_slot("Camera", "Connects a (copy of) camera state")
        , out_texture_format_slot_("OutTexFormat", "texture format of output texture") {
    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &NormalFromDepth::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &NormalFromDepth::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);

    this->m_camera_slot.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->m_camera_slot);

    auto out_tex_formats = new megamol::core::param::EnumParam(0);
    out_tex_formats->SetTypePair(0, "RGBA_32F");
    out_tex_formats->SetTypePair(1, "RGBA_16F");
    out_tex_formats->SetTypePair(2, "RGBA_8UI");

    this->out_texture_format_slot_.SetParameter(out_tex_formats);
    this->out_texture_format_slot_.SetUpdateCallback(&megamol::compositing_gl::NormalFromDepth::setTextureFormatCallback);
    this->MakeSlotAvailable(&this->out_texture_format_slot_);
}

megamol::compositing_gl::NormalFromDepth::~NormalFromDepth() {
    this->Release();
}

bool megamol::compositing_gl::NormalFromDepth::create() {
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    auto shader_options_flags = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);
    if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 0) {
        shader_options_flags->addDefinition("OUT32F");
    } else if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 1) {
        shader_options_flags->addDefinition("OUT16HF");
    } else if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 2) {
        shader_options_flags->addDefinition("OUT8NB");
    }
    try {
        m_normal_from_depth_prgm = core::utility::make_glowl_shader(
            "Compositing_normalFromDepth", *shader_options_flags, "compositing_gl/normalFromDepth.comp.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("NormalFromDepth: " + std::string(e.what())).c_str());
        return false;
    }

    glowl::TextureLayout tx_layout(out_tex_internal_format_, 1, 1, 1, out_tex_format_, out_tex_type_, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("normal_from_depth_output", tx_layout, nullptr);

    return true;
}

void megamol::compositing_gl::NormalFromDepth::release() {}

bool megamol::compositing_gl::NormalFromDepth::getDataCallback(core::Call& caller) {

    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_input = m_input_tex_slot.CallAs<CallTexture2D>();
    auto call_camera = m_camera_slot.CallAs<CallCamera>();

    if (lhs_tc == NULL)
        return false;

    if (call_input != NULL) {
        if (!(*call_input)(0))
            return false;
    }
    if (call_camera != NULL) {
        if (!(*call_camera)(0))
            return false;
    }

    // something has changed in the neath...
    bool something_has_changed = (call_input != NULL ? call_input->hasUpdate() : false) ||
                                 (call_camera != NULL ? call_camera->hasUpdate() : false);

    if (something_has_changed) {
        ++m_version;

        std::function<void(std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt,
            GLenum out_tex_internal_format, GLenum out_tex_format, GLenum out_tex_type)>
            setupOutputTexture = [](std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt,
                                     GLenum out_tex_internal_format, GLenum out_tex_format, GLenum out_tex_type) {
                // set output texture size to primary input texture
                std::array<float, 2> texture_res = {
                    static_cast<float>(src->getWidth()), static_cast<float>(src->getHeight())};

                if (tgt->getWidth() != std::get<0>(texture_res) || tgt->getHeight() != std::get<1>(texture_res)) {
                    glowl::TextureLayout tx_layout(
                        out_tex_internal_format, std::get<0>(texture_res), std::get<1>(texture_res), 1, out_tex_format, out_tex_type, 1);
                    tgt->reload(tx_layout, nullptr);
                }
            };

        if (call_input == NULL) {
            return false;
        }
        auto input_tx2D = call_input->getData();

        setupOutputTexture(input_tx2D, m_output_texture, out_tex_internal_format_, out_tex_format_, out_tex_type_);

        // obtain camera information
        core::view::Camera cam = call_camera->getData();
        glm::mat4 view_mx = cam.getViewMatrix();
        glm::mat4 proj_mx = cam.getProjectionMatrix();

        m_normal_from_depth_prgm->use();

        auto inv_view_mx = glm::inverse(view_mx);
        auto inv_proj_mx = glm::inverse(proj_mx);
        m_normal_from_depth_prgm->setUniform("inv_view_mx", inv_view_mx);
        m_normal_from_depth_prgm->setUniform("inv_proj_mx", inv_proj_mx);

        glActiveTexture(GL_TEXTURE0);
        input_tx2D->bindTexture();
        m_normal_from_depth_prgm->setUniform("src_tx2D", 0);

        m_output_texture->bindImage(0, GL_WRITE_ONLY);

        glDispatchCompute(static_cast<int>(std::ceil(m_output_texture->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(m_output_texture->getHeight() / 8.0f)), 1);

        glUseProgram(0);
    }

    if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_output_texture, m_version);
    }

    return true;
}

bool megamol::compositing_gl::NormalFromDepth::getMetaDataCallback(core::Call& caller) {


    return true;
}

bool megamol::compositing_gl::NormalFromDepth::setTextureFormatCallback(core::param::ParamSlot& slot) {
    switch (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value()) {
    case 0: //RGBA32F
        out_tex_internal_format_ = GL_RGBA32F;
        out_tex_format_ = GL_RGB;
        out_tex_type_ = GL_FLOAT;
        break;
    case 1: //RGBA16F
        out_tex_internal_format_ = GL_RGBA16F;
        out_tex_format_ = GL_RGBA;
        out_tex_type_ = GL_HALF_FLOAT;
        break;
    case 2: //RGBA8UI
        out_tex_internal_format_ = GL_RGBA8_SNORM;
        out_tex_format_ = GL_RGBA;
        out_tex_type_ = GL_UNSIGNED_BYTE;
        break;
    }
    // reinit all textures
    glowl::TextureLayout tx_layout(out_tex_internal_format_, 1, 1, 1, out_tex_format_, out_tex_type_, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);

    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    auto shader_options_flags = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shader_options);
    if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 0) {
        shader_options_flags->addDefinition("OUT32F");
    } else if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 1) {
        shader_options_flags->addDefinition("OUT16HF");
    } else if (this->out_texture_format_slot_.Param<core::param::EnumParam>()->Value() == 2) {
        shader_options_flags->addDefinition("OUT8NB");
    }

    try {
        m_normal_from_depth_prgm = core::utility::make_glowl_shader(
            "Compositing_normalFromDepth", *shader_options_flags, "compositing_gl/normalFromDepth.comp.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("NormalFromDepth: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}
