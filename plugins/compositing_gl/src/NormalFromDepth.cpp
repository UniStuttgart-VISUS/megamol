#include "NormalFromDepth.h"

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/CoreInstance.h"
#include "mmcore_gl/utility/ShaderFactory.h"

using megamol::core::utility::log::Log;

megamol::compositing::NormalFromDepth::NormalFromDepth()
        : m_version(0)
        , m_output_texture(nullptr)
        , m_output_tex_slot("NormalTexture", "Gives access to resulting output normal texture")
        , m_input_tex_slot("DepthTexture", "Connects the depth input texture")
        , m_camera_slot("Camera", "Connects a (copy of) camera state") {
    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &NormalFromDepth::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &NormalFromDepth::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);

    this->m_camera_slot.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->m_camera_slot);
}

megamol::compositing::NormalFromDepth::~NormalFromDepth() {
    this->Release();
}

bool megamol::compositing::NormalFromDepth::create() {
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        m_normal_from_depth_prgm = core::utility::make_glowl_shader(
            "Compositing_normalFromDepth", shader_options, "compositing_gl/normalFromDepth.comp.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("NormalFromDepth: " + std::string(e.what())).c_str());
        return false;
    }

    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("normal_from_depth_output", tx_layout, nullptr);

    return true;
}

void megamol::compositing::NormalFromDepth::release() {}

bool megamol::compositing::NormalFromDepth::getDataCallback(core::Call& caller) {

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

        std::function<void(std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt)>
            setupOutputTexture = [](std::shared_ptr<glowl::Texture2D> src, std::shared_ptr<glowl::Texture2D> tgt) {
                // set output texture size to primary input texture
                std::array<float, 2> texture_res = {
                    static_cast<float>(src->getWidth()), static_cast<float>(src->getHeight())};

                if (tgt->getWidth() != std::get<0>(texture_res) || tgt->getHeight() != std::get<1>(texture_res)) {
                    glowl::TextureLayout tx_layout(
                        GL_RGBA16F, std::get<0>(texture_res), std::get<1>(texture_res), 1, GL_RGBA, GL_HALF_FLOAT, 1);
                    tgt->reload(tx_layout, nullptr);
                }
            };

        if (call_input == NULL) {
            return false;
        }
        auto input_tx2D = call_input->getData();

        setupOutputTexture(input_tx2D, m_output_texture);

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

bool megamol::compositing::NormalFromDepth::getMetaDataCallback(core::Call& caller) {


    return true;
}
