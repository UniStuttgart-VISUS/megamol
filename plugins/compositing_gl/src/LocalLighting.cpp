#include "stdafx.h"
#include "LocalLighting.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/view/light/CallLight.h"
#include "mmcore/view/light/PointLight.h"
#include "mmcore/view/light/DistantLight.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::LocalLighting::LocalLighting() 
    : core::Module()
    , m_version(0)
    , m_output_texture(nullptr)
    , m_point_lights_buffer(nullptr)
    , m_distant_lights_buffer(nullptr)
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_albedo_tex_slot("AlbedoTexture", "Connect to the albedo render target texture")
    , m_normal_tex_slot("NormalTexture", "Connects to the normals render target texture")
    , m_depth_tex_slot("DepthTexture", "Connects to the depth render target texture")
    , m_roughness_metalness_tex_slot("RoughMetalTexture","Connects to the roughness/metalness render target texture")
    , m_lightSlot("lights", "Lights are retrieved over this slot") 
    , m_camera_slot("Camera", "Connects a (copy of) camera state")
{
    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &LocalLighting::getDataCallback);
    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetMetaData", &LocalLighting::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_albedo_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_albedo_tex_slot);

    this->m_normal_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_normal_tex_slot);

    this->m_depth_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_slot);

    this->m_roughness_metalness_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_roughness_metalness_tex_slot);

    this->m_lightSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->m_lightSlot);

    this->m_camera_slot.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->m_camera_slot);
}

megamol::compositing::LocalLighting::~LocalLighting() { this->Release(); }

bool megamol::compositing::LocalLighting::create() {

    try {
        // create shader program
        m_lighting_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::lambert", compute_src))
            return false;
        if (!m_lighting_prgm->Compile(compute_src.Code(), compute_src.Count())) return false;
        if (!m_lighting_prgm->Link()) return false;

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }

    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("lighting_output", tx_layout, nullptr);

    m_point_lights_buffer = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    m_distant_lights_buffer = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}

void megamol::compositing::LocalLighting::release() {}

bool megamol::compositing::LocalLighting::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_albedo = m_albedo_tex_slot.CallAs<CallTexture2D>();
    auto call_normal = m_normal_tex_slot.CallAs<CallTexture2D>();
    auto call_depth = m_depth_tex_slot.CallAs<CallTexture2D>();
    auto call_camera = m_camera_slot.CallAs<CallCamera>();
    auto call_light = m_lightSlot.CallAs<core::view::light::CallLight>();

    if (lhs_tc == nullptr) {
        return false;
    }
    if (call_albedo != nullptr) {
        if (!(*call_albedo)(0)) {
            return false;
        }
    }
    if (call_normal != nullptr) {
        if (!(*call_normal)(0)) {
            return false;
        }
    }
    if (call_depth != nullptr) {
        if (!(*call_depth)(0)) {
            return false;
        }
    }
    if (call_camera != nullptr) {
        if (!(*call_camera)(0)) {
            return false;
        }
    }
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }
    }

    bool all_calls_valid = (call_albedo != nullptr) && (call_normal != nullptr) &&
                           (call_depth != nullptr) && (call_camera != nullptr) && (call_light != nullptr);

    // something has changed in the neath...
    bool something_has_changed =
        call_albedo->hasUpdate() || call_normal->hasUpdate() || call_depth->hasUpdate() || call_camera->hasUpdate();

    if (something_has_changed && all_calls_valid) {
        ++m_version;

        // set output texture size to primary input texture
        auto albedo_tx2D = call_albedo->getData();
        auto normal_tx2D = call_normal->getData();
        auto depth_tx2D = call_depth->getData();
        std::array<float, 2> texture_res = {
            static_cast<float>(albedo_tx2D->getWidth()), static_cast<float>(albedo_tx2D->getHeight())};

        if (m_output_texture->getWidth() != std::get<0>(texture_res) ||
            m_output_texture->getHeight() != std::get<1>(texture_res)) {
            glowl::TextureLayout tx_layout(
                GL_RGBA16F, std::get<0>(texture_res), std::get<1>(texture_res), 1, GL_RGBA, GL_HALF_FLOAT, 1);
            m_output_texture->reload(tx_layout, nullptr);
        }

        // obtain camera information
        core::view::Camera_2 cam = call_camera->getData();
        cam_type::snapshot_type snapshot;
        cam_type::matrix_type view_tmp, proj_tmp;
        cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
        glm::mat4 view_mx = view_tmp;
        glm::mat4 proj_mx = proj_tmp;

        if (call_light->hasUpdate()) {
            auto lights = call_light->getData();

            this->m_point_lights.clear();
            this->m_distant_lights.clear();

            auto point_lights = lights.get<core::view::light::PointLightType>();
            auto distant_lights = lights.get<core::view::light::DistantLightType>();

            for(auto pl : point_lights) {
                m_point_lights.push_back(
                    {pl.position[0], pl.position[1], pl.position[2], pl.intensity});
            }

            for (auto dl : distant_lights) {
                if (dl.eye_direction) {
                    glm::vec3 cam_dir(snapshot.view_vector.x(), snapshot.view_vector.y(), snapshot.view_vector.z());
                    cam_dir = glm::normalize(cam_dir);
                    m_distant_lights.push_back({cam_dir.x, cam_dir.y, cam_dir.z, dl.intensity});
                } else {
                    m_distant_lights.push_back(
                        {dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
                }
            }
        }

        m_point_lights_buffer->rebuffer(m_point_lights);
        m_distant_lights_buffer->rebuffer(m_distant_lights);

        if (m_lighting_prgm != nullptr && m_point_lights_buffer != nullptr && m_distant_lights_buffer != nullptr) {
            m_lighting_prgm->Enable();

            m_point_lights_buffer->bind(1);
            glUniform1i(m_lighting_prgm->ParameterLocation("point_light_cnt"), static_cast<GLint>(m_point_lights.size()));
            m_distant_lights_buffer->bind(2);
            glUniform1i(
                m_lighting_prgm->ParameterLocation("distant_light_cnt"), static_cast<GLint>(m_distant_lights.size()));
            glActiveTexture(GL_TEXTURE0);
            albedo_tx2D->bindTexture();
            glUniform1i(m_lighting_prgm->ParameterLocation("albedo_tx2D"), 0);

            glActiveTexture(GL_TEXTURE1);
            normal_tx2D->bindTexture();
            glUniform1i(m_lighting_prgm->ParameterLocation("normal_tx2D"), 1);

            glActiveTexture(GL_TEXTURE2);
            depth_tx2D->bindTexture();
            glUniform1i(m_lighting_prgm->ParameterLocation("depth_tx2D"), 2);

            auto inv_view_mx = glm::inverse(view_mx);
            auto inv_proj_mx = glm::inverse(proj_mx);
            glUniformMatrix4fv(m_lighting_prgm->ParameterLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(inv_view_mx));
            glUniformMatrix4fv(m_lighting_prgm->ParameterLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(inv_proj_mx));

            m_output_texture->bindImage(0, GL_WRITE_ONLY);

            m_lighting_prgm->Dispatch(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
                static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

            m_lighting_prgm->Disable();
        }
    }

    lhs_tc->setData(m_output_texture, m_version);

    return true; 
}

bool megamol::compositing::LocalLighting::getMetaDataCallback(core::Call& caller) { return true; }
