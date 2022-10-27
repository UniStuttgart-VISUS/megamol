#include "LocalLighting.h"

#include <iostream>

#include <glm/ext.hpp>

#include "compositing_gl/CompositingCalls.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/light/CallLight.h"
#include "mmstd/light/DistantLight.h"
#include "mmstd/light/PointLight.h"
#include "mmstd/light/TriDirectionalLighting.h"

using megamol::core::utility::log::Log;

megamol::compositing::LocalLighting::LocalLighting()
        : core::Module()
        , m_version(0)
        , m_output_texture(nullptr)
        , m_point_lights_buffer(nullptr)
        , m_distant_lights_buffer(nullptr)

        , m_illuminationmode("IlluminationMode", "Sets illumination mode e.g. Lambertian, Phong")

        , m_phong_ambientColor("AmbientColor", "Sets the ambient Color for Blinn-Phong")
        , m_phong_diffuseColor("DiffuseColor", "Sets the diffuse Color for Blinn-Phong")
        , m_phong_specularColor("SpecularColor", "Sets the specular Color for Blinn-Phong")

        , m_phong_k_ambient("AmbientFactor", "Sets the ambient factor for Blinn-Phong")
        , m_phong_k_diffuse("DiffuseFactor", "Sets the diffuse factor for Blinn-Phong")
        , m_phong_k_specular("SpecularFactor", "Sets the specular factor for Blinn-Phong")
        , m_phong_k_exp("ExponentialFactor", "Sets the exponential factor for Blinn-Phong")

        , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
        , m_albedo_tex_slot("AlbedoTexture", "Connect to the albedo render target texture")
        , m_normal_tex_slot("NormalTexture", "Connects to the normals render target texture")
        , m_depth_tex_slot("DepthTexture", "Connects to the depth render target texture")
        , m_roughness_metalness_tex_slot(
              "RoughMetalTexture", "Connects to the roughness/metalness render target texture")
        , m_lightSlot("lights", "Lights are retrieved over this slot")
        , m_camera_slot("Camera", "Connects a (copy of) camera state") {
    this->m_illuminationmode << new megamol::core::param::EnumParam(0);
    this->m_illuminationmode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Lambert");
    this->m_illuminationmode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Blinn-Phong");
    this->MakeSlotAvailable(&this->m_illuminationmode);

    this->m_phong_ambientColor << new megamol::core::param::ColorParam(1.0, 1.0, 1.0, 1.0);
    this->MakeSlotAvailable(&this->m_phong_ambientColor);
    this->m_phong_diffuseColor << new megamol::core::param::ColorParam(1.0, 1.0, 1.0, 1.0);
    this->MakeSlotAvailable(&this->m_phong_diffuseColor);
    this->m_phong_specularColor << new megamol::core::param::ColorParam(1.0, 1.0, 1.0, 1.0);
    this->MakeSlotAvailable(&this->m_phong_specularColor);

    this->m_phong_k_ambient << new megamol::core::param::FloatParam(0.2f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->m_phong_k_ambient);

    this->m_phong_k_diffuse << new megamol::core::param::FloatParam(0.7f, 0.0, 1.0f);
    this->MakeSlotAvailable(&this->m_phong_k_diffuse);

    this->m_phong_k_specular << new megamol::core::param::FloatParam(0.1f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->m_phong_k_specular);

    this->m_phong_k_exp << new megamol::core::param::FloatParam(120.0f, 0.0f, 1000.0f);
    this->MakeSlotAvailable(&this->m_phong_k_exp);

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

megamol::compositing::LocalLighting::~LocalLighting() {
    this->Release();
}

bool megamol::compositing::LocalLighting::create() {

    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        m_lambert_prgm =
            core::utility::make_glowl_shader("compositing_lambert", shader_options, "compositing_gl/lambert.comp.glsl");

        m_phong_prgm =
            core::utility::make_glowl_shader("compositing_phong", shader_options, "compositing_gl/phong.comp.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("LocalLighting: " + std::string(e.what())).c_str());
        return false;
    }

    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("lighting_output", tx_layout, nullptr);

    m_point_lights_buffer =
        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    m_distant_lights_buffer =
        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

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

    bool all_calls_valid = (call_albedo != nullptr) && (call_normal != nullptr) && (call_depth != nullptr) &&
                           (call_camera != nullptr) && (call_light != nullptr);

    if (all_calls_valid) {
        // something has changed in the neath...
        bool something_has_changed =
            call_albedo->hasUpdate() || call_normal->hasUpdate() || call_depth->hasUpdate() || call_camera->hasUpdate();

        if (something_has_changed) {
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
            core::view::Camera cam = call_camera->getData();
            auto cam_pose = cam.get<core::view::Camera::Pose>();
            auto view_mx = cam.getViewMatrix();
            auto proj_mx = cam.getProjectionMatrix();

            if (call_light->hasUpdate()) {
                auto lights = call_light->getData();

                this->m_point_lights.clear();
                this->m_distant_lights.clear();

                auto point_lights = lights.get<core::view::light::PointLightType>();
                auto distant_lights = lights.get<core::view::light::DistantLightType>();
                auto tridirection_lights = lights.get<core::view::light::TriDirectionalLightType>();

                for (auto pl : point_lights) {
                    m_point_lights.push_back({pl.position[0], pl.position[1], pl.position[2], pl.intensity});
                }

                for (auto dl : distant_lights) {
                    if (dl.eye_direction) {
                        auto cam_dir = glm::normalize(cam_pose.direction);
                        m_distant_lights.push_back({cam_dir.x, cam_dir.y, cam_dir.z, dl.intensity});
                    } else {
                        m_distant_lights.push_back({dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
                    }
                }

                for (auto tdl : tridirection_lights) {
                    if (tdl.in_view_space) {
                        auto inverse_view = glm::transpose(glm::mat3(view_mx));
                        auto key_dir =
                            inverse_view * glm::vec3(tdl.key_direction[0], tdl.key_direction[1], tdl.key_direction[2]);
                        auto fill_dir = inverse_view *
                                        glm::vec3(tdl.fill_direction[0], tdl.fill_direction[1], tdl.fill_direction[2]);
                        auto back_dir = inverse_view *
                                        glm::vec3(tdl.back_direction[0], tdl.back_direction[1], tdl.back_direction[2]);
                        m_distant_lights.push_back({key_dir[0], key_dir[1], key_dir[2], tdl.intensity});
                        m_distant_lights.push_back({fill_dir[0], fill_dir[1], fill_dir[2], tdl.intensity});
                        m_distant_lights.push_back({back_dir[0], back_dir[1], back_dir[2], tdl.intensity});
                    } else {
                        m_distant_lights.push_back(
                            {tdl.key_direction[0], tdl.key_direction[1], tdl.key_direction[2], tdl.intensity});
                        m_distant_lights.push_back(
                            {tdl.fill_direction[0], tdl.fill_direction[1], tdl.fill_direction[2], tdl.intensity});
                        m_distant_lights.push_back(
                            {tdl.back_direction[0], tdl.back_direction[1], tdl.back_direction[2], tdl.intensity});
                    }
                }
            }
            m_point_lights_buffer->rebuffer(m_point_lights);
            m_distant_lights_buffer->rebuffer(m_distant_lights);

            // m_illumination mode: Change between Lambert & Blinn Phong
            if (this->m_illuminationmode.Param<core::param::EnumParam>()->Value() == 0) {
                // Lambert: std::cout << "Lambert" << std::endl;
                m_phong_ambientColor.Param<core::param::ColorParam>()->SetGUIVisible(false);
                m_phong_diffuseColor.Param<core::param::ColorParam>()->SetGUIVisible(false);
                m_phong_specularColor.Param<core::param::ColorParam>()->SetGUIVisible(false);

                m_phong_k_ambient.Param<core::param::FloatParam>()->SetGUIVisible(false);
                m_phong_k_diffuse.Param<core::param::FloatParam>()->SetGUIVisible(false);
                m_phong_k_specular.Param<core::param::FloatParam>()->SetGUIVisible(false);
                m_phong_k_exp.Param<core::param::FloatParam>()->SetGUIVisible(false);

                if (m_lambert_prgm != nullptr && m_point_lights_buffer != nullptr &&
                    m_distant_lights_buffer != nullptr) {
                    m_lambert_prgm->use();

                    m_point_lights_buffer->bind(1);
                    glUniform1i(m_lambert_prgm->getUniformLocation("point_light_cnt"),
                        static_cast<GLint>(m_point_lights.size()));
                    m_distant_lights_buffer->bind(2);
                    glUniform1i(m_lambert_prgm->getUniformLocation("distant_light_cnt"),
                        static_cast<GLint>(m_distant_lights.size()));
                    glActiveTexture(GL_TEXTURE0);
                    albedo_tx2D->bindTexture();
                    glUniform1i(m_lambert_prgm->getUniformLocation("albedo_tx2D"), 0);

                    glActiveTexture(GL_TEXTURE1);
                    normal_tx2D->bindTexture();
                    glUniform1i(m_lambert_prgm->getUniformLocation("normal_tx2D"), 1);

                    glActiveTexture(GL_TEXTURE2);
                    depth_tx2D->bindTexture();
                    glUniform1i(m_lambert_prgm->getUniformLocation("depth_tx2D"), 2);

                    auto inv_view_mx = glm::inverse(view_mx);
                    auto inv_proj_mx = glm::inverse(proj_mx);
                    glUniformMatrix4fv(
                        m_lambert_prgm->getUniformLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(inv_view_mx));
                    glUniformMatrix4fv(
                        m_lambert_prgm->getUniformLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(inv_proj_mx));

                    m_output_texture->bindImage(0, GL_WRITE_ONLY);

                    glDispatchCompute(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
                        static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

                    glUseProgram(0);
                }
            } else if (this->m_illuminationmode.Param<core::param::EnumParam>()->Value() == 1) {
                // Blinn-Phong: std::cout << "Blinn Phong" << std::endl;
                // ambient/diffus/specular anhand von Lichtern & keine auswahl mehr todo
                m_phong_ambientColor.Param<core::param::ColorParam>()->SetGUIVisible(true);
                m_phong_diffuseColor.Param<core::param::ColorParam>()->SetGUIVisible(true);
                m_phong_specularColor.Param<core::param::ColorParam>()->SetGUIVisible(true);

                m_phong_k_ambient.Param<core::param::FloatParam>()->SetGUIVisible(true);
                m_phong_k_diffuse.Param<core::param::FloatParam>()->SetGUIVisible(true);
                m_phong_k_specular.Param<core::param::FloatParam>()->SetGUIVisible(true);
                m_phong_k_exp.Param<core::param::FloatParam>()->SetGUIVisible(true);

                if (m_phong_prgm != nullptr && m_point_lights_buffer != nullptr && m_distant_lights_buffer != nullptr) {
                    m_phong_prgm->use();

                    // Phong Parameter to Shader
                    glUniform4fv(m_phong_prgm->getUniformLocation("ambientColor"), 1,
                        m_phong_ambientColor.Param<core::param::ColorParam>()->Value().data());
                    glUniform4fv(m_phong_prgm->getUniformLocation("diffuseColor"), 1,
                        m_phong_diffuseColor.Param<core::param::ColorParam>()->Value().data());
                    glUniform4fv(m_phong_prgm->getUniformLocation("specularColor"), 1,
                        m_phong_specularColor.Param<core::param::ColorParam>()->Value().data());

                    glUniform1f(m_phong_prgm->getUniformLocation("k_amb"),
                        m_phong_k_ambient.Param<core::param::FloatParam>()->Value());
                    glUniform1f(m_phong_prgm->getUniformLocation("k_diff"),
                        m_phong_k_diffuse.Param<core::param::FloatParam>()->Value());
                    glUniform1f(m_phong_prgm->getUniformLocation("k_spec"),
                        m_phong_k_specular.Param<core::param::FloatParam>()->Value());
                    glUniform1f(m_phong_prgm->getUniformLocation("k_exp"),
                        m_phong_k_exp.Param<core::param::FloatParam>()->Value());

                    // Cameraposition
                    glUniform3fv(m_phong_prgm->getUniformLocation("camPos"), 1, glm::value_ptr(cam_pose.position));

                    m_point_lights_buffer->bind(1);
                    glUniform1i(
                        m_phong_prgm->getUniformLocation("point_light_cnt"), static_cast<GLint>(m_point_lights.size()));
                    m_distant_lights_buffer->bind(2);
                    glUniform1i(m_phong_prgm->getUniformLocation("distant_light_cnt"),
                        static_cast<GLint>(m_distant_lights.size()));
                    glActiveTexture(GL_TEXTURE0);
                    albedo_tx2D->bindTexture();
                    glUniform1i(m_phong_prgm->getUniformLocation("albedo_tx2D"), 0);

                    glActiveTexture(GL_TEXTURE1);
                    normal_tx2D->bindTexture();
                    glUniform1i(m_phong_prgm->getUniformLocation("normal_tx2D"), 1);

                    glActiveTexture(GL_TEXTURE2);
                    depth_tx2D->bindTexture();
                    glUniform1i(m_phong_prgm->getUniformLocation("depth_tx2D"), 2);

                    auto inv_view_mx = glm::inverse(view_mx);
                    auto inv_proj_mx = glm::inverse(proj_mx);
                    glUniformMatrix4fv(
                        m_phong_prgm->getUniformLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(inv_view_mx));
                    glUniformMatrix4fv(
                        m_phong_prgm->getUniformLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(inv_proj_mx));

                    m_output_texture->bindImage(0, GL_WRITE_ONLY);

                    glDispatchCompute(static_cast<int>(std::ceil(std::get<0>(texture_res) / 8.0f)),
                        static_cast<int>(std::ceil(std::get<1>(texture_res) / 8.0f)), 1);

                    glUseProgram(0);
                }
            }
        }
    }

    lhs_tc->setData(m_output_texture, m_version);

    return true;
}

bool megamol::compositing::LocalLighting::getMetaDataCallback(core::Call& caller) {
    return true;
}
