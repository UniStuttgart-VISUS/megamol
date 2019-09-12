#include "stdafx.h"
#include "ScreenSpaceEffect.h"

#include <array>
#include <random>

#include <glm/glm.hpp>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::ScreenSpaceEffect::ScreenSpaceEffect()
    : core::Module()
    , m_output_texture(nullptr)
    , m_output_texture_hash(0)
    , m_mode("Mode", "Sets screen space effect mode, e.g. ssao, fxaa...")
    , m_ssao_radius("SSAO Radius", "Sets radius for SSAO")
    , m_ssao_sample_cnt("SSAO Samples", "Sets the number of samples used SSAO")
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_input_tex_slot("InputTexture", "Connects an optional input texture")
    , m_normals_tex_slot("NormalTexture", "Connects the normals render target texture")
    , m_depth_tex_slot("DepthTexture", "Connects the depth render target texture")
    , m_camera_slot("Camera", "Connects a (copy of) camera state") {
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "SSAO");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "FXAA");
    this->MakeSlotAvailable(&this->m_mode);

    this->m_ssao_sample_cnt << new megamol::core::param::IntParam(16, 0, 64);
    this->MakeSlotAvailable(&this->m_ssao_sample_cnt);

    this->m_ssao_radius << new megamol::core::param::FloatParam(0.5f, 0.0f);
    this->MakeSlotAvailable(&this->m_ssao_radius);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &ScreenSpaceEffect::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &ScreenSpaceEffect::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);

    this->m_normals_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_normals_tex_slot);

    this->m_depth_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_depth_tex_slot);

    this->m_camera_slot.SetCompatibleCall<CallCameraDescription>();
    this->MakeSlotAvailable(&this->m_camera_slot);
}

megamol::compositing::ScreenSpaceEffect::~ScreenSpaceEffect() { this->Release(); }

bool megamol::compositing::ScreenSpaceEffect::create() {
    try {
        // create shader program
        m_ssao_prgm = std::make_unique<GLSLComputeShader>();
        m_ssao_blur_prgm = std::make_unique<GLSLComputeShader>();
        m_fxaa_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_ssao_src;
        vislib::graphics::gl::ShaderSource compute_ssao_blur_src;
        vislib::graphics::gl::ShaderSource compute_fxaa_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::ssao", compute_ssao_src)) return false;
        if (!m_ssao_prgm->Compile(compute_ssao_src.Code(), compute_ssao_src.Count())) return false;
        if (!m_ssao_prgm->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::blur", compute_ssao_blur_src))
            return false;
        if (!m_ssao_blur_prgm->Compile(compute_ssao_blur_src.Code(), compute_ssao_blur_src.Count())) return false;
        if (!m_ssao_blur_prgm->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::fxaa", compute_fxaa_src)) return false;
        if (!m_fxaa_prgm->Compile(compute_fxaa_src.Code(), compute_fxaa_src.Count())) return false;
        if (!m_fxaa_prgm->Link()) return false;

    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }

    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_intermediate_texture = std::make_shared<glowl::Texture2D>("screenspace_effect_intermediate", tx_layout, nullptr);

    // quick 'n dirty from https://learnopengl.com/Advanced-Lighting/SSAO
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between 0.0 - 1.0
    std::default_random_engine generator;

    std::vector<float> ssaoKernel;
    for (unsigned int i = 0; i < 64; ++i) {
        glm::vec3 sample(
            randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
        sample = glm::normalize(sample);
        sample *= randomFloats(generator);
        float scale = (float)i / 64.0;
        ssaoKernel.push_back(sample.x);
        ssaoKernel.push_back(sample.y);
        ssaoKernel.push_back(sample.z);
    }

    m_ssao_samples = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, ssaoKernel, GL_DYNAMIC_DRAW);

    std::vector<glm::vec3> ssaoNoise;
    for (unsigned int i = 0; i < 16; i++) {
        glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f);
        ssaoNoise.push_back(noise);
    }

    glowl::TextureLayout tx_layout2(GL_RGB32F, 4, 4, 1, GL_RGB, GL_FLOAT, 1);
    m_ssao_kernelRot_texture = std::make_shared<glowl::Texture2D>("ssao_kernel_rotation", tx_layout2, ssaoNoise.data());

    return true;
}

void megamol::compositing::ScreenSpaceEffect::release() {}

bool megamol::compositing::ScreenSpaceEffect::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_input = m_input_tex_slot.CallAs<CallTexture2D>();
    auto call_normal = m_normals_tex_slot.CallAs<CallTexture2D>();
    auto call_depth = m_depth_tex_slot.CallAs<CallTexture2D>();
    auto call_camera = m_camera_slot.CallAs<CallCamera>();

    if (lhs_tc == NULL) return false;

    // TODO check/update data hash

    if (lhs_tc->getData() == nullptr) {
        lhs_tc->setData(m_output_texture);
    }

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


    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        m_ssao_radius.Param<core::param::FloatParam>()->SetGUIVisible(true);
        m_ssao_sample_cnt.Param<core::param::IntParam>()->SetGUIVisible(true);

        if (call_normal == NULL) return false;
        if (call_depth == NULL) return false;
        if (call_camera == NULL) return false;

        if (!(*call_normal)(0)) return false;
        if (!(*call_depth)(0)) return false;
        if (!(*call_camera)(0)) return false;

        auto normal_tx2D = call_normal->getData();
        auto depth_tx2D = call_depth->getData();

        setupOutputTexture(normal_tx2D, m_intermediate_texture);
        setupOutputTexture(normal_tx2D, m_output_texture);

        // obtain camera information
        core::view::Camera_2 cam = call_camera->getData();
        cam_type::snapshot_type snapshot;
        cam_type::matrix_type view_tmp, proj_tmp;
        cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
        glm::mat4 view_mx = view_tmp;
        glm::mat4 proj_mx = proj_tmp;

        m_ssao_prgm->Enable();

        m_ssao_samples->bind(1);

        glUniform1f(m_ssao_prgm->ParameterLocation("radius"), m_ssao_radius.Param<core::param::FloatParam>()->Value());
        glUniform1i(m_ssao_prgm->ParameterLocation("sample_cnt"), m_ssao_sample_cnt.Param<core::param::IntParam>()->Value());

        glActiveTexture(GL_TEXTURE0);
        normal_tx2D->bindTexture();
        glUniform1i(m_ssao_prgm->ParameterLocation("normal_tx2D"), 0);
        glActiveTexture(GL_TEXTURE1);
        depth_tx2D->bindTexture();
        glUniform1i(m_ssao_prgm->ParameterLocation("depth_tx2D"), 1);
        glActiveTexture(GL_TEXTURE2);
        m_ssao_kernelRot_texture->bindTexture();
        glUniform1i(m_ssao_prgm->ParameterLocation("noise_tx2D"), 2);

        auto inv_view_mx = glm::inverse(view_mx);
        auto inv_proj_mx = glm::inverse(proj_mx);
        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(inv_view_mx));
        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(inv_proj_mx));

        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(view_mx));
        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(proj_mx));

        m_intermediate_texture->bindImage(0, GL_WRITE_ONLY);

        m_ssao_prgm->Dispatch(static_cast<int>(std::ceil(m_output_texture->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(m_output_texture->getHeight() / 8.0f)), 1);

        m_ssao_prgm->Disable();

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        m_ssao_blur_prgm->Enable();

        glActiveTexture(GL_TEXTURE0);
        m_intermediate_texture->bindTexture();
        glUniform1i(m_ssao_prgm->ParameterLocation("src_tx2D"), 0);

        m_output_texture->bindImage(0, GL_WRITE_ONLY);

        m_ssao_blur_prgm->Dispatch(static_cast<int>(std::ceil(m_output_texture->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(m_output_texture->getHeight() / 8.0f)), 1);

        m_ssao_blur_prgm->Disable();

    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        m_ssao_radius.Param<core::param::FloatParam>()->SetGUIVisible(false);
        m_ssao_sample_cnt.Param<core::param::IntParam>()->SetGUIVisible(false);

        if (call_input == NULL) return false;
        if (!(*call_input)(0)) return false;

        auto input_tx2D = call_input->getData();

        setupOutputTexture(input_tx2D, m_output_texture);

        m_fxaa_prgm->Enable();

        glActiveTexture(GL_TEXTURE0);
        input_tx2D->bindTexture();
        glUniform1i(m_fxaa_prgm->ParameterLocation("src_tx2D"), 0);

        m_output_texture->bindImage(0, GL_WRITE_ONLY);

        m_fxaa_prgm->Dispatch(static_cast<int>(std::ceil(m_output_texture->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(m_output_texture->getHeight() / 8.0f)), 1);

        m_fxaa_prgm->Disable();
    }

    return true;
}

bool megamol::compositing::ScreenSpaceEffect::getMetaDataCallback(core::Call& caller) { return true; }
