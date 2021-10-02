#include "stdafx.h"
#include "AntiAliasing.h"

#include <array>
#include <random>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"

#include "SMAAAreaTex.h"
#include "SMAASearchTex.h"

megamol::compositing::AntiAliasing::AntiAliasing() : core::Module()
    , m_version(0)
    , m_output_texture(nullptr)
    , m_output_texture_hash(0)
    , m_mode("Mode", "Sets antialiasing technqiue, e.g. smaa, fxaa, no aa")
    , m_smaa_quality("QualityLevel", "Sets smaa quality level")
    , m_smaa_detection_technique("EdgeDetection", "Sets smaa edge detection base: luma, color, or depth")
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_input_tex_slot("InputTexture", "Connects an optional input texture")
{
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "SMAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "FXAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "No AA");
    this->m_mode.SetUpdateCallback(&megamol::compositing::AntiAliasing::setSettingsCallback);
    this->MakeSlotAvailable(&this->m_mode);

    this->m_smaa_quality << new megamol::core::param::EnumParam(2);
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "LOW");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "MEDIUM");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "HIGH");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "ULTRA");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(4, "CUSTOM");
    this->m_smaa_quality.SetUpdateCallback(&megamol::compositing::AntiAliasing::visibilityCallback);
    this->MakeSlotAvailable(&this->m_smaa_quality);

    this->m_smaa_detection_technique << new megamol::core::param::EnumParam(0);
    this->m_smaa_detection_technique.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "LUMA");
    this->m_smaa_detection_technique.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "COLOR");
    this->m_smaa_detection_technique.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "DEPTH");
    this->MakeSlotAvailable(&this->m_smaa_detection_technique);

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &AntiAliasing::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &AntiAliasing::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_slot);
}

megamol::compositing::AntiAliasing::~AntiAliasing() { this->Release(); }

bool megamol::compositing::AntiAliasing::create() {
    try {
        // create shader program
        m_fxaa_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_edge_detection_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_blending_weight_calculation_prgm = std::make_unique<GLSLComputeShader>();
        m_smaa_neighborhood_blending_prgm = std::make_unique<GLSLComputeShader>();

        vislib::graphics::gl::ShaderSource compute_fxaa_src;
        vislib::graphics::gl::ShaderSource compute_smaa_edge_detection_src;
        vislib::graphics::gl::ShaderSource compute_smaa_blending_weights_src;
        vislib::graphics::gl::ShaderSource compute_smaa_neighborhood_blending_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::fxaa", compute_fxaa_src))
            return false;
        if (!m_fxaa_prgm->Compile(compute_fxaa_src.Code(), compute_fxaa_src.Count()))
            return false;
        if (!m_fxaa_prgm->Link())
            return false;

        std::cout << "check0 \n";

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::edgeDetectionCS", compute_smaa_edge_detection_src))
            return false;
        if (!m_smaa_edge_detection_prgm->Compile(compute_smaa_edge_detection_src.Code(), compute_smaa_edge_detection_src.Count()))
            return false;
        if (!m_smaa_edge_detection_prgm->Link())
            return false;
        std::cout << "check1 \n";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::blendingWeightsCalculationCS", compute_smaa_blending_weights_src))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Compile(compute_smaa_blending_weights_src.Code(), compute_smaa_blending_weights_src.Count()))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Link())
            return false;
        std::cout << "check2 \n";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaa::neighborhoodBlendingCS", compute_smaa_neighborhood_blending_src))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Compile(compute_smaa_neighborhood_blending_src.Code(), compute_smaa_neighborhood_blending_src.Count()))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Link())
            return false;
        std::cout << "check3 \n";
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


    // TODO: we are using 64 bit colors here, see the note in the shader instructions about point sampling
    glowl::TextureLayout tx_layout(GL_RGBA16F, 1, 1, 1, GL_RGBA, GL_HALF_FLOAT, 1);
    m_output_texture = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);

    // texture for smaa
    std::vector<std::pair<GLenum, GLint>> int_params = {
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_LINEAR} };
    m_smaa_layout = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1, int_params, {});
    m_edges_tex = std::make_shared<glowl::Texture2D>("smaa_edges_tex", m_smaa_layout, nullptr);
    m_blend_tex = std::make_shared<glowl::Texture2D>("smaa_blend_tex", m_smaa_layout, nullptr);

    glowl::TextureLayout area_layout(GL_RG8, AREATEX_WIDTH, AREATEX_HEIGHT, 1, GL_RG, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout search_layout(GL_R8, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, GL_RED, GL_UNSIGNED_BYTE, 1, int_params, {});
    m_area_tex = std::make_shared<glowl::Texture2D>("smaa_area_tex", area_layout, areaTexBytes);
    m_search_tex = std::make_shared<glowl::Texture2D>("smaa_search_tex", search_layout, searchTexBytes);

    m_ssbo_constants = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}

void megamol::compositing::AntiAliasing::release() {}

bool megamol::compositing::AntiAliasing::setSettingsCallback(core::param::ParamSlot& slot) {
    // low
    if (slot.Param<core::param::EnumParam>()->Value() == 0) {
        m_smaa_constants.Smaa_threshold = 0.15f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 4;
        m_smaa_constants.Max_search_steps_diag = 8;
        m_smaa_constants.Disable_diag_detection = true;
        m_smaa_constants.Disable_corner_detection = true;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }
    // medium
    else if (slot.Param<core::param::EnumParam>()->Value() == 1) {
        m_smaa_constants.Smaa_threshold = 0.1f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 8;
        m_smaa_constants.Max_search_steps_diag = 8;
        m_smaa_constants.Disable_diag_detection = true;
        m_smaa_constants.Disable_corner_detection = true;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }
    // high
    else if (slot.Param<core::param::EnumParam>()->Value() == 2) {
        m_smaa_constants.Smaa_threshold = 0.1f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 16;
        m_smaa_constants.Max_search_steps_diag = 8;
        m_smaa_constants.Disable_diag_detection = false;
        m_smaa_constants.Disable_corner_detection = false;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }
    // ultra
    else if (slot.Param<core::param::EnumParam>()->Value() == 3) {
        m_smaa_constants.Smaa_threshold = 0.05f;
        m_smaa_constants.Smaa_depth_threshold = 0.1f * m_smaa_constants.Smaa_threshold;
        m_smaa_constants.Max_search_steps = 32;
        m_smaa_constants.Max_search_steps_diag = 16;
        m_smaa_constants.Disable_diag_detection = false;
        m_smaa_constants.Disable_corner_detection = false;
        m_smaa_constants.Corner_rounding = 25;
        m_smaa_constants.Corner_rounding_norm = m_smaa_constants.Corner_rounding / 100.f;
    }

    return true;
}

bool megamol::compositing::AntiAliasing::visibilityCallback(core::param::ParamSlot& slot) {
    if (slot.Param<core::param::EnumParam>()->Value() == 0) {
        m_smaa_quality.Param<core::param::EnumParam>()->SetGUIVisible(true);
        m_smaa_detection_technique.Param<core::param::EnumParam>()->SetGUIVisible(true);
    } else {
        m_smaa_quality.Param<core::param::EnumParam>()->SetGUIVisible(false);
        m_smaa_detection_technique.Param<core::param::EnumParam>()->SetGUIVisible(false);
    }

    return true;
}

void megamol::compositing::AntiAliasing::launchProgram(
    const std::unique_ptr<GLSLComputeShader>& prgm,
    std::shared_ptr<glowl::Texture2D> input,
    const char* uniform_id,
    std::shared_ptr<glowl::Texture2D> output) {
    prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(prgm->ParameterLocation(uniform_id), 0);

    output->bindImage(0, GL_WRITE_ONLY);

    prgm->Dispatch(static_cast<int>(std::ceil(output->getWidth() / 8.0f)),
        static_cast<int>(std::ceil(output->getHeight() / 8.0f)), 1);

    prgm->Disable();
}

bool megamol::compositing::AntiAliasing::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_input = m_input_tex_slot.CallAs<CallTexture2D>();

    if (lhs_tc == NULL) return false;
    
    if(call_input != NULL) { if (!(*call_input)(0)) return false; }

    bool something_has_changed =
        (call_input != NULL ? call_input->hasUpdate() : false);

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

        auto input_tx2D = call_input->getData();
        setupOutputTexture(input_tx2D, m_output_texture);

        // smaa
        if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
            int input_width = input_tx2D->getWidth();
            int input_height = input_tx2D->getHeight();

            // init textures and clear them
            m_smaa_layout.width = input_width;
            m_smaa_layout.height = input_height;
            m_edges_tex->reload(m_smaa_layout, nullptr);
            m_blend_tex->reload(m_smaa_layout, nullptr);
            GLubyte col[4] = { 0, 0, 0, 0 };
            m_edges_tex->clearTexImage(col);
            m_blend_tex->clearTexImage(col);

            m_smaa_constants.Rt_metrics = glm::vec4(
                1.f / (float) input_width, 1.f / (float) input_height, (float) input_width, (float) input_height);
            m_ssbo_constants->rebuffer(&m_smaa_constants, sizeof(m_smaa_constants));

            GLint technique = m_smaa_detection_technique.Param<core::param::EnumParam>()->Value();
            GLint preset = m_smaa_quality.Param<core::param::EnumParam>()->Value();

            // TODO: one program for all? one mega shaders with barriers?
            // edge detection
            m_smaa_edge_detection_prgm->Enable();

            glActiveTexture(GL_TEXTURE0);
            input_tx2D->bindTexture();
            glUniform1i(m_smaa_edge_detection_prgm->ParameterLocation("g_colorTex"), 0);
            glUniform1i(m_smaa_edge_detection_prgm->ParameterLocation("technique"), technique);

            m_edges_tex->bindImage(0, GL_WRITE_ONLY);

            m_ssbo_constants->bind(0);

            m_smaa_edge_detection_prgm->Dispatch(
                static_cast<int>(std::ceil(input_width / 8.0f)), static_cast<int>(std::ceil(input_height / 8.0f)), 1);

            m_smaa_edge_detection_prgm->Disable();


            // blending weights calculation
            m_smaa_blending_weight_calculation_prgm->Enable();

            glActiveTexture(GL_TEXTURE0);
            m_edges_tex->bindTexture();
            glUniform1i(m_smaa_blending_weight_calculation_prgm->ParameterLocation("g_edgesTex"), 0);
            glActiveTexture(GL_TEXTURE1);
            m_area_tex->bindTexture();
            glUniform1i(m_smaa_blending_weight_calculation_prgm->ParameterLocation("g_areaTex"), 1);
            glActiveTexture(GL_TEXTURE2);
            m_search_tex->bindTexture();
            glUniform1i(m_smaa_blending_weight_calculation_prgm->ParameterLocation("g_searchTex"), 2);

            m_blend_tex->bindImage(0, GL_WRITE_ONLY);

            m_ssbo_constants->bind(0);

            m_smaa_blending_weight_calculation_prgm->Dispatch(
                static_cast<int>(std::ceil(input_width / 8.0f)), static_cast<int>(std::ceil(input_height / 8.0f)), 1);

            m_smaa_blending_weight_calculation_prgm->Enable();


            // final step: neighborhood blending
            m_smaa_neighborhood_blending_prgm->Enable();

            glActiveTexture(GL_TEXTURE0);
            input_tx2D->bindTexture();
            glUniform1i(m_smaa_neighborhood_blending_prgm->ParameterLocation("g_colorTex"), 0);
            glActiveTexture(GL_TEXTURE1);
            m_blend_tex->bindTexture();
            glUniform1i(m_smaa_neighborhood_blending_prgm->ParameterLocation("g_blendingWeightsTex"), 1);
            // only used for temporal reprojection
            /*glActiveTexture(GL_TEXTURE2);
            m_velocity_tex->bindTexture();
            glUniform1i(m_smaa_neighborhood_blending_prgm->ParameterLocation("g_velocityTex"), 2);*/

            m_output_texture->bindImage(0, GL_WRITE_ONLY);

            m_ssbo_constants->bind(0);

            m_smaa_neighborhood_blending_prgm->Dispatch(
                static_cast<int>(std::ceil(input_width / 8.0f)), static_cast<int>(std::ceil(input_height / 8.0f)), 1);

            m_smaa_neighborhood_blending_prgm->Disable();


            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);
            // TODO: in smaaneighborhoodblending the reads and writes must be in srgb (and only there!)
        }
        // fxaa
        else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
            if (call_input == NULL)
                return false;

            launchProgram(m_fxaa_prgm, input_tx2D, "src_tx2D", m_output_texture);
        }
        // no aa
        else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
            m_output_texture = input_tx2D;
        }
    }

    if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_output_texture, m_version);
    }

    return true;
}

bool megamol::compositing::AntiAliasing::getMetaDataCallback(core::Call& caller) { return true; }
