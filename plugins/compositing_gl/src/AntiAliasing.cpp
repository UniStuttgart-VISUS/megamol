#include "stdafx.h"
#include "AntiAliasing.h"

#include <array>
#include <random>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

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
    , m_mode("Mode", "Sets screen space effect mode, e.g. ssao, fxaa...")
    , m_smaa_quality("QualityLevel", "Sets smaa quality level")
    , m_smaa_detection_base("EdgeDetection", "Sets smaa edge detection base: luma, color, or depth")
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_input_tex_slot("InputTexture", "Connects an optional input texture")
{
    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "FXAA");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "SMAA");
    this->MakeSlotAvailable(&this->m_mode);

    this->m_smaa_quality << new megamol::core::param::EnumParam(2);
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "LOW");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "MEDIUM");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "HIGH");
    this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "ULTRA");
    //this->m_smaa_quality.Param<megamol::core::param::EnumParam>()->SetTypePair(4, "CUSTOM");
    this->MakeSlotAvailable(&this->m_smaa_quality);

    this->m_smaa_detection_base << new megamol::core::param::EnumParam(0);
    this->m_smaa_detection_base.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "LUMA");
    this->m_smaa_detection_base.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "COLOR");
    this->m_smaa_detection_base.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "DEPTH");
    this->MakeSlotAvailable(&this->m_smaa_detection_base);

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

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaaEdgeDetection", compute_smaa_edge_detection_src))
            return false;
        if (!m_smaa_edge_detection_prgm->Compile(compute_smaa_edge_detection_src.Code(), compute_smaa_edge_detection_src.Count()))
            return false;
        if (!m_smaa_edge_detection_prgm->Link())
            return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaaBlendingWeightsCalculation", compute_smaa_blending_weights_src))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Compile(compute_smaa_blending_weights_src.Code(), compute_smaa_blending_weights_src.Count()))
            return false;
        if (!m_smaa_blending_weight_calculation_prgm->Link())
            return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::smaaEdgeDetection", compute_smaa_neighborhood_blending_src))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Compile(compute_smaa_neighborhood_blending_src.Code(), compute_smaa_neighborhood_blending_src.Count()))
            return false;
        if (!m_smaa_neighborhood_blending_prgm->Link())
            return false;

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
    glowl::TextureLayout smaa_layout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1, int_params, {});
    m_edges_tex = std::make_shared<glowl::Texture2D>("smaa_edges_tex", smaa_layout, nullptr);
    m_blend_tex = std::make_shared<glowl::Texture2D>("smaa_blend_tex", smaa_layout, nullptr);
    // TODO: check textures in nsight or similar to see if textures are correctly loaded
    // TODO: do this in here? or every frame in the corresponding 'if' below
    glowl::TextureLayout area_layout(GL_RG8, AREATEX_WIDTH, AREATEX_HEIGHT, 1, GL_RG, GL_UNSIGNED_BYTE, 1, int_params, {});
    glowl::TextureLayout search_layout(GL_R8, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1, GL_RED, GL_UNSIGNED_BYTE, 1, int_params, {});
    m_area_tex = std::make_shared<glowl::Texture2D>("smaa_area_tex", area_layout, areaTexBytes);
    m_search_tex = std::make_shared<glowl::Texture2D>("smaa_search_tex", search_layout, searchTexBytes);

    return true;
}

void megamol::compositing::AntiAliasing::release() {}

void megamol::compositing::AntiAliasing::launchProgram(
    const std::unique_ptr<vislib::graphics::gl::GLSLComputeShader>& prgm,
    std::shared_ptr<glowl::Texture2D> input,
    const char* input_id,
    std::shared_ptr<glowl::Texture2D> output) {
    prgm->Enable();

    glActiveTexture(GL_TEXTURE0);
    input->bindTexture();
    glUniform1i(prgm->ParameterLocation(input_id), 0);

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

        // fxaa
        if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
            if (call_input == NULL)
                return false;

            setupOutputTexture(input_tx2D, m_output_texture);

            launchProgram(m_fxaa_prgm, input_tx2D, "src_tx2D", m_output_texture);
        }
        // smaa
        else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
            GLubyte col[4] = { 0, 0, 0, 0 };
            m_edges_tex->clearTexImage(col);
            m_blend_tex->clearTexImage(col);

            // edge detection
            launchProgram(m_smaa_edge_detection_prgm, input_tx2D, "src_tx2D", m_edges_tex);


            // blending weights calculation
            m_smaa_blending_weight_calculation_prgm->Enable();
            m_smaa_blending_weight_calculation_prgm->Enable();


            // final step: neighborhood blending
            m_smaa_neighborhood_blending_prgm->Enable();
            m_smaa_neighborhood_blending_prgm->Disable();


            // TODO: in smaaneighborhoodblending the reads and writes must be in srgb (and only there!)
        }
    }

    if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_output_texture, m_version);
    }

    return true;
}

bool megamol::compositing::AntiAliasing::getMetaDataCallback(core::Call& caller) { return true; }
