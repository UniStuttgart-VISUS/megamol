/*
 * ASSAO.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "ASSAO.h"

#include <array>
#include <random>

#include <glm/glm.hpp>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/graphics/gl/ShaderSource.h"

#include "compositing/CompositingCalls.h"


/////////////////////////////////////////////////////////////////////////
// CONSTANTS
/////////////////////////////////////////////////////////////////////////
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)     \
    {                       \
        if (p) {            \
            (p)->Release(); \
            (p) = NULL;     \
        }                   \
    }
#endif
#ifndef SAFE_RELEASE_ARRAY
#define SAFE_RELEASE_ARRAY(p)                 \
    {                                         \
        for (int i = 0; i < _countof(p); i++) \
            if (p[i]) {                       \
                (p[i])->Release();            \
                (p[i]) = NULL;                \
            }                                 \
    }
#endif

#define SSA_STRINGIZIZER(x) SSA_STRINGIZIZER_(x)
#define SSA_STRINGIZIZER_(x) #x

// ** WARNING ** if changing any of the slot numbers, please remember to update the corresponding shader code!
#define SSAO_SAMPLERS_SLOT0 0
#define SSAO_SAMPLERS_SLOT1 1
#define SSAO_SAMPLERS_SLOT2 2
#define SSAO_SAMPLERS_SLOT3 3
#define SSAO_NORMALMAP_OUT_UAV_SLOT 4
#define SSAO_CONSTANTS_BUFFERSLOT 0
#define SSAO_TEXTURE_SLOT0 0
#define SSAO_TEXTURE_SLOT1 1
#define SSAO_TEXTURE_SLOT2 2
#define SSAO_TEXTURE_SLOT3 3
#define SSAO_TEXTURE_SLOT4 4
#define SSAO_LOAD_COUNTER_UAV_SLOT 4

#define SSAO_MAX_TAPS 32
#define SSAO_MAX_REF_TAPS 512
#define SSAO_ADAPTIVE_TAP_BASE_COUNT 5
#define SSAO_ADAPTIVE_TAP_FLEXIBLE_COUNT (SSAO_MAX_TAPS - SSAO_ADAPTIVE_TAP_BASE_COUNT)
#define SSAO_DEPTH_MIP_LEVELS 4

#ifdef INTEL_SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
#define SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION 1
#else
#define SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION 0
#endif

// ** WARNING ** if changing anything here, update the corresponding shader code! ** WARNING **
struct ASSAOConstants {
    glm::vec2 ViewportPixelSize;     // .zw == 1.0 / ViewportSize.xy
    glm::vec2 HalfViewportPixelSize; // .zw == 1.0 / ViewportHalfSize.xy

    glm::vec2 DepthUnpackConsts;
    glm::vec2 CameraTanHalfFOV;

    glm::vec2 NDCToViewMul;
    glm::vec2 NDCToViewAdd;

    glm::ivec2 PerPassFullResCoordOffset;
    glm::vec2 PerPassFullResUVOffset;

    glm::vec2 Viewport2xPixelSize;
    glm::vec2 Viewport2xPixelSize_x_025; // Viewport2xPixelSize * 0.25 (for fusing add+mul into mad)

    float EffectRadius;         // world (viewspace) maximum size of the shadow
    float EffectShadowStrength; // global strength of the effect (0 - 5)
    float EffectShadowPow;
    float EffectShadowClamp;

    float EffectFadeOutMul;                 // effect fade out from distance (ex. 25)
    float EffectFadeOutAdd;                 // effect fade out to distance   (ex. 100)
    float EffectHorizonAngleThreshold;      // limit errors on slopes and caused by insufficient geometry tessellation
                                            // (0.05 to 0.5)
    float EffectSamplingRadiusNearLimitRec; // if viewspace pixel closer than this, don't enlarge shadow sampling
                                            // radius anymore (makes no sense to grow beyond some distance, not
                                            // enough samples to cover everything, so just limit the shadow growth;
                                            // could be SSAOSettingsFadeOutFrom * 0.1 or less)

    float DepthPrecisionOffsetMod;
    float NegRecEffectRadius; // -1.0 / EffectRadius
    float LoadCounterAvgDiv;  // 1.0 / ( halfDepthMip[SSAO_DEPTH_MIP_LEVELS-1].sizeX *
                              // halfDepthMip[SSAO_DEPTH_MIP_LEVELS-1].sizeY )
    float AdaptiveSampleCountLimit;

    float InvSharpness;
    int PassIndex;
    glm::vec2 QuarterResPixelSize; // used for importance map only

    glm::vec4 PatternRotScaleMatrices[5];

    float NormalsUnpackMul;
    float NormalsUnpackAdd;
    float DetailAOStrength;
    float Dummy0;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
    ASSAO_Float4x4 NormalsWorldToViewspaceMatrix;
#endif
};
/////////////////////////////////////////////////////////////////////////

megamol::compositing::ASSAO::ASSAO() : core::Module()
    , m_version(0)
    , m_output_texture_hash(0)
    , m_output_tex_slot("OutputTexture", "Gives access to resulting output texture")
    , m_input_tex_slot("InputTexture", "Connects an optional input texture")
    , m_normals_tex_slot("NormalTexture", "Connects the normals render target texture")
    , m_depth_tex_slot("DepthTexture", "Connects the depth render target texture")
    , m_camera_slot("Camera", "Connects a (copy of) camera state")
    , m_halfDepths{nullptr, nullptr, nullptr, nullptr}
    , m_halfDepthsMipViews{}
    , m_pingPongHalfResultA(nullptr)
    , m_pingPongHalfResultB(nullptr)
    , m_finalResults(nullptr)
    , m_finalResultsArrayViews{nullptr, nullptr, nullptr, nullptr}
    , m_normals(nullptr)
    , m_tx_layout_samplerStatePointClamp()
    , m_tx_layout_samplerStatePointMirror()
    , m_tx_layout_samplerStateLinearClamp()
    , m_tx_layout_samplerStateViewspaceDepthTap()
{
    m_halfDepthsMipViews.resize(4);
    for (auto& v : m_halfDepthsMipViews) {
        for (int i = 0; i < SSAO_DEPTH_MIP_LEVELS; ++i) {
            v.push_back(nullptr);
        }
    }

    this->m_output_tex_slot.SetCallback(CallTexture2D::ClassName(), "GetData", &ASSAO::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &ASSAO::getMetaDataCallback);
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

megamol::compositing::ASSAO::~ASSAO() { this->Release(); }

bool megamol::compositing::ASSAO::create() {
    try {
        {
            // create shader program
            m_prepapre_depths_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_half_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_and_normals_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_and_normals_half_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depth_mip1_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depth_mip2_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depth_mip3_prgm = std::make_unique<GLSLComputeShader>();
            m_generate_q0_prgm = std::make_unique<GLSLComputeShader>();
            m_generate_q1_prgm = std::make_unique<GLSLComputeShader>();
            m_generate_q2_prgm = std::make_unique<GLSLComputeShader>();
            m_smart_blur_prgm = std::make_unique<GLSLComputeShader>();
            m_smart_blur_wide_prgm = std::make_unique<GLSLComputeShader>();
            m_apply_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_blur_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_apply_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_half_apply_prgm = std::make_unique<GLSLComputeShader>();

            vislib::graphics::gl::ShaderSource cs_prepapre_depths;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_half;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_and_normals;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_and_normals_half;
            vislib::graphics::gl::ShaderSource cs_prepare_depth_mip1;
            vislib::graphics::gl::ShaderSource cs_prepare_depth_mip2;
            vislib::graphics::gl::ShaderSource cs_prepare_depth_mip3;
            vislib::graphics::gl::ShaderSource cs_generate_q0;
            vislib::graphics::gl::ShaderSource cs_generate_q1;
            vislib::graphics::gl::ShaderSource cs_generate_q2;
            vislib::graphics::gl::ShaderSource cs_smart_blur;
            vislib::graphics::gl::ShaderSource cs_smart_blur_wide;
            vislib::graphics::gl::ShaderSource cs_apply;
            vislib::graphics::gl::ShaderSource cs_non_smart_blur;
            vislib::graphics::gl::ShaderSource cs_non_smart_apply;
            vislib::graphics::gl::ShaderSource cs_non_smart_half_apply;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepths", cs_prepapre_depths))
                return false;
            if (!m_prepapre_depths_prgm->Compile(cs_prepapre_depths.Code(), cs_prepapre_depths.Count()))
                return false;
            if (!m_prepapre_depths_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsHalf", cs_prepare_depths_half))
                return false;
            if (!m_prepare_depths_half_prgm->Compile(cs_prepare_depths_half.Code(), cs_prepare_depths_half.Count()))
                return false;
            if (!m_prepare_depths_half_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsAndNormals", cs_prepare_depths_and_normals))
                return false;
            if (!m_prepare_depths_and_normals_prgm->Compile(
                    cs_prepare_depths_and_normals.Code(), cs_prepare_depths_and_normals.Count()))
                return false;
            if (!m_prepare_depths_and_normals_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsAndNormalsHalf", cs_prepare_depths_and_normals_half))
                return false;
            if (!m_prepare_depths_and_normals_half_prgm->Compile(
                    cs_prepare_depths_and_normals_half.Code(), cs_prepare_depths_and_normals_half.Count()))
                return false;
            if (!m_prepare_depths_and_normals_half_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthMip1", cs_prepare_depth_mip1))
                return false;
            if (!m_prepare_depth_mip1_prgm->Compile(cs_prepare_depth_mip1.Code(), cs_prepare_depth_mip1.Count()))
                return false;
            if (!m_prepare_depth_mip1_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthMip2", cs_prepare_depth_mip2))
                return false;
            if (!m_prepare_depth_mip2_prgm->Compile(cs_prepare_depth_mip2.Code(), cs_prepare_depth_mip2.Count()))
                return false;
            if (!m_prepare_depth_mip2_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthMip3", cs_prepare_depth_mip3))
                return false;
            if (!m_prepare_depth_mip3_prgm->Compile(cs_prepare_depth_mip3.Code(), cs_prepare_depth_mip3.Count()))
                return false;
            if (!m_prepare_depth_mip3_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSGenerateQ0", cs_generate_q0))
                return false;
            if (!m_generate_q0_prgm->Compile(cs_generate_q0.Code(), cs_generate_q0.Count()))
                return false;
            if (!m_generate_q0_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSGenerateQ1", cs_generate_q1))
                return false;
            if (!m_generate_q1_prgm->Compile(cs_generate_q1.Code(), cs_generate_q1.Count()))
                return false;
            if (!m_generate_q1_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSGenerateQ2", cs_generate_q2))
                return false;
            if (!m_generate_q2_prgm->Compile(cs_generate_q2.Code(), cs_generate_q2.Count()))
                return false;
            if (!m_generate_q2_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSSmartBlur", cs_smart_blur))
                return false;
            if (!m_smart_blur_prgm->Compile(cs_smart_blur.Code(), cs_smart_blur.Count()))
                return false;
            if (!m_smart_blur_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSSmartBlurWide", cs_smart_blur_wide))
                return false;
            if (!m_smart_blur_wide_prgm->Compile(cs_smart_blur_wide.Code(), cs_smart_blur_wide.Count()))
                return false;
            if (!m_smart_blur_wide_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSApply", cs_apply))
                return false;
            if (!m_apply_prgm->Compile(cs_apply.Code(), cs_apply.Count()))
                return false;
            if (!m_apply_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartBlur", cs_non_smart_blur))
                return false;
            if (!m_non_smart_blur_prgm->Compile(cs_non_smart_blur.Code(), cs_non_smart_blur.Count()))
                return false;
            if (!m_non_smart_blur_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartApply", cs_non_smart_apply))
                return false;
            if (!m_non_smart_apply_prgm->Compile(cs_non_smart_apply.Code(), cs_non_smart_apply.Count()))
                return false;
            if (!m_non_smart_apply_prgm->Link())
                return false;

            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartHalfApply", cs_non_smart_half_apply))
                return false;
            if (!m_non_smart_half_apply_prgm->Compile(cs_non_smart_half_apply.Code(), cs_non_smart_half_apply.Count()))
                return false;
            if (!m_non_smart_half_apply_prgm->Link())
                return false;
        }

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
    m_halfDepths[0] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_halfDepths[1] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_halfDepths[2] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_halfDepths[3] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    for (auto& tx : m_halfDepthsMipViews) {
        for (int i = 0; i < tx.size(); ++i) {
            tx[i] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
        }
    }
    m_pingPongHalfResultA = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_pingPongHalfResultB = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_finalResults = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_finalResultsArrayViews[0] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_finalResultsArrayViews[1] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_finalResultsArrayViews[2] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_finalResultsArrayViews[3] = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);
    m_normals = std::make_shared<glowl::Texture2D>("screenspace_effect_output", tx_layout, nullptr);

    return true;
}

void megamol::compositing::ASSAO::release() {}

bool megamol::compositing::ASSAO::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<CallTexture2D*>(&caller);
    auto call_input = m_input_tex_slot.CallAs<CallTexture2D>();
    auto call_normal = m_normals_tex_slot.CallAs<CallTexture2D>();
    auto call_depth = m_depth_tex_slot.CallAs<CallTexture2D>();
    auto call_camera = m_camera_slot.CallAs<CallCamera>();

    if (lhs_tc == NULL) return false;
    
    if(call_input != NULL) { if (!(*call_input)(0)) return false; }
    if(call_normal != NULL) { if (!(*call_normal)(0)) return false; }
    if(call_depth != NULL) { if (!(*call_depth)(0)) return false; }
    if(call_camera != NULL) { if (!(*call_camera)(0)) return false; }

    // something has changed in the neath...
    bool something_has_changed =
        (call_input != NULL ? call_input->hasUpdate() : false) || 
        (call_normal != NULL ? call_normal->hasUpdate() : false ) || 
        (call_depth != NULL ? call_depth->hasUpdate() : false) || 
        (call_camera != NULL ? call_camera->hasUpdate() : false);

    if (something_has_changed) {
        ++m_version;

        if (call_normal == NULL)
            return false;
        if (call_depth == NULL)
            return false;
        if (call_camera == NULL)
            return false;

        auto normal_tx2D = call_normal->getData();
        auto depth_tx2D = call_depth->getData();

        std::array<float, 2> tx_res = {(float) normal_tx2D->getWidth(), (float) normal_tx2D->getHeight()};

        std::vector<std::pair<GLenum, GLint>> int_param = {
            {GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST},
            {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
            {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
            {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}
        };

        m_tx_layout_samplerStatePointClamp =
            glowl::TextureLayout(GL_RGBA16F, tx_res[0], tx_res[1], 1, GL_RGBA, GL_HALF_FLOAT, 1, int_param, {});

        int_param.clear();
        int_param = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
            {GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT}, {GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT}};

        m_tx_layout_samplerStatePointMirror =
            glowl::TextureLayout(GL_RGBA16F, tx_res[0], tx_res[1], 1, GL_RGBA, GL_HALF_FLOAT, 1, int_param, {});

        int_param.clear();
        std::vector<std::pair<GLenum, GLint>> int_param = {{GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
            {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

        m_tx_layout_samplerStateLinearClamp =
            glowl::TextureLayout(GL_RGBA16F, tx_res[0], tx_res[1], 1, GL_RGBA, GL_HALF_FLOAT, 1, int_param, {});

        int_param.clear();
        std::vector<std::pair<GLenum, GLint>> int_param = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST},
            {GL_TEXTURE_MAG_FILTER, GL_NEAREST}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
            {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

        m_tx_layout_samplerStateViewspaceDepthTap =
            glowl::TextureLayout(GL_RGBA16F, tx_res[0], tx_res[1], 1, GL_RGBA, GL_HALF_FLOAT, 1, int_param, {});

        //setupOutputTexture(normal_tx2D, m_output_texture);

        // obtain camera information
        core::view::Camera_2 cam = call_camera->getData();
        cam_type::snapshot_type snapshot;
        cam_type::matrix_type view_tmp, proj_tmp;
        cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
        glm::mat4 view_mx = view_tmp;
        glm::mat4 proj_mx = proj_tmp;

        /*m_ssao_prgm->Enable();

        glActiveTexture(GL_TEXTURE0);
        normal_tx2D->bindTexture();
        glUniform1i(m_ssao_prgm->ParameterLocation("normal_tx2D"), 0);
        glActiveTexture(GL_TEXTURE1);
        depth_tx2D->bindTexture();
        glUniform1i(m_ssao_prgm->ParameterLocation("depth_tx2D"), 1);
        glActiveTexture(GL_TEXTURE2);

        auto inv_view_mx = glm::inverse(view_mx);
        auto inv_proj_mx = glm::inverse(proj_mx);
        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("inv_view_mx"), 1, GL_FALSE, glm::value_ptr(inv_view_mx));
        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("inv_proj_mx"), 1, GL_FALSE, glm::value_ptr(inv_proj_mx));

        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(view_mx));
        glUniformMatrix4fv(m_ssao_prgm->ParameterLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(proj_mx));


        m_ssao_prgm->Dispatch(static_cast<int>(std::ceil(m_output_texture->getWidth() / 8.0f)),
            static_cast<int>(std::ceil(m_output_texture->getHeight() / 8.0f)), 1);

        m_ssao_prgm->Disable();*/

        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

        //m_output_texture->bindImage(0, GL_WRITE_ONLY);
    }
        

    if (lhs_tc->version() < m_version) {
        //lhs_tc->setData(m_output_texture, m_version);
    }

    return true;
}

bool megamol::compositing::ASSAO::getMetaDataCallback(core::Call& caller) { return true; }

void megamol::compositing::ASSAO::updateTextures() {
    // TODO: next
}
