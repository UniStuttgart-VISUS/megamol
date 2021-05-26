/*
 * ASSAO.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// File changes (yyyy-mm-dd)
// 2016-09-07: filip.strugar@intel.com: first commit
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "ASSAO.h"

#include <array>
#include <random>

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
/////////////////////////////////////////////////////////////////////////

megamol::compositing::ASSAO::ASSAO()
    : core::Module()
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
    //, m_normals(nullptr)
    , m_finalOutput(nullptr)
    , m_samplerStatePointClamp()
    , m_samplerStatePointMirror()
    , m_samplerStateLinearClamp()
    , m_samplerStateViewspaceDepthTap()
    , m_depthBufferViewspaceLinearLayout()
    , m_AOResultLayout()
    , m_size(0, 0)
    , m_halfSize(0, 0)
    , m_quarterSize(0, 0)
    , m_fullResOutScissorRect(0, 0, 0, 0)
    , m_halfResOutScissorRect(0, 0, 0, 0)
    , m_depthMipLevels(0)
    , m_inputs(nullptr)
    , m_max_blur_pass_count(6)
    , m_settings()
    , m_ssbo_constants(nullptr)
{
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

// TODO: DELETE ALL MEMORY!
megamol::compositing::ASSAO::~ASSAO() { this->Release(); }

bool megamol::compositing::ASSAO::create() {
    try {
        {
            // create shader program
            m_prepare_depths_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_half_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_and_normals_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depths_and_normals_half_prgm = std::make_unique<GLSLComputeShader>();
            m_prepare_depth_mip_prgms.resize(SSAO_DEPTH_MIP_LEVELS - 1);
            for (auto& cs : m_prepare_depth_mip_prgms) {
                cs = std::make_unique<GLSLComputeShader>();
            }
            for (auto& cs : m_generate_prgms) {
                cs = std::make_unique<GLSLComputeShader>();
            }
            m_smart_blur_prgm = std::make_unique<GLSLComputeShader>();
            m_smart_blur_wide_prgm = std::make_unique<GLSLComputeShader>();
            m_apply_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_blur_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_apply_prgm = std::make_unique<GLSLComputeShader>();
            m_non_smart_half_apply_prgm = std::make_unique<GLSLComputeShader>();

            vislib::graphics::gl::ShaderSource cs_prepare_depths;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_half;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_and_normals;
            vislib::graphics::gl::ShaderSource cs_prepare_depths_and_normals_half;
            std::vector<vislib::graphics::gl::ShaderSource> cs_prepare_depth_mip(SSAO_DEPTH_MIP_LEVELS - 1);
            std::vector<vislib::graphics::gl::ShaderSource> cs_generate(5);
            vislib::graphics::gl::ShaderSource cs_smart_blur;
            vislib::graphics::gl::ShaderSource cs_smart_blur_wide;
            vislib::graphics::gl::ShaderSource cs_apply;
            vislib::graphics::gl::ShaderSource cs_non_smart_blur;
            vislib::graphics::gl::ShaderSource cs_non_smart_apply;
            vislib::graphics::gl::ShaderSource cs_non_smart_half_apply;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepths\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepths", cs_prepare_depths))
                return false;
            if (!m_prepare_depths_prgm->Compile(cs_prepare_depths.Code(), cs_prepare_depths.Count()))
                return false;
            if (!m_prepare_depths_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepthsHalf\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsHalf", cs_prepare_depths_half))
                return false;
            if (!m_prepare_depths_half_prgm->Compile(cs_prepare_depths_half.Code(), cs_prepare_depths_half.Count()))
                return false;
            if (!m_prepare_depths_half_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepthsAndNormals\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsAndNormals", cs_prepare_depths_and_normals))
                return false;
            if (!m_prepare_depths_and_normals_prgm->Compile(
                    cs_prepare_depths_and_normals.Code(), cs_prepare_depths_and_normals.Count()))
                return false;
            if (!m_prepare_depths_and_normals_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSPrepareDepthsAndNormalsHalf\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSPrepareDepthsAndNormalsHalf", cs_prepare_depths_and_normals_half))
                return false;
            if (!m_prepare_depths_and_normals_half_prgm->Compile(
                    cs_prepare_depths_and_normals_half.Code(), cs_prepare_depths_and_normals_half.Count()))
                return false;
            if (!m_prepare_depths_and_normals_half_prgm->Link())
                return false;

            for (int i = 0; i < SSAO_DEPTH_MIP_LEVELS - 1; ++i) {
                std::cout << "Compiling: Compositing::assao::CSPrepareDepthMip" << i + 1 << "\n";
                std::string identifier = "Compositing::assao::CSPrepareDepthMip" + std::to_string(i + 1);
                if (!instance()->ShaderSourceFactory().MakeShaderSource(
                        identifier.c_str(), cs_prepare_depth_mip[i]))
                    return false;
                if (!m_prepare_depth_mip_prgms[i]->Compile(
                        cs_prepare_depth_mip[i].Code(), cs_prepare_depth_mip[i].Count()))
                    return false;
                if (!m_prepare_depth_mip_prgms[i]->Link())
                    return false;
            }

            // one less than cs_generate.size() because the adaptive quality level is not implemented (yet)
            for (int i = 0; i < 4; ++i) { 
                std::cout << "Compiling: Compositing::assao::CSGenerateQ" << i << "\n";
                std::string identifier = "Compositing::assao::CSGenerateQ" + std::to_string(i);
                if (i < 4)
                    identifier = "Compositing::assao::CSGenerateQ" + std::to_string(i);
                else
                    identifier = "Compositing::assao::CSGenerateQ3Base";

                if (!instance()->ShaderSourceFactory().MakeShaderSource(identifier.c_str(), cs_generate[i]))
                    return false;
                if (!m_generate_prgms[i]->Compile(cs_generate[i].Code(), cs_generate[i].Count()))
                    return false;
                if (!m_generate_prgms[i]->Link())
                    return false;
            }

            std::cout << "Compiling: Compositing::assao::CSSmartBlur\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSSmartBlur", cs_smart_blur))
                return false;
            if (!m_smart_blur_prgm->Compile(cs_smart_blur.Code(), cs_smart_blur.Count()))
                return false;
            if (!m_smart_blur_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSSmartBlurWide\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSSmartBlurWide", cs_smart_blur_wide))
                return false;
            if (!m_smart_blur_wide_prgm->Compile(cs_smart_blur_wide.Code(), cs_smart_blur_wide.Count()))
                return false;
            if (!m_smart_blur_wide_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSNonSmartBlur\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartBlur", cs_non_smart_blur))
                return false;
            if (!m_non_smart_blur_prgm->Compile(cs_non_smart_blur.Code(), cs_non_smart_blur.Count()))
                return false;
            if (!m_non_smart_blur_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSApply\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource("Compositing::assao::CSApply", cs_apply))
                return false;
            if (!m_apply_prgm->Compile(cs_apply.Code(), cs_apply.Count()))
                return false;
            if (!m_apply_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSNonSmartApply\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartApply", cs_non_smart_apply))
                return false;
            if (!m_non_smart_apply_prgm->Compile(cs_non_smart_apply.Code(), cs_non_smart_apply.Count()))
                return false;
            if (!m_non_smart_apply_prgm->Link())
                return false;

            std::cout << "Compiling: Compositing::assao::CSNonSmartHalfApply\n";
            if (!instance()->ShaderSourceFactory().MakeShaderSource(
                    "Compositing::assao::CSNonSmartHalfApply", cs_non_smart_half_apply))
                return false;
            if (!m_non_smart_half_apply_prgm->Compile(cs_non_smart_half_apply.Code(), cs_non_smart_half_apply.Count()))
                return false;
            if (!m_non_smart_half_apply_prgm->Link())
                return false;

            std::cout << "Done Compiling\n";
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

    m_depthBufferViewspaceLinearLayout = glowl::TextureLayout(GL_R16F, 1, 1, 1, GL_RED, GL_HALF_FLOAT, 1);
    m_AOResultLayout = glowl::TextureLayout(GL_RG8, 1, 1, 1, GL_RG, GL_UNSIGNED_BYTE, 1);
    m_halfDepths[0] = std::make_shared<glowl::Texture2D>("m_halfDepths0", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[1] = std::make_shared<glowl::Texture2D>("m_halfDepths1", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[2] = std::make_shared<glowl::Texture2D>("m_halfDepths2", m_depthBufferViewspaceLinearLayout, nullptr);
    m_halfDepths[3] = std::make_shared<glowl::Texture2D>("m_halfDepths3", m_depthBufferViewspaceLinearLayout, nullptr);
    for (auto& tx : m_halfDepthsMipViews) {
        for (int i = 0; i < tx.size(); ++i) {
            tx[i] = std::make_shared<glowl::Texture2D>(
                "m_halfDepthsMipViews" + i, m_depthBufferViewspaceLinearLayout, nullptr);
        }
    }
    m_finalOutput = std::make_shared<glowl::Texture2D>("m_finalOutput", m_depthBufferViewspaceLinearLayout, nullptr);
    m_pingPongHalfResultA = std::make_shared<glowl::Texture2D>("m_pingPongHalfResultA", m_AOResultLayout, nullptr);
    m_pingPongHalfResultB = std::make_shared<glowl::Texture2D>("m_pingPongHalfResultB", m_AOResultLayout, nullptr);
    m_finalResults = std::make_shared<glowl::Texture2DArray>("m_finalResults", m_AOResultLayout, nullptr);
    m_finalResultsArrayViews[0] =
        std::make_shared<glowl::Texture2D>("m_finalResultsArrayViews0", m_AOResultLayout, nullptr);
    m_finalResultsArrayViews[1] =
        std::make_shared<glowl::Texture2D>("m_finalResultsArrayViews1", m_AOResultLayout, nullptr);
    m_finalResultsArrayViews[2] =
        std::make_shared<glowl::Texture2D>("m_finalResultsArrayViews2", m_AOResultLayout, nullptr);
    m_finalResultsArrayViews[3] =
        std::make_shared<glowl::Texture2D>("m_finalResultsArrayViews3", m_AOResultLayout, nullptr);
    //m_normals = std::make_shared<glowl::Texture2D>("m_normals", tx_layout, nullptr);

    m_inputs = std::make_shared<ASSAO_Inputs>();

    m_ssbo_constants = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    // TODO: check for correctness
    std::vector<std::pair<GLenum, GLint>> int_params = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_NEAREST}, {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    m_samplerStatePointClamp = std::make_shared<glowl::Sampler>("samplerStatePointClamp", int_params);

    int_params.clear();
    int_params = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        {GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT}, {GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT}};

    m_samplerStatePointMirror = std::make_shared<glowl::Sampler>("samplerStatePointMirror", int_params);

    int_params.clear();
    int_params = {{GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    m_samplerStateLinearClamp = std::make_shared<glowl::Sampler>("samplerStateLinearClamp", int_params);

    int_params.clear();
    int_params = {{GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST}, {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}};

    m_samplerStateViewspaceDepthTap =
        std::make_shared<glowl::Sampler>("samplerStateViewspaceDepthTap", int_params);

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
    bool input_update = call_input->hasUpdate();
    bool normal_update = call_normal->hasUpdate();
    bool depth_update = call_depth->hasUpdate();
    bool camera_update = call_camera->hasUpdate();

    bool something_has_changed =
        (call_input != NULL ? input_update : false) || 
        (call_normal != NULL ? normal_update : false) || 
        (call_depth != NULL ? depth_update : false) || 
        (call_camera != NULL ? camera_update : false);

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
        std::array<int, 2> tx_res_normal = {(int) normal_tx2D->getWidth(), (int) normal_tx2D->getHeight()};
        std::array<int, 2> tx_res_depth = {(int) depth_tx2D->getWidth(), (int) depth_tx2D->getHeight()};

        // obtain camera information
        core::view::Camera_2 cam = call_camera->getData();
        cam_type::snapshot_type snapshot;
        cam_type::matrix_type view_tmp, proj_tmp;
        cam.calc_matrices(snapshot, view_tmp, proj_tmp, core::thecam::snapshot_content::all);
        glm::mat4 view_mx = view_tmp;
        glm::mat4 proj_mx = proj_tmp;

        if (normal_update || depth_update) {

            m_inputs->ViewportWidth = tx_res_normal[0];
            m_inputs->ViewportHeight = tx_res_normal[1];
            m_inputs->normalTexture = normal_tx2D;
            m_inputs->depthTexture = depth_tx2D;
            // TODO: for now we won't use scissortests
            // but the implementation still stays for a more easy migration
            // can be removed or used if (not) needed
            // scissor rectangle stays constant for now
            m_inputs->ScissorLeft = 0;
            m_inputs->ScissorRight = tx_res_normal[0];
            m_inputs->ScissorTop = tx_res_normal[1];
            m_inputs->ScissorBottom = 0;
            m_inputs->ProjectionMatrix = proj_mx;
            m_inputs->ViewMatrix = view_mx;

            updateTextures(m_inputs);

            updateConstants(m_settings, m_inputs, 0);
        }

        {
            // only required when scissors are used
            /*if (m_requiresClear) {
                m_halfDepths[0]->reload(m_halfDepths[0]->getTextureLayout(), nullptr);
                m_halfDepths[1]->reload(m_halfDepths[1]->getTextureLayout(), nullptr);
                m_halfDepths[2]->reload(m_halfDepths[2]->getTextureLayout(), nullptr);
                m_halfDepths[3]->reload(m_halfDepths[3]->getTextureLayout(), nullptr);
                m_pingPongHalfResultA->reload(m_pingPongHalfResultA->getTextureLayout(), nullptr);
                m_pingPongHalfResultB->reload(m_pingPongHalfResultB->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[0]->reload(m_finalResultsArrayViews[0]->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[1]->reload(m_finalResultsArrayViews[1]->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[2]->reload(m_finalResultsArrayViews[2]->getTextureLayout(), nullptr);
                m_finalResultsArrayViews[3]->reload(m_finalResultsArrayViews[3]->getTextureLayout(), nullptr);
                if (m_normals != nullptr)
                    m_normals->reload(m_normals->getTextureLayout(), nullptr);

                m_requiresClear = false;
            }*/

            // TODO: Set effect samplers

            prepareDepths(m_settings, m_inputs);

            generateSSAO(m_settings, m_inputs, false);

            /*if( inputs->OverrideOutputRTV != nullptr )
            {
                // drawing into OverrideOutputRTV
                dx11Context->OMSetRenderTargets( 1, &inputs->OverrideOutputRTV, NULL );
            }
            else
            {
                // restore previous RTs
                d3d11StatesBackup.RestoreRTs( );
            }*/

            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                {m_finalOutput, 0}
            };

            // Apply
            {
                TextureArraySamplerTuple inputFinals =
                    {m_finalResults, "g_finalSSAO", m_samplerStateLinearClamp};

                // TODO: blending states

                if (m_settings.QualityLevel < 0)
                    fullscreenPassDraw(m_non_smart_half_apply_prgm, {}, outputTextures, true, inputFinals);
                else if (m_settings.QualityLevel == 0)
                    fullscreenPassDraw(m_non_smart_apply_prgm, {}, outputTextures, true, inputFinals);
                else
                    fullscreenPassDraw(m_apply_prgm, {}, outputTextures, true, inputFinals);
            }

            // TODO: presumably also unnecessary, see also a few lines above
            // restore previous RTs again (because of the viewport hack)
            // d3d11StatesBackup.RestoreRTs();
        }
    }
        

    if (lhs_tc->version() < m_version) {
        lhs_tc->setData(m_finalOutput, m_version);
    }

    return true;
}

void megamol::compositing::ASSAO::prepareDepths(
    const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs) {
    bool generateNormals = inputs->normalTexture == nullptr;

    std::vector<TextureSamplerTuple> inputTextures = {
        {inputs->depthTexture, (std::string) "g_DepthSource", m_samplerStatePointClamp}};

    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputFourDepths = {
        {m_halfDepths[0], 0}, {m_halfDepths[1], 1}, {m_halfDepths[2], 2}, {m_halfDepths[3], 3}};
    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTwoDepths = {
        {m_halfDepths[0], 0}, {m_halfDepths[3], 3}};


    if (!generateNormals) {
        if (settings.QualityLevel < 0) {
            fullscreenPassDraw(m_prepare_depths_half_prgm, inputTextures, outputTwoDepths);
        } else {
            fullscreenPassDraw(m_prepare_depths_prgm, inputTextures, outputFourDepths);
        }
    } else {
        if (settings.QualityLevel < 0) {
            outputTwoDepths.push_back({inputs->normalTexture, 7});
            fullscreenPassDraw(m_prepare_depths_and_normals_half_prgm, inputTextures, outputTwoDepths);
        } else {
            outputFourDepths.push_back({inputs->normalTexture, 7});
            fullscreenPassDraw(m_prepare_depths_and_normals_prgm, inputTextures, outputFourDepths);
        }
    }

    // only do mipmaps for higher quality levels (not beneficial on quality level 1, and detrimental on quality level 0)
    if (settings.QualityLevel > 1) {

        for (int i = 1; i < m_depthMipLevels; ++i) {
            std::vector<TextureSamplerTuple> inputFourDepthMipsM1 = {
                {m_halfDepthsMipViews[0][i - 1LL], (std::string) "g_ViewspaceDepthSource" , nullptr},
                {m_halfDepthsMipViews[1][i - 1LL], (std::string) "g_ViewspaceDepthSource1", nullptr},
                {m_halfDepthsMipViews[2][i - 1LL], (std::string) "g_ViewspaceDepthSource2", nullptr},
                {m_halfDepthsMipViews[3][i - 1LL], (std::string) "g_ViewspaceDepthSource3", nullptr}};

            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputFourDepthMips = {
                {m_halfDepthsMipViews[0][i], 0}, {m_halfDepthsMipViews[1][i], 1}, {m_halfDepthsMipViews[2][i], 2},
                {m_halfDepthsMipViews[3][i], 3}};

            // dx11Context->RSSetViewports( 1, &viewport ); --> TODO: problem if not set?
            // i dont think so, because we launch a compute shader specifically made for this case

            fullscreenPassDraw(m_prepare_depth_mip_prgms[i - 1LL], inputFourDepthMipsM1, outputFourDepthMips);
        }
    }
}

void megamol::compositing::ASSAO::generateSSAO(
    const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, bool adaptiveBasePass) {

    // omitted viewport and scissor code from intel here

    if (adaptiveBasePass) {
        assert(settings.QualityLevel == 3);
    }

    int passCount = 4;

    for (int pass = 0; pass < passCount; ++pass) {
        if ((settings.QualityLevel < 0) && ((pass == 1) || (pass == 2)))
            continue;

        int blurPasses = settings.BlurPassCount;
        blurPasses = std::min(blurPasses, m_max_blur_pass_count);

        // CHECK FOR ADAPTIVE SSAO
#ifdef INTEL_SSAO_ENABLE_ADAPTIVE_QUALITY
        if (settings.QualityLevel == 3) {
            // if adaptive, at least one blur pass needed as the first pass needs to read the final texture results -
            // kind of awkward
            if (adaptiveBasePass)
                blurPasses = 0;
            else
                blurPasses = Max(1, blurPasses);
        } else
#endif
        if (settings.QualityLevel <= 0) {
            // just one blur pass allowed for minimum quality
            // MM simply uses one blur pass
            blurPasses = std::min(1, settings.BlurPassCount);
        }

        updateConstants(settings, inputs, pass);

        // Generate
        {
            std::shared_ptr<glowl::Texture2D> rts = m_pingPongHalfResultA;
            GLuint binding = 4;
            // no blur?
            if (blurPasses == 0) {
                rts = m_finalResultsArrayViews[pass];
                // we always bind to pingPongHalfResultA since the layout is the same as finalResultsArrayViews
                // --> no distinction required in shader code
                //binding = pass; 
            }

            std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                {rts, binding}}; // TODO: binding correct?

            std::vector<TextureSamplerTuple> inputTextures(3);
            inputTextures[0] = {m_halfDepths[pass], "g_ViewSpaceDepthSource", m_samplerStatePointMirror};
            inputTextures[1] = {inputs->normalTexture, "g_NormalmapSource", nullptr};
            inputTextures[2] = {m_halfDepths[pass], "g_ViewSpaceDepthSourceDepthTapSampler", m_samplerStateViewspaceDepthTap};

            // CHECK FOR ADAPTIVE SSAO
#ifdef INTEL_SSAO_ENABLE_ADAPTIVE_QUALITY
            if (!adaptiveBasePass && (settings.QualityLevel == 3)) {
                inputTextures[3] = {m_loadCounterSRV, "g_LoadCounter"};
                inputTextures[4] = {m_importanceMap.SRV, "g_ImportanceMap"};
                inputTextures[5] = {m_finalResults.SRV, "g_FinalSSAO"};
            }
#endif

            int shaderIndex = std::max(0, !adaptiveBasePass ? settings.QualityLevel : 4);
            fullscreenPassDraw(m_generate_prgms[shaderIndex], inputTextures, outputTextures);
        }

        // Blur
        if (blurPasses > 0) {
            int wideBlursRemaining = std::max(0, blurPasses - 2);

            for (int i = 0; i < blurPasses; ++i) {
                std::shared_ptr<glowl::Texture2D> rts = m_pingPongHalfResultB;
                GLuint binding = 5;
                if (i == blurPasses - 1) {
                    rts = m_finalResultsArrayViews[pass];
                    // we always bind to pingPongHalfResultB since the layout is the same as finalResultsArrayViews
                    // --> no distinction required in shader code
                    // binding = pass; 
                }

                std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>> outputTextures = {
                    {rts, binding}}; // TODO: binding correct?

                std::vector<TextureSamplerTuple> inputTextures = {
                    {m_pingPongHalfResultA, "g_BlurInput", m_samplerStatePointMirror}};

                if (settings.QualityLevel > 0) {
                    if (wideBlursRemaining > 0) {
                        fullscreenPassDraw(m_smart_blur_wide_prgm, inputTextures, outputTextures);
                        wideBlursRemaining--;
                    } else {
                        fullscreenPassDraw(m_smart_blur_prgm, inputTextures, outputTextures);
                    }
                } else {
                    std::get<2>(inputTextures[0]) = m_samplerStateLinearClamp;
                    fullscreenPassDraw(m_non_smart_blur_prgm, inputTextures, outputTextures); // just for quality level 0 (and -1)
                }

                std::swap(m_pingPongHalfResultA, m_pingPongHalfResultB);
            }
        }
    }
}


// TODO: intel originally uses some blending state parameter, but it isnt used here, so we omit this
// but could easily be extended with glEnable(GL_BLEND) and a blendfunc
void megamol::compositing::ASSAO::fullscreenPassDraw(const std::unique_ptr<GLSLComputeShader>& prgm,
    const std::vector<TextureSamplerTuple>& input_textures,
    std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>>& output_textures, bool add_constants,
    const TextureArraySamplerTuple& finals) {
    
    prgm->Enable();

    if (add_constants)
        m_ssbo_constants->bind(0);

    for (int i = 0; i < input_textures.size(); ++i) {
        if (std::get<0>(input_textures[i]) != nullptr) {
            glActiveTexture(GL_TEXTURE0 + i);

            std::get<0>(input_textures[i])->bindTexture();

            if (std::get<2>(input_textures[i]) != nullptr)
                std::get<2>(input_textures[i])->bindSampler(i);

            glUniform1i(prgm->ParameterLocation(std::get<1>(input_textures[i]).c_str()), i);
        }
    }

    if (std::get<0>(finals) != nullptr) {
        glActiveTexture(GL_TEXTURE0);
        std::get<0>(finals)->bindTexture();
        std::get<2>(finals)->bindSampler(0);
        glUniform1i(prgm->ParameterLocation(std::get<1>(finals).c_str()), 0);
    }

    for (int i = 0; i < output_textures.size(); ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        output_textures[i].first->bindTexture();
        output_textures[i].first->bindImage(output_textures[i].second, GL_READ_WRITE);
    }

    // all textures in output_textures should have the same size, so we just use the first
    // TODO: is size for dispatch correct?
    prgm->Dispatch(static_cast<int>(std::ceil(output_textures[0].first->getWidth() / 8)),
        static_cast<int>(std::ceil(output_textures[0].first->getHeight() / 8)), 1);

    prgm->Disable();

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool megamol::compositing::ASSAO::getMetaDataCallback(core::Call& caller) { return true; }

// TODO: the whole functions seems redundant.
// i think it is possible to call it only once at the beginning
// no need for further calls afterwards
void megamol::compositing::ASSAO::updateTextures(const std::shared_ptr<ASSAO_Inputs> inputs) {
    int width = inputs->ViewportWidth;
    int height = inputs->ViewportHeight;

    glowl::TextureLayout depth_layout = inputs->depthTexture->getTextureLayout();
    glowl::TextureLayout normal_layout = inputs->normalTexture->getTextureLayout();


    bool needsUpdate = (m_size.x != width) || (m_size.y != height);

    m_size.x = width;
    m_size.y = height;
    m_halfSize.x = (width + 1) / 2;
    m_halfSize.y = (height + 1) / 2;
    m_quarterSize.x = (m_halfSize.x + 1) / 2;
    m_quarterSize.y = (m_halfSize.y + 1) / 2;

    glm::ivec4 prevScissorRect = m_fullResOutScissorRect;

    if ((inputs->ScissorRight == 0) || (inputs->ScissorTop == 0)) {
        m_fullResOutScissorRect = glm::ivec4(0, 0, width, height);
    } else {
        m_fullResOutScissorRect =
            glm::ivec4(std::max(0, inputs->ScissorLeft),
            std::max(0, inputs->ScissorBottom),
            std::min(width, inputs->ScissorRight),
            std::min(0, inputs->ScissorTop));
    }

    needsUpdate |= (prevScissorRect != m_fullResOutScissorRect);
    if (!needsUpdate)
        return;

    m_halfResOutScissorRect =
        glm::ivec4(m_fullResOutScissorRect.x / 2, m_fullResOutScissorRect.y / 2,
        (m_fullResOutScissorRect.z + 1) / 2, (m_fullResOutScissorRect.w + 1) / 2);

    int blurEnlarge = m_max_blur_pass_count + std::max(0, m_max_blur_pass_count - 2); // +1 for max normal blurs, +2 for wide blurs

    m_halfResOutScissorRect =
        glm::ivec4(std::max(0, m_halfResOutScissorRect.x - blurEnlarge),
        std::max(0, m_halfResOutScissorRect.y - blurEnlarge),
        std::min(m_halfSize.x, m_halfResOutScissorRect.z + blurEnlarge),
        std::min(m_halfSize.y, m_halfResOutScissorRect.w + blurEnlarge));

    float totalSizeInMB = 0.f;

    m_depthMipLevels = SSAO_DEPTH_MIP_LEVELS;

    for (int i = 0; i < 4; i++) {
        if (reCreateIfNeeded(m_halfDepths[i], m_halfSize, m_depthBufferViewspaceLinearLayout)) {
            for (int j = 0; j < m_depthMipLevels; j++)
                m_halfDepthsMipViews[i][j].reset();

            for (int j = 0; j < m_depthMipLevels; j++)
                reCreateMIPViewIfNeeded(m_halfDepthsMipViews[i][j], m_halfDepths[i], j);
        }
    }

    // TODO: is the normal_layout correct?
    // i think i need to use one of the sampler layouts at the beginning of the updateTextures
    // need to investigate this more thoroughly
    reCreateIfNeeded(m_pingPongHalfResultA, m_halfSize, m_AOResultLayout);
    reCreateIfNeeded(m_pingPongHalfResultB, m_halfSize, m_AOResultLayout);
    reCreateIfNeeded(m_finalResults, m_halfSize, m_AOResultLayout);

    for (int i = 0; i < 4; ++i) {
        // TODO: is this correct? intels technically uses ReCreateArrayViewIfNeeded but i think we wont need it
        // since our handling is different
        reCreateArrayIfNeeded(m_finalResultsArrayViews[i], m_finalResults, m_halfSize, m_AOResultLayout, i);
    }

    reCreateIfNeeded(m_finalOutput, m_size, m_depthBufferViewspaceLinearLayout); // is this needed?

    // trigger a full buffers clear first time; only really required when using scissor rects
    //m_requiresClear = true;
}

void megamol::compositing::ASSAO::updateConstants(
    const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, int pass) {
    bool generateNormals = inputs->normalTexture == nullptr;

    // update constants
    if ( m_ssbo_constants->getByteSize() != NULL ) {
        assert(false);
        return;
    } else {
        ASSAO_Constants& consts = m_constants;// = *((ASSAOConstants*) mappedResource.pData);

        const glm::mat4& proj = inputs->ProjectionMatrix;

        consts.ViewportPixelSize = glm::vec2(1.0f / (float) m_size.x, 1.0f / (float) m_size.y);
        consts.HalfViewportPixelSize = glm::vec2(1.0f / (float) m_halfSize.x, 1.0f / (float) m_halfSize.y);

        consts.Viewport2xPixelSize = glm::vec2(consts.ViewportPixelSize.x * 2.0f, consts.ViewportPixelSize.y * 2.0f);
        consts.Viewport2xPixelSize_x_025 =
            glm::vec2(consts.Viewport2xPixelSize.x * 0.25f, consts.Viewport2xPixelSize.y * 0.25f);

        // intel uses a rowmajororder check, but we use a fix column major order with glm, so we wont check for this option
        float depthLinearizeMul = -proj[2][3]; // float depthLinearizeMul = ( clipFar * clipNear ) / ( clipFar - clipNear );
        float depthLinearizeAdd = proj[2][2];  // float depthLinearizeAdd = clipFar / ( clipFar - clipNear );
        // correct the handedness issue. need to make sure this below is correct, but I think it is.
        if (depthLinearizeMul * depthLinearizeAdd < 0)
            depthLinearizeAdd = -depthLinearizeAdd;
        consts.DepthUnpackConsts = glm::vec2(depthLinearizeMul, depthLinearizeAdd);

        float tanHalfFOVY = 1.0f / proj[1][1]; // = tanf( drawContext.Camera.GetYFOV( ) * 0.5f );
        float tanHalfFOVX = 1.0F / proj[0][0]; // = tanHalfFOVY * drawContext.Camera.GetAspect( );
        consts.CameraTanHalfFOV = glm::vec2(tanHalfFOVX, tanHalfFOVY);

        consts.NDCToViewMul = glm::vec2(consts.CameraTanHalfFOV.x * 2.0f, consts.CameraTanHalfFOV.y * -2.0f);
        consts.NDCToViewAdd = glm::vec2(consts.CameraTanHalfFOV.x * -1.0f, consts.CameraTanHalfFOV.y * 1.0f);

        consts.EffectRadius = std::clamp(settings.Radius, 0.0f, 100000.0f);
        consts.EffectShadowStrength = std::clamp(settings.ShadowMultiplier * 4.3f, 0.0f, 10.0f);
        consts.EffectShadowPow = std::clamp(settings.ShadowPower, 0.0f, 10.0f);
        consts.EffectShadowClamp = std::clamp(settings.ShadowClamp, 0.0f, 1.0f);
        consts.EffectFadeOutMul = -1.0f / (settings.FadeOutTo - settings.FadeOutFrom);
        consts.EffectFadeOutAdd = settings.FadeOutFrom / (settings.FadeOutTo - settings.FadeOutFrom) + 1.0f;
        consts.EffectHorizonAngleThreshold = std::clamp(settings.HorizonAngleThreshold, 0.0f, 1.0f);

        // 1.2 seems to be around the best trade off - 1.0 means on-screen radius will stop/slow growing when the camera
        // is at 1.0 distance, so, depending on FOV, basically filling up most of the screen This setting is
        // viewspace-dependent and not screen size dependent intentionally, so that when you change FOV the effect stays
        // (relatively) similar.
        float effectSamplingRadiusNearLimit = (settings.Radius * 1.2f);

        // if the depth precision is switched to 32bit float, this can be set to something closer to 1 (0.9999 is fine)
        consts.DepthPrecisionOffsetMod = 0.9992f;

        // consts.RadiusDistanceScalingFunctionPow     = 1.0f - Clamp( settings.RadiusDistanceScalingFunction,
        // 0.0f, 1.0f );

        int lastHalfDepthMipX = m_halfDepthsMipViews[0][SSAO_DEPTH_MIP_LEVELS - 1]->getWidth();
        int lastHalfDepthMipY = m_halfDepthsMipViews[0][SSAO_DEPTH_MIP_LEVELS - 1]->getHeight();

        // used to get average load per pixel; 9.0 is there to compensate for only doing every 9th InterlockedAdd in
        // PSPostprocessImportanceMapB for performance reasons
        consts.LoadCounterAvgDiv = 9.0f / (float) (m_quarterSize.x * m_quarterSize.y * 255.0);

        // Special settings for lowest quality level - just nerf the effect a tiny bit
        if (settings.QualityLevel <= 0) {
            // consts.EffectShadowStrength     *= 0.9f;
            effectSamplingRadiusNearLimit *= 1.50f;

            if (settings.QualityLevel < 0)
                consts.EffectRadius *= 0.8f;
        }
        effectSamplingRadiusNearLimit /= tanHalfFOVY; // to keep the effect same regardless of FOV

        consts.EffectSamplingRadiusNearLimitRec = 1.0f / effectSamplingRadiusNearLimit;

        consts.AdaptiveSampleCountLimit = settings.AdaptiveQualityLimit;

        consts.NegRecEffectRadius = -1.0f / consts.EffectRadius;

        consts.PerPassFullResCoordOffset = glm::ivec2(pass % 2, pass / 2);
        consts.PerPassFullResUVOffset = glm::vec2(((pass % 2) - 0.0f) / m_size.x, ((pass / 2) - 0.0f) / m_size.y);

        consts.InvSharpness = std::clamp(1.0f - settings.Sharpness, 0.0f, 1.0f);
        consts.PassIndex = pass;
        consts.QuarterResPixelSize = glm::vec2(1.0f / (float) m_quarterSize.x, 1.0f / (float) m_quarterSize.y);

        float additionalAngleOffset =
            settings.TemporalSupersamplingAngleOffset; // if using temporal supersampling approach (like "Progressive
                                                       // Rendering Using Multi-frame Sampling" from GPU Pro 7, etc.)
        float additionalRadiusScale =
            settings.TemporalSupersamplingRadiusOffset; // if using temporal supersampling approach (like "Progressive
                                                        // Rendering Using Multi-frame Sampling" from GPU Pro 7, etc.)
        const int subPassCount = 5;
        for (int subPass = 0; subPass < subPassCount; subPass++) {
            int a = pass;
            int b = subPass;

            int spmap[5]{0, 1, 4, 3, 2};
            b = spmap[subPass];

            float ca, sa;
            float angle0 = ((float) a + (float) b / (float) subPassCount) * (3.1415926535897932384626433832795f) * 0.5f;
            angle0 += additionalAngleOffset;

            ca = ::cosf(angle0);
            sa = ::sinf(angle0);

            float scale = 1.0f + (a - 1.5f + (b - (subPassCount - 1.0f) * 0.5f) / (float) subPassCount) * 0.07f;
            scale *= additionalRadiusScale;

            consts.PatternRotScaleMatrices[subPass] = glm::vec4(scale * ca, scale * -sa, -scale * sa, -scale * ca);
        }

        if (!generateNormals) {
            consts.NormalsUnpackMul = inputs->NormalsUnpackMul;
            consts.NormalsUnpackAdd = inputs->NormalsUnpackAdd;
        } else {
            consts.NormalsUnpackMul = 2.0f;
            consts.NormalsUnpackAdd = -1.0f;
        }
        consts.DetailAOStrength = settings.DetailShadowStrength;
        consts.Dummy0 = 0.0f;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
        if (!generateNormals) {
            consts.NormalsWorldToViewspaceMatrix = inputs->NormalsWorldToViewspaceMatrix;
            if (!inputs->MatricesRowMajorOrder)
                consts.NormalsWorldToViewspaceMatrix.Transpose();
        } else {
            consts.NormalsWorldToViewspaceMatrix.SetIdentity();
        }
#endif

        // probably do something with the ssbo? but could also just be done at this point
        m_ssbo_constants->bind();
        m_ssbo_constants->rebuffer(&m_constants, sizeof(m_constants));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
}

// only resets textures if needed
bool megamol::compositing::ASSAO::reCreateIfNeeded(
    std::shared_ptr<glowl::Texture2D> tex, glm::ivec2 size, const glowl::TextureLayout& ly) {
    if ((size.x == 0) || (size.y == 0)) {
        tex.reset();
    } else {
        if (tex != nullptr) {
            glowl::TextureLayout desc = tex->getTextureLayout();
            if (equalLayoutsWithoutSize(desc, ly) && (desc.width == size.x) && (desc.height == size.y))
                return false;
        }

        glowl::TextureLayout desc = ly;
        desc.width = size.x;
        desc.height = size.y;
        tex->reload(desc, nullptr);
    }

    return true;
}

bool megamol::compositing::ASSAO::reCreateIfNeeded(
    std::shared_ptr<glowl::Texture2DArray> tex, glm::ivec2 size, const glowl::TextureLayout& ly) {
    if ((size.x == 0) || (size.y == 0)) {
        tex.reset();
    } else {
        if (tex != nullptr) {
            glowl::TextureLayout desc = tex->getTextureLayout();
            if (equalLayoutsWithoutSize(desc, ly) && (desc.width == size.x) && (desc.height == size.y))
                return false;
        }

        glowl::TextureLayout desc = ly;
        desc.width = size.x;
        desc.height = size.y;
        //tex->reload(desc, nullptr);
        // TODO: is this okay? Oo
        tex->~Texture2DArray();
        tex = std::make_shared<glowl::Texture2DArray>("m_finalResults", desc, nullptr);
    }

    return true;
}

bool megamol::compositing::ASSAO::reCreateArrayIfNeeded(std::shared_ptr<glowl::Texture2D> tex,
    std::shared_ptr<glowl::Texture2DArray> original, glm::ivec2 size, const glowl::TextureLayout& ly, int arraySlice) {
    if ((size.x == 0) || (size.y == 0)) {
        tex.reset();
    } else {
        if (tex != nullptr) {
            glowl::TextureLayout desc = tex->getTextureLayout();
            if (equalLayoutsWithoutSize(desc, ly) && (desc.width == size.x) && (desc.height == size.y))
                return false;
        }

        glowl::TextureLayout desc = ly;
        desc.width = size.x;
        desc.height = size.y;
        tex->reload(desc, nullptr);
    }

    return true;
}

bool megamol::compositing::ASSAO::reCreateMIPViewIfNeeded(
    std::shared_ptr<glowl::Texture2D> current, std::shared_ptr<glowl::Texture2D> original, int mipViewSlice) {
    if ( equalLayouts(current->getTextureLayout(), original->getTextureLayout()) )
        return true;

    current.reset();

    glowl::TextureLayout current_layout = current->getTextureLayout();
    glowl::TextureLayout original_layout = original->getTextureLayout();

    int new_width = original_layout.width;
    int new_height = original_layout.height;

    for (int i = 0; i < mipViewSlice; ++i) {
        new_width = (new_width + 1) / 2;
        new_height = (new_height + 1) / 2;
    }
    new_width = std::max( new_width, 1 );
    new_height = std::max( new_height, 1 );

    current_layout.width = new_width;
    current_layout.height = new_height;

    current->reload(current_layout, nullptr);

    return true;
}

bool megamol::compositing::ASSAO::equalLayoutsWithoutSize(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs) {
    bool depth = lhs.depth == rhs.depth;
    bool float_parameters = lhs.float_parameters == rhs.float_parameters;
    bool format = lhs.format == rhs.format;
    //bool height = lhs.height == rhs.height;
    bool internal_format = lhs.internal_format == rhs.internal_format;
    bool int_parameters = lhs.int_parameters == rhs.int_parameters;
    bool levels = lhs.levels == rhs.levels;
    bool type = lhs.type == rhs.type;
    //bool width = lhs.width == rhs.width;

    return depth && float_parameters && format /*&& height*/ && internal_format && int_parameters && levels &&
           type /*&& width*/;
}

bool megamol::compositing::ASSAO::equalLayouts(
    const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs) {
    bool depth = lhs.depth == rhs.depth;
    bool float_parameters = lhs.float_parameters == rhs.float_parameters;
    bool format = lhs.format == rhs.format;
    bool height = lhs.height == rhs.height;
    bool internal_format = lhs.internal_format == rhs.internal_format;
    bool int_parameters = lhs.int_parameters == rhs.int_parameters;
    bool levels = lhs.levels == rhs.levels;
    bool type = lhs.type == rhs.type;
    bool width = lhs.width == rhs.width;

    return depth && float_parameters && format && height && internal_format && int_parameters && levels &&
           type && width;
}
