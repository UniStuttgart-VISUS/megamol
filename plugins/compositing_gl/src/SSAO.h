/*
 * ASSAO.h
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

#pragma once

#include <variant>

#include <glm/glm.hpp>
#include <glowl/Sampler.hpp>
#include <glowl/Texture2DView.hpp>
#include <glowl/glowl.h>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/ModuleGL.h"

namespace megamol::compositing_gl {

typedef std::tuple<std::shared_ptr<glowl::Texture2D>, std::string, std::shared_ptr<glowl::Sampler>> TextureSamplerTuple;
typedef std::tuple<std::shared_ptr<glowl::Texture2DView>, std::string, std::shared_ptr<glowl::Sampler>>
    TextureViewSamplerTuple;
typedef std::tuple<std::shared_ptr<glowl::Texture2DArray>, std::string, std::shared_ptr<glowl::Sampler>>
    TextureArraySamplerTuple;

struct ASSAO_Inputs {
    // Custom viewports not supported yet; this is here for future support. ViewportWidth and ViewportHeight must
    // match or be smaller than source depth and normalmap sizes.
    int ViewportX;
    int ViewportY;
    int ViewportWidth;
    int ViewportHeight;

    // Used for expanding UINT normals from [0, 1] to [-1, 1] if needed.
    float NormalsUnpackMul;
    float NormalsUnpackAdd;

    // Incoming textures from callerslots
    bool GenerateNormals;

    // Transformation Matrices
    glm::mat4 ProjectionMatrix;
    glm::mat4 ViewMatrix;

    ASSAO_Inputs() {
        ViewportX = 0; // stays constant
        ViewportY = 0; // stays constant
        ViewportWidth = 0;
        ViewportHeight = 0;
        NormalsUnpackMul = 2.0f;  // stays constant
        NormalsUnpackAdd = -1.0f; // stays constant
        GenerateNormals = false;
        ProjectionMatrix = glm::mat4(1.0f);
        ViewMatrix = glm::mat4(1.0f);
    }

    virtual ~ASSAO_Inputs() {}
};

// effect visual settings
struct ASSAO_Settings {
    float Radius;                // [0.0,  ~ ] World (view) space size of the occlusion sphere.
    float ShadowMultiplier;      // [0.0, 5.0] Effect strength linear multiplier
    float ShadowPower;           // [0.5, 5.0] Effect strength pow modifier
    float ShadowClamp;           // [0.0, 1.0] Effect max limit (applied after multiplier but before blur)
    float HorizonAngleThreshold; // [0.0, 0.2] Limits self-shadowing (makes the sampling area less of a hemisphere, more
                                 // of a spherical cone, to avoid self-shadowing and various artifacts due to low
                                 // tessellation and depth buffer imprecision, etc.)
    float FadeOutFrom;           // [0.0, ~] Distance to start start fading out the effect.
    float FadeOutTo;             // [0.0, ~] Distance at which the effect is faded out.
    int QualityLevel; // [ -1, 3] Effect quality; -1 - lowest (low, half res checkerboard), 0 - low, 1 - medium, 2 -
                      // high, 3 - very high / adaptive; each quality level is roughly 2x more costly than the previous,
                      // except the q3 which is variable but, in general, above q2.
    float AdaptiveQualityLimit; // [0.0, 1.0] (only for Quality Level 3)
    int BlurPassCount; // [0, 6] Number of edge-sensitive smart blur passes to apply. Quality 0 is an exception with
                       // only one 'dumb' blur pass used.
    float Sharpness; // [0.0, 1.0] (How much to bleed over edges; 1: not at all, 0.5: half-half; 0.0: completely ignore
                     // edges)
    float TemporalSupersamplingAngleOffset;  // [0.0, PI] Used to rotate sampling kernel; If using temporal AA /
                                             // supersampling, suggested to rotate by ( (frame%3)/3.0*PI ) or similar.
                                             // Kernel is already symmetrical, which is why we use PI and not 2*PI.
    float TemporalSupersamplingRadiusOffset; // [0.0, 2.0] Used to scale sampling kernel; If using temporal AA /
                                             // supersampling, suggested to scale by ( 1.0f + (((frame%3)-1.0)/3.0)*0.1
                                             // ) or similar.
    float DetailShadowStrength; // [0.0, 5.0] Used for high-res detail AO using neighboring depth pixels: adds a lot of
                                // detail but also reduces temporal stability (adds aliasing).

    ASSAO_Settings() {
        QualityLevel = 2;
        BlurPassCount = 2;
        Radius = 1.2f;
        ShadowMultiplier = 1.0f;
        ShadowPower = 1.50f;
        FadeOutFrom = 50.0f;
        FadeOutTo = 300.0f;
        Sharpness = 0.98f;
        DetailShadowStrength = 0.5f;
        HorizonAngleThreshold = 0.06f;
        ShadowClamp = 0.98f;

        AdaptiveQualityLimit = 0.45f;
        TemporalSupersamplingAngleOffset = 0.0f;
        TemporalSupersamplingRadiusOffset = 1.0f;
    }
};

// ** WARNING ** if changing anything here, update the corresponding shader code! ** WARNING **
struct ASSAO_Constants {
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
    int TransformNormalsToViewSpace;

    glm::mat4 ViewMX;

#if SSAO_ENABLE_NORMAL_WORLD_TO_VIEW_CONVERSION
    ASSAO_Float4x4 NormalsWorldToViewspaceMatrix;
#endif
};

class SSAO : public mmstd_gl::ModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SSAO";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Compositing module that computes screen space ambient occlusion";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

#ifdef MEGAMOL_USE_PROFILING
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        ModuleGL::requested_lifetime_resources(req);
        req.require<frontend_resources::PerformanceManager>();
    }
#endif

    SSAO();
    ~SSAO() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * TODO
     */
    bool getDataCallback(core::Call& caller);

    /**
     * TODO
     */
    bool getMetaDataCallback(core::Call& caller);

private:
    /**
     * De-interleaves the full resolution depth buffer into two or four depths buffers of
     * half or quarter size, respectively, depending on the chosen quality level.
     * If there is no normal texture given, this function additionally approximates
     * normals from depth and creates a corresponding normal texture for further use.
     *
     * \param settings Struct containing various quality tweaks, e.g. radius of occlusion sphere.
     * \param inputs Struct containing essential required inputs such as viewport dimensions or transformation matrices.
     * \param depthTexture The (full resolution) depth texture to de-interleave.
     * \param normalTexture The target texture where possibly generated normals are stored into.
     */
    void prepareDepths(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs,
        std::shared_ptr<glowl::Texture2D> depthTexture, std::shared_ptr<glowl::Texture2D> normalTexture);

    /**
     * Generates the occlusion for all four de-interleaved parts. For each of the four parts,
     * first the occlusion gets generated immediatly followed by a number of blur passes.
     *
     * \param settings Struct containing various quality tweaks, e.g. the number of blur passes.
     * \param inputs Struct containing essential required inputs such as viewport dimensions or transformation matrices.
     * \param adaptiveBasePass Indicates whether the adaptive approach of ASSAO is used or not. Currently not used!
     * \param normalTexture The normal texture to be used for generating the occlusion.
     */
    void generateSSAO(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, bool adaptiveBasePass,
        std::shared_ptr<glowl::Texture2D> normalTexture);

    /**
     * \brief A general function to dispatch the compute shaders depending on the parameters
     * of its signature.
     *
     * \param prgm The program to use containing the compute shader.
     * \param inputTextures Textures functioning as read-only input textures accessed.
     * \param outputTextures Target textures in which the compute shader result is stored into.
     * \param addConstants Indicates whether the compute shader requires the ASSAO_Constants struct (SSBO).
     * \param finals Input texture array containing the four (blurred) de-interleaved occlusion textures.
     *               Only used in the last pass of the ASSAO algorithm.
     */
    template<typename Tuple, typename Tex>
    void fullscreenPassDraw(const std::unique_ptr<glowl::GLSLProgram>& prgm, const std::vector<Tuple>& inputTextures,
        std::vector<std::pair<std::shared_ptr<Tex>, GLuint>>& outputTextures, bool addConstants = true,
        const std::vector<TextureArraySamplerTuple>& finals = {});

    /**
     * \brief Checks both layouts for equality.
     *
     * \return True, if both layouts are equal. False otherwise.
     */
    bool equalLayouts(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs);

    /**
     * \brief Checks both layouts for equality without checking their sizes.
     *
     * @return True, if both layouts are equal. False otherwise.
     */
    bool equalLayoutsWithoutSize(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs);

    /**
     * \brief Updates the required textures (see also SSAO::reCreateIfNeeded) if the inputs (e.g. viewport dimensions) don't match with
     * the texture properties.
     */
    void updateTextures(const std::shared_ptr<ASSAO_Inputs> inputs);

    /**
     * \brief Updates the ASSAO_Constants struct depending on the settings and inputs.
     */
    void updateConstants(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, int pass);

    /**
     * \brief Reload of a texture (see Texture2D::reload) if the given layout ly doesn't match the
     * texture layout of tex. Used inside of SSAO::updateTextures.
     *
     * @return True, if reloading the texture is required, false otherwise.
     */
    bool reCreateIfNeeded(std::shared_ptr<glowl::Texture2D> tex, glm::ivec2 size, const glowl::TextureLayout& ly,
        bool generateMipMaps = false);

    /**
     * \brief Reload of a texture array (see Texture2D::reload) if the given layout ly doesn't match the
     * texture layout of tex. Used inside of SSAO::updateTextures.
     *
     * @return True, if reloading the texture is required, false otherwise.
     */
    bool reCreateIfNeeded(std::shared_ptr<glowl::Texture2DArray> tex, glm::ivec2 size, const glowl::TextureLayout& ly);

    /**
     * \brief Reload of a texture view (see Texture2D::reload) and binding of the view to a texture array slice
     * if the given layout of tex doesn't match the texture layout of original. Used inside of SSAO::updateTextures.
     *
     * \param tex The texture view to reload.
     * \param original The texture array to which the texture view is bound.
     * \param size Half the size of the current full resolution viewport. Only used for boundary checks.
     * \param arraySlice The slice of the texture array to which the texture view is bound.
     *
     * @return True, if reloading the texture is required, false otherwise.
     */
    bool reCreateArrayIfNeeded(std::shared_ptr<glowl::Texture2DView> tex,
        std::shared_ptr<glowl::Texture2DArray> original, glm::ivec2 size, int arraySlice);

    /**
     * \brief Reload of a texture view (see Texture2D::reload) and binding of the view to a texture mip slice
     * if the given layout of tex doesn't match the texture layout of original. Used inside of SSAO::updateTextures.
     *
     * \param tex The texture view to reload.
     * \param original The texture to which mipmap the texture view is bound.
     * \param mipViewSlice The slice of the texture mipmap to which the texture view is bound.
     *
     * @return True, if reloading the texture is required, false otherwise.
     */
    bool reCreateMIPViewIfNeeded(
        std::shared_ptr<glowl::Texture2DView> current, std::shared_ptr<glowl::Texture2D> original, int mipViewSlice);


    // callback functions
    bool settingsCallback(core::param::ParamSlot& slot);
    bool ssaoModeCallback(core::param::ParamSlot& slot);

    // profiling
#ifdef MEGAMOL_USE_PROFILING
    frontend_resources::PerformanceManager::handle_vector timers_;
    frontend_resources::PerformanceManager* perf_manager_ = nullptr;
#endif


    uint32_t version_;

    /////////////////////////////////////////////////////////////////////////
    // COMPUTE SHADER BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::unique_ptr<glowl::GLSLProgram> prepare_depths_prgm_;
    std::unique_ptr<glowl::GLSLProgram> prepare_depths_half_prgm_;
    std::unique_ptr<glowl::GLSLProgram> prepare_depths_and_normals_prgm_;
    std::unique_ptr<glowl::GLSLProgram> prepare_depths_and_normals_half_prgm_;
    std::vector<std::unique_ptr<glowl::GLSLProgram>> prepare_depth_mip_prgms_;
    std::array<std::unique_ptr<glowl::GLSLProgram>, 5> generate_prgms_;
    std::unique_ptr<glowl::GLSLProgram> smart_blur_prgm_;
    std::unique_ptr<glowl::GLSLProgram> smart_blur_wide_prgm_;
    std::unique_ptr<glowl::GLSLProgram> apply_prgm_;
    std::unique_ptr<glowl::GLSLProgram> non_smart_blur_prgm_;
    std::unique_ptr<glowl::GLSLProgram> non_smart_apply_prgm_;
    std::unique_ptr<glowl::GLSLProgram> non_smart_half_apply_prgm_;

    std::unique_ptr<glowl::GLSLProgram> naive_ssao_prgm_;
    std::unique_ptr<glowl::GLSLProgram> simple_blur_prgm_;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // TEXTURE BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::array<std::shared_ptr<glowl::Texture2D>, 4> half_depths_;
    std::vector<std::vector<std::shared_ptr<glowl::Texture2DView>>> half_depths_mip_views_;
    std::shared_ptr<glowl::Texture2D> ping_pong_half_result_a_;
    std::shared_ptr<glowl::Texture2D> ping_pong_half_result_b_;
    std::shared_ptr<glowl::Texture2DArray> final_results_;
    std::array<std::shared_ptr<glowl::Texture2DView>, 4> final_results_array_views_;
    std::shared_ptr<glowl::Texture2D> normals_;
    std::shared_ptr<glowl::Texture2D> final_output_;

    // for naive ssao
    std::shared_ptr<glowl::Texture2D> intermediate_tx2d_;
    /** Texture with random ssao kernel rotation */
    std::shared_ptr<glowl::Texture2D> ssao_kernel_rot_tx2d_;

    // samplers
    std::shared_ptr<glowl::Sampler> sampler_state_point_clamp_;
    std::shared_ptr<glowl::Sampler> sampler_state_point_mirror_;
    std::shared_ptr<glowl::Sampler> sampler_state_linear_clamp_;
    std::shared_ptr<glowl::Sampler> sampler_state_viewspace_depth_tap_;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // TEXTURE VARIABLES
    /////////////////////////////////////////////////////////////////////////
    glowl::TextureLayout depth_buffer_viewspace_linear_layout_;
    glowl::TextureLayout ao_result_layout_;
    glowl::TextureLayout normal_layout_;

    glm::ivec2 size_;
    glm::ivec2 half_size_;
    glm::ivec2 quarter_size_;
    int depth_mip_levels_;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // other constants
    int max_blur_pass_count_;
    ASSAO_Constants constants_;
    std::shared_ptr<glowl::BufferObject> ssbo_constants_;
    /** GPU buffer object for making active (point)lights available in during shading pass */
    std::unique_ptr<glowl::BufferObject> ssao_samples_;
    /////////////////////////////////////////////////////////////////////////

    /** Pointer for assao inputs */
    std::shared_ptr<ASSAO_Inputs> inputs_;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot output_tex_slot_;

    /** Slot for querying normals render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot normals_tex_slot_;

    /** Slot for querying depth render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot depth_tex_slot_;

    /** Slot for querying camera, i.e. a rhs connection */
    megamol::core::CallerSlot camera_slot_;


    /////////////////////////////////////////////////////////////////////////
    core::param::ParamSlot ps_ssao_mode_;

    // paramslots for input settings
    ASSAO_Settings settings_;

    /** Paramslot for radius of occlusion sphere (world space size) */
    core::param::ParamSlot ps_radius_;

    /** Paramslot for effect strength linear multiplier */
    core::param::ParamSlot ps_shadow_multiplier_;

    /** Paramslot for effect strength pow modifier */
    core::param::ParamSlot ps_shadow_power_;

    /** Paramslot for effect max limit (applied after multiplier but before blur) */
    core::param::ParamSlot ps_shadow_clamp_;

    /** Paramslot for self-shadowing limit */
    core::param::ParamSlot ps_horizon_angle_threshold_;

    /** Paramslot for distance to start fading out the effect */
    core::param::ParamSlot ps_fade_out_from_;

    /** Paramslot for distance at which the effect is faded out */
    core::param::ParamSlot ps_fade_out_to_;

    /** Paramslot for the ssao effect quality level */
    core::param::ParamSlot ps_quality_level_;

    /** Paramslot for adaptive quality limit (only for quality level 3) */
    core::param::ParamSlot ps_adaptive_quality_limit_;

    /** Paramslot for number of edge-sensitive smart blur passes to apply */
    core::param::ParamSlot ps_blur_pass_count_;

    /** Paramslot for how much to bleed over edges */
    core::param::ParamSlot ps_sharpness_;

    /** Paramslot for rotating sampling kernel if temporal AA / supersampling is used */
    core::param::ParamSlot ps_temporal_supersampling_angle_offset_;

    /** Paramslot for scaling sampling kernel if temporal AA / supersampling is used */
    core::param::ParamSlot ps_temporal_supersampling_radius_offset_;

    /** Paramslot for high-res detail AO using neighboring depth pixels */
    core::param::ParamSlot ps_detail_shadow_strength_;

    /** Parameter for selecting the ssao radius */
    megamol::core::param::ParamSlot ps_ssao_radius_;

    /** Parameter for selecting the ssao sample count */
    megamol::core::param::ParamSlot ps_ssao_sample_cnt_;

    bool settings_have_changed_;
    bool slot_is_active_;
    bool update_caused_by_normal_slot_change_;
    /////////////////////////////////////////////////////////////////////////
};

template<typename Tuple, typename Tex>
void SSAO::fullscreenPassDraw(const std::unique_ptr<glowl::GLSLProgram>& prgm, const std::vector<Tuple>& inputTextures,
    std::vector<std::pair<std::shared_ptr<Tex>, GLuint>>& outputTextures, bool addConstants,
    const std::vector<TextureArraySamplerTuple>& finals) {
    prgm->use();

    if (addConstants)
        ssbo_constants_->bind(0);

    int cnt = 0;

    for (int i = 0; i < inputTextures.size(); ++i) {
        if (std::get<0>(inputTextures[i]) != nullptr) {
            glActiveTexture(GL_TEXTURE0 + cnt);

            std::get<0>(inputTextures[i])->bindTexture();

            if (std::get<2>(inputTextures[i]) != nullptr)
                std::get<2>(inputTextures[i])->bindSampler(cnt);

            std::string name = std::get<1>(inputTextures[i]);
            GLint loc = prgm->getUniformLocation(name.c_str());
            glUniform1i(loc, cnt);

            ++cnt;
        }
    }

    for (const auto& tex : finals) {
        if (std::get<0>(tex) != nullptr) {
            glActiveTexture(GL_TEXTURE0 + cnt);
            std::get<0>(tex)->bindTexture();
            if (std::get<2>(tex) != nullptr)
                std::get<2>(tex)->bindSampler(cnt);
            glUniform1i(prgm->getUniformLocation(std::get<1>(tex).c_str()), cnt);
        }

        ++cnt;
    }

    for (int i = 0; i < outputTextures.size(); ++i) {
        outputTextures[i].first->bindImage(outputTextures[i].second, GL_WRITE_ONLY);
    }

    // all textures in output_textures should have the same size, so we just use the first
    glDispatchCompute(static_cast<int>(std::ceil(outputTextures[0].first->getWidth() / 8.f)),
        static_cast<int>(std::ceil(outputTextures[0].first->getHeight() / 8.f)), 1);
    ::glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    for (int i = 0; i <= cnt; ++i) {
        glBindSampler(i, 0);
    }
}

} // namespace megamol::compositing_gl
