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

#ifndef ASSAO_H_INCLUDED
#define ASSAO_H_INCLUDED

#include <variant>

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/BufferObject.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture2DArray.hpp"
#include "glowl/Texture2DView.hpp"
#include "glowl/Sampler.hpp"

#include <glm/glm.hpp>

namespace megamol {
namespace compositing {

    typedef std::tuple<std::shared_ptr<glowl::Texture2D>, std::string, std::shared_ptr<glowl::Sampler>>
        TextureSamplerTuple;
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
        ViewportX = 0;              // stays constant
        ViewportY = 0;              // stays constant
        ViewportWidth = 0;
        ViewportHeight = 0;
        NormalsUnpackMul = 2.0f;    // stays constant
        NormalsUnpackAdd = -1.0f;   // stays constant
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
    float FadeOutFrom;           // [0.0,  ~ ] Distance to start start fading out the effect.
    float FadeOutTo;             // [0.0,  ~ ] Distance at which the effect is faded out.
    int QualityLevel; // [ -1,  3 ] Effect quality; -1 - lowest (low, half res checkerboard), 0 - low, 1 - medium, 2 -
                      // high, 3 - very high / adaptive; each quality level is roughly 2x more costly than the previous,
                      // except the q3 which is variable but, in general, above q2.
    float AdaptiveQualityLimit; // [0.0, 1.0] (only for Quality Level 3)
    int BlurPassCount; // [  0,   6] Number of edge-sensitive smart blur passes to apply. Quality 0 is an exception with
                       // only one 'dumb' blur pass used.
    float Sharpness; // [0.0, 1.0] (How much to bleed over edges; 1: not at all, 0.5: half-half; 0.0: completely ignore
                     // edges)
    float TemporalSupersamplingAngleOffset;  // [0.0,  PI] Used to rotate sampling kernel; If using temporal AA /
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
// TODO: watch alignment
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

class ASSAO : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "SSAO"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "Compositing module that computes screen space ambient occlusion"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return true; }

    ASSAO();
    ~ASSAO();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * TODO
     */
    bool getDataCallback(core::Call& caller);

    /**
     * TODO
     */
    bool getMetaDataCallback(core::Call& caller);

private:
    typedef vislib::graphics::gl::GLSLComputeShader GLSLComputeShader;


    void prepareDepths(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs,
        std::shared_ptr<glowl::Texture2D> depthTexture, std::shared_ptr<glowl::Texture2D> normalTexture);
    void generateSSAO(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, bool adaptiveBasePass,
        std::shared_ptr<glowl::Texture2D> depthTexture, std::shared_ptr<glowl::Texture2D> normalTexture);

    template<typename Tuple, typename Tex>
    void fullscreenPassDraw(
        const std::unique_ptr<GLSLComputeShader>& prgm,
        const std::vector<Tuple>& input_textures,
        std::vector<std::pair<std::shared_ptr<Tex>, GLuint>>& output_textures,
        bool add_constants = true,
        const std::vector<TextureArraySamplerTuple>& finals = {}
    );
    
    bool equalLayouts(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs);
    bool equalLayoutsWithoutSize(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs);
    void updateTextures(const std::shared_ptr<ASSAO_Inputs> inputs);
    void updateConstants(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, int pass);
    bool reCreateIfNeeded(std::shared_ptr<glowl::Texture2D> tex, glm::ivec2 size, const glowl::TextureLayout& ly, bool generateMipMaps = false);
    bool reCreateIfNeeded(std::shared_ptr<glowl::Texture2DArray> tex, glm::ivec2 size, const glowl::TextureLayout& ly);
    bool reCreateArrayIfNeeded(std::shared_ptr<glowl::Texture2DView> tex,
        std::shared_ptr<glowl::Texture2DArray> original,
        glm::ivec2 size, const glowl::TextureLayout& ly, int arraySlice);
    bool reCreateMIPViewIfNeeded(std::shared_ptr<glowl::Texture2DView> current,
        std::shared_ptr<glowl::Texture2D> original, int mipViewSlice);


    // callback functions
    bool settingsCallback(core::param::ParamSlot& slot);
    bool ssaoModeCallback(core::param::ParamSlot& slot);


    uint32_t m_version;

    /////////////////////////////////////////////////////////////////////////
    // COMPUTE SHADER BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::unique_ptr<GLSLComputeShader> m_prepareDepthsPrgm;
    std::unique_ptr<GLSLComputeShader> m_prepareDepthsHalfPrgm;
    std::unique_ptr<GLSLComputeShader> m_prepareDepthsAndNormalsPrgm;
    std::unique_ptr<GLSLComputeShader> m_prepareDepthsAndNormalsHalfPrgm;
    std::vector<std::unique_ptr<GLSLComputeShader>> m_prepareDepthMipPrgms;
    std::array<std::unique_ptr<GLSLComputeShader>, 5> m_generatePrgms;
    std::unique_ptr<GLSLComputeShader> m_smartBlurPrgm;
    std::unique_ptr<GLSLComputeShader> m_smartBlurWidePrgm;
    std::unique_ptr<GLSLComputeShader> m_applyPrgm;
    std::unique_ptr<GLSLComputeShader> m_nonSmartBlurPrgm;
    std::unique_ptr<GLSLComputeShader> m_nonSmartApplyPrgm;
    std::unique_ptr<GLSLComputeShader> m_nonSmartHalfApplyPrgm;

    std::unique_ptr<GLSLComputeShader> m_naiveSSAOPrgm;
    std::unique_ptr<GLSLComputeShader> m_naiveSSAOBlurPrgm;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // TEXTURE BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::array<std::shared_ptr<glowl::Texture2D>, 4> m_halfDepths;
    std::vector<std::vector<std::shared_ptr<glowl::Texture2DView>>> m_halfDepthsMipViews;
    std::shared_ptr<glowl::Texture2D> m_pingPongHalfResultA;
    std::shared_ptr<glowl::Texture2D> m_pingPongHalfResultB;
    std::shared_ptr<glowl::Texture2DArray> m_finalResults;
    std::array<std::shared_ptr<glowl::Texture2DView>, 4> m_finalResultsArrayViews;
    std::shared_ptr<glowl::Texture2D> m_normals;
    std::shared_ptr<glowl::Texture2D> m_finalOutput;

    // for naive ssao
    std::shared_ptr<glowl::Texture2D> m_intermediateTx2D;
    /** Texture with random ssao kernel rotation */
    std::shared_ptr<glowl::Texture2D> m_SSAOKernelRotTx2D;

    // samplers
    std::shared_ptr<glowl::Sampler> m_samplerStatePointClamp;
    std::shared_ptr<glowl::Sampler> m_samplerStatePointMirror;
    std::shared_ptr<glowl::Sampler> m_samplerStateLinearClamp;
    std::shared_ptr<glowl::Sampler> m_samplerStateViewspaceDepthTap;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // TEXTURE VARIABLES
    /////////////////////////////////////////////////////////////////////////
    glowl::TextureLayout m_depthBufferViewspaceLinearLayout;
    glowl::TextureLayout m_AOResultLayout;
    glowl::TextureLayout m_normalLayout;

    glm::ivec2 m_size;
    glm::ivec2 m_halfSize;
    glm::ivec2 m_quarterSize;
    int m_depthMipLevels;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // other constants
    int m_maxBlurPassCount;
    ASSAO_Constants m_constants;
    std::shared_ptr<glowl::BufferObject> m_ssboConstants;
    /** GPU buffer object for making active (point)lights available in during shading pass */
    std::unique_ptr<glowl::BufferObject> m_SSAOSamples;
    /////////////////////////////////////////////////////////////////////////

    /** Pointer for assao inputs */
    std::shared_ptr<ASSAO_Inputs> m_inputs;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_outputTexSlot;

    /** Slot for querying normals render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_normalsTexSlot;

    /** Slot for querying depth render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_depthTexSlot;

    /** Slot for querying camera, i.e. a rhs connection */
    megamol::core::CallerSlot m_cameraSlot;


    /////////////////////////////////////////////////////////////////////////
    core::param::ParamSlot m_psSSAOMode;

    // paramslots for input settings
    ASSAO_Settings m_settings;

    /** Paramslot for radius of occlusion sphere (world space size) */
    core::param::ParamSlot m_psRadius;

    /** Paramslot for effect strength linear multiplier */
    core::param::ParamSlot m_psShadowMultiplier;

    /** Paramslot for effect strength pow modifier */
    core::param::ParamSlot m_psShadowPower;

    /** Paramslot for effect max limit (applied after multiplier but before blur) */
    core::param::ParamSlot m_psShadowClamp;

    /** Paramslot for self-shadowing limit */
    core::param::ParamSlot m_psHorizonAngleThreshold;

    /** Paramslot for distance to start fading out the effect */
    core::param::ParamSlot m_psFadeOutFrom;

    /** Paramslot for distance at which the effect is faded out */
    core::param::ParamSlot m_psFadeOutTo;

    /** Paramslot for the ssao effect quality level */
    core::param::ParamSlot m_psQualityLevel;

    /** Paramslot for adaptive quality limit (only for quality level 3) */
    core::param::ParamSlot m_psAdaptiveQualityLimit;

    /** Paramslot for number of edge-sensitive smart blur passes to apply */
    core::param::ParamSlot m_psBlurPassCount;

    /** Paramslot for how much to bleed over edges */
    core::param::ParamSlot m_psSharpness;

    /** Paramslot for rotating sampling kernel if temporal AA / supersampling is used */
    core::param::ParamSlot m_psTemporalSupersamplingAngleOffset;

    /** Paramslot for scaling sampling kernel if temporal AA / supersampling is used */
    core::param::ParamSlot m_psTemporalSupersamplingRadiusOffset;

    /** Paramslot for high-res detail AO using neighboring depth pixels */
    core::param::ParamSlot m_psDetailShadowStrength;

    /** Parameter for selecting the ssao radius */
    megamol::core::param::ParamSlot m_psSSAORadius;

    /** Parameter for selecting the ssao sample count */
    megamol::core::param::ParamSlot m_psSSAOSampleCnt;

    bool m_settingsHaveChanged;
    bool m_slotIsActive;
    bool m_updateCausedByNormalSlotChange;
    /////////////////////////////////////////////////////////////////////////
};

template<typename Tuple, typename Tex>
void ASSAO::fullscreenPassDraw(
    const std::unique_ptr<GLSLComputeShader>& prgm,
    const std::vector<Tuple>& inputTextures,
    std::vector<std::pair<std::shared_ptr<Tex>, GLuint>>& outputTextures,
    bool addConstants,
    const std::vector<TextureArraySamplerTuple>& finals)
{
    prgm->Enable();

    if (addConstants)
        m_ssboConstants->bind(0);

    int cnt = 0;

    for (int i = 0; i < inputTextures.size(); ++i) {
        if (std::get<0>(inputTextures[i]) != nullptr) {
            glActiveTexture(GL_TEXTURE0 + cnt);

            std::get<0>(inputTextures[i])->bindTexture();

            if (std::get<2>(inputTextures[i]) != nullptr)
                std::get<2>(inputTextures[i])->bindSampler(cnt);

            std::string name = std::get<1>(inputTextures[i]);
            GLint loc = prgm->ParameterLocation(name.c_str());
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
            glUniform1i(prgm->ParameterLocation(std::get<1>(tex).c_str()), cnt);
        }

        ++cnt;
    }

    for (int i = 0; i < outputTextures.size(); ++i) {
        outputTextures[i].first->bindImage(outputTextures[i].second, GL_WRITE_ONLY);
    }

    // all textures in output_textures should have the same size, so we just use the first
    // TODO: adjust dispatch size
    prgm->Dispatch(static_cast<int>(std::ceil(outputTextures[0].first->getWidth() / 8.f)),
        static_cast<int>(std::ceil(outputTextures[0].first->getHeight() / 8.f)), 1);

    prgm->Disable();

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    for (int i = 0; i <= cnt; ++i) {
        glBindSampler(i, 0);
    }
}

} // namespace compositing
} // namespace megamol

#endif // !ASSAO_H_INCLUDED
