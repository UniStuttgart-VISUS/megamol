/*
 * ASSAO.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ASSAO_H_INCLUDED
#define ASSAO_H_INCLUDED

#include <variant>

#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/BufferObject.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture2DArray.hpp"

#include <glm/glm.hpp>

namespace megamol {
namespace compositing {

struct ASSAO_Inputs {
    // Output scissor rect - used to draw AO effect to a sub-rectangle, for example, for performance reasons when
    // using wider-than-screen depth input to avoid close-to-border artifacts.
    int ScissorLeft;
    int ScissorTop;
    int ScissorRight;
    int ScissorBottom;

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
    std::shared_ptr<glowl::Texture2D> normalTexture;
    std::shared_ptr<glowl::Texture2D> depthTexture;

    // Transformation Matrices
    glm::mat4 ProjectionMatrix;
    glm::mat4 ViewMatrix;

    ASSAO_Inputs() {
        // ScissorBottom in Direct3D needs to be ScissorTop in OpenGL and vice versa
        ScissorLeft = 0;
        ScissorTop = 0;             // needs to be used as ScissorBottom in host code
        ScissorRight = 0;
        ScissorBottom = 0;          // needs to be used as ScissorTop in host code
        ViewportX = 0;              // stays constant
        ViewportY = 0;              // stays constant
        ViewportWidth = 0;
        ViewportHeight = 0;
        NormalsUnpackMul = 2.0f;    // stays constant
        NormalsUnpackAdd = -1.0f;   // stays constant
        normalTexture = nullptr;
        depthTexture = nullptr;
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
        Radius = 1.2f;
        ShadowMultiplier = 1.0f;
        ShadowPower = 1.50f;
        ShadowClamp = 0.98f;
        HorizonAngleThreshold = 0.06f;
        FadeOutFrom = 50.0f;
        FadeOutTo = 300.0f;
        AdaptiveQualityLimit = 0.45f;
        QualityLevel = 2;
        BlurPassCount = 2;
        Sharpness = 0.98f;
        TemporalSupersamplingAngleOffset = 0.0f;
        TemporalSupersamplingRadiusOffset = 1.0f;
        DetailShadowStrength = 0.5f;
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
    float Dummy0;

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
    static const char* ClassName() { return "ASSAO"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "Compositing module that compute adaptive screen space ambient occlusion"; }

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


    void prepareDepths(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs);
    void generateSSAO(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, bool adaptiveBasePass);
    void fullscreenPassDraw(const std::unique_ptr<GLSLComputeShader>& prgm,
        const std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, std::string>>& input_textures,
        std::vector<std::pair<std::shared_ptr<glowl::Texture2D>, GLuint>>& output_textures, bool add_constants = true,
        std::pair<std::shared_ptr<glowl::Texture2DArray>, std::string> finals = {nullptr, ""});

    bool equalLayouts(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs);
    bool equalLayoutsWithoutSize(const glowl::TextureLayout& lhs, const glowl::TextureLayout& rhs);
    void updateTextures(const std::shared_ptr<ASSAO_Inputs> inputs);
    void updateConstants(const ASSAO_Settings& settings, const std::shared_ptr<ASSAO_Inputs> inputs, int pass);
    bool reCreateIfNeeded(std::shared_ptr<glowl::Texture2D> tex, glm::ivec2 size, const glowl::TextureLayout& ly);
    bool reCreateIfNeeded(std::shared_ptr<glowl::Texture2DArray> tex, glm::ivec2 size, const glowl::TextureLayout& ly);
    bool reCreateArrayIfNeeded(std::shared_ptr<glowl::Texture2D> tex, std::shared_ptr<glowl::Texture2DArray> original,
        glm::ivec2 size, const glowl::TextureLayout& ly, int arraySlice);
    bool reCreateMIPViewIfNeeded(
        std::shared_ptr<glowl::Texture2D> current, std::shared_ptr<glowl::Texture2D> original, int mipViewSlice);


    uint32_t m_version;

    /////////////////////////////////////////////////////////////////////////
    // COMPUTE SHADER BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::unique_ptr<GLSLComputeShader> m_prepare_depths_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depths_half_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depths_and_normals_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depths_and_normals_half_prgm;
    std::vector<std::unique_ptr<GLSLComputeShader>> m_prepare_depth_mip_prgms;
    std::array<std::unique_ptr<GLSLComputeShader>, 5> m_generate_prgms;
    // TODO: redundant, take them out
    // -------------------
    std::unique_ptr<GLSLComputeShader> m_generate_q0_prgm;
    std::unique_ptr<GLSLComputeShader> m_generate_q1_prgm;
    std::unique_ptr<GLSLComputeShader> m_generate_q2_prgm;
    std::unique_ptr<GLSLComputeShader> m_generate_q3_prgm;
    std::unique_ptr<GLSLComputeShader> m_generate_q3_base_prgm;
    // -------------------
    std::unique_ptr<GLSLComputeShader> m_smart_blur_prgm;
    std::unique_ptr<GLSLComputeShader> m_smart_blur_wide_prgm;
    std::unique_ptr<GLSLComputeShader> m_apply_prgm;
    std::unique_ptr<GLSLComputeShader> m_non_smart_blur_prgm;
    std::unique_ptr<GLSLComputeShader> m_non_smart_apply_prgm;
    std::unique_ptr<GLSLComputeShader> m_non_smart_half_apply_prgm;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // TEXTURE BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::array<std::shared_ptr<glowl::Texture2D>, 4> m_halfDepths;
    std::vector<std::vector<std::shared_ptr<glowl::Texture2D>>> m_halfDepthsMipViews;
    std::shared_ptr<glowl::Texture2D> m_pingPongHalfResultA;
    std::shared_ptr<glowl::Texture2D> m_pingPongHalfResultB;
    std::shared_ptr<glowl::Texture2DArray> m_finalResults;
    std::array<std::shared_ptr<glowl::Texture2D>, 4> m_finalResultsArrayViews;
    std::shared_ptr<glowl::Texture2D> m_normals;
    std::shared_ptr<glowl::Texture2D> m_finalOutput;

    glowl::TextureLayout m_tx_layout_samplerStatePointClamp;
    glowl::TextureLayout m_tx_layout_samplerStatePointMirror;
    glowl::TextureLayout m_tx_layout_samplerStateLinearClamp;
    glowl::TextureLayout m_tx_layout_samplerStateViewspaceDepthTap;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // TEXTURE VARIABLES
    /////////////////////////////////////////////////////////////////////////
    glm::ivec2 m_size;
    glm::ivec2 m_halfSize;
    glm::ivec2 m_quarterSize;
    glm::ivec4 m_fullResOutScissorRect;
    glm::ivec4 m_halfResOutScissorRect;
    int m_depthMipLevels;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // other constants
    int m_max_blur_pass_count;
    ASSAO_Settings m_settings;
    ASSAO_Constants m_constants;
    std::shared_ptr<glowl::BufferObject> m_ssbo_constants;
    /////////////////////////////////////////////////////////////////////////

    // used for scissoring only
    // bool m_requiresClear;

    /** TODO */
    std::shared_ptr<ASSAO_Inputs> m_inputs;

    /** Hash value to keep track of update to the output texture */
    size_t m_output_texture_hash;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_output_tex_slot;

    /** Slot for optionally querying an input texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_input_tex_slot;

    /** Slot for querying normals render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_normals_tex_slot;

    /** Slot for querying depth render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_depth_tex_slot;

    /** Slot for querying camera, i.e. a rhs connection */
    megamol::core::CallerSlot m_camera_slot;
};

} // namespace compositing
} // namespace megamol

#endif // !ASSAO_H_INCLUDED
