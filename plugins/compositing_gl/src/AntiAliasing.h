/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glowl/glowl.h>

#include "CompositingOutHandler.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"

#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/special/TextureInspector.h"

namespace megamol::compositing_gl {

/**
 * Struct for all the SMAA configurable settings.
 * For further details see the original code: https://github.com/iryoku/smaa
 * (SMAA.hlsl)
 */
struct SMAAConstants {
    float Smaa_threshold;
    float Smaa_depth_threshold;
    int Max_search_steps;
    int Max_search_steps_diag;

    int Corner_rounding;
    float Corner_rounding_norm;
    float Local_contrast_adaptation_factor;
    int Predication;

    float Predication_threshold;
    float Predication_scale;
    float Predication_strength;
    int Reprojection;

    float Reprojection_weight_scale;
    int Disable_diag_detection;
    int Disable_corner_detection;
    int Decode_velocity;

    glm::vec4 Rt_metrics;

    // set default to "HIGH"
    SMAAConstants() {
        Smaa_threshold = 0.1f; // Range: [0, 0.5]
        Smaa_depth_threshold = 0.1f * Smaa_threshold;
        Max_search_steps = 16;     // Range: [0, 112]
        Max_search_steps_diag = 8; // Range: [0, 20]
        Corner_rounding = 25;      // Range: [0, 100]
        Corner_rounding_norm = Corner_rounding / 100.f;
        Local_contrast_adaptation_factor = 2.f;
        Predication = false;
        Predication_threshold = 0.01f; // Range: depends on input
        Predication_scale = 2.f;       // Range: [1, 5]
        Predication_strength = 0.4f;   // Range: [0, 1]
        Reprojection = false;
        Reprojection_weight_scale = 30.f; // Range: [0, 80]
        Disable_diag_detection = false;
        Disable_corner_detection = false;
        Decode_velocity = false;
        Rt_metrics = glm::vec4(0.f, 0.f, 0.f, 0.f);
    }
};

class AntiAliasing : public mmstd_gl::ModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "AntiAliasing";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Compositing module that computes antialiasing";
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

    AntiAliasing();
    ~AntiAliasing() override;

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
     * Resets previously set GL states
     */
    inline void resetGLStates() {
        glUseProgram(0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(ssbo_constants_->getTarget(), 0);
    }

    /**
     * \brief Sets GUI parameter slot visibility depending on antialiasing technique.
     */
    bool visibilityCallback(core::param::ParamSlot& slot);

    /**
     * \brief Sets Texture format variables and recompiles shaders.
     */
    bool setTextureFormatCallback();

    /**
     * \brief Sets the setting parameter values of the SMAAConstants struct depending
     * on the chosen quality level.
     */
    bool setSettingsCallback(core::param::ParamSlot& slot);

    /**
     * \brief If the quality level is set to custom, this method sets the required parameter
     * values of the SMAAConstants struct.
     */
    bool setCustomSettingsCallback(core::param::ParamSlot& slot);

    /**
     * \brief First pass. Calculates edges of given input.
     *
     * \param input The input texture (texture to smooth) from which the edges are calculated.
     * \param depth The depth texture from corresponding to the input. Used for depth based edge detection.
     * \param edges The output texture to which the edges are written.
     * \param detection_technique The selected technqiue on which edge detection is based: color, luma, or depth.
     */
    void edgeDetection(const std::shared_ptr<glowl::Texture2D>& input, const std::shared_ptr<glowl::Texture2D>& depth,
        const std::shared_ptr<glowl::Texture2D>& edges, GLint detection_technique);

    /**
     * \brief Second pass. Calculates the weights of the found edges from edgeDetection function.
     *
     * \param edges Edge texture from previous edgeDetection pass.
     * \param area Area texture used as lookup texture to properly calculate the coverage area of an edge.
     * \param search Search texture to find the distances to surrounding edges.
     * \param weights The output texture to which the weights are written.
     */
    void blendingWeightCalculation(const std::shared_ptr<glowl::Texture2D>& edges,
        const std::shared_ptr<glowl::Texture2D>& area, const std::shared_ptr<glowl::Texture2D>& search,
        const std::shared_ptr<glowl::Texture2D>& weights);

    /**
     * \brief Final pass. Blends neighborhood pixels using the weight texture from the 2nd pass.
     * Gives final result.
     *
     * \param input The input texture to smooth.
     * \param weights The weights texture from previous blendingWeightCalculation pass. Used to blend edges.
     * \param result The resulting smoothed texture.
     */
    void neighborhoodBlending(const std::shared_ptr<glowl::Texture2D>& input,
        const std::shared_ptr<glowl::Texture2D>& weights, const std::shared_ptr<glowl::Texture2D>& result);

    /**
     * \brief Performs AntiAliasing based on FXAA.
     *
     * \param input The input texture to smooth.
     * \param output The resulting smoothed texture.
     */
    void fxaa(const std::shared_ptr<glowl::Texture2D>& input, const std::shared_ptr<glowl::Texture2D>& output);

    /**
     * Copies a src texture to the tgt texture.
     * Caution: this only works if the tgt texture is of format rgba16f!
     * Use Texture2D::copy otherwise.
     */
    void copyTextureViaShader(
        const std::shared_ptr<glowl::Texture2D>& tgt, const std::shared_ptr<glowl::Texture2D>& src);


    // profiling
#ifdef MEGAMOL_USE_PROFILING
    frontend_resources::PerformanceManager::handle_vector timers_;
    frontend_resources::PerformanceManager* perf_manager_ = nullptr;
#endif

    std::vector<unsigned char> area_;
    std::vector<unsigned char> search_;

    uint32_t version_;

    mmstd_gl::special::TextureInspector tex_inspector_;

    /** Shader program to copy a texture to another */
    std::unique_ptr<glowl::GLSLProgram> copy_prgm_;

    /** Shader program for fxaa */
    std::unique_ptr<glowl::GLSLProgram> fxaa_prgm_;

    /** Shader programs for smaa */
    std::unique_ptr<glowl::GLSLProgram> smaa_edge_detection_prgm_;
    std::unique_ptr<glowl::GLSLProgram> smaa_blending_weight_calculation_prgm_;
    std::unique_ptr<glowl::GLSLProgram> smaa_neighborhood_blending_prgm_;

    /** Configurable settings for smaa */
    SMAAConstants smaa_constants_;
    SMAAConstants constants_;
    std::shared_ptr<glowl::BufferObject> ssbo_constants_;

    /** SMAA depth texture for depth based edge detection */
    std::shared_ptr<glowl::Texture2D> depth_tx2D_;

    /** SMAA intermediate texture layout */
    glowl::TextureLayout smaa_layout_;

    /** Texture that the combination result will be written to */
    std::shared_ptr<glowl::Texture2D> output_tx2D_;

    /** Texture to store edges from edges detection */
    std::shared_ptr<glowl::Texture2D> edges_tx2D_;

    /** Texture holding the blending factors for the coverage areas */
    std::shared_ptr<glowl::Texture2D> blending_weights_tx2D_;

    /** Texture holding the blending factors for the coverage areas */
    std::shared_ptr<glowl::Texture2D> area_tx2D_;

    /** Texture holding the blending factors for the coverage areas */
    std::shared_ptr<glowl::Texture2D> search_tx2D_;

    /** Parameter for selecting the antialiasing technique, e.g. smaa, fxaa, no aa */
    megamol::core::param::ParamSlot mode_;


    /** Parameter for selecting the smaa technique: SMAA 1x, SMAA S2x, SMAA T2x, or SMAA 4x
     * SMAA 1x:  basic version of SMAA
     * SMAA S2x: includes all SMAA 1x features plus spatial multismapling (not implemented yet)
     * SMAA T2x: includes all SMAA 1x features plus temporal multisampling (not implemented yet)
     * SMAA 4x:  includes all SMAA 1x features plus spatial and temporal multi/supersampling (not implemented yet)
     */
    megamol::core::param::ParamSlot smaa_mode_;

    /** Parameter for selecting the smaa quality level
     * as stated in the original work http://www.iryoku.com/smaa/
     * LOW    (60% of the quality)
     * MEDIUM (80% of the quality)
     * HIGH   (95% of the quality)
     * ULTRA  (99% of the quality)
     */
    megamol::core::param::ParamSlot smaa_quality_;

    /** Slot for smaa threshold parameter */
    megamol::core::param::ParamSlot smaa_threshold_;

    /** Slot for smaa maximum search steps parameter */
    megamol::core::param::ParamSlot smaa_max_search_steps_;

    /** Slot for smaa maximum diag search steps parameter */
    megamol::core::param::ParamSlot smaa_max_search_steps_diag_;

    /** Slot for smaa diag detection disable parameter */
    megamol::core::param::ParamSlot smaa_disable_diag_detection_;

    /** Slot for smaa corner detection disable parameter */
    megamol::core::param::ParamSlot smaa_disable_corner_detection_;

    /** Slot for smaa corner rounding parameter */
    megamol::core::param::ParamSlot smaa_corner_rounding_;

    /** Parameter for choosing the edge detection technique: based on Luma, Color, or Depth */
    megamol::core::param::ParamSlot smaa_detection_technique_;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot output_tex_slot_;

    /** Slot for optionally querying an input texture, i.e. a rhs connection */
    megamol::core::CallerSlot input_tex_slot_;

    /** Slot for optionally querying a depth texture, i.e. a rhs connection */
    megamol::core::CallerSlot depth_tex_slot_;

    CompositingOutHandler outHandler_;

    bool settings_have_changed_;
};

} // namespace megamol::compositing_gl
