/*
 * AntiAliasing.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ANTI_ALIASING_H_INCLUDED
#define ANTI_ALIASING_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib_gl/graphics/gl/GLSLComputeShader.h"

#include "glowl/BufferObject.hpp"
#include "glowl/Texture2D.hpp"

#include <glm/ext.hpp>
#include <glm/glm.hpp>

namespace megamol {
namespace compositing {

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

class AntiAliasing : public core::Module {
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

    AntiAliasing();
    ~AntiAliasing();

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
    typedef vislib_gl::graphics::gl::GLSLComputeShader GLSLComputeShader;

    /**
     * \brief Sets GUI parameter slot visibility depending on antialiasing technique.
     */
    bool visibilityCallback(core::param::ParamSlot& slot);

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
#ifdef PROFILING
    frontend_resources::PerformanceManager::handle_vector m_timers;
    frontend_resources::PerformanceManager* m_perf_manager = nullptr;
#endif

    std::vector<unsigned char> m_area;
    std::vector<unsigned char> m_search;

    uint32_t m_version;

    /** Shader program to copy a texture to another */
    std::unique_ptr<GLSLComputeShader> m_copy_prgm;

    /** Shader program for fxaa */
    std::unique_ptr<GLSLComputeShader> m_fxaa_prgm;

    /** Shader programs for smaa */
    std::unique_ptr<GLSLComputeShader> m_smaa_edge_detection_prgm;
    std::unique_ptr<GLSLComputeShader> m_smaa_blending_weight_calculation_prgm;
    std::unique_ptr<GLSLComputeShader> m_smaa_neighborhood_blending_prgm;

    /** Configurable settings for smaa */
    SMAAConstants m_smaa_constants;
    SMAAConstants m_smaa_custom_constants;
    std::shared_ptr<glowl::BufferObject> m_ssbo_constants;

    /** SMAA depth texture for depth based edge detection */
    std::shared_ptr<glowl::Texture2D> m_depth_tx2D;

    /** SMAA intermediate texture layout */
    glowl::TextureLayout m_smaa_layout;

    /** Texture that the combination result will be written to */
    std::shared_ptr<glowl::Texture2D> m_output_tx2D;

    /** Texture to store edges from edges detection */
    std::shared_ptr<glowl::Texture2D> m_edges_tx2D;

    /** Texture holding the blending factors for the coverage areas */
    std::shared_ptr<glowl::Texture2D> m_blending_weights_tx2D;

    /** Texture holding the blending factors for the coverage areas */
    std::shared_ptr<glowl::Texture2D> m_area_tx2D;

    /** Texture holding the blending factors for the coverage areas */
    std::shared_ptr<glowl::Texture2D> m_search_tx2D;

    /** Parameter for selecting the antialiasing technique, e.g. smaa, fxaa, no aa */
    megamol::core::param::ParamSlot m_mode;

    /** Parameter for selecting the smaa technique: SMAA 1x, SMAA S2x, SMAA T2x, or SMAA 4x
     * SMAA 1x:  basic version of SMAA
     * SMAA S2x: includes all SMAA 1x features plus spatial multismapling (not implemented yet)
     * SMAA T2x: includes all SMAA 1x features plus temporal multisampling (not implemented yet)
     * SMAA 4x:  includes all SMAA 1x features plus spatial and temporal multi/supersampling (not implemented yet)
     */
    megamol::core::param::ParamSlot m_smaa_mode;

    /** Parameter for selecting which texture to show, e.g. final output, edges, or weights */
    megamol::core::param::ParamSlot m_smaa_view;

    /** Parameter for selecting the smaa quality level
     * as stated in the original work http://www.iryoku.com/smaa/
     * LOW    (60% of the quality)
     * MEDIUM (80% of the quality)
     * HIGH   (95% of the quality)
     * ULTRA  (99% of the quality)
     */
    megamol::core::param::ParamSlot m_smaa_quality;

    /** Slot for smaa threshold parameter */
    megamol::core::param::ParamSlot m_smaa_threshold;

    /** Slot for smaa maximum search steps parameter */
    megamol::core::param::ParamSlot m_smaa_max_search_steps;

    /** Slot for smaa maximum diag search steps parameter */
    megamol::core::param::ParamSlot m_smaa_max_search_steps_diag;

    /** Slot for smaa diag detection disable parameter */
    megamol::core::param::ParamSlot m_smaa_disable_diag_detection;

    /** Slot for smaa corner detection disable parameter */
    megamol::core::param::ParamSlot m_smaa_disable_corner_detection;

    /** Slot for smaa corner rounding parameter */
    megamol::core::param::ParamSlot m_smaa_corner_rounding;

    /** Parameter for choosing the edge detection technique: based on Luma, Color, or Depth */
    megamol::core::param::ParamSlot m_smaa_detection_technique;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_output_tex_slot;

    /** Slot for optionally querying an input texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_input_tex_slot;

    /** Slot for optionally querying a depth texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_depth_tex_slot;


    bool m_settings_have_changed;
};

} // namespace compositing
} // namespace megamol

#endif // !ANTI_ALIASING_H_INCLUDED
