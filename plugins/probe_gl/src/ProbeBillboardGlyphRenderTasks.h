/*
 * ProbeBillboardGlyphRenderTasks.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <typeindex>

#include "mesh_gl/AbstractGPURenderTaskDataSource.h"

#include "probe/ProbeCollection.h"

#include <imgui.h>

namespace megamol {
namespace probe_gl {

class ProbeBillboardGlyphRenderTasks : public mesh_gl::AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ProbeBillboardGlyphRenderTasks";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    bool create();

    void release();

    ProbeBillboardGlyphRenderTasks();
    ~ProbeBillboardGlyphRenderTasks();

protected:
    bool getDataCallback(core::Call& caller);

    bool getMetaDataCallback(core::Call& caller);

private:
    uint32_t m_version;

    core::CallerSlot m_transfer_function_Slot;

    core::CallerSlot m_probes_slot;

    core::CallerSlot m_event_slot;

    core::param::ParamSlot m_billboard_size_slot;

    core::param::ParamSlot m_rendering_mode_slot;

    core::param::ParamSlot m_use_interpolation_slot;

    core::param::ParamSlot m_show_canvas_slot;

    core::param::ParamSlot m_canvas_color_slot;

    std::shared_ptr<mesh_gl::GPUMaterialCollection> m_material_collection;

    std::shared_ptr<glowl::Mesh> m_billboard_dummy_mesh;

    std::shared_ptr<glowl::Texture2D> m_transfer_function;

    std::array<float, 2> m_tf_range;

    ImGuiContext* m_imgui_context;

    /**
     *
     */

    struct PerFrameData {
        int use_interpolation;

        int show_canvas;

        int padding0;
        int padding1;

        std::array<float, 4> canvas_color;

        GLuint64 tf_texture_handle;
        float tf_min;
        float tf_max;
    };

    struct TexturedGlyphData {
        glm::vec4 position;
        GLuint64 texture_handle;
        float slice_idx;
        float scale;
    };

    struct GlyphVectorProbeData {
        glm::vec4 position;
        glm::vec4 probe_direction;
        float scale;

        int probe_id;
        int state;

        float sample_cnt;
        std::array<float, 4> samples[32];
    };

    struct GlyphScalarProbeData {
        glm::vec4 position;
        glm::vec4 probe_direction;
        float scale;

        float sample_cnt;
        float samples[32];

        int probe_id;
        int state;
    };

    struct GlyphScalarDistributionProbeData {
        glm::vec4 position;
        glm::vec4 probe_direction;
        float scale;

        int probe_id;
        int state;

        float sample_cnt;
        std::array<float, 4> samples[32];
    };

    struct GlyphClusterIDData {
        glm::vec4 position;
        glm::vec4 probe_direction;
        float scale;

        int probe_id;
        int state;

        float sample_cnt;

        int cluster_id;
        int total_cluster_cnt; // we have some space to spare per glyph so why not...
        int padding1;
        int padding2;
    };

    bool m_show_glyphs;

    std::vector<std::pair<std::type_index, size_t>> m_type_index_map;

    std::vector<std::string> m_textured_glyph_identifiers;
    std::vector<std::string> m_vector_probe_glyph_identifiers;
    std::vector<std::string> m_scalar_probe_glyph_identifiers;
    std::vector<std::string> m_scalar_distribution_probe_glyph_identifiers;
    std::vector<std::string> m_clusterID_glyph_identifiers;

    std::vector<TexturedGlyphData> m_textured_glyph_data;
    std::vector<GlyphVectorProbeData> m_vector_probe_glyph_data;
    std::vector<GlyphScalarProbeData> m_scalar_probe_glyph_data;
    std::vector<GlyphScalarDistributionProbeData> m_scalar_distribution_probe_glyph_data;
    std::vector<GlyphClusterIDData> m_clusterID_glyph_data;

    std::vector<glowl::DrawElementsCommand> m_textured_gylph_draw_commands;
    std::vector<glowl::DrawElementsCommand> m_vector_probe_gylph_draw_commands;
    std::vector<glowl::DrawElementsCommand> m_scalar_probe_gylph_draw_commands;
    std::vector<glowl::DrawElementsCommand> m_scalar_distribution_probe_gylph_draw_commands;
    std::vector<glowl::DrawElementsCommand> m_clusterID_gylph_draw_commands;

    bool addAllRenderTasks();

    void updateAllRenderTasks();

    void clearAllRenderTasks();

    template<typename ProbeType>
    TexturedGlyphData createTexturedGlyphData(
        ProbeType const& probe, int probe_id, GLuint64 texture_handle, float slice_idx, float scale);

    GlyphScalarProbeData createScalarProbeGlyphData(probe::FloatProbe const& probe, int probe_id, float scale);

    GlyphScalarDistributionProbeData createScalarDistributionProbeGlyphData(
        probe::FloatDistributionProbe const& probe, int probe_id, float scale);

    GlyphVectorProbeData createVectorProbeGlyphData(probe::Vec4Probe const& probe, int probe_id, float scale);

    GlyphClusterIDData createClusterIDGlyphData(probe::BaseProbe const& probe, int probe_id, float scale);
};

template<typename ProbeType>
inline ProbeBillboardGlyphRenderTasks::TexturedGlyphData ProbeBillboardGlyphRenderTasks::createTexturedGlyphData(
    ProbeType const& probe, int probe_id, GLuint64 texture_handle, float slice_idx, float scale) {
    TexturedGlyphData glyph_data;
    glyph_data.position = glm::vec4(probe.m_position[0] + probe.m_direction[0] * (probe.m_begin * 1.1f),
        probe.m_position[1] + probe.m_direction[1] * (probe.m_begin * 1.1f),
        probe.m_position[2] + probe.m_direction[2] * (probe.m_begin * 1.1f), 1.0f);
    glyph_data.texture_handle = texture_handle;
    glyph_data.slice_idx = slice_idx;
    glyph_data.scale = scale;

    // glyph_data.probe_id = probe_id;

    return glyph_data;
}

} // namespace probe_gl
} // namespace megamol
