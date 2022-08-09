/*
 * ProbeShellElementsRenderTasks.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmstd/event/EventCall.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "ProbeShellElementsRenderer.h"
#include "probe/ProbeCalls.h"


megamol::probe_gl::ProbeShellElementsRenderTasks::ProbeShellElementsRenderTasks()
        : m_show_elements(true)
        , m_probes_slot("Probes", "")
        , m_event_slot("Events", "")
        , m_shading_mode_slot("ShadingMode", "")
        , m_hull_color_slot("ElementsColor", "") {
    // this->m_probes_slot.SetCompatibleCall<megamol::probe::CallProbesDescription>();
    // this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);

    this->m_shading_mode_slot << new megamol::core::param::EnumParam(0);
    this->m_shading_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Color");
    this->m_shading_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "ClusterID");
    this->MakeSlotAvailable(&this->m_shading_mode_slot);

    this->m_hull_color_slot << new megamol::core::param::ColorParam(
        this->m_hull_color[0], this->m_hull_color[1], this->m_hull_color[2], 1.0f);
    this->MakeSlotAvailable(&this->m_hull_color_slot);
}

megamol::probe_gl::ProbeShellElementsRenderTasks::~ProbeShellElementsRenderTasks() {}

void megamol::probe_gl::ProbeShellElementsRenderTasks::createMaterialCollection() {
    material_collection_ = std::make_shared<mesh_gl::GPUMaterialCollection>();
    std::vector<std::filesystem::path> shaderfiles = {
        "probe_gl/hull/dfr_shell_elements_vertex.glsl", "probe_gl/hull/dfr_shell_elements_fragment.glsl"};
    material_collection_->addMaterial(this->instance(), "ProbeShellElements", shaderfiles);
}

void megamol::probe_gl::ProbeShellElementsRenderTasks::createRenderTaskCollection() {
    render_task_collection_ = std::make_shared<mesh_gl::GPURenderTaskCollection>();

    struct PerFrameData {
        int shading_mode;
    };

    std::array<PerFrameData, 1> per_frame_data;
    per_frame_data[0].shading_mode = m_shading_mode_slot.Param<core::param::EnumParam>()->Value();

    render_task_collection_->addPerFrameDataBuffer("", per_frame_data, 1);
}

void megamol::probe_gl::ProbeShellElementsRenderTasks::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {

    if (m_shading_mode_slot.IsDirty()) {
        m_shading_mode_slot.ResetDirty();

        struct PerFrameData {
            int shading_mode;
        };

        std::array<PerFrameData, 1> per_frame_data;
        per_frame_data[0].shading_mode = m_shading_mode_slot.Param<core::param::EnumParam>()->Value();

        render_task_collection_->updatePerFrameDataBuffer("", per_frame_data, 1);
    }

    if (m_hull_color_slot.IsDirty()) {
        m_hull_color_slot.ResetDirty();

        std::array<float, 4> obj_color = this->m_hull_color_slot.Param<core::param::ColorParam>()->Value();

        for (auto& batch : m_per_object_data) {
            for (auto& data : batch) {
                data.color = obj_color;
            }
        }

        for (int i = 0; i < m_batch_meshes.size(); ++i) {
            for (int j = 0; j < m_rt_identifiers[i].size(); ++j) {
                render_task_collection_->updatePerDrawData(
                    m_rt_identifiers[i][j], std::vector<PerObjectData>{m_per_object_data[i][j]});
            }
        }
    }

    bool something_has_changed = force_update;

    if (something_has_changed) {
        render_task_collection_->clear();

        m_rt_identifiers.clear();
        m_draw_commands.clear();
        m_per_object_data.clear();
        m_batch_meshes.clear();

        std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

        int counter = 0;
        for (auto& sub_mesh : mesh_collection_->getSubMeshData()) {
            auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

            if (gpu_batch_mesh != prev_mesh) {
                m_rt_identifiers.emplace_back(std::vector<std::string>());
                m_draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                m_per_object_data.emplace_back(std::vector<PerObjectData>());
                m_batch_meshes.push_back(gpu_batch_mesh);

                prev_mesh = gpu_batch_mesh;
            }

            float scale = 1.0f;
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
            std::array<float, 4> obj_color = this->m_hull_color_slot.Param<core::param::ColorParam>()->Value();

            m_rt_identifiers.back().emplace_back(std::string(FullName()) + "_" + sub_mesh.first);
            m_draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
            m_per_object_data.back().push_back(PerObjectData{obj_xform, obj_color, -1, 0, 0, 0});
        }

        if (m_show_elements) {
            auto shader = material_collection_->getMaterial("ProbeShellElements").shader_program;
            for (int i = 0; i < m_batch_meshes.size(); ++i) {
                render_task_collection_->addRenderTasks(
                    m_rt_identifiers[i], shader, m_batch_meshes[i], m_draw_commands[i], m_per_object_data[i]);
            }
        }
    }

    // check for pending events
    auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
    if (call_event_storage != NULL) {
        if ((!(*call_event_storage)(0))) {
            // TODO throw error
            return;
        }

        auto event_collection = call_event_storage->getData();

        // process toggle show glyph events
        {
            auto pending_deselect_events = event_collection->get<ToggleShowHull>();
            for (auto& evt : pending_deselect_events) {
                m_show_elements = !m_show_elements;

                if (m_show_elements) {
                    // TODO get rid of code copy-pasting...
                    auto shader = material_collection_->getMaterial("ProbeShellElements").shader_program;

                    for (int i = 0; i < m_batch_meshes.size(); ++i) {
                        render_task_collection_->addRenderTasks(
                            m_rt_identifiers[i], shader, m_batch_meshes[i], m_draw_commands[i], m_per_object_data[i]);
                    }
                } else {
                    render_task_collection_->clear();
                }
            }
        }
    }

    // TODO merge meta data stuff, i.e. bounding box
}
