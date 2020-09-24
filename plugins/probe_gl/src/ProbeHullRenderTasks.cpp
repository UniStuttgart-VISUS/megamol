/*
 * ProbeHullRenderTasks.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mmcore/EventCall.h"

#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "ProbeHUllRenderTasks.h"

#include "mesh/MeshCalls.h"

megamol::probe_gl::ProbeHullRenderTasks::ProbeHullRenderTasks()
    : m_version(0), m_show_hull(true), m_event_slot("GetEvents", "") {
    this->m_event_slot.SetCompatibleCall<core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);
}

megamol::probe_gl::ProbeHullRenderTasks::~ProbeHullRenderTasks() {}

bool megamol::probe_gl::ProbeHullRenderTasks::getDataCallback(core::Call& caller) {

    mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;

    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false;

    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection(nullptr);

    if (lhs_rtc->getData() == nullptr) {
        rt_collection = this->m_gpu_render_tasks;
    } else {
        rt_collection = lhs_rtc->getData();
    }

    bool something_has_changed = mtlc->hasUpdate() || mc->hasUpdate();

    if (something_has_changed) {
        ++m_version;

        auto gpu_mtl_storage = mtlc->getData();
        auto gpu_mesh_storage = mc->getData();

        m_draw_commands.clear();
        m_object_transforms.clear();
        m_batch_meshes.clear();

        std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

        for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;

            if (gpu_batch_mesh != prev_mesh) {
                m_draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                m_object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                m_batch_meshes.push_back(gpu_batch_mesh);

                prev_mesh = gpu_batch_mesh;
            }

            float scale = 1.0f;
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

            m_draw_commands.back().push_back(sub_mesh.sub_mesh_draw_command);
            m_object_transforms.back().push_back(obj_xform);
        }

        if (m_show_hull) {
            for (int i = 0; i < m_batch_meshes.size(); ++i) {
                auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
                rt_collection->addRenderTasks(shader, m_batch_meshes[i], m_draw_commands[i], m_object_transforms[i]);

                // TODO add index to index map for removal
            }
        }
    }

    // check for pending events
    auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
    if (call_event_storage != NULL) {
        if ((!(*call_event_storage)(0))) return false;

        auto event_collection = call_event_storage->getData();
        auto gpu_mtl_storage = mtlc->getData();

        // process toggle show glyph events
        {
            auto pending_deselect_events = event_collection->get<ToggleShowHull>();
            for (auto& evt : pending_deselect_events) {
                m_show_hull = !m_show_hull;

                if (m_show_hull) {
                    for (int i = 0; i < m_batch_meshes.size(); ++i) {
                        auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
                        rt_collection->addRenderTasks(
                            shader, m_batch_meshes[i], m_draw_commands[i], m_object_transforms[i]);
                    }
                } else {
                    // TODO this breaks chaining...
                    rt_collection->clear();
                }
            }
        }
    }

    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(rt_collection, 0);
        (*rhs_rtc)(0);
    }

    // TODO merge meta data stuff, i.e. bounding box
    auto mesh_meta_data = mc->getMetaData();

    // TODO set data if necessary
    lhs_rtc->setData(rt_collection, m_version);


    return true;
}

bool megamol::probe_gl::ProbeHullRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller)) return false;

    mesh::CallGPURenderTaskData* lhs_rt_call = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    mesh::CallGPUMeshData* mesh_call = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();

    auto lhs_meta_data = lhs_rt_call->getMetaData();

    if (mesh_call != NULL) {
        auto mesh_meta_data = mesh_call->getMetaData();
        mesh_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        mesh_call->setMetaData(mesh_meta_data);
        if (!(*mesh_call)(1)) return false;
        mesh_meta_data = mesh_call->getMetaData();

        auto bbox = lhs_meta_data.m_bboxs.BoundingBox();
        auto cbbox = lhs_meta_data.m_bboxs.ClipBox();

        auto mesh_bbox = mesh_meta_data.m_bboxs.BoundingBox();
        auto mesh_cbbox = mesh_meta_data.m_bboxs.ClipBox();

        // mesh_bbox.SetSize(vislib::math::Dimension<float, 3>(
        //    2.1f * mesh_bbox.GetSize()[0], 2.1f * mesh_bbox.GetSize()[1], 2.1f * mesh_bbox.GetSize()[2]));

        mesh_cbbox.SetSize(vislib::math::Dimension<float, 3>(
            2.1f * mesh_cbbox.GetSize()[0], 2.1f * mesh_cbbox.GetSize()[1], 2.1f * mesh_cbbox.GetSize()[2]));

        bbox.Union(mesh_bbox);
        cbbox.Union(mesh_cbbox);

        lhs_meta_data.m_bboxs.SetBoundingBox(bbox);
        lhs_meta_data.m_bboxs.SetClipBox(cbbox);
    }

    lhs_rt_call->setMetaData(lhs_meta_data);

    return true;
}
