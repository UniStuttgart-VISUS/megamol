/*
 * ProbeHullRenderTasks.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mmcore/EventCall.h"
#include "mmcore/param/EnumParam.h"

#include "ProbeCalls.h"
#include "ProbeEvents.h"
#include "ProbeGlCalls.h"
#include "ProbeHUllRenderTasks.h"

#include "mesh/MeshCalls.h"

bool megamol::probe_gl::ProbeHullRenderTasks::create() {

    m_rendertask_collection.first = std::make_shared<mesh::GPURenderTaskCollection>();

    struct PerFrameData {
        int shading_mode;
    };

    std::array<PerFrameData, 1> per_frame_data;
    per_frame_data[0].shading_mode = m_shading_mode_slot.Param<core::param::EnumParam>()->Value();

    m_rendertask_collection.first->addPerFrameDataBuffer("", per_frame_data, 1);

    return true;
}

megamol::probe_gl::ProbeHullRenderTasks::ProbeHullRenderTasks()
        : m_version(0)
        , m_show_hull(true)
        //, m_probes_slot("probes","")
        , m_event_slot("GetEvents", "")
        , m_shading_mode_slot("ShadingMode","") {
    //this->m_probes_slot.SetCompatibleCall<megamol::probe::CallProbesDescription>();
    //this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_event_slot.SetCompatibleCall<core::CallEventDescription>();
    this->MakeSlotAvailable(&this->m_event_slot);

    this->m_shading_mode_slot << new megamol::core::param::EnumParam(0);
    this->m_shading_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Grey");
    this->m_shading_mode_slot.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "ClusterID");
    this->MakeSlotAvailable(&this->m_shading_mode_slot);
}

megamol::probe_gl::ProbeHullRenderTasks::~ProbeHullRenderTasks() {}

bool megamol::probe_gl::ProbeHullRenderTasks::getDataCallback(core::Call& caller) {

    mesh::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL)
        return false;

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL)
        return false;
    if (!(*mtlc)(0))
        return false;

    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL)
        return false;
    if (!(*mc)(0))
        return false;

    syncRenderTaskCollection(lhs_rtc);

    if (m_shading_mode_slot.IsDirty()) {
        m_shading_mode_slot.ResetDirty();

        struct PerFrameData {
            int shading_mode;
        };

        std::array<PerFrameData, 1> per_frame_data;
        per_frame_data[0].shading_mode = m_shading_mode_slot.Param<core::param::EnumParam>()->Value();

        m_rendertask_collection.first->updatePerFrameDataBuffer("", per_frame_data, 1);
    }

    bool something_has_changed = mtlc->hasUpdate() || mc->hasUpdate();

    if (something_has_changed) {
        ++m_version;

        for (auto& identifier : m_rendertask_collection.second) {
            m_rendertask_collection.first->deleteRenderTask(identifier);
        }
        m_rendertask_collection.second.clear();

        m_identifiers.clear();
        m_draw_commands.clear();
        m_object_transforms.clear();
        m_batch_meshes.clear();

        auto gpu_mtl_storage = mtlc->getData();
        auto gpu_mesh_storage = mc->getData();

        std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

        for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
            auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

            if (gpu_batch_mesh != prev_mesh) {
                m_draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                m_object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                m_batch_meshes.push_back(gpu_batch_mesh);

                prev_mesh = gpu_batch_mesh;
            }

            float scale = 1.0f;
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

            m_identifiers.emplace_back(std::string(FullName()) + "_" + sub_mesh.first);
            m_draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
            m_object_transforms.back().push_back(obj_xform);
        }

        if (m_show_hull) {
            auto const& shader = gpu_mtl_storage->getMaterial("ProbeHull").shader_program;

            for (int i = 0; i < m_batch_meshes.size(); ++i) {
                m_rendertask_collection.first->addRenderTasks(
                    m_identifiers, shader, m_batch_meshes[i], m_draw_commands[i], m_object_transforms[i]);
                m_rendertask_collection.second.insert(
                    m_rendertask_collection.second.end(), m_identifiers.begin(), m_identifiers.end());
            }
        }
    }

    // check for pending events
    auto call_event_storage = this->m_event_slot.CallAs<core::CallEvent>();
    if (call_event_storage != NULL) {
        if ((!(*call_event_storage)(0)))
            return false;

        auto event_collection = call_event_storage->getData();
        auto gpu_mtl_storage = mtlc->getData();

        // process toggle show glyph events
        {
            auto pending_deselect_events = event_collection->get<ToggleShowHull>();
            for (auto& evt : pending_deselect_events) {
                m_show_hull = !m_show_hull;

                if (m_show_hull) {
                    auto const& shader = gpu_mtl_storage->getMaterial("ProbeHull").shader_program;
                    for (int i = 0; i < m_batch_meshes.size(); ++i) {
                        m_rendertask_collection.first->addRenderTasks(
                            m_identifiers, shader, m_batch_meshes[i], m_draw_commands[i], m_object_transforms[i]);
                        m_rendertask_collection.second.insert(
                            m_rendertask_collection.second.end(), m_identifiers.begin(), m_identifiers.end());
                    }
                } else {
                    for (auto& identifier : m_rendertask_collection.second) {
                        m_rendertask_collection.first->deleteRenderTask(identifier);
                    }
                    m_rendertask_collection.second.clear();
                }
            }
        }
    }

    mesh::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(m_rendertask_collection.first, 0);
        (*rhs_rtc)(0);
    }

    // TODO merge meta data stuff, i.e. bounding box
    auto mesh_meta_data = mc->getMetaData();

    // TODO set data if necessary
    lhs_rtc->setData(m_rendertask_collection.first, m_version);


    return true;
}

bool megamol::probe_gl::ProbeHullRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller))
        return false;

    mesh::CallGPURenderTaskData* lhs_rt_call = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    mesh::CallGPUMeshData* mesh_call = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();

    auto lhs_meta_data = lhs_rt_call->getMetaData();

    if (mesh_call != NULL) {
        auto mesh_meta_data = mesh_call->getMetaData();
        mesh_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        mesh_call->setMetaData(mesh_meta_data);
        if (!(*mesh_call)(1))
            return false;
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
