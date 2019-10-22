/*
 * MeshViewerRenderTasks.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "MeshViewerRenderTasks.h"

#include "mesh/MeshCalls.h"

megamol::mesh::MeshViewerRenderTasks::MeshViewerRenderTasks() {}

megamol::mesh::MeshViewerRenderTasks::~MeshViewerRenderTasks() {}

bool megamol::mesh::MeshViewerRenderTasks::getDataCallback(core::Call& caller) {

    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<CallGPUMaterialData>();
    if (mtlc == NULL) return false;

    if (!(*mtlc)(0)) return false;

    CallGPUMeshData* mc = this->m_mesh_slot.CallAs<CallGPUMeshData>();
    if (mc == NULL) return false;

    if (!(*mc)(0)) return false;

    std::shared_ptr<GPURenderTaskCollection> rt_collection(nullptr);

    if (lhs_rtc->getData() == nullptr) {
        rt_collection = this->m_gpu_render_tasks;
        lhs_rtc->setData(rt_collection);
    } else {
        rt_collection = lhs_rtc->getData();
    }

    auto gpu_mtl_storage = mtlc->getData();
    auto gpu_mesh_storage = mc->getData();

    auto mesh_meta_data = mc->getMetaData();

    if (mesh_meta_data.m_data_hash > m_mesh_cached_hash)
    {
        m_mesh_cached_hash = mesh_meta_data.m_data_hash;

        for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            std::vector<glowl::DrawElementsCommand> draw_commands(1, sub_mesh.sub_mesh_draw_command);

            std::vector<std::array<float,16>> object_transform(1);

            for (int i = 0; i < 1; ++i) {
                float scale = 1.0f;

                object_transform[i][0] = scale;
                object_transform[i][5] = scale;
                object_transform[i][10] = scale;
                object_transform[i][15] = 1.0f;
            }

            rt_collection->addRenderTasks(shader, gpu_batch_mesh, draw_commands, object_transform);
        }

        mesh_meta_data.m_data_hash++;
        mc->setMetaData(mesh_meta_data);
    }


    CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(rt_collection);

        (*rhs_rtc)(0);
    }

    return true;
}
