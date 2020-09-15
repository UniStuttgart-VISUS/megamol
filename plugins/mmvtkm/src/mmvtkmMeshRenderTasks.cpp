/*
 * mmvtkmMeshRenderTasks.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mmvtkm/mmvtkmMeshRenderTasks.h"

#include "mesh/MeshCalls.h"

using namespace megamol;
using namespace megamol::mmvtkm;

mmvtkmMeshRenderTasks ::mmvtkmMeshRenderTasks() : m_version(0) {}

mmvtkmMeshRenderTasks ::~mmvtkmMeshRenderTasks () {}

bool mmvtkmMeshRenderTasks ::getDataCallback(core::Call& caller) {

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
        rt_collection->clear();

        ++m_version;

        auto gpu_mtl_storage = mtlc->getData();
        auto gpu_mesh_storage = mc->getData();

        std::vector<std::vector<glowl::DrawElementsCommand>> draw_commands;
        std::vector<std::vector<std::array<float, 16>>> object_transforms;
        std::vector<std::shared_ptr<glowl::Mesh>> batch_meshes;
        //std::vector<std::vector<mesh::GPURenderTaskCollection::GLState>> batch_states;

        std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

        for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
            
            if (gpu_batch_mesh != prev_mesh) {
                draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                batch_meshes.push_back(gpu_batch_mesh);

                prev_mesh = gpu_batch_mesh;
            }

            float scale = 1.0f;
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

            draw_commands.back().push_back(sub_mesh.sub_mesh_draw_command);
            object_transforms.back().push_back(obj_xform);
        }
        
        for (int i = 0; i < batch_meshes.size(); ++i) {
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
            bool blending_and_depth = i == 0;

			if (i == batch_meshes.size() - 1) {
                rt_collection->addRenderTasks(shader, batch_meshes[i], draw_commands[i], object_transforms[i],
                    {std::pair<std::vector<GLuint>, bool>({GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA}, true),
                        std::pair<std::vector<GLuint>, bool>({GL_DEPTH_TEST}, true),
                        std::pair<std::vector<GLuint>, bool>({GL_CULL_FACE}, false)});
            } else {
                rt_collection->addRenderTasks(
                    shader, batch_meshes[i], draw_commands[i], object_transforms[i]);
			}

            // TODO add index to index map for removal

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
