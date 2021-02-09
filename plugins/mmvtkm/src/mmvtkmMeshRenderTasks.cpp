/*
 * mmvtkmMeshRenderTasks.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
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

    syncRenderTaskCollection(lhs_rtc);

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;

    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false;
    
    bool something_has_changed = mtlc->hasUpdate() || mc->hasUpdate();

    if (something_has_changed) {
        ++m_version;

        for (auto& identifier : m_rendertask_collection.second) {
            m_rendertask_collection.first->deleteRenderTask(identifier);
        }
        m_rendertask_collection.second.clear();

        auto gpu_mtl_storage = mtlc->getData();
        auto gpu_mesh_storage = mc->getData();

        std::vector<std::vector<std::string>> identifiers;
        std::vector<std::vector<glowl::DrawElementsCommand>> draw_commands;
        std::vector<std::vector<std::array<float, 16>>> object_transforms;
        std::vector<std::shared_ptr<glowl::Mesh>> batch_meshes;

        std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

        int gp_batch_idx = -1;
        int cnt = -1;
        for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData()) {
            auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;
            
            if (gpu_batch_mesh != prev_mesh) {
                identifiers.emplace_back(std::vector<std::string>());
                draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                batch_meshes.push_back(gpu_batch_mesh);

                prev_mesh = gpu_batch_mesh;

                ++cnt;
            }

            float scale = 1.0f;
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

            identifiers.back().push_back(std::string(this->FullName()) + sub_mesh.first);
            draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
            object_transforms.back().push_back(obj_xform);

            if(sub_mesh.first == "ghostplane") gp_batch_idx = cnt;
        }

        // swap batch that contains ghostplane to the last position for blending
        if(gp_batch_idx != -1 && gp_batch_idx != batch_meshes.size() - 1) {
            //std::swap(identifiers[gp_batch_idx], identifiers[identifiers.size() - 1]);
            //std::swap(draw_commands[gp_batch_idx], draw_commands[draw_commands.size() - 1]);
            //std::swap(object_transforms[gp_batch_idx], object_transforms[object_transforms.size() - 1]);
        }

		for (int i = 0; i < batch_meshes.size(); ++i) {
            auto const& shader = gpu_mtl_storage->getMaterials().begin()->second.shader_program;

            std::string ghostIdentifier = (std::string)this->FullName() + "ghostplane";
            std::vector<std::string>::iterator it =
                std::find(identifiers[i].begin(), identifiers[i].end(), ghostIdentifier.c_str());
			if (it != identifiers[i].end()) {
                // swap ghostplane to last position within batch for blending
                //int gp_idx = std::distance(identifiers[i].begin(), it);
                //std::cout << identifiers[i].size() << " " << gp_idx << "\n";
                //std::swap(identifiers[i][gp_idx], identifiers[i][identifiers[i].size() - 1]);
                //std::swap(draw_commands[i][gp_idx], draw_commands[i][draw_commands.size() - 1]);
                //std::swap(object_transforms[i][gp_idx], object_transforms[i][object_transforms.size() - 1]);
                //std::cout << "check\n";
				auto set_states = [] { 
					glEnable(GL_BLEND);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					glEnable(GL_DEPTH_TEST);
                    glCullFace(GL_BACK);
					glDisable(GL_CULL_FACE);
				};
				auto reset_states = [] { 
					glDisable(GL_BLEND);
					glBlendFunc(GL_ONE, GL_NONE);
					glDisable(GL_DEPTH_TEST);
				};
				
                m_rendertask_collection.first->addRenderTasks(identifiers[i], shader, batch_meshes[i],
                                    draw_commands[i], object_transforms[i], set_states, reset_states);
                       
            } else {
                m_rendertask_collection.first->addRenderTasks(
                    identifiers[i], shader, batch_meshes[i], draw_commands[i], object_transforms[i]);
			}

            for (const auto& s : identifiers[i]) {
                m_rendertask_collection.second.push_back(s); // new
            }

            // TODO add index to index map for removal

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
