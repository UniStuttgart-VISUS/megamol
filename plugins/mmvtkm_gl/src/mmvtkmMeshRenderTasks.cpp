/*
 * mmvtkmMeshRenderTasks.cpp
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#include "mmvtkm_gl/mmvtkmMeshRenderTasks.h"

#include "mesh_gl/MeshCalls_gl.h"

using namespace megamol;
using namespace megamol::mmvtkm_gl;

mmvtkmMeshRenderTasks ::mmvtkmMeshRenderTasks()
        : m_version(0)
        , m_material_slot("gpuMaterials", "Connects to a material data source") {
    this->m_material_slot.SetCompatibleCall<mesh_gl::CallGPUMaterialDataDescription>();
    this->MakeSlotAvailable(&this->m_material_slot);
}

mmvtkmMeshRenderTasks ::~mmvtkmMeshRenderTasks() {}

bool mmvtkmMeshRenderTasks ::getDataCallback(core::Call& caller) {

    mesh_gl::CallGPURenderTaskData* lhs_rtc = dynamic_cast<mesh_gl::CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == nullptr)
        return false;

    mesh_gl::CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<mesh_gl::CallGPURenderTaskData>();

    auto gpu_render_tasks = std::make_shared<std::vector<std::shared_ptr<mesh_gl::GPURenderTaskCollection>>>();
    if (rhs_rtc != nullptr) {
        if (!(*rhs_rtc)(0)) {
            return false;
        }
        if (rhs_rtc->hasUpdate()) {
            ++m_version;
        }
        gpu_render_tasks = rhs_rtc->getData();
    }
    gpu_render_tasks->push_back(m_rendertask_collection.first);

    mesh_gl::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh_gl::CallGPUMaterialData>();
    mesh_gl::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh_gl::CallGPUMeshData>();

    if (mtlc != nullptr && mc != nullptr) {
        if (!(*mtlc)(0))
            return false;
        if (!(*mc)(0))
            return false;

        bool something_has_changed = mtlc->hasUpdate() || mc->hasUpdate();

        if (something_has_changed) {
            ++m_version;

            clearRenderTaskCollection();

            auto gpu_mtl_collections = mtlc->getData();
            auto gpu_mesh_collections = mc->getData();

            std::vector<std::vector<std::string>> identifiers;
            std::vector<std::vector<glowl::DrawElementsCommand>> draw_commands;
            std::vector<std::vector<std::array<float, 16>>> object_transforms;
            std::vector<std::shared_ptr<glowl::Mesh>> batch_meshes;

            std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

            // extra set of structures for ghostplane
            // need to extract it manually to insert at the end due to transparency
            std::vector<std::string> ghost_identifier;
            std::vector<glowl::DrawElementsCommand> ghost_draw_command;
            std::vector<std::array<float, 16>> ghost_object_transform;
            std::shared_ptr<glowl::Mesh> ghost_batch_mesh;
            bool has_ghostplane = false;

            for (auto& gpu_mesh_collection : *gpu_mesh_collections) {
                for (auto& sub_mesh : gpu_mesh_collection->getSubMeshData()) {
                    auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

                    bool found_ghostplane = sub_mesh.first == "ghostplane";

                    if (gpu_batch_mesh != prev_mesh) {
                        // works because ghostplane is in a single batch
                        // and since this is the case, we can just grab the whole batch
                        if (found_ghostplane) {
                            has_ghostplane = true;
                            ghost_batch_mesh = gpu_batch_mesh;
                        } else {
                            identifiers.emplace_back(std::vector<std::string>());
                            draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                            object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                            batch_meshes.push_back(gpu_batch_mesh);
                        }

                        prev_mesh = gpu_batch_mesh;
                    }

                    float scale = 1.0f;
                    std::array<float, 16> obj_xform = {scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f,
                        scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

                    if (found_ghostplane) {
                        ghost_identifier.push_back(std::string(this->FullName()) + sub_mesh.first);
                        ghost_draw_command.push_back(sub_mesh.second.sub_mesh_draw_command);
                        ghost_object_transform.push_back(obj_xform);
                    } else {
                        identifiers.back().push_back(std::string(this->FullName()) + sub_mesh.first);
                        draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
                        object_transforms.back().push_back(obj_xform);
                    }
                }
            }

            // add ghostplane batch at the end
            if (has_ghostplane) {
                identifiers.push_back(ghost_identifier);
                draw_commands.push_back(ghost_draw_command);
                object_transforms.push_back(ghost_object_transform);
                batch_meshes.push_back(ghost_batch_mesh);
            }

            for (auto& gpu_mtl_collection : *gpu_mtl_collections) {
                for (int i = 0; i < batch_meshes.size(); ++i) {
                    auto const& shader = gpu_mtl_collection->getMaterials().begin()->second.shader_program;

                    std::string ghostIdentifier = (std::string)this->FullName() + "ghostplane";
                    std::vector<std::string>::iterator it =
                        std::find(identifiers[i].begin(), identifiers[i].end(), ghostIdentifier.c_str());

                    if (it != identifiers[i].end()) {
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

                    m_rendertask_collection.second.push_back(std::string(this->FullName()));

                    //for (const auto& s : identifiers[i]) {
                    //    m_rendertask_collection.second.push_back(s); // new
                    //}

                    // TODO add index to index map for removal
                }
            }
        }
    } else {
        clearRenderTaskCollection();
    }

    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}
