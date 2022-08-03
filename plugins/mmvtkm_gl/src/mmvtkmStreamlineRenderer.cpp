/*
 * mmvtkmMeshRenderTasks.cpp
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#include "mmvtkm_gl/mmvtkmStreamlineRenderer.h"

using namespace megamol;
using namespace megamol::mmvtkm_gl;

void megamol::mmvtkm_gl::mmvtkmStreamlineRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<mesh_gl::GPUMaterialCollection>();
    material_collection_->addMaterial(
        this->instance(), "gltfMaterial", {"mmvtkm_gl/streamline.vert.glsl", "mmvtkm_gl/streamline.frag.glsl"});
}

void megamol::mmvtkm_gl::mmvtkmStreamlineRenderer::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {
    bool something_has_changed = force_update;

    if (something_has_changed) {
        render_task_collection_->clear();

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

        for (auto& sub_mesh : mesh_collection_->getSubMeshData()) {
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
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

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

        // add ghostplane batch at the end
        if (has_ghostplane) {
            identifiers.push_back(ghost_identifier);
            draw_commands.push_back(ghost_draw_command);
            object_transforms.push_back(ghost_object_transform);
            batch_meshes.push_back(ghost_batch_mesh);
        }

        for (int i = 0; i < batch_meshes.size(); ++i) {
            auto const& shader = material_collection_->getMaterials().begin()->second.shader_program;

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

                render_task_collection_->addRenderTasks(identifiers[i], shader, batch_meshes[i], draw_commands[i],
                    object_transforms[i], set_states, reset_states);

            } else {
                render_task_collection_->addRenderTasks(
                    identifiers[i], shader, batch_meshes[i], draw_commands[i], object_transforms[i]);
            }

            //for (const auto& s : identifiers[i]) {
            //    m_rendertask_collection.second.push_back(s); // new
            //}

            // TODO add index to index map for removal
        }
    }
}
