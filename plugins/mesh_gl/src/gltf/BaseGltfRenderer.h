/*
 * BaseGltfRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef BASE_GLTF_RENDERER_H_INCLUDED
#define BASE_GLTF_RENDERER_H_INCLUDED

#include "mmcore/CallerSlot.h"

#include "mesh/MeshCalls.h"
#include "mesh_gl/BaseMeshRenderer.h"

namespace megamol {
namespace mesh_gl {

class BaseGltfRenderer : public BaseMeshRenderer {
public:
    BaseGltfRenderer();
    ~BaseGltfRenderer() = default;

protected:
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

private:
    /** Slot to retrieve the gltf model */
    megamol::core::CallerSlot glTF_callerSlot_;
};

inline BaseGltfRenderer::BaseGltfRenderer()
        : BaseMeshRenderer()
        , glTF_callerSlot_("gltfModels", "Connects a collection of loaded glTF files") {
    glTF_callerSlot_.SetCompatibleCall<mesh::CallGlTFDataDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->glTF_callerSlot_);
}

inline void BaseGltfRenderer::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {

    bool something_has_changed = force_update;

    mesh::CallGlTFData* gltf_call = this->glTF_callerSlot_.CallAs<mesh::CallGlTFData>();

    if (gltf_call != nullptr) {

        if (!(*gltf_call)(0)) {
            //return false;
        }

        something_has_changed |= gltf_call->hasUpdate();

        if (something_has_changed) {
            render_task_collection_->clear();

            auto model = gltf_call->getData().second;

            for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++) {
                if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1) {
                    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;

                    if (model->nodes[node_idx].matrix.size() != 0) // has matrix transform
                    {
                        // TODO
                    } else {
                        auto& translation = model->nodes[node_idx].translation;
                        auto& scale = model->nodes[node_idx].scale;
                        auto& rotation = model->nodes[node_idx].rotation;

                        if (translation.size() != 0) {
                            object_transform.SetAt(0, 3, translation[0]);
                            object_transform.SetAt(1, 3, translation[1]);
                            object_transform.SetAt(2, 3, translation[2]);
                        }

                        if (scale.size() != 0) {}

                        if (rotation.size() != 0) {}
                    }

                    auto primitive_cnt = model->meshes[model->nodes[node_idx].mesh].primitives.size();
                    for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx) {
                        std::string sub_mesh_identifier = gltf_call->getData().first +
                                                          model->meshes[model->nodes[node_idx].mesh].name + "_" +
                                                          std::to_string(primitive_idx);

                        GPUMeshCollection::SubMeshData sub_mesh = mesh_collection_->getSubMesh(sub_mesh_identifier);

                        if (sub_mesh.mesh != nullptr) {
                            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;
                            auto const& shader = material_collection_->getMaterials().begin()->second.shader_program;

                            std::string rt_identifier(std::string(this->FullName()) + "_" + sub_mesh_identifier);
                            render_task_collection_->addRenderTask(rt_identifier, shader, gpu_batch_mesh,
                                sub_mesh.sub_mesh_draw_command, object_transform);
                        }
                    }
                }
            }
        }
    } else {
        render_task_collection_->clear();
    }
}


} // namespace mesh_gl
} // namespace megamol

#endif // !BASE_GLTF_RENDERER_H_INCLUDED
