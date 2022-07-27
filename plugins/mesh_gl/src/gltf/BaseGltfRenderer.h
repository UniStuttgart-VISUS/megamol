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
#include "mesh_gl/BaseRenderTaskRenderer.h"

namespace megamol {
namespace mesh_gl {

template<const char* NAME, const char* DESC>
class BaseGltfRenderer : public BaseRenderTaskRenderer<NAME,DESC> {
public:
    using BaseRenderTaskRenderer<NAME, DESC>::material_collection_;
    using BaseRenderTaskRenderer<NAME, DESC>::mesh_collection_;
    using BaseRenderTaskRenderer<NAME, DESC>::render_task_collection_;

    BaseGltfRenderer();
    ~BaseGltfRenderer() = default;

protected:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call);

    bool updateMeshCollection() override;

    void updateRenderTaskCollection(bool force_update) override;

private:
    /** Slot to retrieve the gltf model */
    megamol::core::CallerSlot glTF_callerSlot_;

    /** Slot to retrieve the mesh data of the gltf model */
    megamol::core::CallerSlot mesh_slot_;
};

template<const char* NAME, const char* DESC>
inline BaseGltfRenderer<NAME, DESC>::BaseGltfRenderer()
        : BaseRenderTaskRenderer<NAME,DESC>()
        , glTF_callerSlot_("gltfModels", "Connects a collection of loaded glTF files")
        , mesh_slot_("meshes", "Connects a mesh data access collection")
{
    glTF_callerSlot_.SetCompatibleCall<mesh::CallGlTFDataDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->glTF_callerSlot_);
    mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->mesh_slot_);
}

template<const char* NAME, const char* DESC>
bool BaseGltfRenderer<NAME, DESC>::GetExtents(mmstd_gl::CallRender3DGL& call) {

    mmstd_gl::CallRender3DGL* cr = &call; // dynamic_cast<mmstd_gl::CallRender3DGL*>(&call);
    if (cr == nullptr) {
        return false;
    }

    auto gltf_call = glTF_callerSlot_.CallAs<mesh::CallGlTFData>();
    if (gltf_call != nullptr) {
        if (!(*gltf_call)(1)) {
            return false;
        }
    }

    mesh::CallMesh* src_mesh_call = mesh_slot_.CallAs<mesh::CallMesh>();
    if (src_mesh_call != nullptr) {

        auto src_meta_data = src_mesh_call->getMetaData();
        src_meta_data.m_frame_ID = static_cast<int>(cr->Time());
        src_mesh_call->setMetaData(src_meta_data);
        if (!(*src_mesh_call)(1)) {
            return false;
        }
        src_meta_data = src_mesh_call->getMetaData();

        cr->SetTimeFramesCount(src_meta_data.m_frame_cnt);
        cr->AccessBoundingBoxes() = src_meta_data.m_bboxs;
    }

    return true;
}

template<const char* NAME, const char* DESC>
inline bool BaseGltfRenderer<NAME, DESC>::updateMeshCollection() {
    bool something_has_changed = false;

    mesh::CallMesh* mc = this->mesh_slot_.CallAs<mesh::CallMesh>();
    if (mc != nullptr) {

        if (!(*mc)(0)) {
            return false;
        }

        something_has_changed = mc->hasUpdate(); // something has changed in the neath...

        if (something_has_changed) {
            mesh_collection_->clear();
            mesh_collection_->addMeshes(*(mc->getData()));
        }
    } else {
        if (mesh_collection_->getMeshes().size() > 0) {
            mesh_collection_->clear();
            something_has_changed = true;
        }
    }

    return something_has_changed;
}

template<const char* NAME, const char* DESC>
inline void BaseGltfRenderer<NAME, DESC>::updateRenderTaskCollection(bool force_update) {

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
