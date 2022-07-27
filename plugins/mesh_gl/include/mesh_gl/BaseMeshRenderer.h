/*
 * BaseMeshRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef BASE_MESH_RENDERER_H_INCLUDED
#define BASE_MESH_RENDERER_H_INCLUDED

#include "mmcore/CallerSlot.h"

#include "mesh/MeshCalls.h"
#include "mesh_gl/BaseRenderTaskRenderer.h"

namespace megamol {
namespace mesh_gl {

template<const char* NAME, const char* DESC>
class BaseMeshRenderer : public BaseRenderTaskRenderer<NAME, DESC> {
public:
    using BaseRenderTaskRenderer<NAME, DESC>::material_collection_;
    using BaseRenderTaskRenderer<NAME, DESC>::mesh_collection_;
    using BaseRenderTaskRenderer<NAME, DESC>::render_task_collection_;

    BaseMeshRenderer();
    ~BaseMeshRenderer() = default;

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
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    bool updateMeshCollection() override;

    /** Slot to retrieve mesh data */
    megamol::core::CallerSlot mesh_slot_;
};

template<const char* NAME, const char* DESC>
inline BaseMeshRenderer<NAME, DESC>::BaseMeshRenderer()
        : BaseRenderTaskRenderer<NAME, DESC>()
        , mesh_slot_("meshes", "Connects a mesh data access collection") {
    mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->mesh_slot_);
}

template<const char* NAME, const char* DESC>
bool BaseMeshRenderer<NAME, DESC>::GetExtents(mmstd_gl::CallRender3DGL& call) {

    mmstd_gl::CallRender3DGL* cr = &call; // dynamic_cast<mmstd_gl::CallRender3DGL*>(&call);
    if (cr == nullptr) {
        return false;
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
inline bool BaseMeshRenderer<NAME, DESC>::updateMeshCollection() {
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

} // namespace mesh_gl
} // namespace megamol

#endif // !BASE_MESH_RENDERER_H_INCLUDED
