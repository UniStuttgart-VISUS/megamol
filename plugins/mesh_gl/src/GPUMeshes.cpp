#include "GPUMeshes.h"

#include "mesh_gl/MeshCalls_gl.h"

megamol::mesh_gl::GPUMeshes::GPUMeshes()
        : m_version(0)
        , m_mesh_slot("meshes", "Connect mesh data for upload to the GPU") {
    this->m_mesh_slot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);
}

megamol::mesh_gl::GPUMeshes::~GPUMeshes() {
    this->Release();
}

bool megamol::mesh_gl::GPUMeshes::getDataCallback(core::Call& caller) {
    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    CallGPUMeshData* rhs_mesh_call = this->m_mesh_rhs_slot.CallAs<CallGPUMeshData>();

    if (lhs_mesh_call == nullptr) {
        return false;
    }
    
    auto gpu_mesh_collection = std::make_shared<std::vector<std::shared_ptr<GPUMeshCollection>>>();
    // if there is a mesh connection to the right, pass on the mesh collection
    if (rhs_mesh_call != nullptr) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++m_version;
        }
        gpu_mesh_collection = rhs_mesh_call->getData();
    }
    gpu_mesh_collection->push_back(m_mesh_collection.first);

    mesh::CallMesh* mc = this->m_mesh_slot.CallAs<mesh::CallMesh>();
    if (mc != nullptr) {

        if (!(*mc)(0))
            return false;

        bool something_has_changed = mc->hasUpdate(); // something has changed in the neath...

        if (something_has_changed) {
            ++m_version;

            clearMeshCollection();

            auto mesh_collection = mc->getData();

            auto meshes = mesh_collection->accessMeshes();

            for (auto& mesh : meshes) {
                m_mesh_collection.first->addMesh(mesh.first,mesh.second);
                m_mesh_collection.second.push_back(mesh.first);
            }
        }

        auto lhs_meta_data = lhs_mesh_call->getMetaData();
        core::Spatial3DMetaData rhs_meta_data;
        auto src_meta_data = mc->getMetaData();

        if (rhs_mesh_call != nullptr) {
            rhs_meta_data = rhs_mesh_call->getMetaData();
        } else {
            rhs_meta_data.m_frame_cnt = src_meta_data.m_frame_cnt;
        }

        lhs_meta_data.m_frame_cnt = std::min(src_meta_data.m_frame_cnt, rhs_meta_data.m_frame_cnt);

        auto bbox = src_meta_data.m_bboxs.BoundingBox();
        bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
        lhs_meta_data.m_bboxs.SetBoundingBox(bbox);

        auto cbbox = src_meta_data.m_bboxs.ClipBox();
        cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
        lhs_meta_data.m_bboxs.SetClipBox(cbbox);

        lhs_mesh_call->setMetaData(lhs_meta_data);
    } else {
        clearMeshCollection();

        ++m_version;
    }

    if (lhs_mesh_call->version() < m_version) {
        lhs_mesh_call->setData(gpu_mesh_collection, m_version);
    }

    return true;
}

bool megamol::mesh_gl::GPUMeshes::getMetaDataCallback(core::Call& caller) {
    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    CallGPUMeshData* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallGPUMeshData>();
    mesh::CallMesh* src_mesh_call = m_mesh_slot.CallAs<mesh::CallMesh>();

    if (lhs_mesh_call == NULL)
        return false;
    if (src_mesh_call == NULL)
        return false;

    auto lhs_meta_data = lhs_mesh_call->getMetaData();
    auto src_meta_data = src_mesh_call->getMetaData();
    core::Spatial3DMetaData rhs_meta_data;

    src_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    src_mesh_call->setMetaData(src_meta_data);
    if (!(*src_mesh_call)(1))
        return false;
    src_meta_data = src_mesh_call->getMetaData();

    if (rhs_mesh_call != NULL) {
        rhs_meta_data = rhs_mesh_call->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        rhs_mesh_call->setMetaData(rhs_meta_data);
        if (!(*rhs_mesh_call)(1))
            return false;
        rhs_meta_data = rhs_mesh_call->getMetaData();
    } else {
        rhs_meta_data.m_frame_cnt = 1;
    }

    lhs_meta_data.m_frame_cnt = std::min(src_meta_data.m_frame_cnt, rhs_meta_data.m_frame_cnt);

    auto bbox = src_meta_data.m_bboxs.BoundingBox();
    bbox.Union(rhs_meta_data.m_bboxs.BoundingBox());
    lhs_meta_data.m_bboxs.SetBoundingBox(bbox);

    auto cbbox = src_meta_data.m_bboxs.ClipBox();
    cbbox.Union(rhs_meta_data.m_bboxs.ClipBox());
    lhs_meta_data.m_bboxs.SetClipBox(cbbox);

    lhs_mesh_call->setMetaData(lhs_meta_data);

    return true;
}
