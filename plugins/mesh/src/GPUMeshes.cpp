#include "stdafx.h"
#include "GPUMeshes.h"

#include "mesh/MeshCalls.h"

megamol::mesh::GPUMeshes::GPUMeshes()
    : m_version(0)
    , m_mesh_slot("CallMeshes", "Connects mesh data to be uploaded to the GPU")
{
    this->m_mesh_slot.SetCompatibleCall<CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);
}

megamol::mesh::GPUMeshes::~GPUMeshes() { this->Release(); }

bool megamol::mesh::GPUMeshes::getDataCallback(core::Call& caller) {

    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    if (lhs_mesh_call == NULL) return false;

    std::shared_ptr<GPUMeshCollection> mesh_collection(nullptr);

    if (lhs_mesh_call->getData() == nullptr) {
        // no incoming mesh -> use your own mesh storage
        mesh_collection = this->m_gpu_meshes;
    } else {
        // incoming mesh -> use it (delete local?)
        mesh_collection = lhs_mesh_call->getData();
    }

    CallMesh* mc = this->m_mesh_slot.CallAs<CallMesh>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false;

    bool something_has_changed = mc->hasUpdate(); // something has changed in the neath...

    if (something_has_changed) {
        ++m_version;

        if (!m_mesh_collection_indices.empty()) {
            // TODO delete all exisiting render task from this module
            for (auto& submesh_idx : m_mesh_collection_indices) {
                // mesh_collection->deleteSubMesh()
            }

            m_mesh_collection_indices.clear();
        }

        auto meshes = mc->getData()->accessMesh();

        for (auto& mesh : meshes) {

            // check if primtives type
            GLenum primtive_type = GL_TRIANGLES;
            if (mesh.primitive_type == MeshDataAccessCollection::QUADS) {
                primtive_type = GL_PATCHES;
            }

            std::vector<glowl::VertexLayout::Attribute> attribs;
            std::vector<std::pair<uint8_t*, uint8_t*>> vb_iterators;
            std::pair<uint8_t*, uint8_t*> ib_iterators;

            ib_iterators = {mesh.indices.data, mesh.indices.data + mesh.indices.byte_size};

            for (auto attrib : mesh.attributes) {

                attribs.push_back(glowl::VertexLayout::Attribute(attrib.component_cnt,
                    MeshDataAccessCollection::convertToGLType(attrib.component_type), GL_FALSE /*ToDO*/,
                    attrib.offset));

                // TODO vb_iterators
                vb_iterators.push_back({attrib.data, attrib.data + attrib.byte_size});
            }

            glowl::VertexLayout vertex_descriptor(0, attribs);
            mesh_collection->addMesh(vertex_descriptor, vb_iterators, ib_iterators,
                MeshDataAccessCollection::convertToGLType(mesh.indices.type), GL_STATIC_DRAW, primtive_type);
        }
    }

    // update meta data to lhs
    auto lhs_meta_data = lhs_mesh_call->getMetaData();
    core::Spatial3DMetaData rhs_meta_data;
    auto src_meta_data = mc->getMetaData();

    // if there is a mesh connection to the right, pass on the mesh collection
    CallGPUMeshData* rhs_mesh_call = this->m_mesh_rhs_slot.CallAs<CallGPUMeshData>();
    if (rhs_mesh_call != NULL) {
        rhs_mesh_call->setData(mesh_collection,0);

        if (!(*rhs_mesh_call)(0)) return false;

        if (rhs_mesh_call->hasUpdate()){
            ++m_version;
            rhs_mesh_call->getData();
        }

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

    if (lhs_mesh_call->version() < m_version)
    {
        lhs_mesh_call->setData(mesh_collection, m_version);
    }

    return true;
}

bool megamol::mesh::GPUMeshes::getMetaDataCallback(core::Call& caller)
{
    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    CallGPUMeshData* rhs_mesh_call = m_mesh_rhs_slot.CallAs<CallGPUMeshData>();
    CallMesh* src_mesh_call = m_mesh_slot.CallAs<CallMesh>();

    if (lhs_mesh_call == NULL) return false;
    if (src_mesh_call == NULL) return false;

    auto lhs_meta_data = lhs_mesh_call->getMetaData();
    auto src_meta_data = src_mesh_call->getMetaData();
    core::Spatial3DMetaData rhs_meta_data;

    src_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    src_mesh_call->setMetaData(src_meta_data);
    if (!(*src_mesh_call)(1)) return false;
    src_meta_data = src_mesh_call->getMetaData();

    if (rhs_mesh_call != NULL)
    {
        rhs_meta_data = rhs_mesh_call->getMetaData();
        rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
        rhs_mesh_call->setMetaData(rhs_meta_data);
        if (!(*rhs_mesh_call)(1)) return false;
        rhs_meta_data = rhs_mesh_call->getMetaData();
    }
    else
    {
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
