#include "stdafx.h"
#include "GPUMeshes.h"

#include "mesh/MeshCalls.h"

megamol::mesh::GPUMeshes::GPUMeshes()
    : m_mesh_slot("CallGlTFData", "Connects the data source with a loaded glTF file")
    , m_mesh_cached_hash(0) 
{
    this->m_mesh_slot.SetCompatibleCall<CallMeshDescription>();
    this->MakeSlotAvailable(&this->m_mesh_slot);
}

megamol::mesh::GPUMeshes::~GPUMeshes() { this->Release(); }

bool megamol::mesh::GPUMeshes::create() {
    m_gpu_meshes = std::make_shared<GPUMeshCollection>();

    m_bbox = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    return true;
}

bool megamol::mesh::GPUMeshes::getDataCallback(core::Call& caller) { 

    CallGPUMeshData* lhs_mesh_call = dynamic_cast<CallGPUMeshData*>(&caller);
    if (lhs_mesh_call == NULL) return false;

    std::shared_ptr<GPUMeshCollection> mesh_collection(nullptr);

    if (lhs_mesh_call->getData() == nullptr) {
        // no incoming mesh -> use your own mesh storage
        mesh_collection = this->m_gpu_meshes;
        lhs_mesh_call->setData(mesh_collection);
    } else {
        // incoming mesh -> use it (delete local?)
        mesh_collection = lhs_mesh_call->getData();
    }

    CallMesh* mc = this->m_mesh_slot.CallAs<CallMesh>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false;

    if (mc->getMetaData().m_data_hash > m_mesh_cached_hash) {
        m_mesh_cached_hash = mc->getMetaData().m_data_hash;

        if (!m_mesh_collection_indices.empty()) {
            // TODO delete all exisiting render task from this module
            for (auto& submesh_idx : m_mesh_collection_indices) {
                // mesh_collection->deleteSubMesh()
            }

            m_mesh_collection_indices.clear();
        }

        auto& meta_data = mc->getMetaData();

        m_bbox[0] = meta_data.m_bboxs.ObjectSpaceBBox().Left();
        m_bbox[1] = meta_data.m_bboxs.ObjectSpaceBBox().Bottom();
        m_bbox[2] = meta_data.m_bboxs.ObjectSpaceBBox().Back();
        m_bbox[3] = meta_data.m_bboxs.ObjectSpaceBBox().Right();
        m_bbox[4] = meta_data.m_bboxs.ObjectSpaceBBox().Top();
        m_bbox[5] = meta_data.m_bboxs.ObjectSpaceBBox().Front();

        auto meshes = mc->getData()->accessMesh();

        for (auto& mesh : meshes)
        {

            std::vector<glowl::VertexLayout::Attribute> attribs;
            std::vector < std::pair<uint8_t*, uint8_t* >> vb_iterators;
            std::pair<uint8_t*, uint8_t*> ib_iterators;

            ib_iterators = {mesh.indices.data, mesh.indices.data + mesh.indices.byte_size};

            for (auto attrib : mesh.attributes) {

                attribs.push_back(
                    glowl::VertexLayout::Attribute(attrib.component_cnt , 
                        MeshDataAccessCollection::convertToGLType(attrib.component_type),
                        GL_FALSE /*ToDO*/, attrib.offset));

                // TODO vb_iterators
                vb_iterators.push_back({attrib.data, attrib.data + attrib.byte_size});
            }

            glowl::VertexLayout vertex_descriptor(0,attribs);
            mesh_collection->addMesh(
                vertex_descriptor,
                vb_iterators,
                ib_iterators,
                MeshDataAccessCollection::convertToGLType(mesh.indices.type),
                GL_STATIC_DRAW,
                GL_TRIANGLES);
        }

    }

    // if there is a mesh connection to the right, pass on the mesh collection
    CallGPUMeshData* rhs_mesh_call = this->m_mesh_callerSlot.CallAs<CallGPUMeshData>();
    if (rhs_mesh_call != NULL) {
        rhs_mesh_call->setData(mesh_collection);

        if (!(*rhs_mesh_call)(0)) return false;
    }
}
