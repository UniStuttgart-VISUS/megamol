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

megamol::mesh::GPUMeshes::~GPUMeshes() {}

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
        // no incoming material -> use your own material storage
        mesh_collection = this->m_gpu_meshes;
        lhs_mesh_call->setData(mesh_collection);
    } else {
        // incoming material -> use it (delete local?)
        mesh_collection = lhs_mesh_call->getData();
    }

    CallMesh* mc = this->m_mesh_slot.CallAs<CallMesh>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false;

    if (mc->getMetaData().m_data_hash > m_mesh_cached_hash) {
        m_mesh_cached_hash = mc->getMetaData().m_data_hash;


        //TODO 


    }
}
