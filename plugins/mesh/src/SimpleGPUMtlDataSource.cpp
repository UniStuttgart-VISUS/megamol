#include "mesh/SimpleGPUMtlDataSource.h"

#include "mesh/MeshCalls.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh::SimpleGPUMtlDataSource::SimpleGPUMtlDataSource()
    : m_version(0)
    , m_btf_filename_slot("BTF filename", "The name of the btf file to load") {
    this->m_btf_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_btf_filename_slot);
}

megamol::mesh::SimpleGPUMtlDataSource::~SimpleGPUMtlDataSource() {}

bool megamol::mesh::SimpleGPUMtlDataSource::create() { return true; }

bool megamol::mesh::SimpleGPUMtlDataSource::getDataCallback(core::Call& caller) {
    CallGPUMaterialData* lhs_mtl_call = dynamic_cast<CallGPUMaterialData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    syncMaterialCollection(lhs_mtl_call);

    if (this->m_btf_filename_slot.IsDirty()) {
        m_btf_filename_slot.ResetDirty();

        ++m_version;

        auto vislib_filename = m_btf_filename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.PeekBuffer());

        for (auto& idx : m_material_collection.second) {
            m_material_collection.first->deleteMaterial(idx);
        }

        m_material_collection.first->addMaterial(this->instance(), filename, filename);
        m_material_collection.second.push_back(filename);
    }

    if (lhs_mtl_call->version() < m_version) {
        lhs_mtl_call->setData(m_material_collection.first, m_version);
    }


    // if there is a material connection to the right, pass on the material collection
    CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<CallGPUMaterialData>();
    if (rhs_mtl_call != NULL) {
        rhs_mtl_call->setData(m_material_collection.first, 0);
    }

    return true;
}

bool megamol::mesh::SimpleGPUMtlDataSource::getMetaDataCallback(core::Call& caller)
{ 
    return true;
}
