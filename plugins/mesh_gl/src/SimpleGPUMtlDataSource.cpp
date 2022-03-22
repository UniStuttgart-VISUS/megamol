#include "SimpleGPUMtlDataSource.h"

#include "mesh_gl/MeshCalls_gl.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh_gl::SimpleGPUMtlDataSource::SimpleGPUMtlDataSource()
        : m_version(0)
        , m_btf_filename_slot("BTF filename", "The name of the btf file to load") {
    this->m_btf_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_btf_filename_slot);
}

megamol::mesh_gl::SimpleGPUMtlDataSource::~SimpleGPUMtlDataSource() {}

bool megamol::mesh_gl::SimpleGPUMtlDataSource::getDataCallback(core::Call& caller) {
    CallGPUMaterialData* lhs_mtl_call = dynamic_cast<CallGPUMaterialData*>(&caller);
    CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<CallGPUMaterialData>();

    if (lhs_mtl_call == nullptr) {
        return false;
    }

    std::vector<std::shared_ptr<GPUMaterialCollection>> gpu_mtl_collections;
    // if there is a material connection to the right, issue callback
    if (rhs_mtl_call != nullptr) {
        (*rhs_mtl_call)(0);
        if (rhs_mtl_call->hasUpdate()) {
            ++m_version;
        }
        gpu_mtl_collections = rhs_mtl_call->getData();
    }
    gpu_mtl_collections.push_back(m_material_collection.first);

    if (this->m_btf_filename_slot.IsDirty()) {
        m_btf_filename_slot.ResetDirty();

        ++m_version;

        clearMaterialCollection();

        auto vislib_filename = m_btf_filename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.generic_u8string());

        m_material_collection.first->addMaterial(this->instance(), filename, filename);
        m_material_collection.second.push_back(filename);
    }

    lhs_mtl_call->setData(gpu_mtl_collections, m_version);

    return true;
}

bool megamol::mesh_gl::SimpleGPUMtlDataSource::getMetaDataCallback(core::Call& caller) {
    return true;
}
