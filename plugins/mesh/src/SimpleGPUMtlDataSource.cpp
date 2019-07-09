#include "mesh/SimpleGPUMtlDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mesh/CallGPUMaterialData.h"


megamol::mesh::SimpleGPUMtlDataSource::SimpleGPUMtlDataSource()
    : m_btf_filename_slot("BTF filename", "The name of the btf file to load") {
    this->m_btf_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_btf_filename_slot);
}

megamol::mesh::SimpleGPUMtlDataSource::~SimpleGPUMtlDataSource() {}

bool megamol::mesh::SimpleGPUMtlDataSource::create() { return true; }

bool megamol::mesh::SimpleGPUMtlDataSource::getDataCallback(core::Call& caller) {
    CallGPUMaterialData* lhs_mtl_call = dynamic_cast<CallGPUMaterialData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    std::shared_ptr<GPUMaterialCollecton> mtl_collection(nullptr);

    if (lhs_mtl_call->getMaterialStorage() == nullptr) {
        // no incoming material -> use your own material storage
        mtl_collection = this->m_gpu_materials;
        lhs_mtl_call->setMaterialStorage(mtl_collection);
    } else {
        // incoming material -> use it (delete local?)
        mtl_collection = lhs_mtl_call->getMaterialStorage();
    }

    // clear update?

    if (this->m_btf_filename_slot.IsDirty()) {
        m_btf_filename_slot.ResetDirty();

        auto vislib_filename = m_btf_filename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.PeekBuffer());

        mtl_collection->clearMaterials();

        mtl_collection->addMaterial(this->instance(), filename);
    }

    // set update?

    // if there is a material connection to the right, pass on the material collection
    CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<CallGPUMaterialData>();
    if (rhs_mtl_call != NULL) {
        rhs_mtl_call->setMaterialStorage(mtl_collection);
    }

    return true;
}