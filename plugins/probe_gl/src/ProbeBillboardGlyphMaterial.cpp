#include "ProbeBillboardGlyphMaterial.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

megamol::probe_gl::ProbeBillboardGlyphMaterial::ProbeBillboardGlyphMaterial() 
    : m_probes_slot("GetProbes", "Slot for accessing a probe collection"), m_probes_cached_hash(0) {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

}

megamol::probe_gl::ProbeBillboardGlyphMaterial::~ProbeBillboardGlyphMaterial() {}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::create() { return true; }

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getDataCallback(core::Call& caller) {

    mesh::CallGPUMaterialData* lhs_mtl_call = dynamic_cast<mesh::CallGPUMaterialData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    // no incoming material -> use your own material storage
    if (lhs_mtl_call->getData() == nullptr) lhs_mtl_call->setData(this->m_gpu_materials);
    std::shared_ptr<mesh::GPUMaterialCollecton> mtl_collection = lhs_mtl_call->getData();

    // if there is a material connection to the right, pass on the material collection
    mesh::CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<mesh::CallGPUMaterialData>();
    if (rhs_mtl_call != NULL) rhs_mtl_call->setData(mtl_collection);

    // ToDo the actual data getting

    return true; 
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getMetaDataCallback(core::Call& caller) {

    //if (!mesh::AbstractGPUMaterialDataSource::getMetaDataCallback(caller)) return false;

    auto probe_call = m_probes_slot.CallAs<probe::CallProbes>();
    if (!(*probe_call)(1)) return false;

    return true;
}
