#include "ProbeBillboardGlyphRenderTasks.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::ProbeBillboardGlyphRenderTasks() 
    : m_probes_slot("GetProbes", "Slot for accessing a probe collection"), m_probes_cached_hash(0) {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

}

megamol::probe_gl::ProbeBillboardGlyphRenderTasks::~ProbeBillboardGlyphRenderTasks() {}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getDataCallback(core::Call& caller) { 

    mesh::CallGPURenderTaskData* lhs_mtl_call = dynamic_cast<mesh::CallGPURenderTaskData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    // no incoming material -> use your own material storage
    if (lhs_mtl_call->getData() == nullptr) lhs_mtl_call->setData(this->m_gpu_render_tasks);
    std::shared_ptr<mesh::GPURenderTaskCollection> rt_collection = lhs_mtl_call->getData();

    // if there is a material connection to the right, pass on the material collection
    mesh::CallGPURenderTaskData* rhs_mtl_call = this->m_renderTask_rhs_slot.CallAs<mesh::CallGPURenderTaskData>();
    if (rhs_mtl_call != NULL) rhs_mtl_call->setData(rt_collection);

    return true; 
}

bool megamol::probe_gl::ProbeBillboardGlyphRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!mesh::AbstractGPURenderTaskDataSource::getMetaDataCallback(caller)) return false;

    auto probe_call = m_probes_slot.CallAs<probe::CallProbes>();
    if (!(*probe_call)(1)) return false;

    return true;

}
