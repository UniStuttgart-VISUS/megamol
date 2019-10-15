#include "ProbeRenderTasks.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

megamol::mesh::ProbeRenderTasks::ProbeRenderTasks() 
: m_probes_slot("GetProbes", "Slot for accessing a probe collection"), m_probes_cached_hash(0) {

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);
}

megamol::mesh::ProbeRenderTasks::~ProbeRenderTasks() {}

bool megamol::mesh::ProbeRenderTasks::getDataCallback(core::Call& caller) { 

    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    mesh::CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<mesh::CallGPUMaterialData>();
    if (mtlc == NULL) return false;

    if (!(*mtlc)(0)) return false;

    mesh::CallGPUMeshData* mc = this->m_mesh_slot.CallAs<mesh::CallGPUMeshData>();
    if (mc == NULL) return false;

    if (!(*mc)(0)) return false; //TODO only call callback when hash is outdated?

    probe::CallProbes* pc = dynamic_cast<probe::CallProbes*>(&caller);
    if (pc == NULL) return false;
    
    std::shared_ptr<GPURenderTaskCollection> rt_collection(nullptr);

    if (lhs_rtc->getData() == nullptr) {
        rt_collection = this->m_gpu_render_tasks;
        lhs_rtc->setData(rt_collection);
    } else {
        rt_collection = lhs_rtc->getData();
    }

    auto probe_meta_data = pc->getMetaData();

    if (probe_meta_data.m_data_hash > m_probes_cached_hash)
    {
        if (!(*pc)(0)) return false;
        auto probes = pc->getData();

        auto probe_cnt = probes->getProbeCount();

        for (int probe_idx = 0; probe_idx < probe_cnt; ++probe_idx)
        {
            try {
                auto probe = probes->getProbe<probe::FloatProbe>(probe_idx);

                // TODO create and add new render task for probe
            } 
            catch (std::bad_variant_access&) {
                // TODO log error, dont add new render task
            }
            
        }

    }

    return true; 
}

bool megamol::mesh::ProbeRenderTasks::getMetaDataCallback(core::Call& caller) {

    if (!AbstractGPURenderTaskDataSource::getMetaDataCallback(caller)) return false;

    return true;
}
