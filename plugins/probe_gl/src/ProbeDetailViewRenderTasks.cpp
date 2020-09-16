#include "ProbeDetailViewRenderTasks.h"

#include "ProbeCalls.h"
#include "ProbeGlCalls.h"
#include "mmcore/view/CallGetTransferFunction.h"

bool megamol::probe_gl::ProbeDetailViewRenderTasks::create() { return false; }

void megamol::probe_gl::ProbeDetailViewRenderTasks::release() {}

megamol::probe_gl::ProbeDetailViewRenderTasks::ProbeDetailViewRenderTasks()
    : m_version(0)
    , m_transfer_function_Slot("GetTransferFunction", "Slot for accessing a transfer function")
    , m_probes_slot("GetProbes", "Slot for accessing a probe collection")
    , m_probe_manipulation_slot("GetProbeManipulation", "")
    , m_ui_mesh(nullptr)
    , m_probes_mesh(nullptr)
    , m_tf_min(0.0f)
    , m_tf_max(1.0f) {
    this->m_transfer_function_Slot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->m_transfer_function_Slot);

    this->m_probes_slot.SetCompatibleCall<probe::CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probes_slot);

    this->m_probe_manipulation_slot.SetCompatibleCall<probe_gl::CallProbeInteractionDescription>();
    this->MakeSlotAvailable(&this->m_probe_manipulation_slot);
}

megamol::probe_gl::ProbeDetailViewRenderTasks::~ProbeDetailViewRenderTasks() {}

bool megamol::probe_gl::ProbeDetailViewRenderTasks::getDataCallback(core::Call& caller) { return false; }

bool megamol::probe_gl::ProbeDetailViewRenderTasks::getMetaDataCallback(core::Call& caller) { return false; }

megamol::probe_gl::ProbeDetailViewRenderTasks::VectorProbeData
megamol::probe_gl::ProbeDetailViewRenderTasks::createVectorProbeData(
    probe::Vec4Probe const& probe, int probe_id, float scale) {
    return VectorProbeData();
}
