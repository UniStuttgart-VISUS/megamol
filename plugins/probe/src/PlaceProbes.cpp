#include "PlaceProbes.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

megamol::probe::PlaceProbes::PlaceProbes() 
    : Module()
    , m_mesh_call("", "")
    , m_probe_call("", "") 
{

    this->m_probe_call.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(0), &PlaceProbes::getData);
    this->m_probe_call.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &PlaceProbes::getMetaData);
    this->MakeSlotAvailable(&this->m_probe_call);

    this->m_mesh_call.SetCompatibleCall<mesh::CallGPUMeshDataDescription>();
    this->MakeSlotAvailable(&this->m_mesh_call);

    /* Feasibility test */
    m_probes->addProbe(FloatProbe());

    auto retrieved_probe = m_probes->getProbe<FloatProbe>(0);

    retrieved_probe.probe();
}

megamol::probe::PlaceProbes::~PlaceProbes() {}

bool megamol::probe::PlaceProbes::create() { return false; }

void megamol::probe::PlaceProbes::release() {}

bool megamol::probe::PlaceProbes::getData(core::Call& call) { return false; }

bool megamol::probe::PlaceProbes::getMetaData(core::Call& call) { return false; }
