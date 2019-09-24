#include "PlaceProbes.h"

megamol::probe::PlaceProbes::PlaceProbes() 
: Module(), m_mesh_call("","") {

    /* Feasibility test */
    m_probes.addProbe(FloatProbe());

    auto retrieved_probe = m_probes.getProbe<FloatProbe>(0);

    retrieved_probe.probe();
}

megamol::probe::PlaceProbes::~PlaceProbes() {}

bool megamol::probe::PlaceProbes::create() { return false; }

void megamol::probe::PlaceProbes::release() {}

bool megamol::probe::PlaceProbes::getData(core::Call& call) { return false; }

bool megamol::probe::PlaceProbes::getMetaData(core::Call& call) { return false; }
