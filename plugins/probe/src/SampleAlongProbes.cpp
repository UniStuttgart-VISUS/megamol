#include "adios_plugin/CallADIOSData.h" 
#include "SampleAlongProbes.h"
#include "ProbeCalls.h"

megamol::probe::SampleAlongPobes::SampleAlongPobes() 
    : Module()
    , m_probe_lhs_slot("deployProbe", "")
    , m_probe_rhs_slot("getProbe", "")
    , m_adios_rhs_slot("getData","") {

    this->m_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(0), &SampleAlongPobes::getData);
    this->m_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &SampleAlongPobes::getMetaData);
    this->MakeSlotAvailable(&this->m_probe_lhs_slot);

    this->m_probe_rhs_slot.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probe_rhs_slot);

    this->m_adios_rhs_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->m_adios_rhs_slot);
}

megamol::probe::SampleAlongPobes::~SampleAlongPobes() {}

bool megamol::probe::SampleAlongPobes::create() { return false; }

void megamol::probe::SampleAlongPobes::release() {}

bool megamol::probe::SampleAlongPobes::getData(core::Call& call) { return true; }

bool megamol::probe::SampleAlongPobes::getMetaData(core::Call& call) { return true; }
