#include "SampleAlongProbes.h"

#include "ProbeCalls.h"

megamol::probe::SampleAlongPobes::SampleAlongPobes() 
    : Module()
    , m_probe_lhs_call("", "")
    , m_probe_rhs_call("", "") 
{
    this->m_probe_lhs_call.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(0), &SampleAlongPobes::getData);
    this->m_probe_lhs_call.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &SampleAlongPobes::getMetaData);
    this->MakeSlotAvailable(&this->m_probe_lhs_call);

    this->m_probe_rhs_call.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probe_rhs_call);
}

megamol::probe::SampleAlongPobes::~SampleAlongPobes() {}

bool megamol::probe::SampleAlongPobes::create() { return false; }

void megamol::probe::SampleAlongPobes::release() {}

bool megamol::probe::SampleAlongPobes::getData(core::Call& call) { return false; }

bool megamol::probe::SampleAlongPobes::getMetaData(core::Call& call) { return false; }
