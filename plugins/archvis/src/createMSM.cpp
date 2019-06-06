#include "CreateMSM.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

#include "ScaleModel.h"

megamol::archvis::CreateMSM::CreateMSM() : Module()
    , m_MSM(nullptr)
    , m_getData_slot("msmmodel", "Provides the MSM Model.")
    , m_node_floatTable_slot("nodes", "Node float table input call.")
    , m_element_floatTable_slot("elements", "Element float table input call.")
    , m_displacement_floatTable_slot("displacements", "Displacement float table input call.")
    , m_node_input_hash(0)
    , m_element_input_hash(0)
    , m_displacement_input_hash(0)
    , m_my_hash(0)
{
    this->m_getData_slot.SetCallback(MSMDataCall::ClassName(), "GetData", &CreateMSM::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    // TODO GetExtents?

    this->m_node_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->m_node_floatTable_slot);

    this->m_element_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->m_element_floatTable_slot);

    this->m_displacement_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->m_displacement_floatTable_slot);
}

megamol::archvis::CreateMSM::~CreateMSM() {
}

bool megamol::archvis::CreateMSM::create(void) { return false; }

bool megamol::archvis::CreateMSM::getDataCallback(core::Call& caller) { return false; }

void megamol::archvis::CreateMSM::release() {}
