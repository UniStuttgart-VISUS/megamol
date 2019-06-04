#include "CreateFEMModel.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

megamol::archvis::CreateFEMModel::CreateFEMModel() 
    : Module()
    , m_getData_slot("femmodel", "Provides the FEM Model.")
    , m_node_floatTable_slot("nodes", "Node float table input call.")
    , m_element_floatTable_slot("elements", "Element float table input call.")
    , m_displacement_floatTable_slot("displacements", "Displacement float table input call.")
{
    this->m_getData_slot.SetCallback(FEMDataCall::ClassName(), "GetData", &CreateFEMModel::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    //TODO GetExtents?

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

megamol::archvis::CreateFEMModel::~CreateFEMModel() {}

bool megamol::archvis::CreateFEMModel::create(void) { return false; }

bool megamol::archvis::CreateFEMModel::getDataCallback(core::Call& caller) {

    auto node_ft =
        this->m_node_floatTable_slot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
    auto element_ft =
        this->m_element_floatTable_slot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();

    // node and element data are mandatory, return false is either is not available
    if (node_ft == NULL || element_ft == NULL) {
        return false;
    }

    (*node_ft)();
    (*element_ft)();


    return true;
}

void megamol::archvis::CreateFEMModel::release() {}
