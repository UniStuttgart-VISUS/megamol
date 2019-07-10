#include "CreateMSM.h"

#include "mmstd_datatools/table/TableDataCall.h"

#include "ScaleModel.h"

megamol::archvis::CreateMSM::CreateMSM() : Module()
    , m_MSM(nullptr)
    , m_getData_slot("msmmodel", "Provides the MSM Model.")
    , m_node_floatTable_slot("nodes", "Node float table input call.")
    , m_element_floatTable_slot("elements", "Element float table input call.")
    , m_inputElement_floatTable_slot("inputElements", "Input element float table input call.")
    , m_displacement_floatTable_slot("displacements", "Displacement float table input call.")
    , m_node_input_hash(0)
    , m_element_input_hash(0)
    , m_inputElement_input_hash(0)
    , m_displacement_input_hash(0)
    , m_my_hash(0)
{
    this->m_getData_slot.SetCallback(MSMDataCall::ClassName(), "GetData", &CreateMSM::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    // TODO GetExtents?

    this->m_node_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->m_node_floatTable_slot);

    this->m_element_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->m_element_floatTable_slot);

    this->m_inputElement_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->m_inputElement_floatTable_slot);

    this->m_displacement_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->m_displacement_floatTable_slot);
}

megamol::archvis::CreateMSM::~CreateMSM() {
}

bool megamol::archvis::CreateMSM::create(void) { 
    return true;
}

bool megamol::archvis::CreateMSM::getDataCallback(core::Call& caller) {

    MSMDataCall* msm_call = dynamic_cast<MSMDataCall*>(&caller);
    if (msm_call == NULL) {
        return false;
    }

    auto node_ft = this->m_node_floatTable_slot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    auto element_ft = this->m_element_floatTable_slot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    auto inputElement_ft =
        this->m_inputElement_floatTable_slot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    auto displacement_ft =
        this->m_displacement_floatTable_slot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();

    // node and element data are mandatory, return false is either is not available
    if (node_ft == NULL || element_ft == NULL || inputElement_ft == NULL || displacement_ft == NULL) {
        return false;
    }

    (*node_ft)();
    (*element_ft)();
    (*inputElement_ft)();
    (*displacement_ft)();

    if (this->m_node_input_hash == node_ft->DataHash() &&
        this->m_element_input_hash == element_ft->DataHash() &&
        this->m_inputElement_input_hash == inputElement_ft->DataHash() &&
        this->m_displacement_input_hash == displacement_ft->DataHash())
    {
        if (m_MSM != nullptr) {
            msm_call->setMSM(m_MSM);
            msm_call->SetDataHash(this->m_my_hash);
        }

        return true;
    }

    if (node_ft->GetColumnsCount() != 3) {
        return false;
    }

    std::vector<ScaleModel::Vec3> nodes;
    nodes.reserve(node_ft->GetRowsCount());

    auto const node_accessor = node_ft->GetData();
    for (int node_idx = 0; node_idx < node_ft->GetRowsCount(); ++node_idx) {
        auto curr_idx = node_idx * 3;
        nodes.push_back({node_accessor[curr_idx + 0], node_accessor[curr_idx + 1], node_accessor[curr_idx + 2]});
    }


    
    auto displ_row_cnt = displacement_ft->GetRowsCount();
    auto displ_col_cnt = displacement_ft->GetColumnsCount();
    std::vector<ScaleModel::Vec3> deformations;
    deformations.reserve(displ_row_cnt * (displ_col_cnt/3));
    auto const displ_accessor = displacement_ft->GetData();
    for (int row_idx = 0; row_idx < displ_row_cnt; ++row_idx) {
    
        for (int col_idx = 0; col_idx < displ_col_cnt; col_idx = col_idx + 3)
        {
            auto curr_idx = col_idx + (row_idx * displ_col_cnt);
    
            deformations.push_back({
                displ_accessor[curr_idx + 1], 
                displ_accessor[curr_idx + 2], 
                displ_accessor[curr_idx + 3]
            });
        }
    }

    std::vector<std::tuple<int, int, int, int, int>> elements;
    elements.reserve(element_ft->GetRowsCount());

    auto const elem_accesssor = element_ft->GetData();
    if (element_ft->GetColumnsCount() == 4) {

        for (int elem_idx = 0; elem_idx < element_ft->GetRowsCount(); ++elem_idx) {

            auto curr_idx = elem_idx * 4;

            int type = (elem_idx % 13) == 0 ? 2 : (elem_idx % 13) < 5 ? 0 : 1;

            elements.push_back({
                type, static_cast<int>(elem_accesssor[curr_idx + 0]), static_cast<int>(elem_accesssor[curr_idx + 1]),
                    static_cast<int>(elem_accesssor[curr_idx + 2]), static_cast<int>(elem_accesssor[curr_idx + 3])
                });
        }

    } else {
        return false;
    }

    std::vector<int> inputElements;
    inputElements.reserve(inputElement_ft->GetRowsCount());
    auto const inputElem_accesssor = inputElement_ft->GetData();
    if (inputElement_ft->GetColumnsCount() == 1) {
    
        for (int iE_idx = 0; iE_idx < inputElement_ft->GetRowsCount(); ++iE_idx)
        {
            inputElements.push_back(inputElem_accesssor[iE_idx]);
        }

    } else {
        return false;
    }


    m_MSM = std::make_shared<ScaleModel>(nodes, elements, inputElements);
    m_MSM->updateNodeDisplacements(deformations);

    this->m_my_hash++;
    this->m_node_input_hash = node_ft->DataHash();
    this->m_element_input_hash = element_ft->DataHash();
    this->m_displacement_input_hash = displacement_ft->DataHash();

    msm_call->setMSM(m_MSM);
    msm_call->SetDataHash(this->m_my_hash);


    return true;
}

void megamol::archvis::CreateMSM::release() {}
