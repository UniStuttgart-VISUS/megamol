#include "CreateFEMModel.h"

#include "mmstd_datatools/table/TableDataCall.h"
#include "mmcore/param/IntParam.h"

#include "FEMModel.h"

megamol::archvis::CreateFEMModel::CreateFEMModel() : Module()
    , m_FEM_model(nullptr)
    , m_fem_param_0("First FEM model parameter", "First input for parameterized FEM model.")
    , m_fem_param_1("Second FEM model parameter", "Second input for parameterized FEM model.")
    , m_getData_slot("femmodel", "Provides the FEM Model.")
    , m_node_floatTable_slot("nodes", "Node float table input call.")
    , m_element_floatTable_slot("elements", "Element float table input call.")
    , m_deformation_floatTable_slot("displacements", "Displacement float table input call.")
    , m_node_input_hash(0)
    , m_element_input_hash(0)
    , m_deform_input_hash(0)
    , m_my_hash(0) {
    this->m_getData_slot.SetCallback(FEMDataCall::ClassName(), "GetData", &CreateFEMModel::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    //TODO GetExtents?

    this->m_fem_param_0 << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->m_fem_param_0);

    this->m_fem_param_1 << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->m_fem_param_1);

    this->m_node_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->m_node_floatTable_slot);

    this->m_element_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->m_element_floatTable_slot);

    this->m_deformation_floatTable_slot
        .SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->m_deformation_floatTable_slot);
}

megamol::archvis::CreateFEMModel::~CreateFEMModel() {}

bool megamol::archvis::CreateFEMModel::create(void) {
    return true;
}

bool megamol::archvis::CreateFEMModel::getDataCallback(core::Call& caller) {

    FEMDataCall* cd = dynamic_cast<FEMDataCall*>(&caller);
    if (cd == NULL){
        return false;
    }

    auto node_ft =
        this->m_node_floatTable_slot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    auto element_ft =
        this->m_element_floatTable_slot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
    auto deformation_ft =
        this->m_deformation_floatTable_slot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();

    // node and element data are mandatory, return false is either is not available
    if (node_ft == NULL || element_ft == NULL || deformation_ft == NULL) {
        return false;
    }

    // update frame id for faked parameter changing...
    
    auto p0 = this->m_fem_param_0.Param<core::param::IntParam>()->Value();
    auto p1 = this->m_fem_param_1.Param<core::param::IntParam>()->Value();

    unsigned int frame_id = p0 * 11 + p1;

    deformation_ft->SetFrameID(frame_id);

    (*node_ft)();
    (*element_ft)();
    (*deformation_ft)();

    if ( this->m_node_input_hash == node_ft->DataHash() 
        && this->m_element_input_hash == element_ft->DataHash()
        && this->m_deform_input_hash == deformation_ft->DataHash()) {

        if (m_FEM_model != nullptr)
        {
            cd->setFEMData(m_FEM_model);
            cd->SetDataHash(this->m_my_hash);
        }
        
        return true;
    }

    if (node_ft->GetColumnsCount() != 4){
        return false;
    }

    std::vector<FEMModel::Vec3> nodes;
    nodes.reserve(node_ft->GetRowsCount());

    std::vector<FEMModel::Vec4> deformations;
    deformations.reserve(deformation_ft->GetRowsCount());

    auto const node_accessor = node_ft->GetData();
    for (int node_idx = 0; node_idx < node_ft->GetRowsCount(); ++node_idx) {
        auto curr_idx = node_idx * 4;
        nodes.push_back({
            node_accessor[curr_idx + 1],
            node_accessor[curr_idx + 2],
            node_accessor[curr_idx + 3]
            }
        );
    }

    auto const deform_accessor = deformation_ft->GetData();
    for (int deform_idx = 0; deform_idx < deformation_ft->GetRowsCount(); ++deform_idx) {
        auto curr_idx = deform_idx * 13;
        deformations.push_back({
            deform_accessor[curr_idx + 4],
            deform_accessor[curr_idx + 5],
            deform_accessor[curr_idx + 6],
            0.0f //padding
            }
        );
    }

    auto const elem_accesssor = element_ft->GetData();
    if (element_ft->GetColumnsCount() == 9)
    {
        std::vector<std::array<size_t, 8>> elements;
        elements.reserve(element_ft->GetRowsCount());

        for (int elem_idx = 0; elem_idx < element_ft->GetRowsCount(); ++elem_idx) {

            auto curr_idx = elem_idx * 9;
            elements.push_back({
                    static_cast<size_t>(elem_accesssor[curr_idx + 1]),
                    static_cast<size_t>(elem_accesssor[curr_idx + 2]),
                    static_cast<size_t>(elem_accesssor[curr_idx + 3]),
                    static_cast<size_t>(elem_accesssor[curr_idx + 4]),
                    static_cast<size_t>(elem_accesssor[curr_idx + 5]),
                    static_cast<size_t>(elem_accesssor[curr_idx + 6]),
                    static_cast<size_t>(elem_accesssor[curr_idx + 7]),
                    static_cast<size_t>(elem_accesssor[curr_idx + 8])
                }
            );
        }

        m_FEM_model = std::make_shared<FEMModel>(nodes, elements);
        m_FEM_model->setNodeDeformations(deformations);
    }
    else
    {
        return false;
    }

    this->m_my_hash++;
    this->m_node_input_hash = node_ft->DataHash();
    this->m_element_input_hash = element_ft->DataHash();
    this->m_deform_input_hash = deformation_ft->DataHash();

    cd->setFEMData(m_FEM_model);
    cd->SetDataHash(this->m_my_hash);

    return true;
}

void megamol::archvis::CreateFEMModel::release() {}
