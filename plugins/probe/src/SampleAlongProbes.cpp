/*
 * SampleAlongProbes.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "SampleAlongProbes.h"
#include "CallKDTree.h"
#include "ProbeCalls.h"
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/param/FlexEnumParam.h"

namespace megamol {
namespace probe {

SampleAlongPobes::SampleAlongPobes()
    : Module()
    , _probe_lhs_slot("deployProbe", "")
    , _probe_rhs_slot("getProbe", "")
    , _adios_rhs_slot("getData", "")
    , _full_tree_rhs_slot("getTree", "")
    , _parameter_to_sample_slot("ParameterToSample", "")
    , _num_samples_per_probe_slot("NumSamplesPerProbe", "") {

    this->_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(0), &SampleAlongPobes::getData);
    this->_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &SampleAlongPobes::getMetaData);
    this->MakeSlotAvailable(&this->_probe_lhs_slot);

    this->_probe_rhs_slot.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->_probe_rhs_slot);

    this->_adios_rhs_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_adios_rhs_slot);

    this->_full_tree_rhs_slot.SetCompatibleCall<CallKDTreeDescription>();
    this->MakeSlotAvailable(&this->_full_tree_rhs_slot);

    core::param::FlexEnumParam* paramEnum = new core::param::FlexEnumParam("undef");
    this->_parameter_to_sample_slot << paramEnum;
    this->_parameter_to_sample_slot.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_parameter_to_sample_slot);

    this->_num_samples_per_probe_slot << new core::param::IntParam(10);
    this->_num_samples_per_probe_slot.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_num_samples_per_probe_slot);
}

SampleAlongPobes::~SampleAlongPobes() { this->Release(); }

bool SampleAlongPobes::create() { return true; }

void SampleAlongPobes::release() {}

bool SampleAlongPobes::getData(core::Call& call) {
    
    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr) return false;

    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto ct = this->_full_tree_rhs_slot.CallAs<CallKDTree>();
    if (ct == nullptr) return false;

    auto cprobes = this->_probe_rhs_slot.CallAs<CallProbes>();
    if (cprobes == nullptr) return false;


    auto meta_data = cp->getMetaData();
    auto tree_meta_data = ct->getMetaData();
    auto probes_meta_data = cprobes->getMetaData();


    std::vector<std::string> toInq;
    std::string var_str =
        std::string(this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString());
    toInq.clear();
    toInq.emplace_back(
        std::string(this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString()));

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquire(var)) return false;
    }

    if (cd->getDataHash() != _old_datahash)
        if (!(*cd)(0)) return false;
    
    if (tree_meta_data.m_data_hash != this->_full_tree_cached_hash)
        if (!(*ct)(0)) return false;

    if (probes_meta_data.m_data_hash != this->_probe_cached_hash)
        if (!(*cprobes)(0)) return false;

    tree_meta_data = ct->getMetaData();
    probes_meta_data = cprobes->getMetaData();

    // do sampling
    _probes = cprobes->getData();
    auto tree = ct->getData();
    if (cd->getData(var_str)->getType() == "double") {
        std::vector<double> data = cd->getData(var_str)->GetAsDouble();
        doSampling(tree, data);
    
    } else if (cd->getData(var_str)->getType() == "float") {
        std::vector<float> data = cd->getData(var_str)->GetAsFloat();
        doSampling(tree, data);
    }


    // put data into probes
    meta_data.m_bboxs = probes_meta_data.m_bboxs;
    cp->setMetaData(meta_data);


    cp->setData(_probes);
    _old_datahash = cd->getDataHash();
    this->_full_tree_cached_hash = tree_meta_data.m_data_hash;
    this->_probe_cached_hash = probes_meta_data.m_data_hash;
    _trigger_recalc = false;

    return true;
}

bool SampleAlongPobes::getMetaData(core::Call& call) {

    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr) return false;

    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto ct = this->_full_tree_rhs_slot.CallAs<CallKDTree>();
    if (ct == nullptr) return false;

    auto cplaceprobes = this->_probe_rhs_slot.CallAs<CallProbes>();
    if (cplaceprobes == nullptr) return false;

    auto meta_data = cp->getMetaData();
    if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_trigger_recalc) return true;

    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    if (!(*cd)(1)) return false;
    if (!(*ct)(1)) return false;
    if (!(*cplaceprobes)(1)) return false;

    // get adios meta data
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }

    // put metadata in mesh call
    meta_data.m_frame_cnt = cd->getFrameCount();
    meta_data.m_data_hash++;
    cp->setMetaData(meta_data);

    return true;
}

bool SampleAlongPobes::paramChanged(core::param::ParamSlot& p) {
    
    _trigger_recalc = true;
    return true;
}

} // namespace probe
} // namespace megamol
