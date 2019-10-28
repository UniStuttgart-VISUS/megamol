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
    , m_probe_lhs_slot("deployProbe", "")
    , m_probe_rhs_slot("getProbe", "")
    , m_adios_rhs_slot("getData", "")
    , m_full_tree_rhs_slot("getTree", "")
    , m_parameter_to_sample_slot("ParameterToSample", "") {

    this->m_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(0), &SampleAlongPobes::getData);
    this->m_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &SampleAlongPobes::getMetaData);
    this->MakeSlotAvailable(&this->m_probe_lhs_slot);

    this->m_probe_rhs_slot.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->m_probe_rhs_slot);

    this->m_adios_rhs_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->m_adios_rhs_slot);

    this->m_full_tree_rhs_slot.SetCompatibleCall<CallKDTreeDescription>();
    this->MakeSlotAvailable(&this->m_full_tree_rhs_slot);

    core::param::FlexEnumParam* paramEnum = new core::param::FlexEnumParam("undef");
    this->m_parameter_to_sample_slot << paramEnum;
    this->m_parameter_to_sample_slot.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->m_parameter_to_sample_slot);
}

SampleAlongPobes::~SampleAlongPobes() { this->Release(); }

bool SampleAlongPobes::create() { return true; }

void SampleAlongPobes::release() {}

void SampleAlongPobes::doSampling(const std::shared_ptr<ProbeCollection>& probes,
    const std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>>& tree, std::vector<float>& data) {
}

bool SampleAlongPobes::getData(core::Call& call) {
    
    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr) return false;

    auto cd = this->m_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto ct = this->m_full_tree_rhs_slot.CallAs<CallKDTree>();
    if (ct == nullptr) return false;

    auto cprobes = this->m_probe_rhs_slot.CallAs<CallProbes>();
    if (cprobes == nullptr) return false;


    auto meta_data = cp->getMetaData();
    auto tree_meta_data = ct->getMetaData();
    auto probes_meta_data = cprobes->getMetaData();


    std::vector<std::string> toInq;
    toInq.clear();
    toInq.emplace_back(
        std::string(this->m_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString()));

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquire(var)) return false;
    }

    if (cd->getDataHash() != _old_datahash)
        if (!(*cd)(0)) return false;
    
    if (tree_meta_data.m_data_hash != this->m_full_tree_cached_hash)
        if (!(*ct)(0)) return false;

    if (probes_meta_data.m_data_hash != this->m_probe_cached_hash)
        if (!(*cprobes)(0)) return false;



    // do sampling
    auto probes = cprobes->getData();
    auto tree = ct->getData();
    auto data = cd->getData(this->m_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->Value())->GetAsFloat();
    doSampling(probes, tree, data);


    // put data into probes
    meta_data.m_bboxs = probes_meta_data.m_bboxs;
    cp->setMetaData(meta_data);


    cp->setData(probes);
    _old_datahash = cd->getDataHash();
    this->m_full_tree_cached_hash = tree_meta_data.m_data_hash;
    this->m_probe_cached_hash = probes_meta_data.m_data_hash;
    m_recalc = false;

    return true;



}

bool megamol::probe::SampleAlongPobes::getMetaData(core::Call& call) {

    auto cp = dynamic_cast<mesh::CallMesh*>(&call);
    if (cp == nullptr) return false;

    auto cd = this->m_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto ct = this->m_full_tree_rhs_slot.CallAs<CallKDTree>();
    if (ct == nullptr) return false;

    auto meta_data = cp->getMetaData();
    if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !m_recalc) return true;

    // get metadata from adios

    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    if (!(*cd)(1)) return false;
    if (!(*ct)(1)) return false;

    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->m_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }

    // put metadata in mesh call
    meta_data.m_frame_cnt = cd->getFrameCount();
    meta_data.m_data_hash++;
    cp->setMetaData(meta_data);

    return true;
}

bool SampleAlongPobes::paramChanged(core::param::ParamSlot& p) {
    
    m_recalc = true;
    return true;
}

} // namespace probe
} // namespace megamol
