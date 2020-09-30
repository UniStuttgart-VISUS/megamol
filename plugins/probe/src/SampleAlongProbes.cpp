/*
 * SampleAlongProbes.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "SampleAlongProbes.h"
#include "CallKDTree.h"
#include "ProbeCalls.h"
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"

namespace megamol {
namespace probe {

SampleAlongPobes::SampleAlongPobes()
    : Module()
    , _version(0)
    , _old_datahash(0)
    , _probe_lhs_slot("deployProbe", "")
    , _probe_rhs_slot("getProbe", "")
    , _adios_rhs_slot("getData", "")
    , _full_tree_rhs_slot("getTree", "")
    , _parameter_to_sample_slot("ParameterToSample", "")
    , _num_samples_per_probe_slot("NumSamplesPerProbe", "Note: Tighter sample placement leads to reduced sampling radius.")
    , _sample_radius_factor_slot("SampleRadiusFactor", "Multiplier for base sampling distance.")
    , _sampling_mode("SamplingMode", "")
    , _vec_param_to_samplex_x("ParameterToSampleX", "")
    , _vec_param_to_samplex_y("ParameterToSampleY", "")
    , _vec_param_to_samplex_z("ParameterToSampleZ", "")
    , _vec_param_to_samplex_w("ParameterToSampleW", "") {

    this->_probe_lhs_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &SampleAlongPobes::getData);
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

    this->_sample_radius_factor_slot << new core::param::FloatParam(1.0f);
    this->_sample_radius_factor_slot.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_sample_radius_factor_slot);

    this->_num_samples_per_probe_slot << new core::param::IntParam(10);
    this->_num_samples_per_probe_slot.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_num_samples_per_probe_slot);

    this->_sampling_mode << new megamol::core::param::EnumParam(0);
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Scalar");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Vector");
    this->_sampling_mode.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_sampling_mode);
	
	core::param::FlexEnumParam* paramEnum_1 = new core::param::FlexEnumParam("undef");
    this->_vec_param_to_samplex_x << paramEnum_1;
    this->_vec_param_to_samplex_x.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_vec_param_to_samplex_x);
	
	core::param::FlexEnumParam* paramEnum_2 = new core::param::FlexEnumParam("undef");
    this->_vec_param_to_samplex_y << paramEnum_2;
    this->_vec_param_to_samplex_y.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_vec_param_to_samplex_y);
	
	core::param::FlexEnumParam* paramEnum_3 = new core::param::FlexEnumParam("undef");
    this->_vec_param_to_samplex_z << paramEnum_3;
    this->_vec_param_to_samplex_z.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_vec_param_to_samplex_z);
	
	core::param::FlexEnumParam* paramEnum_4 = new core::param::FlexEnumParam("undef");
    this->_vec_param_to_samplex_w << paramEnum_4;
    this->_vec_param_to_samplex_w.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_vec_param_to_samplex_w);
}

SampleAlongPobes::~SampleAlongPobes() { this->Release(); }

bool SampleAlongPobes::create() { return true; }

void SampleAlongPobes::release() {}

bool SampleAlongPobes::getData(core::Call& call) {

    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr) return false;

    // query adios data
    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    std::vector<std::string> toInq;
    std::string var_str =
        std::string(this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString());

    std::string x_var_str =
        std::string(this->_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->ValueString());
    std::string y_var_str =
        std::string(this->_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->ValueString());
    std::string z_var_str =
        std::string(this->_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->ValueString());
    std::string w_var_str =
        std::string(this->_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->ValueString());

    toInq.clear();
    if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(
            std::string(this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(
            std::string(this->_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(
            std::string(this->_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(
            std::string(this->_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(
            std::string(this->_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquire(var)) return false;
    }

    if (cd->getDataHash() != _old_datahash || _trigger_recalc) {
        if (!(*cd)(0)) return false;
    }

    // query kd tree data
    auto ct = this->_full_tree_rhs_slot.CallAs<CallKDTree>();
    if (ct == nullptr) return false;
    if (!(*ct)(0)) return false;

    // query probe data
    auto cprobes = this->_probe_rhs_slot.CallAs<CallProbes>();
    if (cprobes == nullptr) return false;
    if (!(*cprobes)(0)) return false;

    bool something_has_changed = (cd->getDataHash() != _old_datahash) || ct->hasUpdate() || cprobes->hasUpdate() || _trigger_recalc;

    auto meta_data = cp->getMetaData();
    auto tree_meta_data = ct->getMetaData();
    auto probes_meta_data = cprobes->getMetaData();

    tree_meta_data = ct->getMetaData();
    probes_meta_data = cprobes->getMetaData();

    if (something_has_changed) {
        ++_version;

		if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0) {
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
		}
		else
		{
			//vector sampling
            _probes = cprobes->getData();
            auto tree = ct->getData();

			if (cd->getData(x_var_str)->getType() == "double" && cd->getData(y_var_str)->getType() == "double" &&
                cd->getData(z_var_str)->getType() == "double" && cd->getData(w_var_str)->getType() == "double")
			{
                std::vector<double> data_x = cd->getData(x_var_str)->GetAsDouble();
                std::vector<double> data_y = cd->getData(y_var_str)->GetAsDouble();
                std::vector<double> data_z = cd->getData(z_var_str)->GetAsDouble();
                std::vector<double> data_w = cd->getData(w_var_str)->GetAsDouble();
                doVectorSamling(tree, data_x, data_y, data_z, data_w);
			}
			else if (cd->getData(x_var_str)->getType() == "float" && cd->getData(y_var_str)->getType() == "float" &&
                cd->getData(z_var_str)->getType() == "float" && cd->getData(w_var_str)->getType() == "float"	)
			{
                std::vector<float> data_x = cd->getData(x_var_str)->GetAsFloat();
                std::vector<float> data_y = cd->getData(y_var_str)->GetAsFloat();
                std::vector<float> data_z = cd->getData(z_var_str)->GetAsFloat();
                std::vector<float> data_w = cd->getData(w_var_str)->GetAsFloat();
                doVectorSamling(tree, data_x, data_y, data_z, data_w);
			}
		}
    }

    // put data into probes
    if (probes_meta_data.m_bboxs.IsBoundingBoxValid()) {
        meta_data.m_bboxs = probes_meta_data.m_bboxs;
    } else if (tree_meta_data.m_bboxs.IsBoundingBoxValid()) {
        meta_data.m_bboxs = tree_meta_data.m_bboxs;
    }
    cp->setMetaData(meta_data);

    cp->setData(_probes, _version);
    _old_datahash = cd->getDataHash();
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
    if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_trigger_recalc)
        return true;

    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    if (!(*cd)(1)) return false;
    if (!(*ct)(1)) return false;
    if (!(*cplaceprobes)(1)) return false;

    // get adios meta data
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->AddValue(var);
    }

    // put metadata in mesh call
    meta_data.m_frame_cnt = cd->getFrameCount();
    cp->setMetaData(meta_data);

    return true;
}

bool SampleAlongPobes::paramChanged(core::param::ParamSlot& p) {

    _trigger_recalc = true;
    return true;
}

} // namespace probe
} // namespace megamol
