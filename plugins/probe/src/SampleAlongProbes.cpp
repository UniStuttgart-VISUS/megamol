/*
 * SampleAlongProbes.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "SampleAlongProbes.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCalls.h"


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
        , _num_samples_per_probe_slot(
              "NumSamplesPerProbe", "Note: Tighter sample placement leads to reduced sampling radius.")
        , _sample_radius_factor_slot("SampleRadiusFactor", "Multiplier for base sampling distance.")
        , _sampling_mode("SamplingMode", "")
        , _weighting("weighting", "")
        , _vec_param_to_samplex_x("ParameterToSampleX", "")
        , _vec_param_to_samplex_y("ParameterToSampleY", "")
        , _vec_param_to_samplex_z("ParameterToSampleZ", "")
        , _vec_param_to_samplex_w("ParameterToSampleW", "")
        , _volume_rhs_slot("getVolumeData", "") {

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

    this->_volume_rhs_slot.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->_volume_rhs_slot);

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
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "VolumeTrilin");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "Tetrahedral");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(4, "Nearest");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(5, "TetrahedralVector");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(6, "ScalarDistribution");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(7, "VolumeRadius");
    this->_sampling_mode.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_sampling_mode);

    this->_weighting << new megamol::core::param::EnumParam(0);
    this->_weighting.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "distance_based");
    this->_weighting.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "max_value");
    this->_weighting.SetUpdateCallback(&SampleAlongPobes::paramChanged);
    this->MakeSlotAvailable(&this->_weighting);

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

SampleAlongPobes::~SampleAlongPobes() {
    this->Release();
}

bool SampleAlongPobes::create() {
    return true;
}

void SampleAlongPobes::release() {}

bool SampleAlongPobes::getData(core::Call& call) {

    bool something_has_changed = false;
    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr)
        return false;

    // query adios data
    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto cv = this->_volume_rhs_slot.CallAs<geocalls::VolumetricDataCall>();
    auto ct = this->_full_tree_rhs_slot.CallAs<CallKDTree>();
    auto cprobes = this->_probe_rhs_slot.CallAs<CallProbes>();


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

    core::Spatial3DMetaData meta_data = cp->getMetaData();
    core::Spatial3DMetaData tree_meta_data;
    core::Spatial3DMetaData probes_meta_data;

    // only for particle data
    if (cd != nullptr && ct != nullptr) {

        toInq.clear();
        if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 3 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 4 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 6) {
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
            if (!cd->inquireVar(var))
                return false;
        }

        if (cd->getDataHash() != _old_datahash || _trigger_recalc) {
            if (!(*cd)(0))
                return false;
        }

        // query kd tree data
        if (!(*ct)(0))
            return false;


        tree_meta_data = ct->getMetaData();

        something_has_changed = something_has_changed || (cd->getDataHash() != _old_datahash) || ct->hasUpdate();
    } else if (cv != nullptr) {

        // get volume data
        if (!(*cv)(geocalls::VolumetricDataCall::IDX_GET_DATA))
            return false;

        meta_data.m_frame_cnt = cv->FrameCount();
        _vol_metadata = cv->GetMetadata();

        if (_vol_metadata->Components > 1) {
            core::utility::log::Log::DefaultLog.WriteError(
                "[SampleAlongProbes] Volume data has more than one component. Not supported.");
            return false;
        }

        something_has_changed = something_has_changed || (cv->DataHash() != _old_volume_datahash);
    } else {
        return false;
    }

    // query probe data
    if (cprobes == nullptr)
        return false;
    if (!(*cprobes)(0))
        return false;

    something_has_changed = something_has_changed || cprobes->hasUpdate() || _trigger_recalc;

    probes_meta_data = cprobes->getMetaData();
    _probes = cprobes->getData();

    if (something_has_changed) {
        ++_version;

        if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 3 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 4 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 6) {
            if (cd == nullptr || ct == nullptr) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[SampleAlongProbes] Scalar mode selected but no particle data connected.");
                return false;
            }
            // scalar sampling
            auto tree = ct->getData();
            if (cd->getData(var_str)->getType() == "double") {
                std::vector<double> data = cd->getData(var_str)->GetAsDouble();
                if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0) {
                    doScalarSampling(tree, data);
                } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 3) {
                    doTetrahedralSampling(tree, data);
                } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 6) {
                    doScalarDistributionSampling(tree, data);
                } else {
                    doNearestNeighborSampling(tree, data);
                }

            } else if (cd->getData(var_str)->getType() == "float") {
                std::vector<float> data = cd->getData(var_str)->GetAsFloat();
                if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0) {
                    doScalarSampling(tree, data);
                } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 3) {
                    doTetrahedralSampling(tree, data);
                } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 6) {
                    doScalarDistributionSampling(tree, data);
                } else {
                    doNearestNeighborSampling(tree, data);
                }
            }
        } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 1 ||
                   _sampling_mode.Param<core::param::EnumParam>()->Value() == 5) {
            if (cd == nullptr || ct == nullptr) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[SampleAlongProbes] Vector mode selected but no particle data connected.");
                return false;
            }
            // vector sampling
            auto tree = ct->getData();
            if (cd->getData(x_var_str)->getType() == "double" && cd->getData(y_var_str)->getType() == "double" &&
                cd->getData(z_var_str)->getType() == "double" && cd->getData(w_var_str)->getType() == "double") {
                std::vector<double> data_x = cd->getData(x_var_str)->GetAsDouble();
                std::vector<double> data_y = cd->getData(y_var_str)->GetAsDouble();
                std::vector<double> data_z = cd->getData(z_var_str)->GetAsDouble();
                std::vector<double> data_w = cd->getData(w_var_str)->GetAsDouble();

                if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 1) {
                    doVectorSamling(tree, data_x, data_y, data_z, data_w);
                } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 5) {
                    doTetrahedralVectorSamling(tree, data_x, data_y, data_z, data_w);
                }
            } else if (cd->getData(x_var_str)->getType() == "float" && cd->getData(y_var_str)->getType() == "float" &&
                       cd->getData(z_var_str)->getType() == "float" && cd->getData(w_var_str)->getType() == "float") {
                std::vector<float> data_x = cd->getData(x_var_str)->GetAsFloat();
                std::vector<float> data_y = cd->getData(y_var_str)->GetAsFloat();
                std::vector<float> data_z = cd->getData(z_var_str)->GetAsFloat();
                std::vector<float> data_w = cd->getData(w_var_str)->GetAsFloat();

                if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 1) {
                    doVectorSamling(tree, data_x, data_y, data_z, data_w);
                } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 5) {
                    doTetrahedralVectorSamling(tree, data_x, data_y, data_z, data_w);
                }
            }
        } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 2) {
            if (cv == nullptr) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[SampleAlongProbes] Volume mode selected but no volume data connected.");
                return false;
            }

            auto type = _vol_metadata->ScalarType;
            auto type_length = _vol_metadata->ScalarLength;

            if (type == geocalls::FLOATING_POINT) {
                auto data = reinterpret_cast<float*>(cv->GetData());
                doVolumeTrilinSampling(data);
            } else if (type == geocalls::UNSIGNED_INTEGER) {
                if (type_length < 4) {
                    auto data = reinterpret_cast<unsigned char*>(cv->GetData());
                    doVolumeTrilinSampling(data);
                } else {
                    auto data = reinterpret_cast<unsigned int*>(cv->GetData());
                    doVolumeTrilinSampling(data);
                }
            } else if (type == geocalls::SIGNED_INTEGER) {
                if (type_length < 4) {
                    auto data = reinterpret_cast<char*>(cv->GetData());
                    doVolumeTrilinSampling(data);
                } else {
                    auto data = reinterpret_cast<int*>(cv->GetData());
                    doVolumeTrilinSampling(data);
                }
            }
        } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 7) {
            if (cv == nullptr) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[SampleAlongProbes] Volume mode selected but no volume data connected.");
                return false;
            }

            auto type = _vol_metadata->ScalarType;
            auto type_length = _vol_metadata->ScalarLength;

            if (type == geocalls::FLOATING_POINT) {
                auto data = reinterpret_cast<float*>(cv->GetData());
                doVolumeRadiusSampling(data);
            } else if (type == geocalls::UNSIGNED_INTEGER) {
                if (type_length < 4) {
                    auto data = reinterpret_cast<unsigned char*>(cv->GetData());
                    doVolumeRadiusSampling(data);
                } else {
                    auto data = reinterpret_cast<unsigned int*>(cv->GetData());
                    doVolumeRadiusSampling(data);
                }
            } else if (type == geocalls::SIGNED_INTEGER) {
                if (type_length < 4) {
                    auto data = reinterpret_cast<char*>(cv->GetData());
                    doVolumeRadiusSampling(data);
                } else {
                    auto data = reinterpret_cast<int*>(cv->GetData());
                    doVolumeRadiusSampling(data);
                }
            } else {
                core::utility::log::Log::DefaultLog.WriteError("[SampleAlongProbes]: Volume data type not supported.");
                return false;
            }
        }
    }

    // put data into probes

    if (cd != nullptr) {
        _old_datahash = cd->getDataHash();
        meta_data.m_bboxs = tree_meta_data.m_bboxs;
    }
    if (cv != nullptr) {
        _old_volume_datahash = cv->DataHash();
        meta_data.m_bboxs.SetBoundingBox({_vol_metadata->Origin[0], _vol_metadata->Origin[1],
            _vol_metadata->Origin[2] + _vol_metadata->Extents[2], _vol_metadata->Origin[0] + _vol_metadata->Extents[0],
            _vol_metadata->Origin[1] + _vol_metadata->Extents[1], _vol_metadata->Origin[2]});
    }
    cp->setMetaData(meta_data);
    cp->setData(_probes, _version);
    _trigger_recalc = false;


    return true;
}

bool SampleAlongPobes::getMetaData(core::Call& call) {

    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr)
        return false;

    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto cv = this->_volume_rhs_slot.CallAs<geocalls::VolumetricDataCall>();
    auto ct = this->_full_tree_rhs_slot.CallAs<CallKDTree>();
    auto cprobes = this->_probe_rhs_slot.CallAs<CallProbes>();
    if (cprobes == nullptr)
        return false;

    auto meta_data = cp->getMetaData();
    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_trigger_recalc)
    //    return true;

    if (cd != nullptr && ct != nullptr) {
        cd->setFrameIDtoLoad(meta_data.m_frame_ID);
        if (!(*cd)(1))
            return false;
        if (!(*ct)(1))
            return false;
        meta_data.m_frame_cnt = cd->getFrameCount();

        // get adios meta data
        auto vars = cd->getAvailableVars();
        for (auto var : vars) {
            this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_vec_param_to_samplex_x.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_vec_param_to_samplex_y.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_vec_param_to_samplex_z.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->_vec_param_to_samplex_w.Param<core::param::FlexEnumParam>()->AddValue(var);
        }
    } else if (cv != nullptr) {
        cv->SetFrameID(meta_data.m_frame_ID);
        if (!(*cv)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
            return false;
        if (!(*cv)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
            return false;
        meta_data.m_frame_cnt = cv->FrameCount();
    } else {
        return false;
    }

    auto probes_meta_data = cprobes->getMetaData();
    probes_meta_data.m_frame_ID = meta_data.m_frame_ID;
    cprobes->setMetaData(probes_meta_data);
    if (!(*cprobes)(1))
        return false;

    // put metadata in mesh call
    cp->setMetaData(meta_data);

    return true;
}

bool SampleAlongPobes::paramChanged(core::param::ParamSlot& p) {

    _trigger_recalc = true;
    return true;
}

} // namespace probe
} // namespace megamol
