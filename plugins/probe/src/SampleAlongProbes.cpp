/*
 * SampleAlongProbes.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "SampleAlongProbes.h"
#include "CallKDTree.h"
#include "ProbeCalls.h"
#include "adios_plugin/CallADIOSData.h"
#include "glm/glm.hpp"
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

    this->_volume_rhs_slot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
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
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(2, "Volume");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(3, "Tetrahedral");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(4, "Nearest");
    this->_sampling_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(5, "TetrahedralVector");
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

SampleAlongPobes::~SampleAlongPobes() { this->Release(); }

bool SampleAlongPobes::create() { return true; }

void SampleAlongPobes::release() {}

void SampleAlongPobes::doVolumeSampling() {
    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();
    const float sample_radius_factor = this->_sample_radius_factor_slot.Param<core::param::FloatParam>()->Value();

    glm::vec3 origin = {_vol_metadata->Origin[0], _vol_metadata->Origin[1], _vol_metadata->Origin[2]};
    glm::vec3 spacing = {*_vol_metadata->SliceDists[0], *_vol_metadata->SliceDists[1], *_vol_metadata->SliceDists[2]};
    float min_spacing = std::min(std::min(spacing.x, spacing.y), spacing.z);

    float global_min = std::numeric_limits<float>::max();
    float global_max = -std::numeric_limits<float>::max();
    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); i++) {

        FloatProbe probe;

        auto visitor = [&probe, i, samples_per_probe, sample_radius_factor, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::Vec4Probe>) {

                probe.m_timestamp = arg.m_timestamp;
                probe.m_value_name = arg.m_value_name;
                probe.m_position = arg.m_position;
                probe.m_direction = arg.m_direction;
                probe.m_begin = arg.m_begin;
                probe.m_end = arg.m_end;
                probe.m_cluster_id = arg.m_cluster_id;

                auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
                auto radius = 0.5 * sample_step * sample_radius_factor;
                probe.m_sample_radius = radius;

                _probes->setProbe(i, probe);

            } else if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                probe = arg;

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(i);
        std::visit(visitor, generic_probe);

        auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
        auto radius = 0.5 * sample_step * sample_radius_factor;
        auto grid_radius = glm::vec3(radius) / spacing;
        std::array<int, 3> num_grid_points_per_dim = {grid_radius.x * 2, grid_radius.y * 2, grid_radius.z * 2};

        bool get_nearest = false;
        for (int i = 0; i < num_grid_points_per_dim.size(); ++i) {
            if (num_grid_points_per_dim[i] < 1) {
                num_grid_points_per_dim[i] = 1;
                get_nearest = true;
            }
        }

        std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();
        float min_value = std::numeric_limits<float>::max();
        float max_value = -std::numeric_limits<float>::max();
        float min_data = std::numeric_limits<float>::max();
        float max_data = -std::numeric_limits<float>::max();
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);
        

        for (int j = 0; j < samples_per_probe; j++) {

            glm::vec3 sample_point;
            sample_point.x = probe.m_position[0] + j * sample_step * probe.m_direction[0];
            sample_point.y = probe.m_position[1] + j * sample_step * probe.m_direction[1];
            sample_point.z = probe.m_position[2] + j * sample_step * probe.m_direction[2];

            // calculate in which cell (i,j,k) the point resides in
            glm::vec3 grid_point = (sample_point - origin) / spacing;

            glm::vec3 start = {std::roundf(grid_point.x - grid_radius.x),
                std::roundf(grid_point.y - grid_radius.y),
                std::roundf(grid_point.z - grid_radius.z)};
            auto end = grid_point + grid_radius;

            float value = 0;
            int num_samples = 0;
            for (int k = 0; k < num_grid_points_per_dim[0]; ++k) {
                for (int l = 0; l < num_grid_points_per_dim[1]; ++l) {
                    for (int m = 0; m < num_grid_points_per_dim[2]; ++m) {
                        auto pos = start + glm::vec3(k,l,m);
                        auto dif = pos - grid_point;
                        if ((std::abs(dif.x) <= grid_radius.x && std::abs(dif.y) <= grid_radius.y &&
                            std::abs(dif.z) <= grid_radius.z) || get_nearest) {
                            int index = pos.x + _vol_metadata->Resolution[1] *
                                                 (pos.y + _vol_metadata->Resolution[2] * pos.z);
                            auto current_data =
                                _volume_data[index];
                            value += current_data;
                            min_data = std::min(min_data, current_data);
                            max_data = std::max(max_data, current_data);

                            num_samples++;
                        }
                    }
                }
            }
            if (value != 0) value /= num_samples;
            if (this->_weighting.Param<megamol::core::param::EnumParam>()->Value() == 0) {
                samples->samples[j] = value;
            } else {
                samples->samples[j] = max_data;
            }
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            avg_value += value;
        }
        if (avg_value != 0) avg_value /= samples_per_probe;
        if (!std::isfinite(avg_value)) {
            core::utility::log::Log::DefaultLog.WriteError(
                "[SampleAlongProbes] Non-finite value in sampled.");
        }
        if (this->_weighting.Param<megamol::core::param::EnumParam>()->Value() == 0) {
            samples->average_value = avg_value;
            samples->max_value = max_value;
            samples->min_value = min_value;
        } else {
            samples->average_value = max_data;
            samples->max_value = max_data;
            samples->min_value = max_data;
        }
        global_min = std::min(global_min, samples->min_value);
        global_max = std::max(global_max, samples->max_value);
    } // end for probes
    _probes->setGlobalMinMax(global_min, global_max);
}

bool SampleAlongPobes::getData(core::Call& call) {

    bool something_has_changed = false;
    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr) return false;

    // query adios data
    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto cv = this->_volume_rhs_slot.CallAs<core::misc::VolumetricDataCall>();
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
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 4) {
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
        if (!(*ct)(0)) return false;


        tree_meta_data = ct->getMetaData();

        something_has_changed = something_has_changed || (cd->getDataHash() != _old_datahash) || ct->hasUpdate();
    } else if (cv != nullptr) {

        // get volume data
        if (!(*cv)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

        meta_data.m_frame_cnt = cv->FrameCount();
        _vol_metadata = cv->GetMetadata();

        if (_vol_metadata->Components > 1) {
            core::utility::log::Log::DefaultLog.WriteError(
                "[SampleAlongProbes] Volume data has more than one component. Not supported.");
            return false;
        }

        _volume_data = static_cast<float*>(cv->GetData());

        something_has_changed = something_has_changed || (cv->DataHash() != _old_volume_datahash);
    } else {
        return false;
    }

    // query probe data
    if (cprobes == nullptr) return false;
    if (!(*cprobes)(0)) return false;

    something_has_changed = something_has_changed || cprobes->hasUpdate() || _trigger_recalc;

    probes_meta_data = cprobes->getMetaData();
    _probes = cprobes->getData();

    if (something_has_changed) {
        ++_version;

        if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 3 ||
            _sampling_mode.Param<core::param::EnumParam>()->Value() == 4) {
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
                } else {
                    doNearestNeighborSampling(tree, data);
                }

            } else if (cd->getData(var_str)->getType() == "float") {
                std::vector<float> data = cd->getData(var_str)->GetAsFloat();
                if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 0) {
                    doScalarSampling(tree, data);
                } else if (_sampling_mode.Param<core::param::EnumParam>()->Value() == 3) {
                    doTetrahedralSampling(tree, data);
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
        } else {
            if (cv == nullptr) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[SampleAlongProbes] Volume mode selected but no volume data connected.");
                return false;
            }

            doVolumeSampling();
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
    if (cp == nullptr) return false;

    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto cv = this->_volume_rhs_slot.CallAs<core::misc::VolumetricDataCall>();
    auto ct = this->_full_tree_rhs_slot.CallAs<CallKDTree>();
    auto cprobes = this->_probe_rhs_slot.CallAs<CallProbes>();
    if (cprobes == nullptr) return false;

    auto meta_data = cp->getMetaData();
    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_trigger_recalc)
    //    return true;

    if (cd != nullptr && ct != nullptr) {
        cd->setFrameIDtoLoad(meta_data.m_frame_ID);
        if (!(*cd)(1)) return false;
        if (!(*ct)(1)) return false;
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
        if (!(*cv)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
        if (!(*cv)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;
        meta_data.m_frame_cnt = cv->FrameCount();
    } else {
        return false;
    }

    auto probes_meta_data = cprobes->getMetaData();
    probes_meta_data.m_frame_ID = meta_data.m_frame_ID;
    cprobes->setMetaData(probes_meta_data);
    if (!(*cprobes)(1)) return false;

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
