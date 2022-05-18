/*
 * ProbesToTable.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ProbesToTable.h"
#include "probe/ProbeCalls.h"

namespace megamol {
namespace probe {

ProbeToTable::ProbeToTable() : Module(), _getDataSlot("getData", ""), _deployTableSlot("deployTable", "") {

    this->_deployTableSlot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(0), &ProbeToTable::getData);
    this->_deployTableSlot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(1), &ProbeToTable::getMetaData);
    this->MakeSlotAvailable(&this->_deployTableSlot);

    this->_getDataSlot.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->_getDataSlot);
}

ProbeToTable::~ProbeToTable() {
    this->Release();
}

bool ProbeToTable::create() {
    return true;
}

void ProbeToTable::release() {}

bool ProbeToTable::InterfaceIsDirty() {
    return false;
}


bool ProbeToTable::getData(core::Call& call) {

    datatools::table::TableDataCall* ctd = dynamic_cast<datatools::table::TableDataCall*>(&call);
    if (ctd == nullptr)
        return false;

    CallProbes* cpd = this->_getDataSlot.CallAs<CallProbes>();
    if (cpd == nullptr)
        return false;

    auto meta_data = cpd->getMetaData();
    // maybe get meta data was not called jet
    if (meta_data.m_frame_cnt == 0) {
        this->getMetaData(call);
        meta_data = cpd->getMetaData();
    }

    if (!(*cpd)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ProbeToTable] Error during GetData");
        return false;
    }

    auto probe_data = cpd->getData();

    if (cpd->hasUpdate() || (meta_data.m_frame_ID != _currentFrame)) {

        auto num_probes = probe_data->getProbeCount();
        bool distrib_probe = false;
        {
            auto const test_probe = probe_data->getGenericProbe(0);
            distrib_probe = std::holds_alternative<FloatDistributionProbe>(test_probe);
        }

        std::vector<std::vector<float>> raw_data;
        std::vector<float> mins;
        std::vector<float> maxes;
        std::vector<std::string> var_names;
        if (distrib_probe) {
            auto probe_one = probe_data->getProbe<FloatDistributionProbe>(0);
            auto num_samples = probe_one.getSamplingResult()->samples.size();
            _rows = num_probes;

            var_names = {"id", "position_x", "position_y", "position_z", "direction_x", "direction_y", "direction_z",
                "begin", "end", "timestamp", "sample_radius", "cluster_id"};
            auto fixed_var_names_index = var_names.size();
            _fixed_cols = var_names.size();

            for (int i = 0; i < num_samples; ++i) {
                std::string var_mean = "sample_value_" + std::to_string(i);
                var_names.emplace_back(var_mean);
                std::string var_lower = "sample_value_lower_" + std::to_string(i);
                var_names.emplace_back(var_lower);
                std::string var_upper = "sample_value_upper_" + std::to_string(i);
                var_names.emplace_back(var_upper);
            }

            _total_cols = var_names.size();
            _colinfo.resize(_total_cols);
            raw_data.resize(num_probes);
            mins.resize(var_names.size(), std::numeric_limits<float>::max());
            maxes.resize(var_names.size(), std::numeric_limits<float>::min());
            for (int i = 0; i < num_probes; ++i) {
                raw_data[i].resize(_total_cols);

                int current_col = 0;
                auto probe = probe_data->getProbe<FloatDistributionProbe>(i);
                raw_data[i][current_col] = i;
                mins[current_col] = std::min(mins[current_col], static_cast<float>(i));
                maxes[current_col] = std::max(maxes[current_col], static_cast<float>(i));
                current_col += 1;

                for (int n = 0; n < probe.m_position.size(); ++n) {
                    raw_data[i][current_col] = probe.m_position[n];
                    mins[current_col] = std::min(mins[current_col], probe.m_position[n]);
                    maxes[current_col] = std::max(maxes[current_col], probe.m_position[n]);
                    current_col += 1;
                }

                for (int n = 0; n < probe.m_position.size(); ++n) {
                    raw_data[i][current_col] = probe.m_direction[n];
                    mins[current_col] = std::min(mins[current_col], probe.m_direction[n]);
                    maxes[current_col] = std::max(maxes[current_col], probe.m_direction[n]);
                    current_col += 1;
                }

                raw_data[i][current_col] = probe.m_begin;
                mins[current_col] = std::min(mins[current_col], probe.m_begin);
                maxes[current_col] = std::max(maxes[current_col], probe.m_begin);
                current_col += 1;

                raw_data[i][current_col] = probe.m_end;
                mins[current_col] = std::min(mins[current_col], probe.m_end);
                maxes[current_col] = std::max(maxes[current_col], probe.m_end);
                current_col += 1;

                raw_data[i][current_col] = probe.m_timestamp;
                mins[current_col] = std::min(mins[current_col], static_cast<float>(probe.m_timestamp));
                maxes[current_col] = std::max(maxes[current_col], static_cast<float>(probe.m_timestamp));
                current_col += 1;

                raw_data[i][current_col] = probe.m_sample_radius;
                mins[current_col] = std::min(mins[current_col], probe.m_sample_radius);
                maxes[current_col] = std::max(maxes[current_col], probe.m_sample_radius);
                current_col += 1;

                raw_data[i][current_col] = probe.m_cluster_id;
                mins[current_col] = std::min(mins[current_col], static_cast<float>(probe.m_cluster_id));
                maxes[current_col] = std::max(maxes[current_col], static_cast<float>(probe.m_cluster_id));
                current_col += 1;

                auto result = probe.getSamplingResult()->samples;
                for (int k = 0; k < num_samples; ++k) {
                    raw_data[i][fixed_var_names_index + 3 * k + 0] = result[k].mean;
                    raw_data[i][fixed_var_names_index + 3 * k + 1] = result[k].lower_bound;
                    raw_data[i][fixed_var_names_index + 3 * k + 2] = result[k].upper_bound;

                    mins[fixed_var_names_index + k] = std::min(mins[fixed_var_names_index + k], result[k].lower_bound);
                    maxes[fixed_var_names_index + k] =
                        std::max(maxes[fixed_var_names_index + k], result[k].upper_bound);
                }
            }
        } else {
            auto probe_one = probe_data->getProbe<FloatProbe>(0);
            auto num_samples = probe_one.getSamplingResult()->samples.size();
            _rows = num_probes;

            var_names = {"id", "position_x", "position_y", "position_z", "direction_x", "direction_y", "direction_z",
                "begin", "end", "timestamp", "sample_radius", "cluster_id"};
            auto fixed_var_names_index = var_names.size();
            _fixed_cols = var_names.size();

            for (int i = 0; i < num_samples; ++i) {
                std::string var = "sample_value_" + std::to_string(i);
                var_names.emplace_back(var);
            }

            _total_cols = var_names.size();
            _colinfo.resize(_total_cols);
            raw_data.resize(num_probes);
            mins.resize(var_names.size(), std::numeric_limits<float>::max());
            maxes.resize(var_names.size(), std::numeric_limits<float>::min());
            for (int i = 0; i < num_probes; ++i) {
                raw_data[i].resize(_total_cols);

                int current_col = 0;
                auto probe = probe_data->getProbe<FloatProbe>(i);
                raw_data[i][current_col] = i;
                mins[current_col] = std::min(mins[current_col], static_cast<float>(i));
                maxes[current_col] = std::max(maxes[current_col], static_cast<float>(i));
                current_col += 1;

                for (int n = 0; n < probe.m_position.size(); ++n) {
                    raw_data[i][current_col] = probe.m_position[n];
                    mins[current_col] = std::min(mins[current_col], probe.m_position[n]);
                    maxes[current_col] = std::max(maxes[current_col], probe.m_position[n]);
                    current_col += 1;
                }

                for (int n = 0; n < probe.m_position.size(); ++n) {
                    raw_data[i][current_col] = probe.m_direction[n];
                    mins[current_col] = std::min(mins[current_col], probe.m_direction[n]);
                    maxes[current_col] = std::max(maxes[current_col], probe.m_direction[n]);
                    current_col += 1;
                }

                raw_data[i][current_col] = probe.m_begin;
                mins[current_col] = std::min(mins[current_col], probe.m_begin);
                maxes[current_col] = std::max(maxes[current_col], probe.m_begin);
                current_col += 1;

                raw_data[i][current_col] = probe.m_end;
                mins[current_col] = std::min(mins[current_col], probe.m_end);
                maxes[current_col] = std::max(maxes[current_col], probe.m_end);
                current_col += 1;

                raw_data[i][current_col] = probe.m_timestamp;
                mins[current_col] = std::min(mins[current_col], static_cast<float>(probe.m_timestamp));
                maxes[current_col] = std::max(maxes[current_col], static_cast<float>(probe.m_timestamp));
                current_col += 1;

                raw_data[i][current_col] = probe.m_sample_radius;
                mins[current_col] = std::min(mins[current_col], probe.m_sample_radius);
                maxes[current_col] = std::max(maxes[current_col], probe.m_sample_radius);
                current_col += 1;

                raw_data[i][current_col] = probe.m_cluster_id;
                mins[current_col] = std::min(mins[current_col], static_cast<float>(probe.m_cluster_id));
                maxes[current_col] = std::max(maxes[current_col], static_cast<float>(probe.m_cluster_id));
                current_col += 1;

                auto result = probe.getSamplingResult()->samples;
                for (int k = 0; k < num_samples; ++k) {
                    raw_data[i][fixed_var_names_index + k] = result[k];
                    mins[fixed_var_names_index + k] = std::min(mins[fixed_var_names_index + k], result[k]);
                    maxes[fixed_var_names_index + k] = std::max(maxes[fixed_var_names_index + k], result[k]);
                }
            }
        }
        for (int i = 0; i < _total_cols; i++) {
            _colinfo[i].SetName(var_names[i]);
            _colinfo[i].SetMaximumValue(maxes[i]);
            _colinfo[i].SetMinimumValue(mins[i]);
            _colinfo[i].SetType(datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
        }

        _floatBlob.resize(_rows * _total_cols);
#pragma omp parallel for
        for (int i = 0; i < _rows; ++i) {
            for (int j = 0; j < _total_cols; ++j) {
                // if (j >= raw_data[i].size()) {
                //    _floatBlob[_total_cols * i + j] = 0.0f;
                //} else {
                _floatBlob[_total_cols * i + j] = raw_data[i][j];
            }
        }
    }

    if (_floatBlob.empty())
        return false;

    _currentFrame = ctd->GetFrameID();
    ctd->Set(_total_cols, _rows, _colinfo.data(), _floatBlob.data());
    ctd->SetDataHash(_datahash++);

    return true;
}

bool ProbeToTable::getMetaData(core::Call& call) {

    datatools::table::TableDataCall* ctd = dynamic_cast<datatools::table::TableDataCall*>(&call);
    if (ctd == nullptr)
        return false;

    CallProbes* cpd = this->_getDataSlot.CallAs<CallProbes>();
    if (cpd == nullptr)
        return false;

    // get metadata from probes
    auto meta_data = cpd->getMetaData();
    meta_data.m_frame_ID = ctd->GetFrameID();
    cpd->setMetaData(meta_data);

    if (!(*cpd)(1))
        return false;

    // put metadata in table call
    meta_data = cpd->getMetaData();
    ctd->SetFrameCount(meta_data.m_frame_cnt);

    return true;
}

} // namespace probe
} // namespace megamol
