/*
 * TableToProbes.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "TableToProbes.h"
#include "mmcore/param/BoolParam.h"
#include "probe/MeshUtilities.h"
#include "probe/ProbeCalls.h"


namespace megamol {
namespace probe {

megamol::probe::TableToProbes::TableToProbes()
        : Module()
        , _version(0)
        , _table_slot("getTable", "")
        , _probe_slot("deployProbes", "")
        , _accumulate_clustered_slot("accumulate_clustered", "") {

    this->_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &TableToProbes::getData);
    this->_probe_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(1), &TableToProbes::getMetaData);
    this->MakeSlotAvailable(&this->_probe_slot);


    this->_table_slot.SetCompatibleCall<datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->_table_slot);


    this->_accumulate_clustered_slot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->_accumulate_clustered_slot);

    /* Feasibility test */
    _probes = std::make_shared<ProbeCol>();
    _probes->addProbe(FloatProbe());

    auto retrieved_probe = _probes->getProbe<FloatProbe>(0);

    float data;
    retrieved_probe.probe(&data);

    auto result = retrieved_probe.getSamplingResult();
}

megamol::probe::TableToProbes::~TableToProbes() {
    this->Release();
}

bool megamol::probe::TableToProbes::create() {
    return true;
}

void megamol::probe::TableToProbes::release() {}

bool megamol::probe::TableToProbes::getData(core::Call& call) {

    auto* pc = dynamic_cast<CallProbes*>(&call);
    datatools::table::TableDataCall* ct = this->_table_slot.CallAs<datatools::table::TableDataCall>();


    if (ct == nullptr)
        return false;
    if (!(*ct)(0))
        return false;


    bool something_changed = ct->DataHash() != _table_data_hash;

    auto probe_meta_data = pc->getMetaData();

    // here something really happens
    if (something_changed) {
        ++_version;

        _table = ct->GetData();
        _col_info = ct->GetColumnsInfos();
        _num_cols = ct->GetColumnsCount();
        _num_rows = ct->GetRowsCount();

        this->generateProbes();
    }

    // set bbox
    probe_meta_data.m_bboxs = _bbox;

    pc->setData(this->_probes, _version);

    pc->setMetaData(probe_meta_data);
    _table_data_hash = ct->DataHash();
    return true;
}

bool megamol::probe::TableToProbes::getMetaData(core::Call& call) {

    auto* pc = dynamic_cast<CallProbes*>(&call);
    datatools::table::TableDataCall* ct = this->_table_slot.CallAs<datatools::table::TableDataCall>();

    if (ct == nullptr)
        return false;

    // set frame id before callback
    auto probe_meta_data = pc->getMetaData();

    ct->SetFrameID(probe_meta_data.m_frame_ID);

    if (!(*ct)(1))
        return false;


    probe_meta_data.m_frame_cnt = ct->GetFrameCount();
    //probe_meta_data.m_bboxs // normally not available here

    pc->setMetaData(probe_meta_data);

    return true;
}

bool megamol::probe::TableToProbes::generateProbes() {

    _probes = std::make_shared<ProbeCol>();

    assert(_table != nullptr);

    // check for probe type
    bool distrib_probe = false;
    for (uint32_t i = 0; i < _num_cols; i++) {
        if (_col_info[i].Name().find("sample_value_lower") != std::string::npos) {
            distrib_probe = true;
        }
    }

    std::string check_num_str;
    if (distrib_probe) {
        check_num_str = "sample_value_lower";
    } else {
        check_num_str = "sample_value";
    }

    uint32_t samples_per_probe = 0;
    std::map<std::string, uint32_t> col_to_id_map;
    for (uint32_t i = 0; i < _num_cols; i++) {
        col_to_id_map[_col_info[i].Name()] = i;
        if (_col_info[i].Name().find(check_num_str) != std::string::npos) {
            samples_per_probe += 1;
        }
    }

    assert(col_to_id_map.find("cluster_id") != col_to_id_map.end());
    std::vector<int> cluster_ids(_num_rows);
    std::map<int, uint32_t> cluster_id_count;

    for (uint32_t i = 0; i < _num_rows; ++i) {
        cluster_ids[i] = static_cast<uint32_t>(_table[_num_cols * i + col_to_id_map["cluster_id"]]);
        auto it = cluster_id_count.find(cluster_ids[i]);
        if (it == cluster_id_count.end()) {
            cluster_id_count[cluster_ids[i]] = 1;
        } else {
            cluster_id_count[cluster_ids[i]] += 1;
        }
    }

    float min_x = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();

    float global_min_value = std::numeric_limits<float>::max();
    float global_max_value = -std::numeric_limits<float>::max();

    if (this->_accumulate_clustered_slot.Param<core::param::BoolParam>()->Value()) {

        _accum_probes.resize(cluster_id_count.size());
        std::vector<std::array<float, 3>> accum_pos(cluster_id_count.size(), {0, 0, 0});
        std::vector<std::array<float, 3>> accum_dir(cluster_id_count.size(), {0, 0, 0});
        std::vector<float> accum_begin(cluster_id_count.size());
        std::vector<float> accum_end(cluster_id_count.size());
        float sample_radius = 0;
        float time_stamp = 0;

        for (uint32_t i = 0; i < _num_rows; ++i) {
            if (_accum_probes[cluster_ids[i]].empty()) {
                _accum_probes[cluster_ids[i]].resize(samples_per_probe, 0);
            }

            // calc bbox
            min_x = std::min(min_x, _table[_num_cols * i + col_to_id_map["position_x"]]);
            max_x = std::max(max_x, _table[_num_cols * i + col_to_id_map["position_x"]]);
            min_y = std::min(min_y, _table[_num_cols * i + col_to_id_map["position_y"]]);
            max_y = std::max(max_y, _table[_num_cols * i + col_to_id_map["position_y"]]);
            min_z = std::min(min_z, _table[_num_cols * i + col_to_id_map["position_z"]]);
            max_z = std::max(max_z, _table[_num_cols * i + col_to_id_map["position_z"]]);

            accum_pos[cluster_ids[i]][0] += _table[_num_cols * i + col_to_id_map["position_x"]];
            accum_pos[cluster_ids[i]][1] += _table[_num_cols * i + col_to_id_map["position_y"]];
            accum_pos[cluster_ids[i]][2] += _table[_num_cols * i + col_to_id_map["position_z"]];

            accum_dir[cluster_ids[i]][0] += _table[_num_cols * i + col_to_id_map["direction_x"]];
            accum_dir[cluster_ids[i]][1] += _table[_num_cols * i + col_to_id_map["direction_y"]];
            accum_dir[cluster_ids[i]][2] += _table[_num_cols * i + col_to_id_map["direction_z"]];

            accum_begin[cluster_ids[i]] += _table[_num_cols * i + col_to_id_map["begin"]];

            accum_end[cluster_ids[i]] += _table[_num_cols * i + col_to_id_map["end"]];

            sample_radius = _table[_num_cols * i + col_to_id_map["sample_radius"]];

            time_stamp = _table[_num_cols * i + col_to_id_map["timestamp"]];

            for (int j = 0; j < samples_per_probe; j++) {
                std::string sv = "sample_value_" + std::to_string(j);
                auto value = _table[_num_cols * i + col_to_id_map[sv]];
                _accum_probes[cluster_ids[i]][j] += value;
            }
        }

        for (uint32_t i = 0; i < _accum_probes.size(); i++) {

            FloatProbe probe;
            probe.m_position = {accum_pos[i][0] / cluster_id_count[i], accum_pos[i][1] / cluster_id_count[i],
                accum_pos[i][2] / cluster_id_count[i]};
            probe.m_direction = {accum_dir[i][0] / cluster_id_count[i], accum_dir[i][1] / cluster_id_count[i],
                accum_dir[i][2] / cluster_id_count[i]};
            probe.m_begin = accum_begin[i] / cluster_id_count[i];
            probe.m_end = accum_end[i] / cluster_id_count[i];
            probe.m_sample_radius = sample_radius;
            probe.m_timestamp = time_stamp;
            probe.m_cluster_id = cluster_ids[i];

            std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();
            samples->samples.resize(samples_per_probe);
            float min_value = std::numeric_limits<float>::max();
            float max_value = -std::numeric_limits<float>::min();
            float avg_value = 0.0f;
            for (int j = 0; j < samples_per_probe; j++) {
                auto value = _accum_probes[i][j];
                samples->samples[j] = value;
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
                avg_value += value;
            }
            avg_value /= samples_per_probe;

            samples->average_value = avg_value;
            samples->max_value = max_value;
            samples->min_value = min_value;
            global_max_value = std::max(samples->max_value, global_max_value);
            global_min_value = std::min(samples->min_value, global_min_value);

            this->_probes->addProbe(std::move(probe));
        }

    } else {

        for (uint32_t i = 0; i < _num_rows; ++i) {

            // calc bbox
            min_x = std::min(min_x, _table[_num_cols * i + col_to_id_map["position_x"]]);
            max_x = std::max(max_x, _table[_num_cols * i + col_to_id_map["position_x"]]);
            min_y = std::min(min_y, _table[_num_cols * i + col_to_id_map["position_y"]]);
            max_y = std::max(max_y, _table[_num_cols * i + col_to_id_map["position_y"]]);
            min_z = std::min(min_z, _table[_num_cols * i + col_to_id_map["position_z"]]);
            max_z = std::max(max_z, _table[_num_cols * i + col_to_id_map["position_z"]]);

            if (distrib_probe) {
                FloatDistributionProbe probe;
                probe.m_position = {_table[_num_cols * i + col_to_id_map["position_x"]],
                    _table[_num_cols * i + col_to_id_map["position_y"]],
                    _table[_num_cols * i + col_to_id_map["position_z"]]};
                probe.m_direction = {_table[_num_cols * i + col_to_id_map["direction_x"]],
                    _table[_num_cols * i + col_to_id_map["direction_y"]],
                    _table[_num_cols * i + col_to_id_map["direction_z"]]};
                probe.m_begin = _table[_num_cols * i + col_to_id_map["begin"]];
                probe.m_end = _table[_num_cols * i + col_to_id_map["end"]];
                probe.m_sample_radius = _table[_num_cols * i + col_to_id_map["sample_radius"]];
                probe.m_timestamp = _table[_num_cols * i + col_to_id_map["timestamp"]];
                probe.m_cluster_id = _table[_num_cols * i + col_to_id_map["cluster_id"]];

                std::shared_ptr<FloatDistributionProbe::SamplingResult> samples = probe.getSamplingResult();

                samples->samples.resize(samples_per_probe);
                float min_value = std::numeric_limits<float>::max();
                float max_value = -std::numeric_limits<float>::min();
                float avg_value = 0.0f;
                for (int j = 0; j < samples_per_probe; j++) {
                    std::string sv = "sample_value_" + std::to_string(j);
                    std::string svl = "sample_value_lower_" + std::to_string(j);
                    std::string svu = "sample_value_upper_" + std::to_string(j);

                    auto mean = _table[_num_cols * i + col_to_id_map[sv]];
                    auto lower = _table[_num_cols * i + col_to_id_map[svl]];
                    auto upper = _table[_num_cols * i + col_to_id_map[svu]];
                    samples->samples[j].mean = mean;
                    samples->samples[j].lower_bound = lower;
                    samples->samples[j].upper_bound = upper;
                    min_value = std::min(min_value, mean);
                    max_value = std::max(max_value, mean);
                    avg_value += mean;
                }

                avg_value /= samples_per_probe;

                samples->average_value = avg_value;
                samples->max_value = max_value;
                samples->min_value = min_value;

                global_max_value = std::max(samples->max_value, global_max_value);
                global_min_value = std::min(samples->min_value, global_min_value);


                this->_probes->addProbe(std::move(probe));
            } else {
                FloatProbe probe;
                probe.m_position = {_table[_num_cols * i + col_to_id_map["position_x"]],
                    _table[_num_cols * i + col_to_id_map["position_y"]],
                    _table[_num_cols * i + col_to_id_map["position_z"]]};
                probe.m_direction = {_table[_num_cols * i + col_to_id_map["direction_x"]],
                    _table[_num_cols * i + col_to_id_map["direction_y"]],
                    _table[_num_cols * i + col_to_id_map["direction_z"]]};
                probe.m_begin = _table[_num_cols * i + col_to_id_map["begin"]];
                probe.m_end = _table[_num_cols * i + col_to_id_map["end"]];
                probe.m_sample_radius = _table[_num_cols * i + col_to_id_map["sample_radius"]];
                probe.m_timestamp = _table[_num_cols * i + col_to_id_map["timestamp"]];
                probe.m_cluster_id = _table[_num_cols * i + col_to_id_map["cluster_id"]];

                std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();

                samples->samples.resize(samples_per_probe);
                float min_value = std::numeric_limits<float>::max();
                float max_value = -std::numeric_limits<float>::min();
                float avg_value = 0.0f;
                for (int j = 0; j < samples_per_probe; j++) {
                    std::string sv = "sample_value_" + std::to_string(j);
                    auto value = _table[_num_cols * i + col_to_id_map[sv]];
                    samples->samples[j] = value;
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                    avg_value += value;
                }
                avg_value /= samples_per_probe;

                samples->average_value = avg_value;
                samples->max_value = max_value;
                samples->min_value = min_value;

                global_max_value = std::max(samples->max_value, global_max_value);
                global_min_value = std::min(samples->min_value, global_min_value);


                this->_probes->addProbe(std::move(probe));
            }
        }
    }
    _probes->setGlobalMinMax(global_min_value, global_max_value);
    _bbox.SetBoundingBox({min_x, min_y, max_z, max_x, max_y, min_z});

    return true;
}


} // namespace probe
} // namespace megamol
