/*
 * FilterProbes.cpp
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "FilterProbes.h"
#include "glm/glm.hpp"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/BoolParam.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCalls.h"


namespace megamol {
namespace probe {

FilterProbes::FilterProbes()
        : Module()
        , _version(0)
        , _probe_rhs_slot("getProbes", "")
        , _probe_lhs_slot("deployMesh", "")
        , _center_param("centerProbes", "") {

    this->_probe_rhs_slot.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->_probe_rhs_slot);


    this->_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(0), &FilterProbes::getData);
    this->_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &FilterProbes::getMetaData);
    this->MakeSlotAvailable(&this->_probe_lhs_slot);
    this->_probe_lhs_slot.SetNecessity(core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    this->_center_param << new core::param::BoolParam(true);
    this->_center_param.SetUpdateCallback(&FilterProbes::parameterChanged);
    this->MakeSlotAvailable(&this->_center_param);
}

FilterProbes::~FilterProbes() {
    this->Release();
}

bool FilterProbes::create() {
    return true;
}

void FilterProbes::release() {}

bool FilterProbes::getData(core::Call& call) {

    CallProbes* lhsProbesCall = dynamic_cast<CallProbes*>(&call);
    if (lhsProbesCall == nullptr)
        return false;

    auto rhsProbesCall = this->_probe_rhs_slot.CallAs<probe::CallProbes>();
    if (rhsProbesCall != nullptr) {
        auto meta_data = rhsProbesCall->getMetaData();
        if (!(*rhsProbesCall)(0))
            return false;
        const bool rhs_dirty = rhsProbesCall->hasUpdate();
        if (rhs_dirty || _recalc) {
            ++_version;
            _recalc = false;
            _filtered_probe_collection = std::make_shared<ProbeCol>();

            auto const probe_count = rhsProbesCall->getData()->getProbeCount();
            auto const probes = rhsProbesCall->getData();

            auto minmax = probes->getGlobalMinMax<float>();
            _filtered_probe_collection->setGlobalMinMax(minmax[0], minmax[1]);

            std::map < int, std::vector<int>> representant_of_cluster;

            for (auto i = 0; i < probe_count; i++) {
                auto generic_probe = probes->getGenericProbe(i);

                int cluster_id;
                bool representant = false;

                auto visitor = [&cluster_id, &representant](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                        cluster_id = arg.m_cluster_id;
                        representant = arg.m_representant;
                    } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                        cluster_id = arg.m_cluster_id;
                        representant = arg.m_representant;
                    } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {
                        cluster_id = arg.m_cluster_id;
                        representant = arg.m_representant;
                    } else if constexpr (std::is_same_v<T, probe::FloatDistributionProbe>) {
                        cluster_id = arg.m_cluster_id;
                        representant = arg.m_representant;
                    } else if constexpr (std::is_same_v<T, probe::BaseProbe>) {
                        cluster_id = arg.m_cluster_id;
                        representant = arg.m_representant;
                    } else {
                        // unknown probe type, throw error? do nothing?
                    }
                };

                std::visit(visitor, generic_probe);
                if (representant) {
                    representant_of_cluster[cluster_id].emplace_back(i);
                    _filtered_probe_collection->addProbe(generic_probe);
                }
            }

            auto center_probes = _center_param.Param<core::param::BoolParam>()->Value();
            if (center_probes) {

                // custom algorithm to find regions and determine mean probe for region
                std::vector<std::array<float, 3>> probe_positions(probe_count);
                std::map<int,std::pair<std::vector<int>, std::vector<std::array<float, 3>>>> cluster_positions;
                std::map<int, std::vector<int>> probes_with_this_clusterid;
                for (auto i = 0; i < probe_count; i++) {
                    auto generic_probe = rhsProbesCall->getData()->getGenericProbe(i);

                    int cluster_id;
                    std::array<float, 3> pos;

                    auto visitor = [&cluster_id, &pos](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, probe::FloatProbe>) {
                            cluster_id = arg.m_cluster_id;
                            pos = arg.m_position;
                        } else if constexpr (std::is_same_v<T, probe::IntProbe>) {
                            cluster_id = arg.m_cluster_id;
                            pos = arg.m_position;
                        } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {
                            cluster_id = arg.m_cluster_id;
                            pos = arg.m_position;
                        } else if constexpr (std::is_same_v<T, probe::FloatDistributionProbe>) {
                            cluster_id = arg.m_cluster_id;
                            pos = arg.m_position;
                        } else if constexpr (std::is_same_v<T, probe::BaseProbe>) {
                            cluster_id = arg.m_cluster_id;
                            pos = arg.m_position;
                        } else {
                            // unknown probe type, throw error? do nothing?
                        }
                    };

                    std::visit(visitor, generic_probe);

                    probe_positions[i] = pos;
                    cluster_positions[cluster_id].first.emplace_back(i);
                    cluster_positions[cluster_id].second.emplace_back(pos);
                    probes_with_this_clusterid[cluster_id].emplace_back(i);
                }

                int i = 0;
                for (auto& cluster : cluster_positions) {

                    auto const num_probes_in_this_cluster = cluster.second.first.size();
                    std::array<float,3> center_of_mass = {0,0,0};

                    for (auto& pos : cluster.second.second) {
                        center_of_mass[0] += pos[0];
                        center_of_mass[1] += pos[1];
                        center_of_mass[2] += pos[2];
                    }
                    center_of_mass[0] /= static_cast<float>(num_probes_in_this_cluster);
                    center_of_mass[1] /= static_cast<float>(num_probes_in_this_cluster);
                    center_of_mass[2] /= static_cast<float>(num_probes_in_this_cluster);

                    auto kd_tree = my_kd_tree_t(
                        3 /*dim*/, cluster.second.second, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
                    kd_tree.buildIndex();

                    std::vector<size_t> ret_index(1);
                    std::vector<float> sqr_dist(1);
                    nanoflann::KNNResultSet<float> resultSet(1);
                    resultSet.init(ret_index.data(), sqr_dist.data());


                    kd_tree.findNeighbors(resultSet, &center_of_mass[0], nanoflann::SearchParams());

                    auto new_probe_id = cluster.second.first[ret_index[0]];

                    newPosDir(i, new_probe_id);

                    ++i;
                }

#if 0
                
                // calculate middle probe distance
                auto avg_dist = calculateAverageDistance(probe_positions, 10);

                // check if cluster is connected region
                // generate one summed glyph for each region
                for (auto& cluster : probes_with_this_clusterid) {
                    if (cluster.second.size() > 1) {
                        std::vector<std::array<float, 3>> cluster_positions;
                        cluster_positions.reserve(cluster.second.size());
                        std::vector<int> id_translator;
                        id_translator.reserve(cluster.second.size());
                        for (auto probe_id : cluster.second) {
                            cluster_positions.emplace_back(probe_positions[probe_id]);
                            id_translator.emplace_back(probe_id);
                        }

                        std::mt19937 rnd;
                        rnd.seed(std::random_device()());
                        std::uniform_int_distribution<int> dist(0, cluster.second.size());
                        auto start_index = dist(rnd);


                        std::map<int, std::array<float, 3>> to_check_map;
                        to_check_map[start_index] = cluster_positions[start_index];

                        // store region
                        std::vector<std::map<int, std::array<float, 3>>> region;

                        // if there is more than one re
                        while (!cluster_positions.empty()) {
                            region.emplace_back(std::map<int, std::array<float, 3>>());
                            //if (!cluster_positions.empty() && to_check_map.empty()) {
                            //    start_index = dist(rnd);
                            //    to_check_map[start_index] = cluster_positions[start_index];
                            //}

                            // check connection in region
                            while (!to_check_map.empty()) {

                                auto old_to_check_map = to_check_map;

                                auto kd_tree = my_kd_tree_t(
                                    3 /*dim*/, cluster_positions, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
                                kd_tree.buildIndex();

                                
                                for (auto sample_point : to_check_map) {
                                    std::vector<std::pair<size_t, float>> res;
                                    auto num_results = kd_tree.radiusSearch(
                                        &sample_point.second[0], 2 * avg_dist, res, nanoflann::SearchParams(10, 0.01f, true));

                                    for (auto entry: res) {
                                        to_check_map[std::get<0>(entry)] = cluster_positions[std::get<0>(entry)];
                                        region.back()[std::get<0>(entry)] = cluster_positions[std::get<0>(entry)];
                                    }
                                }

                                // erase already checked points
                                for (auto old_id : old_to_check_map) {
                                    to_check_map.erase(old_id.first);
                                    cluster_positions.erase(std::find(cluster_positions.begin(), cluster_positions.end(), old_id.second));
                                }
                            }
                        }


                        //current_probe_type sum_probe;

                    }
                }
#endif
            }
        }
    }

    lhsProbesCall->setData(_filtered_probe_collection, _version);

    return true;
}

bool FilterProbes::getMetaData(core::Call& call) {

    CallProbes* lhsProbesCall = dynamic_cast<CallProbes*>(&call);
    if (lhsProbesCall == nullptr)
        return false;

    auto rhsProbesCall = this->_probe_rhs_slot.CallAs<probe::CallProbes>();
    if (rhsProbesCall != nullptr)
        return false;


    auto rhs_meta_data = rhsProbesCall->getMetaData();
    auto lhs_meta_data = lhsProbesCall->getMetaData();

    rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;

    rhsProbesCall->setMetaData(rhs_meta_data);
    if (!(*rhsProbesCall)(1))
        return false;
    rhs_meta_data = rhsProbesCall->getMetaData();

    lhs_meta_data.m_frame_cnt = rhs_meta_data.m_frame_cnt;
    lhs_meta_data.m_bboxs = rhs_meta_data.m_bboxs;

    // put metadata in mesh call
    lhsProbesCall->setMetaData(lhs_meta_data);

    return true;
}

bool FilterProbes::parameterChanged(core::param::ParamSlot& p) {
    _recalc = true;
    return true;
}

float FilterProbes::calculateAverageDistance(std::vector<std::array<float, 3>> const& input_data, int const num_neighbors) {
    auto kd_tree = my_kd_tree_t(3 /*dim*/, input_data, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    kd_tree.buildIndex();
    int const num_results = num_neighbors + 1;
    float distance = 0.0f;
    int normalizer = 0;
    for (auto& pos : input_data) {
        std::vector<size_t> ret_index(num_results);
        std::vector<float> sqr_dist(num_results);
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(ret_index.data(), sqr_dist.data());


        kd_tree.findNeighbors(resultSet, &pos[0], nanoflann::SearchParams());
        for (int i = 1; i < num_results; ++i) {
            if (sqr_dist[i] > 0.0f) {
                distance += std::sqrtf(sqr_dist[i]);
                ++normalizer;
                break;
            }
        }
    }

    return distance / normalizer;
}

float FilterProbes::getDistance(std::array<float, 3>const& point1, std::array<float, 3> const&  point2) {
    return std::sqrtf(
        std::powf(point2[0] - point1[0], 2) + std::powf(point2[0] - point1[0], 2) + std::powf(point2[0] - point1[0], 2));
}

bool FilterProbes::newPosDir(int const id_filtered, int const id_all) {

    auto all_probes = _probe_rhs_slot.CallAs<CallProbes>()->getData();
    auto const gen_probe = all_probes->getGenericProbe(id_all);

    if (std::holds_alternative<Vec4Probe>(gen_probe)) {
        auto probe = all_probes->getProbe<Vec4Probe>(id_all);
        auto probe_f = _filtered_probe_collection->getProbe<Vec4Probe>(id_filtered);
        probe_f.m_begin = probe.m_begin;
        probe_f.m_direction = probe.m_direction;
        probe_f.m_position = probe.m_position;
        probe_f.m_end = probe.m_end;
        _filtered_probe_collection->setProbe(id_filtered, probe_f);
    } else if (std::holds_alternative<FloatDistributionProbe>(gen_probe)) {
        auto probe = all_probes->getProbe<FloatDistributionProbe>(id_all);
        auto probe_f = _filtered_probe_collection->getProbe<FloatDistributionProbe>(id_filtered);
        probe_f.m_begin = probe.m_begin;
        probe_f.m_direction = probe.m_direction;
        probe_f.m_position = probe.m_position;
        probe_f.m_end = probe.m_end;
        _filtered_probe_collection->setProbe(id_filtered, probe_f);
    } else if (std::holds_alternative<FloatProbe>(gen_probe)) {
        auto probe = all_probes->getProbe<FloatProbe>(id_all);
        auto probe_f = _filtered_probe_collection->getProbe<FloatProbe>(id_filtered);
        probe_f.m_begin = probe.m_begin;
        probe_f.m_direction = probe.m_direction;
        probe_f.m_position = probe.m_position;
        probe_f.m_end = probe.m_end;
        _filtered_probe_collection->setProbe(id_filtered, probe_f);
    } else {
        auto probe = all_probes->getProbe<IntProbe>(id_all);
        auto probe_f = _filtered_probe_collection->getProbe<IntProbe>(id_filtered);
        probe_f.m_begin = probe.m_begin;
        probe_f.m_direction = probe.m_direction;
        probe_f.m_position = probe.m_position;
        probe_f.m_end = probe.m_end;
        _filtered_probe_collection->setProbe(id_filtered, probe_f);
    }

    return true;
}

} // namespace probe
} // namespace megamol
