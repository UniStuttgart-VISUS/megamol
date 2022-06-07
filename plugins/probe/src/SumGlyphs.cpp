/*
 * SumGlyphs.cpp
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "SumGlyphs.h"
#include "glm/glm.hpp"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCalls.h"


namespace megamol {
namespace probe {

SumGlyphs::SumGlyphs()
        : Module()
        , _version(0)
        , _probe_rhs_slot("getProbes", "")
        , _probe_lhs_slot("deployMesh", "") {

    this->_probe_rhs_slot.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->_probe_rhs_slot);


    this->_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(0), &SumGlyphs::getData);
    this->_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &SumGlyphs::getMetaData);
    this->MakeSlotAvailable(&this->_probe_lhs_slot);
    this->_probe_lhs_slot.SetNecessity(core::AbstractCallSlotPresentation::SLOT_REQUIRED);
}

SumGlyphs::~SumGlyphs() {
    this->Release();
}

bool SumGlyphs::create() {
    return true;
}

void SumGlyphs::release() {}

bool SumGlyphs::getData(core::Call& call) {

    CallProbes* lhsProbesCall = dynamic_cast<CallProbes*>(&call);
    if (lhsProbesCall == nullptr)
        return false;

    if (!(*lhsProbesCall)(0))
        return false;
    const bool lhs_dirty = lhsProbesCall->hasUpdate();

    auto rhsProbesCall = this->_probe_rhs_slot.CallAs<probe::CallProbes>();
    if (rhsProbesCall != nullptr) {
        auto meta_data = rhsProbesCall->getMetaData();
        if (!(*rhsProbesCall)(0))
            return false;
        const bool rhs_dirty = rhsProbesCall->hasUpdate();
        if (rhs_dirty || lhs_dirty) {
            ++_version;
            _sum_probe_collection = std::make_shared<ProbeCollection>();

            auto const probe_count = rhsProbesCall->getData()->getProbeCount();
            auto const probes = rhsProbesCall->getData();
            std::vector<std::array<float, 3>> probe_positions(probe_count);
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
                probes_with_this_clusterid[cluster_id].emplace_back(i);
            }
            //using current_probe_type =  std::decay_t<decltype(probes->getGenericProbe(0))>;



            // calculate middle probe distance
            auto avg_dist = calculateAverageDistance(probe_positions, 2);

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

                    while (!cluster_positions.empty()) {
                        region.emplace_back(std::map<int, std::array<float, 3>>());
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


                } else {
                    _sum_probe_collection->addProbe(probes->getProbe<FloatProbe>(cluster.second[0]));
                }
            }



        }
    }

    lhsProbesCall->setData(_sum_probe_collection, _version);

    return true;
}

bool SumGlyphs::getMetaData(core::Call& call) {

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

bool SumGlyphs::parameterChanged(core::param::ParamSlot& p) {
    _recalc = true;
    return true;
}

float SumGlyphs::calculateAverageDistance(std::vector<std::array<float, 3>> const& input_data, int const num_neighbors) {
    auto kd_tree = my_kd_tree_t(3 /*dim*/, input_data, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    kd_tree.buildIndex();
    int const num_results = num_neighbors + 1;
    float distance = 0.0f;
    for (auto& pos : input_data) {
        std::vector<size_t> ret_index(num_results);
        std::vector<float> sqr_dist(num_results);
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(ret_index.data(), sqr_dist.data());


        kd_tree.findNeighbors(resultSet, &pos[0], nanoflann::SearchParams(10));
        for (int i = 1; i < num_results; ++i) {
            distance += std::sqrtf(sqr_dist[i]);
        }
    }

    return distance / static_cast<float>(input_data.size()*num_neighbors);
}

float SumGlyphs::getDistance(std::array<float, 3>const& point1, std::array<float, 3> const&  point2) {
    return std::sqrtf(
        std::powf(point2[0] - point1[0], 2) + std::powf(point2[0] - point1[0], 2) + std::powf(point2[0] - point1[0], 2));
}

} // namespace probe
} // namespace megamol
