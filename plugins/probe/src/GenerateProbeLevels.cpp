/*
 * GenerateProbeLevels.cpp
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "GenerateProbeLevels.h"
#include "probe/ProbeCalls.h"


namespace megamol {
namespace probe {

GenerateProbeLevels::GenerateProbeLevels()
        : Module()
        , _version(0)
        , _probe_rhs_slot("getProbes", "")
        , _probe_lhs_slot("deployProbes", "")
{
    _probe_rhs_slot.SetCompatibleCall<probe::CallProbesDescription>();
    MakeSlotAvailable(&_probe_rhs_slot);
    _probe_rhs_slot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    _probe_lhs_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &GenerateProbeLevels::getData);
    _probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &GenerateProbeLevels::getMetaData);
    MakeSlotAvailable(&_probe_lhs_slot);
}

GenerateProbeLevels::~GenerateProbeLevels() {
    this->Release();
}

bool GenerateProbeLevels::create() {
    return true;
}

void GenerateProbeLevels::release() {}

float GenerateProbeLevels::getAvgDist() {
    int const neighbors = 5;
    int const num_samples = 5;
    std::vector<size_t> ret_index(neighbors);
    std::vector<float> out_dist_sqr(neighbors);
    nanoflann::KNNResultSet<float> resultSet(neighbors);

    std::mt19937 rnd;
    rnd.seed(std::random_device()());
    std::uniform_int_distribution<int> dist(0, _probe_positions.size());

    float avg_dist = 0;
    for (int i = 0; i < num_samples; i++) {
        auto const sample_pos = _probe_positions[dist(rnd)];

        _probe_tree->findNeighbors(resultSet, &sample_pos[0], nanoflann::SearchParams());

        float distance = 0;
        for (auto current_dist : out_dist_sqr) {
            distance += current_dist * current_dist;
        }
        distance /= (neighbors -1);
        avg_dist = distance;
    }
    avg_dist /= num_samples;

    return avg_dist;
}

bool GenerateProbeLevels::getData(core::Call& call) {

    auto const lhs_probes = dynamic_cast<CallProbes*>(&call);
    if (lhs_probes == nullptr)
        return false;

    auto const rhs_probes = _probe_rhs_slot.CallAs<probe::CallProbes>();
    if (rhs_probes == nullptr)
        return false;

    auto meta_data = rhs_probes->getMetaData();
    if (!(*rhs_probes)(0))
        return false;

    bool const rhs_dirty = rhs_probes->hasUpdate();
    if (rhs_dirty) {
        ++_version;

        auto tmp_center = meta_data.m_bboxs.BoundingBox().CalcCenter();
        _center = {tmp_center.GetX(), tmp_center.GetY(), tmp_center.GetZ()};
        auto const probe_count = rhs_probes->getData()->getProbeCount();
        _probe_positions.resize(probe_count);
        for (auto i = 0; i < probe_count; i++) {
            auto const base_probe = rhs_probes->getData()->getBaseProbe(i);
            _probe_positions[i] = base_probe.m_position;
        }

        _probe_dataKD = std::make_shared<const data2KD>(_probe_positions);
        _probe_tree = std::make_shared<my_kd_tree_t>(
            3 /*dim*/, *_probe_dataKD, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        _probe_tree->buildIndex();

        //float const avg_dist = getAvgDist();

        //// estimate pack of 4
        //int const neighbors = 10 + 1;
        //std::vector<size_t> ret_index(neighbors);
        //std::vector<float> out_dist_sqr(neighbors);
        //nanoflann::KNNResultSet<float> resultSet(neighbors);

        //std::mt19937 rnd;
        //rnd.seed(std::random_device()());
        //std::uniform_int_distribution<int> dist(0, _probe_positions.size());
        //int rnd_probe_idx = dist(rnd);
        //auto rnd_probe_pos = _probe_positions[rnd_probe_idx];

        //_probe_tree->findNeighbors(resultSet, &rnd_probe_pos[0], nanoflann::SearchParams());


        // map surface to sphere
        // calcMolecularMap()...

        // transform x,y,z -> r,theta,phi
        if (!calcSphericalCoordinates())
            return false;

        // do a mercator projection
        if (!calcMercatorProjection())
            return false;

        // apply force directed distribution
        if (!doRelaxation()) {
            return false;
        }

        // start with grouping from bottom left
        if (!calcLevels(rhs_probes->getData()))
            return false;

    }


    for (int i = 0; i < lhs_probes->getData()->getNumLevels(); i++) {
            lhs_probes->getData()->setLevel(i, _levels[i]);
    }

    return true;
}

bool GenerateProbeLevels::getMetaData(core::Call& call) {

    auto const rhs_probe = this->_probe_rhs_slot.CallAs<probe::CallProbes>();
    auto const lhs_probe = dynamic_cast<CallProbes*>(&call);
    if (lhs_probe == nullptr)
        return false;

    if (rhs_probe == nullptr)
        return false;

    auto lhs_meta_data = lhs_probe->getMetaData();
    auto rhs_meta_data = rhs_probe->getMetaData();
    rhs_meta_data.m_frame_ID = lhs_meta_data.m_frame_ID;
    rhs_probe->setMetaData(rhs_meta_data);

    if (!(*rhs_probe)(1))
        return false;
    rhs_meta_data = rhs_probe->getMetaData();

    lhs_meta_data.m_frame_cnt = rhs_meta_data.m_frame_cnt;
    lhs_meta_data.m_bboxs = rhs_meta_data.m_bboxs;

    // put metadata in mesh call
    lhs_probe->setMetaData(rhs_meta_data);

    return true;
}

bool GenerateProbeLevels::parameterChanged(core::param::ParamSlot& p) {
    _recalc = true;
    return true;
}

bool GenerateProbeLevels::calcSphericalCoordinates() {
    if (_probe_positions.empty()) {
        core::utility::log::Log::DefaultLog.WriteError("[GenerateProbeLevels] Probe positions empty. Abort.");
        return false;
    }

    _probe_positions_spherical_coordinates.clear();
    _probe_positions_spherical_coordinates.resize(_probe_positions.size());
    for (int i = 0; i < _probe_positions.size(); ++i) {
        glm::vec3 const pos = glm::vec3(_probe_positions[i][0], _probe_positions[i][1], _probe_positions[i][2]) -
                              glm::vec3(_center[0], _center[1], _center[2]);

        auto const r = glm::length(pos);
        auto const theta = std::acosf(pos.z/ r);
        auto const phi = std::atan2f(pos.y, pos.x);

        _probe_positions_spherical_coordinates[i][0] = r;
        _probe_positions_spherical_coordinates[i][1] = theta;
        _probe_positions_spherical_coordinates[i][2] = phi;
    }

    return true;
}

bool GenerateProbeLevels::calcMercatorProjection() {

    if (_probe_positions_spherical_coordinates.empty()) {
        core::utility::log::Log::DefaultLog.WriteError("[GenerateProbeLevels] Calculate spherical coordinates first. Abort.");
        return false;
    }
    _probe_positions_mercator.clear();
    _probe_positions_mercator.resize(_probe_positions_spherical_coordinates.size());

    std::array<float, 4> bounds;
    bounds[0] = std::numeric_limits<float>::max();
    bounds[1] = std::numeric_limits<float>::max();
    bounds[2] = std::numeric_limits<float>::lowest();
    bounds[3] = std::numeric_limits<float>::lowest();

    constexpr float pi = 3.14159265359f;

    int inf_count = 0;
    for (int i = 0; i < _probe_positions_spherical_coordinates.size(); i++) {
        // theta -> phi , phi -> lambda
        auto const phi = _probe_positions_spherical_coordinates[i][1];
        auto const lambda = _probe_positions_spherical_coordinates[i][2];

        //_probe_positions_mercator[i][0] = lambda;
        //_probe_positions_mercator[i][1] = 0.5f * std::logf((1.0f + std::sinf(phi)) / (1.0f - std::sinf(phi)));

        _probe_positions_mercator[i][0] = phi;
        _probe_positions_mercator[i][1] = std::sin(lambda);

        if (!std::isfinite(_probe_positions_mercator[i][1])) {
            _probe_positions_mercator[i][1] = std::copysignf(5.0f, _probe_positions_mercator[i][1]);
            inf_count++;
        }

        bounds[0] = std::min(bounds[0], _probe_positions_mercator[i][0]);
        bounds[1] = std::min(bounds[1], _probe_positions_mercator[i][1]);
        bounds[2] = std::max(bounds[2], _probe_positions_mercator[i][0]);
        bounds[3] = std::max(bounds[3], _probe_positions_mercator[i][1]);

    }

    _mercator_bounds = bounds;

    return true;
}

bool GenerateProbeLevels::calcLevels(std::shared_ptr<ProbeCol> inputProbes) {

    auto const rhs_probe = this->_probe_rhs_slot.CallAs<probe::CallProbes>();

    if (_probe_positions_mercator.empty()) {
        return false;
    }

    auto const grid_dim_x = 20;
    auto const grid_dim_y = 20;

    auto const grid_step_x = (_mercator_bounds[2] - _mercator_bounds[0]) / static_cast<float>(grid_dim_x);
    auto const grid_step_y = (_mercator_bounds[3] - _mercator_bounds[1]) / static_cast<float>(grid_dim_y);

    std::vector<std::vector<size_t>> level_by_id;
    level_by_id.resize(grid_dim_x * grid_dim_y);

    // span grid over projection and use one grid cell as first superlevel
    for (int i = 0; i < _probe_positions_mercator.size(); i ++) {

        int id_x = grid_dim_x *
            (_probe_positions_mercator[i][0] - _mercator_bounds[0]) / (_mercator_bounds[2] - _mercator_bounds[0]);
        int id_y = grid_dim_y *
            (_probe_positions_mercator[i][1] - _mercator_bounds[1]) / (_mercator_bounds[3] - _mercator_bounds[1]);

        id_x = (id_x == grid_dim_x) ? id_x - 1 : id_x;
        id_y = (id_y == grid_dim_y) ? id_y - 1 : id_y;

        auto const n = grid_dim_x * id_y + id_x;
        level_by_id[n].emplace_back(i);
    }

    // generate level from level_by_id
    ProbeCol::ProbeLevel level;
    level.probes.reserve(level_by_id.size());


    for (int n = 0; n < level_by_id.size(); n++) {
        if (level_by_id.empty()) {
            core::utility::log::Log::DefaultLog.WriteWarn("[GenerateProbeLevels] Level found to be empty.");
        }
        // since we do a relaxation, the mapping from projection to sphere is not bijective anymore
        /*
        auto const id_x = n % grid_dim_x;
        auto const id_y = n / grid_dim_x;
        std::array<float,2> level_midpoint = {id_x * grid_step_x + 0.5f * grid_step_x, id_y * grid_step_y + 0.5f * grid_step_y};
        auto const radius = _probe_positions_spherical_coordinates[level_by_id[n][0]][0];
        auto const new_probe_pos_spherical = calcInverseMercatorProjection(level_midpoint, radius);
        auto const new_probe_pos = calcInverseSphericalProjection(new_probe_pos_spherical);
        */

        glm::vec3 probe_center_of_mass = {0.0f,0.0f,0.0f};
        for (auto const probe_id : level_by_id[n]) {
            // calc center of mass
            probe_center_of_mass[0] += _probe_positions[probe_id][0];
            probe_center_of_mass[1] += _probe_positions[probe_id][1];
            probe_center_of_mass[2] += _probe_positions[probe_id][2];
        }
        probe_center_of_mass /= static_cast<float>(level_by_id[n].size());

        std::vector<float> dists_to_com;
        dists_to_com.reserve(level_by_id[n].size());
        for (auto const probe_id : level_by_id[n]) {
            // calc dists to center of mass
            dists_to_com.emplace_back(glm::length(probe_center_of_mass - to_vec3(_probe_positions[probe_id])));
        }

        auto min_element = std::min_element(dists_to_com.begin(), dists_to_com.end());
        auto min_element_pos = min_element - dists_to_com.begin();

        auto const min_dist_id = level_by_id[n][min_element_pos];

        if (std::holds_alternative<FloatDistributionProbe>(inputProbes->getGenericProbe(0))) {

            FloatDistributionProbe probe;
            probe = rhs_probe->getData()->getProbe<FloatDistributionProbe>(min_dist_id);
            auto sampling_result = probe.getSamplingResult();
            float new_min = 0.0f;
            float new_max = 0.0f;
            float new_avg = 0.0f;
            for (auto const probe_id : level_by_id[n]) {
                auto const& current_probe = rhs_probe->getData()->getProbe<FloatDistributionProbe>(probe_id);
                auto const& current_sampling_result = current_probe.getSamplingResult();
                new_min += current_sampling_result->min_value;
                new_max += current_sampling_result->max_value;
                new_avg += current_sampling_result->average_value;
            }
            new_min /= static_cast<float>(level_by_id[n].size());
            new_max /= static_cast<float>(level_by_id[n].size());
            new_avg /= static_cast<float>(level_by_id[n].size());

            sampling_result->min_value = new_min;
            sampling_result->max_value = new_max;
            sampling_result->average_value = new_avg;

            level.probes.emplace_back(probe);
            level.level_index = 0;
            level.sub_probes = level_by_id;

        } else {
            // Vector distribution probe
        }
    }

    // generate next level by joining cells
    // starting from left bottom




    return true;
}


std::array<float, 3> GenerateProbeLevels::calcInverseMercatorProjection(std::array<float,2> const& coords, float const& r) {

    std::array<float,3> res;

    res[0] = r;
    res[1] = std::atanf(std::sinhf(coords[1]));
    res[2] = coords[0]; // lambda
    
    return res;
}

std::array<float, 3> GenerateProbeLevels::calcInverseSphericalProjection(std::array<float, 3> const& coords) {

    std::array<float,3> res;

    res[0] = coords[0] * std::sinf(coords[1]) *
             std::cosf(coords[2]);
    res[1] = coords[0] * std::sinf(coords[1]) * std::sinf(coords[2]);
    res[2] = coords[0] * std::cosf(coords[1]);

    return res;
}

bool GenerateProbeLevels::doRelaxation() {

    auto const w = _mercator_bounds[2] - _mercator_bounds[0];
    auto const h = _mercator_bounds[3] - _mercator_bounds[1];

    float delta_t = std::max(w,h) / 1000.0f;

    float const radius = std::min(w,h)/10.0f;

    int const iterations = 150;

    for (int n = 0; n < iterations; n++) {

        auto const old_probe_positions_mercator = _probe_positions_mercator;

        _probe_mercator_dataKD =
            std::make_shared<const PBC_adaptor>(_probe_positions_mercator, _mercator_bounds, true, true);
        _probe_mercator_tree = std::make_shared<my_2d_kd_tree_t>(
            2 /*dim*/, *_probe_mercator_dataKD, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        _probe_mercator_tree->buildIndex();

        for (int i = 0; i < _probe_positions_mercator.size(); i++) {

            auto& current_pos = _probe_positions_mercator[i]; 

            std::vector<std::pair<size_t, float>> res;
            auto num_neighbors = _probe_mercator_tree->radiusSearch(
                _probe_positions_mercator[i].data(), std::powf(radius, 2), res, nanoflann::SearchParams());

            // calc force and dir
            glm::vec2 tot_force(0.0f, 0.0f);
            for (auto const& neighbor : res) {

                auto const dist = std::sqrtf(neighbor.second);
                // filter self interaction
                if (dist < (radius / 1000.0f)) {
                    continue;
                }

                // check if pbc has to be applied into force calculation
                auto const hand_calced_dist =
                    glm::length(to_vec2(current_pos) -
                                to_vec2(_probe_positions_mercator[neighbor.first]));
                auto neighbor_pos = to_vec2(_probe_positions_mercator[neighbor.first]);
                if (hand_calced_dist > radius) {
                    // apply pbc
                    auto const dist_x = current_pos[0] - neighbor_pos[0];
                    auto const dist_y = current_pos[1] - neighbor_pos[1];
                    if ((std::abs(dist_x) > radius && dist_x > 0)) {
                        neighbor_pos[0] += w;
                    } else if ((std::abs(dist_x) > radius && dist_x < 0)) {
                        neighbor_pos[0] -= w;
                    }

                    if ((std::abs(dist_y) > radius && dist_y > 0)) {
                        neighbor_pos[1] += h;
                    } else if ((std::abs(dist_y) > radius && dist_y < 0)) {
                        neighbor_pos[1] -= h;
                    }
                }

                auto const dir = glm::normalize(to_vec2(current_pos) - neighbor_pos);
                auto const current_force = coulomb_force(dist);
                tot_force[0] += dir[0] * current_force;
                tot_force[1] += dir[1] * current_force;

            }
            // move the probe according to verlet
            
            current_pos[0] = 2 * current_pos[0] -
                                               old_probe_positions_mercator[i][0] +
                                               tot_force[0] * std::powf(delta_t, 2.0f);
            current_pos[1] = 2 * current_pos[1] -
                                               old_probe_positions_mercator[i][1] +
                                               tot_force[1] * std::powf(delta_t, 2.0f);



            // check if repositioning would kick probe out of bounds
            // since we use pbc in the neighbor search, the probes should not get kicked out too hard
            if (current_pos[0] < _mercator_bounds[0] ||
                current_pos[1] < _mercator_bounds[1] ||
                current_pos[0] > _mercator_bounds[2] ||
                current_pos[1] > _mercator_bounds[3]) {

                // reset pos
                current_pos = old_probe_positions_mercator[i];

            }
        }
    }

    return true;
}

} // namespace probe
} // namespace megamol
