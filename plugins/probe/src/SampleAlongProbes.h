/*
 * SampleAlongProbes.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "geometry_calls/VolumetricDataCall.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/ProbeCollection.h"
#include "probe/CallKDTree.h"

#include "CGAL/Delaunay_triangulation_3.h"
#include "CGAL/Delaunay_triangulation_cell_base_3.h"
#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
#include "CGAL/Triangulation_vertex_base_3.h"
#include "CGAL/Triangulation_vertex_base_with_info_3.h"

#include <glm/glm.hpp>

namespace megamol {
namespace probe {

class SampleAlongPobes : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SampleAlongProbes";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    SampleAlongPobes();
    virtual ~SampleAlongPobes();

protected:
    virtual bool create();
    virtual void release();

    uint32_t _version;

    core::CalleeSlot _probe_lhs_slot;

    core::CallerSlot _probe_rhs_slot;
    size_t _probe_cached_hash;

    core::CallerSlot _adios_rhs_slot;
    size_t _adios_cached_hash;

    core::CallerSlot _full_tree_rhs_slot;
    size_t _full_tree_cached_hash;

    core::CallerSlot _volume_rhs_slot;
    size_t _volume_cached_hash;

    core::param::ParamSlot _parameter_to_sample_slot;
    core::param::ParamSlot _num_samples_per_probe_slot;
    core::param::ParamSlot _sample_radius_factor_slot;

    core::param::ParamSlot _sampling_mode;
    core::param::ParamSlot _weighting;
    core::param::ParamSlot _vec_param_to_samplex_x;
    core::param::ParamSlot _vec_param_to_samplex_y;
    core::param::ParamSlot _vec_param_to_samplex_z;
    core::param::ParamSlot _vec_param_to_samplex_w;

private:
    template<typename T>
    void doScalarSampling(const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data);

    template<typename T>
    void doScalarDistributionSampling(
        const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data);

    template<typename T>
    void doVolumeRadiusSampling(T* data);

    template<typename T>
    void doVolumeTrilinSampling(T* data);

    template<typename T>
    void doVectorSamling(const std::shared_ptr<my_kd_tree_t>& tree, const std::vector<T>& data_x,
        const std::vector<T>& data_y, const std::vector<T>& data_z, const std::vector<T>& data_w);

    template<typename T>
    void doTetrahedralSampling(const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data);

    template<typename T>
    void doTetrahedralVectorSamling(const std::shared_ptr<my_kd_tree_t>& tree,
        const std::vector<T>& data_x, const std::vector<T>& data_y, const std::vector<T>& data_z,
        const std::vector<T>& data_w);

    template<typename T>
    void doNearestNeighborSampling(const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data);

    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);

    std::shared_ptr<ProbeCollection> _probes;

    const geocalls::VolumetricDataCall::Metadata* _vol_metadata;

    size_t _old_datahash;
    size_t _old_volume_datahash;
    bool _trigger_recalc;
    bool paramChanged(core::param::ParamSlot& p);
};


template<typename T>
void SampleAlongPobes::doScalarSampling(const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data) {

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();
    const float sample_radius_factor = this->_sample_radius_factor_slot.Param<core::param::FloatParam>()->Value();

    float global_min = std::numeric_limits<float>::max();
    float global_max = -std::numeric_limits<float>::max();
    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); i++) {

        FloatProbe probe;

        auto visitor = [&probe, i, samples_per_probe, sample_radius_factor, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::Vec4Probe> ||
                          std::is_same_v<T, probe::FloatDistributionProbe>) {

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

        std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();

        float min_value = std::numeric_limits<float>::max();
        float max_value = -std::numeric_limits<float>::max();
        float min_data = std::numeric_limits<float>::max();
        float max_data = -std::numeric_limits<float>::max();
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);

        for (int j = 0; j < samples_per_probe; j++) {

            std::array<float,3> sample_point;
            sample_point[0] = probe.m_position[0] + j * sample_step * probe.m_direction[0];
            sample_point[1] = probe.m_position[1] + j * sample_step * probe.m_direction[1];
            sample_point[2] = probe.m_position[2] + j * sample_step * probe.m_direction[2];

            std::vector<std::pair<size_t, float>> res;

            auto num_neighbors =
                tree->radiusSearch(&sample_point[0], radius, res, nanoflann::SearchParams(10, 0.01f, true));
            if (num_neighbors == 0) {
                std::vector<size_t> ret_index(1);
                std::vector<float> out_dist_sqr(1);
                nanoflann::KNNResultSet<float> resultSet(1);
                resultSet.init(ret_index.data(), out_dist_sqr.data());
                num_neighbors = tree->findNeighbors(resultSet ,&sample_point[0], nanoflann::SearchParams(10));

                res.resize(1);
                res[0].first = ret_index[0];
                res[0].second = out_dist_sqr[0];
            }

            // accumulate values
            float value = 0;
            for (int n = 0; n < num_neighbors; n++) {
                auto distance_weight = res[n].second / radius;
                value += data[res[n].first] * distance_weight;
                min_data = std::min(min_data, static_cast<float>(data[res[n].first]));
                max_data = std::max(max_data, static_cast<float>(data[res[n].first]));
            } // end num_neighbors
            value /= num_neighbors;
            if (this->_weighting.Param<megamol::core::param::EnumParam>()->Value() == 0) {
                samples->samples[j] = value;
            } else {
                samples->samples[j] = max_data;
            }
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            avg_value += value;
        } // end num samples per probe
        avg_value /= samples_per_probe;
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

template<typename T>
inline void SampleAlongPobes::doScalarDistributionSampling(
    const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data) {

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();
    const float sample_radius_factor = this->_sample_radius_factor_slot.Param<core::param::FloatParam>()->Value();

    float global_min = std::numeric_limits<float>::max();
    float global_max = -std::numeric_limits<float>::max();
    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); i++) {

        FloatDistributionProbe probe;

        auto visitor = [&probe, i, samples_per_probe, sample_radius_factor, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::FloatProbe> ||
                          std::is_same_v<T, probe::Vec4Probe>) {

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

            } else if constexpr (std::is_same_v<T, probe::FloatDistributionProbe>) {
                probe = arg;

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(i);
        std::visit(visitor, generic_probe);

        auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
        auto radius = 0.5 * sample_step * sample_radius_factor;

        std::shared_ptr<FloatDistributionProbe::SamplingResult> samples = probe.getSamplingResult();

        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::min();
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);

        for (int j = 0; j < samples_per_probe; j++) {

            std::array<float,3> sample_point;
            sample_point[0] = probe.m_position[0] + j * sample_step * probe.m_direction[0];
            sample_point[1] = probe.m_position[1] + j * sample_step * probe.m_direction[1];
            sample_point[2] = probe.m_position[2] + j * sample_step * probe.m_direction[2];

            std::vector<std::pair<size_t, float>> res;

            auto num_neighbors =
                tree->radiusSearch(&sample_point[0], radius, res, nanoflann::SearchParams(10, 0.01f, true));
            if (num_neighbors == 0) {
                std::vector<size_t> ret_index(1);
                std::vector<float> out_dist_sqr(1);
                nanoflann::KNNResultSet<float> resultSet(1);
                resultSet.init(ret_index.data(), out_dist_sqr.data());
                num_neighbors = tree->findNeighbors(resultSet, &sample_point[0], nanoflann::SearchParams(10));

                res.resize(1);
                res[0].first = ret_index[0];
                res[0].second = out_dist_sqr[0];
            }

            // accumulate values
            float value = 0.0f;
            float min_data = std::numeric_limits<float>::max();
            float max_data = std::numeric_limits<float>::min();
            for (int n = 0; n < num_neighbors; n++) {
                value += data[res[n].first];
                min_data = std::min(min_data, static_cast<float>(data[res[n].first]));
                max_data = std::max(max_data, static_cast<float>(data[res[n].first]));
            } // end num_neighbors
            value /= num_neighbors;

            samples->samples[j].mean = value;
            samples->samples[j].lower_bound = min_data;
            samples->samples[j].upper_bound = max_data;

            min_value = std::min(min_value, min_data);
            max_value = std::max(max_value, max_data);
            avg_value += value;
        } // end num samples per probe

        global_min = std::min(global_min, min_value);
        global_max = std::max(global_max, max_value);
    } // end for probes
    _probes->setGlobalMinMax(global_min, global_max);
}

template<typename T>
inline void SampleAlongPobes::doVectorSamling(const std::shared_ptr<my_kd_tree_t>& tree,
    const std::vector<T>& data_x, const std::vector<T>& data_y, const std::vector<T>& data_z,
    const std::vector<T>& data_w) {

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();
    const float sample_radius_factor = this->_sample_radius_factor_slot.Param<core::param::FloatParam>()->Value();

    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); i++) {

        Vec4Probe probe;

        auto visitor = [&probe, i, samples_per_probe, sample_radius_factor, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::FloatProbe> ||
                          std::is_same_v<T, probe::FloatDistributionProbe>) {

                probe.m_timestamp = arg.m_timestamp;
                probe.m_value_name = arg.m_value_name;
                probe.m_position = arg.m_position;
                probe.m_direction = arg.m_direction;
                probe.m_begin = arg.m_begin;
                probe.m_end = arg.m_end;
                probe.m_cluster_id = arg.m_cluster_id;

                auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
                auto radius = sample_step * sample_radius_factor;
                probe.m_sample_radius = radius;

                _probes->setProbe(i, probe);

            } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {
                probe = arg;

                auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
                auto radius = sample_step * sample_radius_factor;
                probe.m_sample_radius = radius;

                _probes->setProbe(i, probe);

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(i);
        std::visit(visitor, generic_probe);

        auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
        auto radius = sample_step * sample_radius_factor;

        std::shared_ptr<Vec4Probe::SamplingResult> samples = probe.getSamplingResult();

        float min_value = std::numeric_limits<float>::max();
        float max_value = -std::numeric_limits<float>::max();
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);

        for (int j = 0; j < samples_per_probe; j++) {

            std::array<float,3> sample_point;
            sample_point[0] = probe.m_position[0] + j * sample_step * probe.m_direction[0];
            sample_point[1] = probe.m_position[1] + j * sample_step * probe.m_direction[1];
            sample_point[2] = probe.m_position[2] + j * sample_step * probe.m_direction[2];

            std::vector<std::pair<size_t, float>> res;

            auto num_neighbors =
                tree->radiusSearch(&sample_point[0], radius, res, nanoflann::SearchParams(10, 0.01f, true));
            if (num_neighbors == 0) {
                std::vector<size_t> ret_index(1);
                std::vector<float> out_dist_sqr(1);
                nanoflann::KNNResultSet<float> resultSet(1);
                resultSet.init(ret_index.data(), out_dist_sqr.data());
                num_neighbors = tree->findNeighbors(resultSet, &sample_point[0], nanoflann::SearchParams(10));

                res.resize(1);
                res[0].first = ret_index[0];
                res[0].second = out_dist_sqr[0];
            }


            // accumulate values
            float value_x = 0, value_y = 0, value_z = 0, value_w = 0;
            for (int n = 0; n < num_neighbors; n++) {
                value_x += data_x[res[n].first];
                value_y += data_y[res[n].first];
                value_z += data_z[res[n].first];
                value_w += data_w[res[n].first];
            } // end num_neighbors
            samples->samples[j][0] = value_x / num_neighbors;
            ;
            samples->samples[j][1] = value_y / num_neighbors;
            ;
            samples->samples[j][2] = value_z / num_neighbors;
            ;
            samples->samples[j][3] = value_w / num_neighbors;
            ;
            //min_value = std::min(min_value, value);
            //max_value = std::max(max_value, value);
            //avg_value += value;
        } // end num samples per probe
        //avg_value /= samples_per_probe;
        //samples->average_value = avg_value;
        //samples->max_value = max_value;
        //samples->min_value = min_value;
    } // end for probes
}


template<typename T>
void SampleAlongPobes::doTetrahedralSampling(
    const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data) {

    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb = CGAL::Triangulation_vertex_base_with_info_3<T, K>;
    using Cb = CGAL::Delaunay_triangulation_cell_base_3<K>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
    using Triangulation = CGAL::Delaunay_triangulation_3<K, Tds>;
    using Point = Triangulation::Point;
    using Segment = Triangulation::Segment;
    using Tetrahedron = Triangulation::Tetrahedron;

    Triangulation tri;

    {
        
        auto const num_points = tree->dataset.derived().size();
        auto const& cloud = tree->dataset.derived();
        std::vector<std::pair<Point, T>> points(num_points);
        std::transform(cloud.cbegin(), cloud.cend(), data.cbegin(), points.begin(),
            [](std::array<float, 3> const& p, T const& val) { return std::make_pair(Point(p[0], p[1], p[2]), val); });
        tri = Triangulation(points.cbegin(), points.cend());
    }

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();
    const float sample_radius_factor = this->_sample_radius_factor_slot.Param<core::param::FloatParam>()->Value();

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();
    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); ++i) {

        FloatProbe probe;

        auto visitor = [&probe, i, samples_per_probe, sample_radius_factor, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::Vec4Probe> ||
                          std::is_same_v<T, probe::FloatDistributionProbe>) {

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

        std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();

        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::lowest();
        /*float min_data = std::numeric_limits<float>::max();
        float max_data = -std::numeric_limits<float>::max();*/
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);

        for (int j = 0; j < samples_per_probe; ++j) {

            Point sample_point(probe.m_position[0] + static_cast<float>(j) * sample_step * probe.m_direction[0],
                probe.m_position[1] + static_cast<float>(j) * sample_step * probe.m_direction[1],
                probe.m_position[2] + static_cast<float>(j) * sample_step * probe.m_direction[2]);

            T val = std::numeric_limits<T>::signaling_NaN();

            auto cell = tri.locate(sample_point);
            if (!tri.is_infinite(cell)) {
                Tetrahedron tet_c = Tetrahedron(cell->vertex(0)->point(), cell->vertex(1)->point(),
                    cell->vertex(2)->point(), cell->vertex(3)->point());
                Tetrahedron tet_0 = Tetrahedron(
                    sample_point, cell->vertex(1)->point(), cell->vertex(2)->point(), cell->vertex(3)->point());
                Tetrahedron tet_1 = Tetrahedron(
                    cell->vertex(0)->point(), sample_point, cell->vertex(2)->point(), cell->vertex(3)->point());
                Tetrahedron tet_2 = Tetrahedron(
                    cell->vertex(0)->point(), cell->vertex(1)->point(), sample_point, cell->vertex(3)->point());
                Tetrahedron tet_3 = Tetrahedron(
                    cell->vertex(0)->point(), cell->vertex(1)->point(), cell->vertex(2)->point(), sample_point);

                auto const V_c = tet_c.volume();

                auto const V_0 = tet_0.volume();
                auto const V_1 = tet_1.volume();
                auto const V_2 = tet_2.volume();
                auto const V_3 = tet_3.volume();

                auto const a_0 = V_0 / V_c;
                auto const a_1 = V_1 / V_c;
                auto const a_2 = V_2 / V_c;
                auto const a_3 = V_3 / V_c;

                auto const val_0 = cell->vertex(0)->info();
                auto const val_1 = cell->vertex(1)->info();
                auto const val_2 = cell->vertex(2)->info();
                auto const val_3 = cell->vertex(3)->info();

                val = a_0 * val_0 + a_1 * val_1 + a_2 * val_2 + a_3 * val_3;
            }
            samples->samples[j] = val;

            min_value = std::min<decltype(min_value)>(min_value, val);
            max_value = std::max<decltype(max_value)>(max_value, val);
            avg_value += val;
        } // end num samples per probe

        avg_value /= samples_per_probe;
        samples->average_value = avg_value;
        samples->max_value = max_value;
        samples->min_value = min_value;
        global_min = std::min(global_min, samples->min_value);
        global_max = std::max(global_max, samples->max_value);
    } // end for probes
    _probes->setGlobalMinMax(global_min, global_max);
    _probes->shuffle_probes();
}

template<typename T>
inline void SampleAlongPobes::doTetrahedralVectorSamling(const std::shared_ptr<my_kd_tree_t>& tree,
    const std::vector<T>& data_x, const std::vector<T>& data_y, const std::vector<T>& data_z,
    const std::vector<T>& data_w) {

    using InfoType = std::array<T, 4>;

    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb = CGAL::Triangulation_vertex_base_with_info_3<InfoType, K>;
    using Cb = CGAL::Delaunay_triangulation_cell_base_3<K>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
    using Triangulation = CGAL::Delaunay_triangulation_3<K, Tds>;
    using Point = Triangulation::Point;
    using Segment = Triangulation::Segment;
    using Tetrahedron = Triangulation::Tetrahedron;

    Triangulation tri;

    {
        auto const num_points = tree->dataset.derived().size();
        auto const& cloud = tree->dataset.derived();
        std::vector<std::pair<Point, InfoType>> points(num_points);
        for (int i = 0; i < num_points; ++i) {

            auto const& p = cloud[i];
            T const& val_x = data_x[i];
            T const& val_y = data_y[i];
            T const& val_z = data_z[i];
            T const& val_w = data_w[i];

            points[i] = std::make_pair(Point(p[0], p[1], p[2]), std::array<T, 4>({val_x, val_y, val_z, val_w}));
        }
        tri = Triangulation(points.cbegin(), points.cend());
    }

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();
    const float sample_radius_factor = this->_sample_radius_factor_slot.Param<core::param::FloatParam>()->Value();

    std::vector<char> invalid_probes(_probes->getProbeCount(), 1);

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();
    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); ++i) {

        Vec4Probe probe;

        auto visitor = [&probe, i, samples_per_probe, sample_radius_factor, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::FloatProbe> ||
                          std::is_same_v<T, probe::FloatDistributionProbe>) {

                probe.m_timestamp = arg.m_timestamp;
                probe.m_value_name = arg.m_value_name;
                probe.m_position = arg.m_position;
                probe.m_direction = arg.m_direction;
                probe.m_begin = arg.m_begin;
                probe.m_end = arg.m_end;
                probe.m_cluster_id = arg.m_cluster_id;

                auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
                auto radius = sample_step * sample_radius_factor;
                probe.m_sample_radius = radius;

                _probes->setProbe(i, probe);

            } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {
                probe = arg;

                auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
                auto radius = sample_step * sample_radius_factor;
                probe.m_sample_radius = radius;

                _probes->setProbe(i, probe);

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(i);
        std::visit(visitor, generic_probe);

        auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
        auto radius = 0.5 * sample_step * sample_radius_factor;

        std::shared_ptr<Vec4Probe::SamplingResult> samples = probe.getSamplingResult();

        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::lowest();
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);

        for (int j = 0; j < samples_per_probe; ++j) {

            Point sample_point(probe.m_position[0] + static_cast<float>(j) * sample_step * probe.m_direction[0],
                probe.m_position[1] + static_cast<float>(j) * sample_step * probe.m_direction[1],
                probe.m_position[2] + static_cast<float>(j) * sample_step * probe.m_direction[2]);

            InfoType val = {std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN(),
                std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN()};

            auto cell = tri.locate(sample_point);
            if (!tri.is_infinite(cell)) {
                invalid_probes[i] = 0;

                Tetrahedron tet_c = Tetrahedron(cell->vertex(0)->point(), cell->vertex(1)->point(),
                    cell->vertex(2)->point(), cell->vertex(3)->point());
                Tetrahedron tet_0 = Tetrahedron(
                    sample_point, cell->vertex(1)->point(), cell->vertex(2)->point(), cell->vertex(3)->point());
                Tetrahedron tet_1 = Tetrahedron(
                    cell->vertex(0)->point(), sample_point, cell->vertex(2)->point(), cell->vertex(3)->point());
                Tetrahedron tet_2 = Tetrahedron(
                    cell->vertex(0)->point(), cell->vertex(1)->point(), sample_point, cell->vertex(3)->point());
                Tetrahedron tet_3 = Tetrahedron(
                    cell->vertex(0)->point(), cell->vertex(1)->point(), cell->vertex(2)->point(), sample_point);

                auto const V_c = tet_c.volume();

                auto const V_0 = tet_0.volume();
                auto const V_1 = tet_1.volume();
                auto const V_2 = tet_2.volume();
                auto const V_3 = tet_3.volume();

                auto const a_0 = V_0 / V_c;
                auto const a_1 = V_1 / V_c;
                auto const a_2 = V_2 / V_c;
                auto const a_3 = V_3 / V_c;

                auto const val_0 = cell->vertex(0)->info();
                auto const val_1 = cell->vertex(1)->info();
                auto const val_2 = cell->vertex(2)->info();
                auto const val_3 = cell->vertex(3)->info();

                std::get<0>(val) = a_0 * std::get<0>(val_0) + a_1 * std::get<0>(val_1) + a_2 * std::get<0>(val_2) +
                                   a_3 * std::get<0>(val_3);
                std::get<1>(val) = a_0 * std::get<1>(val_0) + a_1 * std::get<1>(val_1) + a_2 * std::get<1>(val_2) +
                                   a_3 * std::get<1>(val_3);
                std::get<2>(val) = a_0 * std::get<2>(val_0) + a_1 * std::get<2>(val_1) + a_2 * std::get<2>(val_2) +
                                   a_3 * std::get<2>(val_3);
                std::get<3>(val) = a_0 * std::get<3>(val_0) + a_1 * std::get<3>(val_1) + a_2 * std::get<3>(val_2) +
                                   a_3 * std::get<3>(val_3);
            }
            std::array<float, 4> sample = {std::get<0>(val), std::get<1>(val), std::get<2>(val), std::get<3>(val)};
            samples->samples[j] = sample;

            min_value = std::min(min_value, std::get<3>(sample));
            max_value = std::max(max_value, std::get<3>(sample));
            avg_value += std::get<3>(sample);
        } // end num samples per probe

        avg_value /= samples_per_probe;

        global_min = std::min(global_min, min_value);
        global_max = std::max(global_max, max_value);
    } // end for probes
    _probes->setGlobalMinMax(global_min, global_max);
    _probes->erase_probes(invalid_probes);
    _probes->shuffle_probes();
}


template<typename T>
void SampleAlongPobes::doNearestNeighborSampling(
    const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data) {

    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb = CGAL::Triangulation_vertex_base_with_info_3<T, K>;
    using Cb = CGAL::Delaunay_triangulation_cell_base_3<K>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
    using Triangulation = CGAL::Delaunay_triangulation_3<K, Tds>;
    using Point = Triangulation::Point;
    using Segment = Triangulation::Segment;
    using Tetrahedron = Triangulation::Tetrahedron;

    Triangulation tri;

    {
        auto const num_points = tree->dataset.derived().size();
        auto const& cloud = tree->dataset.derived();
        std::vector<std::pair<Point, T>> points(num_points);
        std::transform(cloud.cbegin(), cloud.cend(), data.cbegin(), points.begin(),
            [](std::array<float, 3> const& p, T const& val) { return std::make_pair(Point(p[0], p[1], p[2]), val); });
        tri = Triangulation(points.begin(), points.end());
    }

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();
    const float sample_radius_factor = this->_sample_radius_factor_slot.Param<core::param::FloatParam>()->Value();

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();
    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); ++i) {

        FloatProbe probe;

        auto visitor = [&probe, i, samples_per_probe, sample_radius_factor, this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, probe::BaseProbe> || std::is_same_v<T, probe::Vec4Probe> ||
                          std::is_same_v<T, probe::FloatDistributionProbe>) {

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

        std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();

        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::lowest();
        /*float min_data = std::numeric_limits<float>::max();
        float max_data = -std::numeric_limits<float>::max();*/
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);

        for (int j = 0; j < samples_per_probe; ++j) {

            Point sample_point(probe.m_position[0] + static_cast<float>(j) * sample_step * probe.m_direction[0],
                probe.m_position[1] + static_cast<float>(j) * sample_step * probe.m_direction[1],
                probe.m_position[2] + static_cast<float>(j) * sample_step * probe.m_direction[2]);

            T val = std::numeric_limits<T>::signaling_NaN();

            auto cell = tri.locate(sample_point);
            if (!tri.is_infinite(cell)) {
                auto vertex = tri.nearest_vertex_in_cell(sample_point, cell);

                val = vertex->info();
            }

            samples->samples[j] = val;

            min_value = std::min<decltype(min_value)>(min_value, val);
            max_value = std::max<decltype(max_value)>(max_value, val);
            avg_value += val;
        } // end num samples per probe

        avg_value /= samples_per_probe;
        /*if (this->_weighting.Param<megamol::core::param::EnumParam>()->Value() == 0) {
            samples->average_value = avg_value;
            samples->max_value = max_value;
            samples->min_value = min_value;
        } else {
            samples->average_value = max_data;
            samples->max_value = max_data;
            samples->min_value = max_data;
        }*/
        global_min = std::min(global_min, samples->min_value);
        global_max = std::max(global_max, samples->max_value);
    } // end for probes
    _probes->setGlobalMinMax(global_min, global_max);
}

template<typename T>
void SampleAlongPobes::SampleAlongPobes::doVolumeRadiusSampling(T* data) {
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

            glm::vec3 start = {std::roundf(grid_point.x - grid_radius.x), std::roundf(grid_point.y - grid_radius.y),
                std::roundf(grid_point.z - grid_radius.z)};
            auto end = grid_point + grid_radius;

            float value = 0;
            int num_samples = 0;
            for (int k = 0; k < num_grid_points_per_dim[0]; ++k) {
                for (int l = 0; l < num_grid_points_per_dim[1]; ++l) {
                    for (int m = 0; m < num_grid_points_per_dim[2]; ++m) {
                        auto pos = start + glm::vec3(k, l, m);
                        auto dif = pos - grid_point;
                        if ((std::abs(dif.x) <= grid_radius.x && std::abs(dif.y) <= grid_radius.y &&
                                std::abs(dif.z) <= grid_radius.z) ||
                            get_nearest) {
                            int index =
                                pos.z + _vol_metadata->Resolution[1] * (pos.y + _vol_metadata->Resolution[2] * pos.x);
                            assert(index < _vol_metadata->Resolution[0] * _vol_metadata->Resolution[1] *
                                               _vol_metadata->Resolution[2]);
                            float current_data = data[index];
                            value += current_data;
                            min_data = std::min(min_data, current_data);
                            max_data = std::max(max_data, current_data);

                            num_samples++;
                        }
                    }
                }
            }
            if (value != 0)
                value /= num_samples;
            if (this->_weighting.Param<megamol::core::param::EnumParam>()->Value() == 0) {
                samples->samples[j] = value;
            } else {
                samples->samples[j] = max_data;
            }
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            avg_value += value;
        }
        if (avg_value != 0)
            avg_value /= samples_per_probe;
        if (!std::isfinite(avg_value)) {
            core::utility::log::Log::DefaultLog.WriteError("[SampleAlongProbes] Non-finite value in sampled.");
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

template<typename T>
void SampleAlongPobes::SampleAlongPobes::doVolumeTrilinSampling(T* data) {
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

        std::shared_ptr<FloatProbe::SamplingResult> samples = probe.getSamplingResult();
        float min_value = std::numeric_limits<float>::max();
        float max_value = -std::numeric_limits<float>::max();
        float avg_value = 0.0f;
        samples->samples.resize(samples_per_probe);


        for (int j = 0; j < samples_per_probe; j++) {

            glm::vec3 sample_point;
            sample_point.x = probe.m_position[0] + j * sample_step * probe.m_direction[0];
            sample_point.y = probe.m_position[1] + j * sample_step * probe.m_direction[1];
            sample_point.z = probe.m_position[2] + j * sample_step * probe.m_direction[2];

            auto xd = sample_point.x -
                      std::floorf(sample_point.x) / (std::ceilf(sample_point.x) - std::floorf(sample_point.x));
            auto yd = sample_point.y -
                      std::floorf(sample_point.y) / (std::ceilf(sample_point.y) - std::floorf(sample_point.y));
            auto zd = sample_point.z -
                      std::floorf(sample_point.z) / (std::ceilf(sample_point.z) - std::floorf(sample_point.z));

            auto c000 = data[static_cast<size_t>(std::floor(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::floor(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::floor(sample_point.x)))];
            auto c001 = data[static_cast<size_t>(std::ceil(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::floor(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::floor(sample_point.x)))];
            auto c010 = data[static_cast<size_t>(std::floor(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::ceil(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::floor(sample_point.x)))];
            auto c011 = data[static_cast<size_t>(std::ceil(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::ceil(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::floor(sample_point.x)))];
            auto c100 = data[static_cast<size_t>(std::floor(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::floor(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::ceil(sample_point.x)))];
            auto c101 = data[static_cast<size_t>(std::ceil(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::floor(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::ceil(sample_point.x)))];
            auto c110 = data[static_cast<size_t>(std::floor(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::ceil(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::ceil(sample_point.x)))];
            auto c111 = data[static_cast<size_t>(std::ceil(sample_point.z)) +
                             _vol_metadata->Resolution[1] *
                                 (static_cast<size_t>(std::ceil(sample_point.y)) +
                                     _vol_metadata->Resolution[2] * static_cast<size_t>(std::ceil(sample_point.x)))];

            auto c00 = c000 * (1 - xd) + c100 * xd;
            auto c01 = c001 * (1 - xd) + c101 * xd;
            auto c10 = c010 * (1 - xd) + c110 * xd;
            auto c11 = c011 * (1 - xd) + c111 * xd;

            auto c0 = c00 * (1 - yd) + c10 * yd;
            auto c1 = c01 * (1 - yd) + c11 * yd;

            auto value = c0 * (1 - zd) + c1 * zd;
            samples->samples[j] = value;

            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            avg_value += value;
        }
        if (avg_value != 0)
            avg_value /= samples_per_probe;
        if (!std::isfinite(avg_value)) {
            core::utility::log::Log::DefaultLog.WriteError("[SampleAlongProbes] Non-finite value in sampled.");
        }

        samples->average_value = avg_value;
        samples->max_value = max_value;
        samples->min_value = min_value;

        global_min = std::min(global_min, samples->min_value);
        global_max = std::max(global_max, samples->max_value);
    } // end for probes
    _probes->setGlobalMinMax(global_min, global_max);
}


} // namespace probe
} // namespace megamol
