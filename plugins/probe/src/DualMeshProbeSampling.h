/*
 * DualMeshProbeSampling.h
 * Copyright (C) 2022 by MegaMol Team
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
#include "mesh/MeshCalls.h"

#include "CGAL/Delaunay_triangulation_3.h"
#include "CGAL/Delaunay_triangulation_cell_base_3.h"
#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
#include "CGAL/Triangulation_vertex_base_3.h"
#include "CGAL/Triangulation_vertex_base_with_info_3.h"

#include <glm/glm.hpp>

namespace megamol::probe {

inline glm::vec3 to_vec3(std::array<float,3> input) {
    return glm::vec3(input[0], input[1], input[2]);
}

inline auto log = core::utility::log::Log::DefaultLog;

class DualMeshProbeSampling : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "DualMeshProbeSampling";
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

    DualMeshProbeSampling();
    virtual ~DualMeshProbeSampling();

protected:
    virtual bool create();
    virtual void release();

    uint32_t _version;

    core::CalleeSlot _probe_lhs_slot;

    core::CalleeSlot _debug_lhs_slot;

    core::CallerSlot _probe_rhs_slot;
    size_t _probe_cached_hash;

    core::CallerSlot _adios_rhs_slot;
    size_t _adios_cached_hash;

    core::CallerSlot _full_tree_rhs_slot;
    size_t _full_tree_cached_hash;

    core::CallerSlot _mesh_rhs_slot;

    core::param::ParamSlot _parameter_to_sample_slot;
    core::param::ParamSlot _num_samples_per_probe_slot;

    core::param::ParamSlot _vec_param_to_samplex_x;
    core::param::ParamSlot _vec_param_to_samplex_y;
    core::param::ParamSlot _vec_param_to_samplex_z;
    core::param::ParamSlot _vec_param_to_samplex_w;

private:

    template<typename T>
    void doScalarDistributionSampling(
        const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data);

    template<typename T>
    void doVectorSamling(const std::shared_ptr<my_kd_tree_t>& tree, const std::vector<T>& data_x,
        const std::vector<T>& data_y, const std::vector<T>& data_z, const std::vector<T>& data_w);

    template<typename T>
    bool isInsideSamplingArea(float& min_radius, const std::array<T, 3>& _point, const std::vector<std::array<T, 3>>& _top, const std::vector<std::array<T, 3>>& bottom,
        const std::array<T, 3>& _probe, const std::array<T, 3>& _probe_dir, const bool check_bottom = true,
        const bool includes_probe = false);

    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);
    bool getParticleMetaData(core::Call& call);
    bool getParticleData(core::Call& call);

    std::shared_ptr<ProbeCol> _probes;

    size_t _old_datahash;
    size_t _old_volume_datahash;
    bool _trigger_recalc;
    bool paramChanged(core::param::ParamSlot& p);
    bool createMeshTree(std::shared_ptr<mesh::MeshDataAccessCollection> mesh);
    bool calcDualMesh(std::shared_ptr<mesh::MeshDataAccessCollection> mesh);

    std::shared_ptr<my_kd_tree_t> _mesh_tree;
    std::shared_ptr<const data2KD> _mesh_dataKD;
    std::vector<std::array<float, 3>> _mesh_vertex_data;

    std::vector<std::vector<std::array<float, 3>>> _dual_mesh_vertices;

};


template<typename T>
inline void DualMeshProbeSampling::doScalarDistributionSampling(
    const std::shared_ptr<my_kd_tree_t>& tree, std::vector<T>& data) {

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();

    T global_min = std::numeric_limits<T>::max();
    T global_max = std::numeric_limits<T>::lowest();
    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); i++) {

        FloatDistributionProbe probe;

        auto visitor = [&probe, i, samples_per_probe, this](auto&& arg) {
            using PROBE_TYPE = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<PROBE_TYPE, probe::BaseProbe> ||
                          std::is_same_v<PROBE_TYPE, probe::FloatProbe> ||
                          std::is_same_v<PROBE_TYPE, probe::Vec4Probe>) {

                probe.m_timestamp = arg.m_timestamp;
                probe.m_value_name = arg.m_value_name;
                probe.m_position = arg.m_position;
                probe.m_direction = arg.m_direction;
                probe.m_begin = arg.m_begin;
                probe.m_end = arg.m_end;
                probe.m_orig_end = arg.m_orig_end;
                probe.m_vert_ids = arg.m_vert_ids;
                probe.m_geo_ids = arg.m_geo_ids;
                probe.m_representant = arg.m_representant;
                probe.m_cluster_id = arg.m_cluster_id;
                probe.m_placement_method = arg.m_placement_method;

                _probes->setProbe(i, probe);

            } else if constexpr (std::is_same_v<PROBE_TYPE, probe::FloatDistributionProbe>) {
                probe = arg;

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(i);
        std::visit(visitor, generic_probe);

        auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
        auto probe_end_point = glm::vec3(to_vec3(probe.m_direction) * probe.m_orig_end + to_vec3(probe.m_position));

        std::shared_ptr<FloatDistributionProbe::SamplingResult> samples = probe.getSamplingResult();

        T probe_min = std::numeric_limits<T>::max();
        T probe_max = std::numeric_limits<T>::lowest();
        T probe_avg = 0.0;
        samples->samples.resize(samples_per_probe);
        // _dual_mesh_vertices is sorted after probe ids
        auto dual_mesh_top_vertices = _dual_mesh_vertices[i];
        auto dual_mesh_bottom_vertices = _dual_mesh_vertices[i];
        for (int j = 0; j < samples_per_probe; j++) {

            std::array<float,3> sample_center;
            sample_center[0] = probe.m_position[0] + (j * sample_step + sample_step/2.0f) * probe.m_direction[0];
            sample_center[1] = probe.m_position[1] + (j * sample_step + sample_step/2.0f) * probe.m_direction[1];
            sample_center[2] = probe.m_position[2] + (j * sample_step + sample_step/2.0f) * probe.m_direction[2];

            // DEBUG USE: auto sampledist_to_probe = glm::length(to_vec3(sample_center) - to_vec3(probe.m_position));
            float min_radius = std::numeric_limits<float>::max();
            float _ = 0.0f;
            float radius = sample_step/2.0f;
            for (int d = 0; d < dual_mesh_top_vertices.size(); d++) {
                // determine max radius of sampling geometry
                auto const dist = glm::length(to_vec3(dual_mesh_top_vertices[d]) -
                                              to_vec3(sample_center));
                radius = std::max(radius, dist);
                // calculate bottom of the sampling geometry
                glm::vec3 shift_dir;
                if (probe.m_placement_method == BaseProbe::CENTERPOINT) {
                    shift_dir = probe_end_point - to_vec3(dual_mesh_top_vertices[d]);
                } else if (probe.m_placement_method == BaseProbe::VERTEX_NORMAL) {
                    shift_dir = to_vec3(probe.m_direction);
                } else {
                    log.WriteError("[DualMeshSampling] Probe placement method unknown or not implemented.");
                    return;
                }
                shift_dir = glm::normalize(shift_dir);
                dual_mesh_bottom_vertices[d][0] += sample_step * shift_dir[0];
                dual_mesh_bottom_vertices[d][1] += sample_step * shift_dir[1];
                dual_mesh_bottom_vertices[d][2] += sample_step * shift_dir[2];
            }

            std::vector<std::pair<size_t, float>> res;
            auto num_neighbors =
                tree->radiusSearch(sample_center.data(), std::powf(radius,2), res, nanoflann::SearchParams());

            if (num_neighbors == 0) {
                log.WriteError("[DualMeshSampling] No samples in radius found!");
                return;
            }

            bool check_bottom = true;
            bool includes_probe = false;
            if (j == 0 && probe.m_placement_method != BaseProbe::VERTEX_NORMAL) {
                includes_probe = true;
            }
            if ((j == samples_per_probe - 1) && (probe.m_placement_method == BaseProbe::CENTERPOINT ||
                                                    probe.m_placement_method == BaseProbe::CENTERLINE)) {
                check_bottom = false;
            }

            // clac min radius
            if (!isInsideSamplingArea(min_radius, sample_center, dual_mesh_top_vertices, dual_mesh_bottom_vertices,
                        probe.m_position, probe.m_direction, check_bottom, includes_probe)) {
                log.WriteError("[DualMeshSampling] ERROR: Sample point found to be not in geometry. Check equations!");
                return;
            }

            // only keep samplepoints that are within the boundaries of the dual mesh sampling geometry
            std::vector<std::pair<size_t, float>> points_to_keep;
            points_to_keep.reserve(res.size());
            std::vector<float> kept_dists;
            for (auto& id: res) {
                std::array<float, 3> const point = tree->dataset.derived()[id.first];
                if (std::sqrtf(id.second) < min_radius) {
                    points_to_keep.emplace_back(id);
                    kept_dists.emplace_back(std::sqrtf(id.second));
                }else {
                    if (isInsideSamplingArea(_, point, dual_mesh_top_vertices, dual_mesh_bottom_vertices,
                        probe.m_position, probe.m_direction, check_bottom, includes_probe)) {
                    points_to_keep.emplace_back(id);
                    kept_dists.emplace_back(std::sqrtf(id.second));
                    }
                }
            }
            points_to_keep.shrink_to_fit();

            // accumulate values
            T sample_avg = 0.0;
            T sample_min = std::numeric_limits<T>::max();
            T sample_max = std::numeric_limits<T>::lowest();
            samples->samples[j].values.resize(points_to_keep.size());
            samples->samples[j].value_depth.resize(points_to_keep.size());
            for (int n = 0; n < points_to_keep.size(); n++) {
                T const current_value = data[points_to_keep[n].first];
                sample_avg += current_value;
                sample_min = std::min(sample_min, current_value);
                sample_max = std::max(sample_max, current_value);
                samples->samples[j].values[n] = current_value;
                samples->samples[j].value_depth[n] =
                    glm::dot(to_vec3(probe.m_position) - to_vec3(tree->dataset.derived()[points_to_keep[n].first]), to_vec3(probe.m_direction));
            } // end points_to_keep.size()
            sample_avg /= static_cast<T>(points_to_keep.size());

            if (!std::isfinite(sample_avg)) {
                log.WriteError("[DualMeshSampling] Non-finite values detected.");
                return;
            }

            samples->samples[j].mean = sample_avg;
            samples->samples[j].lower_bound = sample_min;
            samples->samples[j].upper_bound = sample_max;

            probe_min = std::min(probe_min, sample_min);
            probe_max = std::max(probe_max, sample_max);
            probe_avg += sample_avg;

            dual_mesh_top_vertices = dual_mesh_bottom_vertices;
        } // end num samples per probe
        samples->max_value = probe_max;
        samples->min_value = probe_min;
        samples->average_value = probe_avg / static_cast<T>(samples_per_probe);

        global_min = std::min(global_min, probe_min);
        global_max = std::max(global_max, probe_max);
    } // end for probes
    _probes->setGlobalMinMax(global_min, global_max);
}

template<typename T>
inline void DualMeshProbeSampling::doVectorSamling(const std::shared_ptr<my_kd_tree_t>& tree,
    const std::vector<T>& data_x, const std::vector<T>& data_y, const std::vector<T>& data_z,
    const std::vector<T>& data_w) {

    const int samples_per_probe = this->_num_samples_per_probe_slot.Param<core::param::IntParam>()->Value();

    //#pragma omp parallel for
    for (int32_t i = 0; i < static_cast<int32_t>(_probes->getProbeCount()); i++) {

        Vec4Probe probe;

        auto visitor = [&probe, i, samples_per_probe, this](auto&& arg) {
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

                _probes->setProbe(i, probe);

            } else if constexpr (std::is_same_v<T, probe::Vec4Probe>) {
                probe = arg;

                auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);

                _probes->setProbe(i, probe);

            } else {
                // unknown/incompatible probe type, throw error? do nothing?
            }
        };

        auto generic_probe = _probes->getGenericProbe(i);
        std::visit(visitor, generic_probe);

        auto sample_step = probe.m_end / static_cast<float>(samples_per_probe);
        auto radius = 1.0f;

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
            samples->samples[j][1] = value_y / num_neighbors;
            samples->samples[j][2] = value_z / num_neighbors;
            samples->samples[j][3] = value_w / num_neighbors;
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
inline bool DualMeshProbeSampling::isInsideSamplingArea(float& min_radius, const std::array<T, 3>& _point,
    const std::vector<std::array<T, 3>>& _top,
    const std::vector<std::array<T, 3>>& bottom, const std::array<T, 3>& _probe, const std::array<T, 3>& _probe_dir, const bool check_bottom, const bool includes_probe) {

    assert(bottom.size() >= 3);
    assert(_top.size() >= 3);

    glm::vec3 const point = to_vec3(_point);
    glm::vec3 const probe = to_vec3(_probe);
    std::vector<std::array<T, 3>> top = _top;
    top.emplace_back(_top[0]);
    // check bottom
    if (check_bottom) {
        // calculate normal
        glm::vec3 const bottom_normal = to_vec3(_probe_dir);
        // check direction of diff vec
        auto dot = glm::dot(bottom_normal, to_vec3(bottom[0]) - point);
        min_radius = std::min(min_radius, std::abs(dot));
        if (dot < 0) {
            return false;
        }
    }

    // check sides
    // DEBUG CODE
    //glm::vec3 com(0.0f,0.0f,0.0f);
    //for (int i = 0; i < _top.size(); i++) {
    //    com += to_vec3(top[i]);
    //}
    //com /= _top.size();
    //std::vector<glm::vec3> top_star(_top.size());
    //for (int i = 0; i < _top.size(); i++) {
    //    top_star[i] = to_vec3(top[i]) - com;
    //}
    for (int i = 0; i < _top.size(); i++) {
        glm::vec3 const side_normal = glm::normalize(glm::cross(to_vec3(bottom[i]) - to_vec3(top[i]), to_vec3(top[i + 1]) - to_vec3(top[i])));
        // check direction of diff vec
        auto dot = glm::dot(to_vec3(top[i]) - point, side_normal);
        min_radius = std::min(min_radius, std::abs(dot));
        if (dot < 0) {
            return false;
        }
    }

    // check top
    if (includes_probe) {
        for (int i = 0; i < _top.size(); i++) {
            glm::vec3 const top_normal =
                glm::normalize(glm::cross(to_vec3(top[i+1]) - to_vec3(top[i]), probe - to_vec3(top[i])));
            // check direction of diff vec
            auto dot = glm::dot(probe - point, top_normal);
            min_radius = std::min(min_radius, std::abs(dot));
            if (dot < 0) {
                return false;
            }
        }
    } else {
        // calculate normal
        glm::vec3 const top_normal = -1.0f * to_vec3(_probe_dir);
        // check direction of diff vec
        auto dot = glm::dot(to_vec3(top[0]) - point, top_normal);
        min_radius = std::min(min_radius, std::abs(dot));
        if (dot < 0) {
            return false;
        }
    }


    return true;
}


} // namespace megamol::probe
