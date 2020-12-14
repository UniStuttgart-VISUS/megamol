#pragma once

#include <memory>
#include <vector>

#include "PointcloudHelpers.h"

#include <nanoflann.hpp>

namespace megamol::stdplugin::datatools::clustering {

using index_t = std::size_t;

using cluster_result_t = std::vector<index_t>;

template<typename T, int DIM>
using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<T, genericPointcloud<T, DIM>>,
    genericPointcloud<T, DIM>, DIM>;

using cluster_type_ut = index_t;

enum class cluster_type : cluster_type_ut { UNDEFINED = 0, NOISE = 1 };

template<typename T>
using search_res_t = std::vector<std::pair<index_t, T>>;

// see https://de.wikipedia.org/wiki/DBSCAN for algorithm

template<typename T, int DIM>
inline void expand_cluster(std::shared_ptr<kd_tree_t<T, DIM>> const& D, index_t P, search_res_t<T> Nvec, index_t C,
    T eps, index_t minPts, cluster_result_t& clusters, std::vector<char>& visited,
    nanoflann::SearchParams const& params) {
    auto const& data = D->dataset;

    clusters[P] = C;

    search_res_t<T> tmp_res(minPts);

    for (typename search_res_t<T>::size_type vec_idx = 0; vec_idx < Nvec.size(); ++vec_idx) {
        auto const idx = Nvec[vec_idx].first;
        if (visited[idx] == 0) {
            visited[idx] = 1;
            auto query = data.get_position(idx);
            auto const N = D->radiusSearch(query, eps, tmp_res, params);
            if (N >= minPts) {
                Nvec.insert(Nvec.cend(), tmp_res.cbegin(), tmp_res.cend());
            }
        }
        if (clusters[idx] <= static_cast<cluster_type_ut>(cluster_type::NOISE)) {
            clusters[idx] = C;
        }
    }
}

template<typename T, int DIM>
inline cluster_result_t DBSCAN(std::shared_ptr<kd_tree_t<T, DIM>> const& D, T eps, index_t minPts) {
    auto const& data = D->dataset;
    auto const num_points = data.kdtree_get_point_count();
    cluster_result_t clusters(num_points, static_cast<cluster_type_ut>(cluster_type::UNDEFINED));
    std::vector<char> visited(num_points, 0);
    nanoflann::SearchParams params;
    params.sorted = false;

    search_res_t<T> tmp_res(minPts);

    index_t cluster_idx = static_cast<cluster_type_ut>(cluster_type::NOISE);

    for (std::remove_const_t<decltype(num_points)> idx = 0; idx < num_points; ++idx) {
        if (visited[idx] > 0)
            continue; 
        visited[idx] = 1;
        auto query = data.get_position(idx);
        auto const N = D->radiusSearch(query, eps, tmp_res, params);
        if (N < minPts) {
            clusters[idx] = static_cast<cluster_type_ut>(cluster_type::NOISE);
        } else {
            ++cluster_idx;
            expand_cluster(D, idx, tmp_res, cluster_idx, eps, minPts, clusters, visited, params);
        }
    }

    return clusters;
}

} // namespace megamol::stdplugin::datatools::clustering
