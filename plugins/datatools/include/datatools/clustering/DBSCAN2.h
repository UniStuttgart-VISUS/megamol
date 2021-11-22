#pragma once

#include <algorithm>
#include <queue>
#include <stack>

#include "ann_interface.h"

namespace megamol {
namespace datatools {
namespace clustering {

using cluster_id_t = int;
using clusters_t = std::vector<cluster_id_t>;

enum class CLUSTER_IDS : int { UNVISITED = -3, VISITED = -2, NOISE = -1 };

inline void DBSCAN_expand(ANNkd_tree& kd_tree, float srad, int minPts, cluster_id_t id, clusters_t& cluster,
    clusters_t& visited, ANNidx idx) {
    cluster[idx] = id;

    if (visited[idx] == static_cast<int>(CLUSTER_IDS::VISITED))
        return;
    visited[idx] = static_cast<int>(CLUSTER_IDS::VISITED);

    auto const n = kd_tree.nPoints();
    auto const d = kd_tree.theDim();
    auto const points = kd_tree.thePoints();

    ann_point p(d, points[idx]);

    auto const nn = kd_tree.annkFRSearch(p, srad, 0);
    if (nn >= minPts) {
        std::vector<ANNidx> indices(nn, ANN_NULL_IDX);
        kd_tree.annkFRSearch(p, srad, nn, indices.data());
        indices.erase(std::remove(indices.begin(), indices.end(), idx), indices.end());

        for (auto const& el : indices) {
            if (cluster[el] < 0)
                DBSCAN_expand(kd_tree, srad, minPts, id, cluster, visited, el);
        }
    }
}

inline clusters_t DBSCAN_scan(ANNkd_tree& kd_tree, float eps, int minPts, int& numClusters) {
    auto const n = kd_tree.nPoints();
    auto const d = kd_tree.theDim();
    auto const points = kd_tree.thePoints();

    auto const srad = eps * eps;

    clusters_t cluster = clusters_t(n, static_cast<int>(CLUSTER_IDS::UNVISITED));
    clusters_t visited = clusters_t(n, static_cast<int>(CLUSTER_IDS::UNVISITED));

    cluster_id_t current_id = 0;

    for (int idx = 0; idx < n; ++idx) {
        if (visited[idx] != static_cast<int>(CLUSTER_IDS::UNVISITED))
            continue;

        visited[idx] = static_cast<int>(CLUSTER_IDS::VISITED);

        ann_point p(d, points[idx]);

        auto const nn = kd_tree.annkFRSearch(p, srad, 0);
        if (nn >= minPts) {
            std::vector<ANNidx> indices(nn, ANN_NULL_IDX);
            kd_tree.annkFRSearch(p, srad, nn, indices.data());
            indices.erase(std::remove(indices.begin(), indices.end(), idx), indices.end());

            cluster[idx] = current_id;

            for (auto const& el : indices) {
                DBSCAN_expand(kd_tree, srad, minPts, current_id, cluster, visited, el);
            }

            ++current_id;
        } else {
            cluster[idx] = static_cast<int>(CLUSTER_IDS::NOISE);
            //continue;
        }
    }

    numClusters = current_id;
    return cluster;
}


inline void DBSCAN_expand_it(ANNkd_tree& kd_tree, std::vector<ANNidx>& indices, float srad, int minPts, cluster_id_t id,
    clusters_t& cluster, clusters_t& visited, ANNidx idx) {
    //if (cluster[idx] > static_cast<int>(CLUSTER_IDS::NOISE)) return;
    cluster[idx] = id;

    auto const n = kd_tree.nPoints();
    auto const d = kd_tree.theDim();
    auto const points = kd_tree.thePoints();

    std::sort(indices.begin(), indices.end());

    std::queue<ANNidx> jobs;
    for (auto const& el : indices) {
        jobs.push(el);
    }

    while (!jobs.empty()) {
        auto el = jobs.front();
        jobs.pop();

        if (visited[el] == static_cast<int>(CLUSTER_IDS::VISITED)) {
            if (cluster[el] < 0)
                cluster[el] = id;
            continue;
        }

        visited[el] = static_cast<int>(CLUSTER_IDS::VISITED);
        cluster[el] = id;

        ann_point p(d, points[el]);

        auto const nn = kd_tree.annkFRSearch(p, srad, 0);
        if (nn >= minPts) {
            std::vector<ANNidx> new_indices(nn, ANN_NULL_IDX);
            std::vector<ANNidx> new_jobs(nn);
            kd_tree.annkFRSearch(p, srad, nn, new_indices.data());
            std::sort(new_indices.begin(), new_indices.end());
            new_jobs.erase(std::set_difference(new_indices.begin(), new_indices.end(), indices.begin(), indices.end(),
                               new_jobs.begin()),
                new_jobs.end());

            for (auto const& el : new_jobs) {
                if (visited[el] == static_cast<int>(CLUSTER_IDS::VISITED) && cluster[el] < 0)
                    cluster[el] = id;

                if (visited[el] != static_cast<int>(CLUSTER_IDS::VISITED))
                    jobs.push(el);
            }
        }
    }
}


inline clusters_t DBSCAN_scan_it(ANNkd_tree& kd_tree, float eps, int minPts, int& numClusters) {
    auto const n = kd_tree.nPoints();
    auto const d = kd_tree.theDim();
    auto const points = kd_tree.thePoints();

    auto const srad = eps * eps;

    clusters_t cluster = clusters_t(n, static_cast<int>(CLUSTER_IDS::UNVISITED));
    clusters_t visited = clusters_t(n, static_cast<int>(CLUSTER_IDS::UNVISITED));

    cluster_id_t current_id = 0;

    for (int idx = 0; idx < n; ++idx) {
        if (visited[idx] != static_cast<int>(CLUSTER_IDS::UNVISITED))
            continue;

        visited[idx] = static_cast<int>(CLUSTER_IDS::VISITED);

        ann_point p(d, points[idx]);

        auto const nn = kd_tree.annkFRSearch(p, srad, 0);
        if (nn >= minPts) {
            std::vector<ANNidx> indices(nn, ANN_NULL_IDX);
            kd_tree.annkFRSearch(p, srad, nn, indices.data());
            //indices.erase(std::remove(indices.begin(), indices.end(), idx), indices.end());

            //cluster[idx] = current_id;

            DBSCAN_expand_it(kd_tree, indices, srad, minPts, current_id, cluster, visited, idx);

            ++current_id;
        } else {
            cluster[idx] = static_cast<int>(CLUSTER_IDS::NOISE);
        }
    }

    numClusters = current_id;
    return cluster;
}


} // namespace clustering
} // namespace datatools
} // namespace megamol
