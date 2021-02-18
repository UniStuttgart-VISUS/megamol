#pragma once

#include "ann_interface.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace clustering {

using cluster_id_t = int;
using clusters_t = std::vector<cluster_id_t>;

enum class CLUSTER_IDS : int { UNDEFINED = -2, NOISE = -1 };

inline void DBSCAN_expand(ANNkd_tree& kd_tree, float srad, int minPts, cluster_id_t id, clusters_t& cluster, ANNidx idx) {
    if (cluster[idx] > static_cast<int>(CLUSTER_IDS::NOISE)) return;
    cluster[idx] = id;

    auto const n = kd_tree.nPoints();
    auto const d = kd_tree.theDim();
    auto const points = kd_tree.thePoints();

    ann_point p(d, points[idx]);

    auto const nn = kd_tree.annkFRSearch(p, srad, 0);
    if (nn >= minPts) {
        std::vector<ANNidx> indices(nn, ANN_NULL_IDX);
        kd_tree.annkFRSearch(p, srad, nn, indices.data());

        for (auto const& el : indices) {
            DBSCAN_expand(kd_tree, srad, minPts, id, cluster, el);
        }
    }
}

inline clusters_t DBSCAN_scan(ANNkd_tree& kd_tree, float eps, int minPts, int& numClusters) {
    auto const n = kd_tree.nPoints();
    auto const d = kd_tree.theDim();
    auto const points = kd_tree.thePoints();

    auto const srad = eps * eps;

    clusters_t cluster = clusters_t(n, static_cast<int>(CLUSTER_IDS::UNDEFINED));

    cluster_id_t current_id = 0;

    for (int idx = 0; idx < n; ++idx) {
        if (cluster[idx] != static_cast<int>(CLUSTER_IDS::UNDEFINED)) continue;

        ann_point p(d, points[idx]);

        auto const nn = kd_tree.annkFRSearch(p, srad, 0);
        if (nn >= minPts) {
            std::vector<ANNidx> indices(nn, ANN_NULL_IDX);
            kd_tree.annkFRSearch(p, srad, nn, indices.data());

            cluster[idx] = current_id;

            for (auto const& el : indices) {
                DBSCAN_expand(kd_tree, srad, minPts, current_id, cluster, el);
            }

            ++current_id;
        } else {
            cluster[idx] = static_cast<int>(CLUSTER_IDS::NOISE);
            continue;
        }
    }

    numClusters = current_id;
    return cluster;
}


} // namespace clustering
} // namespace datatools
} // namespace stdplugin
} // namespace megamol
