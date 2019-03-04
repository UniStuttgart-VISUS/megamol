#pragma once

#include "mmstd_datatools.h"

#include "PointcloudHelpers.h"

#include "nanoflann.hpp"

namespace megamol {
namespace stdplugin {
namespace datatools {

template <typename T, bool NOISE, int DIM = -1> class DBSCAN {
public:
    using cluster_set_t = std::vector<std::vector<T>>;

    DBSCAN(size_t const numPts, size_t const stride, std::vector<T>& pts, vislib::math::Cuboid<float> const& bbox, size_t const minPts, float const sigma)
        : minpts_(minPts), sigma_(sigma), numPts_(numPts), stride_(stride), ptsC_(numPts, stride, pts, bbox) {
        kdTree_ = std::make_shared<kd_tree_t>(DIM, ptsC_, nanoflann::KDTreeSingleIndexAdaptorParams());
        kdTree_->buildIndex();
    }

    cluster_set_t Scan();

private:
    using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<T, genericPointCloud<T>>,
        genericPointCloud<T>, DIM>;

    void expandCluster(size_t const qidx, std::vector<std::pair<size_t, T>>& matches, std::vector<T>& cluster,
        std::vector<bool>& marked, std::vector<bool>& noCluster, std::vector<size_t>& noise);

    std::shared_ptr<kd_tree_t> kdTree_;

    size_t minpts_;

    T sigma_;

    size_t numPts_;

    size_t stride_;

    genericPointCloud<T> ptsC_;
}; // end class DBSCAN


template <typename T, bool NOISE, int DIM> typename DBSCAN<T, NOISE, DIM>::cluster_set_t DBSCAN<T, NOISE, DIM>::Scan() {
    std::vector<bool> marked(numPts_, false);
    std::vector<bool> noCluster(numPts_, true);
    std::vector<size_t> noise;
    noise.reserve(numPts_/1000);

    std::array<T, DIM> qp;
    //qp[DIM];

    T sr = sigma_ * sigma_;

    std::vector<std::pair<size_t, T>> matches;
    matches.reserve(100);

    nanoflann::SearchParams sp;
    sp.sorted = false;

    cluster_set_t ret;
    ret.reserve(25);

    for (size_t idx = 0; idx < numPts_; ++idx) {
        if (!marked[idx]) {
            matches.clear();
            marked[idx] = true;
            for (int d = 0; d < DIM; ++d) {
                qp[d] = ptsC_.kdtree_get_pt(idx, d);
            }
            kdTree_->radiusSearch(qp.data(), sr, matches, sp);
            matches.erase(
                std::remove_if(matches.begin(), matches.end(), [&idx](auto const& el) { return el.first == idx; }),
                matches.end());
            auto numMatches = matches.size();
            if (numMatches >= minpts_) {
                ret.emplace_back();
                expandCluster(idx, matches, ret.back(), marked, noCluster, noise);
            } else {
                noise.push_back(idx);
            }
        }
    }

    if (NOISE) {
        // return noise points as separate clusters
        for (auto const& el : noise) {
            ret.emplace_back();
            auto& cluster = ret.back();
            for (size_t d = 0; d < stride_; ++d) {
                cluster.push_back(ptsC_.kdtree_get_pt(el, d));
            }
        }
    }

    return ret;
}


template <typename T, bool NOISE, int DIM>
void DBSCAN<T, NOISE, DIM>::expandCluster(size_t const qidx, std::vector<std::pair<size_t, T>>& matches,
    std::vector<T>& cluster, std::vector<bool>& marked, std::vector<bool>& noCluster, std::vector<size_t>& noise) {
    std::array<T, DIM> qp;
    //T qp[DIM];

    T sr = sigma_ * sigma_;

    std::vector<std::pair<size_t, T>> local_matches;
    local_matches.reserve(100);

    nanoflann::SearchParams sp;
    sp.sorted = false;

    cluster.reserve(100 * stride_);
    for (size_t d = 0; d < stride_; ++d) {
        cluster.push_back(ptsC_.kdtree_get_pt(qidx, d));
    }
    for (size_t midx = 0; midx < matches.size(); ++midx) {
        auto idx = matches[midx].first;
        if (!marked[idx]) {
            marked[idx] = true;
            for (int d = 0; d < DIM; ++d) {
                qp[d] = ptsC_.kdtree_get_pt(idx, d);
            }
            kdTree_->radiusSearch(qp.data(), sr, local_matches, sp);
            local_matches.erase(std::remove_if(local_matches.begin(), local_matches.end(),
                                    [&idx](auto const& el) { return el.first == idx; }),
                local_matches.end());
            auto numMatches = local_matches.size();
            if (numMatches >= numPts_) {
                matches.insert(matches.end(), local_matches.begin(), local_matches.end());
                /*for (auto const& n : local_matches) {
                    for (size_t d = 0; d < stride_; ++d) {
                        cluster.push_back(ptsC_.kdtree_get_pt(n.first, d));
                    }
                }*/
            }
        }
        if (noCluster[idx]) {
            noCluster[idx] = false;
            for (size_t d = 0; d < stride_; ++d) {
                cluster.push_back(ptsC_.kdtree_get_pt(idx, d));
            }
            auto it = std::find(noise.begin(), noise.end(), idx);
            if (it != noise.end()) {
                noise.erase(it);
            }
        }
    }
}

} // end namespace datatools
} // end namespace stdplugin
} // end namespace megamol
