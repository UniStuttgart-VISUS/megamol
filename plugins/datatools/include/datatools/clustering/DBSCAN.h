#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "vislib/sys/ConsoleProgressBar.h"

#include "datatools/PointcloudHelpers.h"

#include <nanoflann.hpp>

namespace megamol::datatools::clustering {

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

template<typename T, int DIM>
inline void expand_cluster_with_similarity(std::shared_ptr<kd_tree_t<T, DIM>> const& D, index_t P, search_res_t<T> Nvec,
    index_t C, T eps, index_t minPts, cluster_result_t& clusters, std::vector<char>& visited,
    nanoflann::SearchParams const& params, std::function<bool(index_t, index_t)> const& similarity) {
    auto const& data = D->dataset;

    clusters[P] = C;

    search_res_t<T> tmp_res(minPts);

    for (typename search_res_t<T>::size_type vec_idx = 0; vec_idx < Nvec.size(); ++vec_idx) {
        auto const idx = Nvec[vec_idx].first;
        if (visited[idx] == 0) {
            visited[idx] = 1;
            auto query = data.get_position(idx);
            auto N = D->radiusSearch(query, eps, tmp_res, params);
            N = std::count_if(tmp_res.cbegin(), tmp_res.cend(),
                [idx, &similarity](auto const& el) { return similarity(idx, el.first); });
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
inline cluster_result_t DBSCAN_with_similarity(std::shared_ptr<kd_tree_t<T, DIM>> const& D, T eps, index_t minPts,
    std::function<bool(index_t, index_t)> const& similarity) {
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
        auto N = D->radiusSearch(query, eps, tmp_res, params);
        N = std::count_if(
            tmp_res.cbegin(), tmp_res.cend(), [idx, &similarity](auto const& el) { return similarity(idx, el.first); });
        if (N < minPts) {
            clusters[idx] = static_cast<cluster_type_ut>(cluster_type::NOISE);
        } else {
            ++cluster_idx;
            expand_cluster_with_similarity(
                D, idx, tmp_res, cluster_idx, eps, minPts, clusters, visited, params, similarity);
        }
    }

    return clusters;
}

template<typename T, int DIM>
inline void expand_cluster_with_similarity_and_score(std::shared_ptr<kd_tree_t<T, DIM>> const& D, index_t P, T P_score,
    search_res_t<T> Nvec, index_t C, T eps, index_t minPts, cluster_result_t& clusters, std::vector<char>& visited,
    nanoflann::SearchParams const& params, std::function<bool(index_t, index_t)> const& similarity,
    std::function<index_t(index_t, std::vector<index_t> const&)> const& score,
    std::unordered_map<index_t, std::vector<index_t>>& cluster_identity) {
    auto const& data = D->dataset;

    clusters[P] = C;
    cluster_identity[C].push_back(P);

    search_res_t<T> tmp_res(minPts);

    index_t pivot = P;
    T pivot_score = P_score;
    index_t pivot_size = 1;

    for (typename search_res_t<T>::size_type vec_idx = 0; vec_idx < Nvec.size(); ++vec_idx) {
        auto const idx = Nvec[vec_idx].first;
        if (visited[idx] == 0) {
            visited[idx] = 1;
            auto query = data.get_position(idx);
            auto N = D->radiusSearch(query, eps, tmp_res, params);
            // auto old_N = N;
            tmp_res.erase(std::remove_if(tmp_res.begin(), tmp_res.end(),
                              [pivot, &similarity](auto const& el) { return !similarity(pivot, el.first); }),
                tmp_res.end());
            N = tmp_res.size();
            // core::utility::log::Log::DefaultLog.WriteInfo("Size of Neighborhood %d was %d", N, old_N);
            if (N >= minPts) {
                Nvec.insert(Nvec.cend(), tmp_res.cbegin(), tmp_res.cend());
                // adapt pivot
                /*std::vector<std::pair<index_t, T>> scores(tmp_res.size());
                std::transform(tmp_res.cbegin(), tmp_res.cend(), scores.begin(), [&score, pivot_score, pivot_size,
                &tmp_res](auto const& el) { auto const score_val = score(el.first, tmp_res); auto const change =
                (score_val - pivot_score) / static_cast<T>(pivot_size + 1); return std::make_pair(el.first, change);
                });
                auto const min_el = std::min_element(scores.cbegin(), scores.cend(),
                    [](auto const& lhs, auto const& rhs) { return std::abs(lhs.second) < std::abs(rhs.second); });
                pivot = min_el->first;
                pivot_score += min_el->second;
                ++pivot_size;*/
                // pivot_score = score(pivot, idx, pivot_score, pivot_size);
            }
        }
        if (clusters[idx] <= static_cast<cluster_type_ut>(cluster_type::NOISE)) {
            clusters[idx] = C;
            cluster_identity[C].push_back(idx);
            pivot = score(pivot, cluster_identity[C]);
        }
    }
}

template<typename T, int DIM>
inline cluster_result_t DBSCAN_with_similarity_and_score(std::shared_ptr<kd_tree_t<T, DIM>> const& D, T eps,
    index_t minPts, std::function<bool(index_t, index_t)> const& similarity,
    std::function<index_t(index_t, std::vector<index_t> const&)> const& score) {
    auto const& data = D->dataset;
    auto const num_points = data.kdtree_get_point_count();
    cluster_result_t clusters(num_points, static_cast<cluster_type_ut>(cluster_type::UNDEFINED));
    std::vector<char> visited(num_points, 0);
    nanoflann::SearchParams params;
    params.sorted = false;

    search_res_t<T> tmp_res(minPts);

    index_t cluster_idx = static_cast<cluster_type_ut>(cluster_type::NOISE);

    std::unordered_map<index_t, std::vector<index_t>> cluster_identity;

    vislib::sys::ConsoleProgressBar cpb;

    cpb.Start("DBSCAN", num_points);

    for (std::remove_const_t<decltype(num_points)> idx = 0; idx < num_points; ++idx) {
        if (visited[idx] > 0)
            continue;
        visited[idx] = 1;
        auto query = data.get_position(idx);
        auto N = D->radiusSearch(query, eps, tmp_res, params);
        // auto old_N = N;
        tmp_res.erase(std::remove_if(tmp_res.begin(), tmp_res.end(),
                          [idx, &similarity](auto const& el) { return !similarity(idx, el.first); }),
            tmp_res.end());
        N = tmp_res.size();
        // core::utility::log::Log::DefaultLog.WriteInfo("Size of Neighborhood %d was %d", N, old_N);
        if (N < minPts) {
            clusters[idx] = static_cast<cluster_type_ut>(cluster_type::NOISE);
        } else {
            ++cluster_idx;
            T const idx_score = 0;
            //= score(idx, tmp_res);
            expand_cluster_with_similarity_and_score(D, idx, idx_score, tmp_res, cluster_idx, eps, minPts, clusters,
                visited, params, similarity, score, cluster_identity);
        }
        cpb.Set(idx);
    }

    cpb.Stop();

    return clusters;
}


template<typename T, int DIM>
inline void expand_GROWING_with_similarity_and_score(std::shared_ptr<kd_tree_t<T, DIM>> const& D, index_t P, T P_score,
    search_res_t<T> Nvec, index_t C, T eps, index_t minPts, cluster_result_t& clusters, std::vector<char>& visited,
    nanoflann::SearchParams const& params, std::function<bool(index_t, index_t)> const& similarity,
    std::function<index_t(index_t, std::vector<index_t> const&)> const& score,
    std::unordered_map<index_t, std::vector<index_t>>& cluster_identity) {
    auto const& data = D->dataset;

    clusters[P] = C;
    cluster_identity[C].push_back(P);

    search_res_t<T> tmp_res;
    tmp_res.reserve(minPts);

    index_t pivot = P;
    T pivot_score = P_score;
    index_t pivot_size = 1;

    while (!Nvec.empty()) {
        search_res_t<T> nextNvec;
        nextNvec.reserve(minPts);
        for (typename search_res_t<T>::size_type vec_idx = 0; vec_idx < Nvec.size(); ++vec_idx) {
            auto const idx = Nvec[vec_idx].first;
            if (visited[idx] == 0) {
                visited[idx] = 1;
                auto query = data.get_position(idx);
                auto N = D->radiusSearch(query, eps, tmp_res, params);
                // auto old_N = N;
                tmp_res.erase(std::remove_if(tmp_res.begin(), tmp_res.end(),
                                  [pivot, &similarity](auto const& el) { return !similarity(pivot, el.first); }),
                    tmp_res.end());
                N = tmp_res.size();
                /*core::utility::log::Log::DefaultLog.WriteInfo(
                    "Size of Neighborhood %d was %d and Nvec %d", N, old_N, Nvec.size());*/
                if (N > 0) {
                    tmp_res.erase(std::remove_if(tmp_res.begin(), tmp_res.end(),
                                      [&visited](auto const& el) { return visited[el.first] > 0; }),
                        tmp_res.end());
                    if (!tmp_res.empty()) {
                        nextNvec.insert(nextNvec.cend(), tmp_res.cbegin(), tmp_res.cend());
                        std::sort(nextNvec.begin(), nextNvec.end(),
                            [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
                        nextNvec.erase(std::unique(nextNvec.begin(), nextNvec.end(),
                                           [](auto const& lhs, auto const& rhs) { return lhs.first == rhs.first; }),
                            nextNvec.end());
                    }
                    // adapt pivot
                    /*std::vector<std::pair<index_t, T>> scores(tmp_res.size());
                    std::transform(tmp_res.cbegin(), tmp_res.cend(), scores.begin(), [&score, pivot_score, pivot_size,
                    &tmp_res](auto const& el) { auto const score_val = score(el.first, tmp_res); auto const change =
                    (score_val - pivot_score) / static_cast<T>(pivot_size + 1); return std::make_pair(el.first, change);
                    });
                    auto const min_el = std::min_element(scores.cbegin(), scores.cend(),
                        [](auto const& lhs, auto const& rhs) { return std::abs(lhs.second) < std::abs(rhs.second); });
                    pivot = min_el->first;
                    pivot_score += min_el->second;
                    ++pivot_size;*/
                    // pivot_score = score(pivot, idx, pivot_score, pivot_size);
                }
            }
            if (clusters[idx] <= static_cast<cluster_type_ut>(cluster_type::NOISE)) {
                clusters[idx] = C;
                cluster_identity[C].push_back(idx);
            }
            // pivot = score(pivot, cluster_identity[C]);
        }
        pivot = score(pivot, cluster_identity[C]);
        Nvec = nextNvec;
    }
}


template<typename T, int DIM>
inline void expand_GROWING_with_similarity(std::shared_ptr<kd_tree_t<T, DIM>> const& D, index_t P, T P_score,
    std::list<typename search_res_t<T>::value_type> Nvec, index_t C, T eps, index_t minPts, cluster_result_t& clusters,
    std::vector<char>& visited, nanoflann::SearchParams const& params,
    std::function<bool(index_t, index_t)> const& similarity,
    std::function<index_t(index_t, std::vector<index_t> const&)> const& score,
    std::unordered_map<index_t, std::vector<index_t>>& cluster_identity) {
    auto const& data = D->dataset;

    clusters[P] = C;

    std::list<index_t> current_cluster;
    current_cluster.push_back(P);

    while (!Nvec.empty()) {
        auto const current_el = Nvec.front();
        Nvec.pop_front();

        auto const idx = current_el.first;
        if (visited[idx] == 1)
            continue;

        visited[idx] = 1;
        search_res_t<T> tmp_res;
        auto query = data.get_position(idx);
        auto N = D->radiusSearch(query, eps, tmp_res, params);
        std::copy_if(tmp_res.begin(), tmp_res.end(), std::back_inserter(Nvec),
            [idx, &similarity, &current_cluster](auto const& el) {
                if (idx == el.first)
                    return false;
                if (similarity(idx, el.first)) {
                    /*for (auto const& cl : current_cluster) {
                        if (!similarity(idx, cl)) {
                            return false;
                        }
                    }*/
                    return true;
                }
                return false;
            });
        clusters[idx] = C;
        current_cluster.push_back(idx);
    }


    //cluster_identity[C].push_back(P);

    //search_res_t<T> tmp_res;
    //tmp_res.reserve(minPts);

    //index_t pivot = P;
    //T pivot_score = P_score;
    //index_t pivot_size = 1;

    //while (!Nvec.empty()) {
    //    search_res_t<T> nextNvec;
    //    nextNvec.reserve(minPts);
    //    for (search_res_t<T>::size_type vec_idx = 0; vec_idx < Nvec.size(); ++vec_idx) {
    //        auto const idx = Nvec[vec_idx].first;
    //        if (visited[idx] == 0) {
    //            visited[idx] = 1;
    //            auto query = data.get_position(idx);
    //            auto N = D->radiusSearch(query, eps, tmp_res, params);
    //            // auto old_N = N;
    //            tmp_res.erase(std::remove_if(tmp_res.begin(), tmp_res.end(),
    //                              [pivot, &similarity](auto const& el) { return !similarity(pivot, el.first); }),
    //                tmp_res.end());
    //            N = tmp_res.size();
    //            /*core::utility::log::Log::DefaultLog.WriteInfo(
    //                "Size of Neighborhood %d was %d and Nvec %d", N, old_N, Nvec.size());*/
    //            if (N > 0) {
    //                tmp_res.erase(std::remove_if(tmp_res.begin(), tmp_res.end(),
    //                                  [&visited](auto const& el) { return visited[el.first] > 0; }),
    //                    tmp_res.end());
    //                if (!tmp_res.empty()) {
    //                    nextNvec.insert(nextNvec.cend(), tmp_res.cbegin(), tmp_res.cend());
    //                    std::sort(nextNvec.begin(), nextNvec.end(),
    //                        [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
    //                    nextNvec.erase(std::unique(nextNvec.begin(), nextNvec.end(),
    //                                       [](auto const& lhs, auto const& rhs) { return lhs.first == rhs.first; }),
    //                        nextNvec.end());
    //                }
    //                // adapt pivot
    //                /*std::vector<std::pair<index_t, T>> scores(tmp_res.size());
    //                std::transform(tmp_res.cbegin(), tmp_res.cend(), scores.begin(), [&score, pivot_score, pivot_size,
    //                &tmp_res](auto const& el) { auto const score_val = score(el.first, tmp_res); auto const change =
    //                (score_val - pivot_score) / static_cast<T>(pivot_size + 1); return std::make_pair(el.first, change);
    //                });
    //                auto const min_el = std::min_element(scores.cbegin(), scores.cend(),
    //                    [](auto const& lhs, auto const& rhs) { return std::abs(lhs.second) < std::abs(rhs.second); });
    //                pivot = min_el->first;
    //                pivot_score += min_el->second;
    //                ++pivot_size;*/
    //                // pivot_score = score(pivot, idx, pivot_score, pivot_size);
    //            }
    //        }
    //        if (clusters[idx] <= static_cast<cluster_type_ut>(cluster_type::NOISE)) {
    //            clusters[idx] = C;
    //            cluster_identity[C].push_back(idx);
    //        }
    //        // pivot = score(pivot, cluster_identity[C]);
    //    }
    //    pivot = score(pivot, cluster_identity[C]);
    //    Nvec = nextNvec;
    //}
}


template<typename T, int DIM>
inline cluster_result_t GROWING_with_similarity_and_score(std::shared_ptr<kd_tree_t<T, DIM>> const& D, T eps,
    index_t minPts, std::function<bool(index_t, index_t)> const& similarity,
    std::function<index_t(index_t, std::vector<index_t> const&)> const& score) {
    auto const& data = D->dataset;
    auto const num_points = data.kdtree_get_point_count();
    cluster_result_t clusters(num_points, static_cast<cluster_type_ut>(cluster_type::UNDEFINED));
    std::vector<char> visited(num_points, 0);
    nanoflann::SearchParams params;
    params.sorted = false;

    search_res_t<T> tmp_res(minPts);

    index_t cluster_idx = static_cast<cluster_type_ut>(cluster_type::NOISE);

    std::unordered_map<index_t, std::vector<index_t>> cluster_identity;

    vislib::sys::ConsoleProgressBar cpb;

    cpb.Start("GROWING", num_points);

    for (std::remove_const_t<decltype(num_points)> idx = 0; idx < num_points; ++idx) {
        cpb.Set(idx);
        if (visited[idx] > 0)
            continue;
        visited[idx] = 1;
        auto query = data.get_position(idx);
        auto N = D->radiusSearch(query, eps, tmp_res, params);

        std::list<typename search_res_t<T>::value_type> candidates;
        std::copy_if(tmp_res.begin(), tmp_res.end(), std::back_inserter(candidates),
            [idx, &similarity](auto const& el) { return (idx != el.first) && similarity(idx, el.first); });

        /*tmp_res.erase(std::remove_if(tmp_res.begin(), tmp_res.end(),
                          [idx, &similarity](auto const& el) { return !similarity(idx, el.first); }),
            tmp_res.end());
        N = tmp_res.size();*/

        N = candidates.size();

        if (N > 0) {
            ++cluster_idx;
            clusters[idx] = cluster_idx;

            T const idx_score = 0;
            /*expand_GROWING_with_similarity_and_score(D, idx, idx_score, tmp_res, cluster_idx, eps, minPts, clusters,
                visited, params, similarity, score, cluster_identity);*/
            expand_GROWING_with_similarity(D, idx, idx_score, candidates, cluster_idx, eps, minPts, clusters, visited,
                params, similarity, score, cluster_identity);
        }
    }

    cpb.Stop();

    return clusters;
}


} // namespace megamol::datatools::clustering
