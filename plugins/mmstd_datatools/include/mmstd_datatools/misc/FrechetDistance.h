#pragma once

#include <vector>

// http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

namespace {
template<typename T, typename DIST>
inline T c(std::vector<T>& ca, std::vector<T>::size_type i, std::vector<T>::size_type j, DIST const& dist,
    std::vector<T> const& a, std::vector<T> const& b) {
    if (ca(i, j) > -1) {
        return ca(i, j);
    } else if (i == 0 && j == 0) {
        ca(i, j) = dist(a[0], b[0]);
    } else if (i > 0 && j == 0) {
        ca(i, j) = std::max<T>(c(i - 1, 0), dist(a[i], b[0]));
    } else if (i == 0 && j > 0) {
        ca(i, j) = std::max<T>(c(0, j - 1), dist(a[0], b[j]));
    } else if (i > 0 && j > 0) {
        ca(i, j) = std::max<T>(std::min<T>(c(i - 1, j), std::min<T>(c(i - 1, j - 1), c(i, j - 1))), dist(a[i], b[j]));
    } else {
        return std::numeric_limits<T>::infinite();
    }
    return ca(i, j);
}
} // namespace

namespace megamol::stdplugin::datatools::misc {

template<typename T>
inline T std_dist(T const& a, T const& b) {
    return std::abs(a - b);
}

template<typename T, typename DIST>
inline T frechet_distance(std::vector<T> const& a, std::vector<T> const& b, DIST const& dist) {
    auto const total_size = a.size() * b.size();
    std::vector<T> ca(total_size, -1);
    return c(ca, a.size() - 1, b.size() - 1, dist, a, b);
}

} // namespace megamol::stdplugin::datatools::misc
