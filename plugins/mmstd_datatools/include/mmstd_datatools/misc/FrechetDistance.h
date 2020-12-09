#pragma once

#include <vector>

// http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

namespace {
template<typename T>
inline T c(std::vector<T>& ca, typename std::vector<T>::size_type width, typename std::vector<T>::size_type i,
    typename std::vector<T>::size_type j, std::vector<T> const& a, std::vector<T> const& b) {
    auto dist = [](T const& a, T const& b) -> T { return std::abs(a - b); };
    if (ca[i + j * a.size()] > static_cast<T>(-1)) {
        return ca[i + j * a.size()];
    } else if (i == 0 && j == 0) {
        ca[i + j * a.size()] = dist(a[0], b[0]);
    } else if (i > 0 && j == 0) {
        ca[i + j * a.size()] = std::max<T>(c(ca, width, i - 1, 0, a, b), dist(a[i], b[0]));
    } else if (i == 0 && j > 0) {
        ca[i + j * a.size()] = std::max<T>(c(ca, width, 0, j - 1, a, b), dist(a[0], b[j]));
    } else if (i > 0 && j > 0) {
        ca[i + j * a.size()] =
            std::max<T>(std::min<T>(c(ca, width, i - 1, j, a, b),
                            std::min<T>(c(ca, width, i - 1, j - 1, a, b), c(ca, width, i, j - 1, a, b))),
                dist(a[i], b[j]));
    } else {
        return std::numeric_limits<T>::infinity();
    }
    return ca[i + j * a.size()];
}
} // namespace

namespace megamol::stdplugin::datatools::misc {

template<typename T>
inline T frechet_distance(std::vector<T> const& a, std::vector<T> const& b) {
    auto const total_size = a.size() * b.size();
    std::vector<T> ca(total_size, static_cast<T>(-1));
    return c<T>(ca, a.size(), a.size() - 1, b.size() - 1, a, b);
}

} // namespace megamol::stdplugin::datatools::misc
