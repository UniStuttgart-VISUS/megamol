#pragma once

#include <functional>
#include <vector>

// http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

namespace {
template<typename T, typename V>
inline T c(std::vector<T>& ca, typename std::vector<V>::size_type width, typename std::vector<V>::size_type i,
    typename std::vector<V>::size_type j, std::vector<V> const& a, std::vector<V> const& b,
    std::function<T(V const&, V const&)> const& dist) {
    // auto dist = [](T const& a, T const& b) -> T { return std::abs(a - b); };
    if (ca[i + j * a.size()] > static_cast<T>(-1)) {
        return ca[i + j * a.size()];
    } else if (i == 0 && j == 0) {
        ca[i + j * a.size()] = dist(a[0], b[0]);
    } else if (i > 0 && j == 0) {
        ca[i + j * a.size()] = std::max<T>(c(ca, width, i - 1, 0, a, b, dist), dist(a[i], b[0]));
    } else if (i == 0 && j > 0) {
        ca[i + j * a.size()] = std::max<T>(c(ca, width, 0, j - 1, a, b, dist), dist(a[0], b[j]));
    } else if (i > 0 && j > 0) {
        ca[i + j * a.size()] =
            std::max<T>(std::min<T>(c(ca, width, i - 1, j, a, b, dist),
                            std::min<T>(c(ca, width, i - 1, j - 1, a, b, dist), c(ca, width, i, j - 1, a, b, dist))),
                dist(a[i], b[j]));
    } else {
        return std::numeric_limits<T>::infinity();
    }
    return ca[i + j * a.size()];
}

template<typename T>
inline T c(std::vector<T>& ca, std::size_t width, std::size_t i, std::size_t j,
    std::function<T(std::size_t, std::size_t)> const& dist) {
    // auto dist = [](T const& a, T const& b) -> T { return std::abs(a - b); };
    if (ca[i + j * width] > static_cast<T>(-1.0f)) {
        return ca[i + j * width];
    } else if (i == 0 && j == 0) {
        ca[i + j * width] = dist(0, 0);
    } else if (i > 0 && j == 0) {
        ca[i + j * width] = std::max<T>(c(ca, width, i - 1, 0, dist), dist(i, 0));
    } else if (i == 0 && j > 0) {
        ca[i + j * width] = std::max<T>(c(ca, width, 0, j - 1, dist), dist(0, j));
    } else if (i > 0 && j > 0) {
        ca[i + j * width] =
            std::max<T>(std::min<T>(c(ca, width, i - 1, j, dist),
                            std::min<T>(c(ca, width, i - 1, j - 1, dist), c(ca, width, i, j - 1, dist))),
                dist(i, j));
    } else {
        return std::numeric_limits<T>::infinity();
    }
    return ca[i + j * width];
}
} // namespace

namespace megamol::datatools::misc {

template<typename T, typename V>
inline T frechet_distance(
    std::vector<V> const& a, std::vector<V> const& b, std::function<T(V const&, V const&)> const& dist) {
    auto const total_size = a.size() * b.size();
    std::vector<T> ca(total_size, static_cast<T>(-1));
    return c<T, V>(ca, a.size(), a.size() - 1, b.size() - 1, a, b, dist);
}

template<typename T>
inline T frechet_distance(std::size_t sample_count, std::function<T(std::size_t, std::size_t)> const& dist) {
    auto const total_size = sample_count * sample_count;
    std::vector<T> ca(total_size, static_cast<T>(-1));

    ca[0] = dist(0, 0);
    for (std::size_t i = 1; i < sample_count; ++i) {
        ca[i] = std::max<T>(ca[i - 1], dist(i, 0));
        ca[i * sample_count] = std::max<T>(ca[(i - 1) * sample_count], dist(0, i));
    }
    for (std::size_t i = 1; i < sample_count; ++i) {
        for (std::size_t j = 1; j < sample_count; ++j) {
            auto const c_0 = ca[i - 1 + j * sample_count];
            auto const c_1 = ca[i - 1 + (j - 1) * sample_count];
            auto const c_2 = ca[i + (j - 1) * sample_count];
            ca[i + j * sample_count] = std::max<T>(std::min<T>(c_0, std::min<T>(c_1, c_2)), dist(i, j));
        }
    }

    //return ca[sample_count - 1 + (sample_count - 1) * sample_count];
    return ca[sample_count * sample_count - 1];
}

template<typename T>
inline T frechet_distance_2(std::size_t sample_count, std::function<T(std::size_t, std::size_t)> const& dist) {
    auto const total_size = sample_count * sample_count;
    std::vector<T> ca(total_size, static_cast<T>(-1));
    return c<T>(ca, sample_count, sample_count - 1, sample_count - 1, dist);
}

} // namespace megamol::datatools::misc
