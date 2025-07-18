/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <random>

#include <glm/glm.hpp>

namespace megamol::core::utility {

constexpr double pi = 3.14159265359;

// http://corysimon.github.io/articles/uniformdistn-on-sphere/
template<typename T>
glm::vec<3, T> sample_on_sphere(T r, std::uniform_real_distribution<T>& distr, std::mt19937& rng) {
    float theta = static_cast<T>(2.0) * pi * distr(rng);
    float phi = std::acos(static_cast<T>(1.0) - static_cast<T>(2.0) * distr(rng));
    float x = r * std::sin(phi) * std::cos(theta);
    float y = r * std::sin(phi) * std::sin(theta);
    float z = r * std::cos(phi);
    return glm::vec3(x, y, z);
}

// https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
template<typename T>
glm::vec<3, T> sample_on_sphere_ex(T r, std::uniform_real_distribution<T>& distr, std::mt19937& rng) {
    glm::vec3 ret(0);
    do {
        ret = glm::vec3(distr(rng), distr(rng), distr(rng));
    } while (glm::length(ret) > 1.f);
    return r * glm::normalize(ret);
}

template<typename T>
std::vector<glm::vec<3, T>> sample_on_sphere(T r, unsigned int num_samples, unsigned int seed) {
    auto distr = std::uniform_real_distribution<T>();
    auto rng = std::mt19937(seed);
    std::vector<glm::vec<3, T>> ret(num_samples);
    std::generate(ret.begin(), ret.end(), [&r, &distr, &rng]() { return sample_on_sphere(r, distr, rng); });
    return ret;
}

template<typename T>
std::vector<glm::vec<3, T>> sample_on_sphere_ex(T r, unsigned int num_samples, unsigned int seed) {
    auto distr = std::uniform_real_distribution<T>(-1.f, 1.f);
    auto rng = std::mt19937(seed);
    std::vector<glm::vec<3, T>> ret(num_samples);
    std::generate(ret.begin(), ret.end(), [&r, &distr, &rng]() { return sample_on_sphere_ex(r, distr, rng); });
    return ret;
}
} // namespace megamol::core::utility
