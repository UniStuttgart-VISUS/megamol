/*
 * integrator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "Eigen/Dense"

#include "data/tpf_grid.h"

namespace megamol {
namespace flowvis {

/**
 * Runge-Kutta 4 for fixed step size
 *
 * @tparam dimension Dimension of the vector field domain
 *
 * @param vector_field Vector field
 * @param point Point to advect (will be modified)
 * @param delta Time step
 * @param forward Forward integration if true, reverse integration otherwise
 */
template <int dimension>
void advect_point_rk4(const tpf::data::grid<float, float, dimension, dimension>& vector_field,
    Eigen::Matrix<float, dimension, 1>& point, const float delta, const bool forward);

/**
 * Runge-Kutta-Fehlberg 4-5 for dynamic step size
 *
 * @tparam dimension Dimension of the vector field domain
 * @tparam components Number of components (dimension of the vectors)
 *
 * @param vector_field Vector field
 * @param point Point to advect (will be modified)
 * @param delta Initial time step; returns the adapted time step
 * @param max_error Maximum allowed error, exceeding leads to time step adaption
 * @param forward Forward integration if true, reverse integration otherwise
 */
template <int dimension>
void advect_point_rk45(const tpf::data::grid<float, float, dimension, dimension>& vector_field,
    Eigen::Matrix<float, dimension, 1>& point, float& delta, const float max_error, const bool forward);

} // namespace flowvis
} // namespace megamol

template <int dimension>
void megamol::flowvis::advect_point_rk4(const tpf::data::grid<float, float, dimension, dimension>& vector_field,
    Eigen::Matrix<float, dimension, 1>& point, const float delta, const bool forward) {

    // Calculate step size
    const auto max_velocity = vector_field.interpolate(point).norm();
    const auto min_cellsize = vector_field.get_cell_sizes(*vector_field.find_cell(point)).minCoeff();

    const auto steps_per_cell = max_velocity > 0.0f ? min_cellsize / max_velocity : 0.0f;

    // Integration parameters
    const auto sign = forward ? 1.0f : -1.0f;

    // Calculate Runge-Kutta coefficients
    const auto k1 = steps_per_cell * delta * sign * vector_field.interpolate(point);
    const auto k2 = steps_per_cell * delta * sign * vector_field.interpolate(point + 0.5f * k1);
    const auto k3 = steps_per_cell * delta * sign * vector_field.interpolate(point + 0.5f * k2);
    const auto k4 = steps_per_cell * delta * sign * vector_field.interpolate(point + k3);

    // Advect and store position
    Eigen::Matrix<float, dimension, 1> advection = (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);

    if (advection.norm() > max_velocity) {
        advection = advection.normalized() * max_velocity;
    }

    point += advection;
}

template <int dimension>
void megamol::flowvis::advect_point_rk45(const tpf::data::grid<float, float, dimension, dimension>& vector_field,
    Eigen::Matrix<float, dimension, 1>& point, float& delta, const float max_error, const bool forward) {

    // Cash-Karp parameters
    constexpr float b_21 = 0.2f;
    constexpr float b_31 = 0.075f;
    constexpr float b_41 = 0.3f;
    constexpr float b_51 = -11.0f / 54.0f;
    constexpr float b_61 = 1631.0f / 55296.0f;
    constexpr float b_32 = 0.225f;
    constexpr float b_42 = -0.9f;
    constexpr float b_52 = 2.5f;
    constexpr float b_62 = 175.0f / 512.0f;
    constexpr float b_43 = 1.2f;
    constexpr float b_53 = -70.0f / 27.0f;
    constexpr float b_63 = 575.0f / 13824.0f;
    constexpr float b_54 = 35.0f / 27.0f;
    constexpr float b_64 = 44275.0f / 110592.0f;
    constexpr float b_65 = 253.0f / 4096.0f;

    constexpr float c_1 = 37.0f / 378.0f;
    constexpr float c_2 = 0.0f;
    constexpr float c_3 = 250.0f / 621.0f;
    constexpr float c_4 = 125.0f / 594.0f;
    constexpr float c_5 = 0.0f;
    constexpr float c_6 = 512.0f / 1771.0f;

    constexpr float c_1s = 2825.0f / 27648.0f;
    constexpr float c_2s = 0.0f;
    constexpr float c_3s = 18575.0f / 48384.0f;
    constexpr float c_4s = 13525.0f / 55296.0f;
    constexpr float c_5s = 277.0f / 14336.0f;
    constexpr float c_6s = 0.25f;

    // Constants
    constexpr float grow_exponent = -0.2f;
    constexpr float shrink_exponent = -0.25f;
    constexpr float max_growth = 5.0f;
    constexpr float max_shrink = 0.1f;
    constexpr float safety = 0.9f;

    // Integration parameters
    const auto sign = forward ? 1.0f : -1.0f;

    // Calculate Runge-Kutta coefficients
    bool decreased = false;

    do {
        const auto k1 = delta * sign * vector_field.interpolate(point);
        const auto k2 = delta * sign * vector_field.interpolate(point + b_21 * k1);
        const auto k3 = delta * sign * vector_field.interpolate(point + b_31 * k1 + b_32 * k2);
        const auto k4 = delta * sign * vector_field.interpolate(point + b_41 * k1 + b_42 * k2 + b_43 * k3);
        const auto k5 = delta * sign * vector_field.interpolate(point + b_51 * k1 + b_52 * k2 + b_53 * k3 + b_54 * k4);
        const auto k6 = delta * sign * vector_field.interpolate(point + b_61 * k1 + b_62 * k2 + b_63 * k3 + b_64 * k4 + b_65 * k5);

        // Calculate error estimate
        const auto fifth_order = point + c_1 * k1 + c_2 * k2 + c_3 * k3 + c_4 * k4 + c_5 * k5 + c_6 * k6;
        const auto fourth_order = point + c_1s * k1 + c_2s * k2 + c_3s * k3 + c_4s * k4 + c_5s * k5 + c_6s * k6;

        const auto difference = (fifth_order - fourth_order).cwiseAbs();
        const auto scale = vector_field.interpolate(point).cwiseAbs();

        const auto error = std::max(0.0f, difference.cwiseQuotient(scale).maxCoeff()) / max_error;

        // Set new, adapted time step
        if (error > 1.0f) {
            // Error too large, reduce time step
            delta *= std::max(max_shrink, safety * std::pow(error, shrink_exponent));
            decreased = true;
        } else {
            // Error (too) small, increase time step
            delta *= std::min(max_growth, safety * std::pow(error, grow_exponent));
            decreased = false;
        }

        // Set output
        point = fifth_order;
    } while (decreased);
}
