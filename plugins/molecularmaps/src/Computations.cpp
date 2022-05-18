/*
 * Computations.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "Computations.h"
#include "stdafx.h"

using namespace megamol::molecularmaps;
using namespace megamol::protein_calls;

/**
 * Initalise the access patterns for the submatrices.
 */
std::vector<int> Computations::access_columns_circle = [] {
    std::vector<int> retval;
    retval.push_back(0);
    retval.push_back(1);
    retval.push_back(0);
    retval.push_back(2);
    retval.push_back(1);
    retval.push_back(2);
    return retval;
}();

std::vector<int> Computations::access_columns_sphere = [] {
    std::vector<int> retval;
    retval.push_back(0);
    retval.push_back(1);
    retval.push_back(2);
    retval.push_back(0);
    retval.push_back(1);
    retval.push_back(3);
    retval.push_back(0);
    retval.push_back(2);
    retval.push_back(3);
    retval.push_back(1);
    retval.push_back(2);
    retval.push_back(3);
    return retval;
}();

double Computations::pi_half = vislib::math::PI_DOUBLE / 2.0;

/*
 * Computations::~Computations
 */
Computations::~Computations(void) {}

/*
 * Computations::AngularDistance
 */
double Computations::AngularDistance(const vec3d& p_start_pivot, const vec3d& p_pivot, const vec4d& p_vertToTest) {
    // Compute the vector from the pivot to the end vertex and normalise it.
    vec3d end_pivot = vec3d(p_vertToTest.PeekComponents()) - p_pivot;
    Computations::NormaliseVector(end_pivot);

    // Compute the angle between the voroVert and the vertToTest.
    double x = p_start_pivot.Dot(end_pivot);
    double fac = 2.0 * x - 1.0;
    double fac_sqrt = fac * fac;
    double fac_sqrt_3 = fac_sqrt * fac;
    double val = Computations::pi_half -
                 (0.5689111419 - 0.2644381021 * x - 0.4212611542 * fac_sqrt + 0.1475622352 * fac_sqrt_3) /
                     (2.006022274 - 2.343685222 * x + 0.3316406750 * fac_sqrt + 0.02607135626 * fac_sqrt_3);

    return val;
}

/*
 * Computations::Computations
 */
Computations::Computations(void) {}

/*
 * Computations::ComputeGateCenter
 */
size_t Computations::ComputeGateCenter(const std::array<vec4d, 4>& p_gate_vector, std::array<vec4d, 2>& p_gate_center,
    std::array<vec4d, 2>& p_bary_solutions, std::array<vec3d, 2>& p_circles) {
    // Compute the coordinates of the center circle(s) of the three gate vertices.
    auto bary_cnt = Computations::incircleBaryCoords(p_gate_vector, p_bary_solutions, p_circles);

    // If no incircle was found, process the next gate.
    if (bary_cnt < 1)
        return 0;

    // Compute the real middle spheres based on the barycentric coordinates.
    for (size_t i = 0; i < bary_cnt; i++) {
        p_gate_center[i].Set(0.0, 0.0, 0.0, 0.0);
        p_gate_center[i] += p_bary_solutions[i].GetX() * p_gate_vector[0];
        p_gate_center[i] += p_bary_solutions[i].GetY() * p_gate_vector[1];
        p_gate_center[i] += p_bary_solutions[i].GetZ() * p_gate_vector[2];
        p_gate_center[i].SetW(p_bary_solutions[i].GetW());
    }

    return bary_cnt;
}

/*
 * Computations::ComputePivot
 */
vec3d Computations::ComputePivot(const std::array<vec4d, 4>& p_gate_spheres) {
    vec3d pivot = vec3d(p_gate_spheres[0].PeekComponents());
    if (p_gate_spheres[1].GetW() < p_gate_spheres[2].GetW()) {
        if (p_gate_spheres[1].GetW() < p_gate_spheres[0].GetW()) {
            pivot = vec3d(p_gate_spheres[1].PeekComponents());
        }

    } else {
        if (p_gate_spheres[2].GetW() < p_gate_spheres[0].GetW()) {
            pivot = vec3d(p_gate_spheres[2].PeekComponents());
        }
    }

    return pivot;
}

/*
 * Computations::ComputeVoronoiCircle
 */
size_t Computations::ComputeVoronoiCircle(
    const std::array<vec3d, 3>& p_circles, std::array<vec3d, 2>& p_result_circles) {
    // Look for the smallest radius.
    double radius = p_circles[0].GetZ();
    size_t circle_id = 0;
    for (size_t i = 1; i < p_circles.size(); i++) {
        if (p_circles[i].GetZ() < radius) {
            radius = p_circles[i].GetZ();
            circle_id = i;
        }
    }

    // Create linear system matrix and compute the right hand side.
    mat2x3d A;
    evec2d rhs = evec2d(0.0, 0.0);
    size_t row = 0;
    for (size_t i = 0; i < p_circles.size(); i++) {
        if (i == circle_id)
            continue;
        for (size_t j = 0; j < 3; j++) {
            // Compute the value.
            double val = p_circles[i].PeekComponents()[j] - p_circles[circle_id].PeekComponents()[j];

            // Add it to the matrix.
            A(row, j) = 2.0f * val;

            if (j != 2) {
                // Add it to the right hand side.
                rhs[row] += (val * val);

            } else {
                // Add it to the right hand side.
                rhs[row] -= val * val;
            }
        }
        row++;
    }

    // Look for the first determinate that is non zero.
    double det;
    uint free_variable;
    std::array<double, 4> m_values;
    std::array<double, 2> ui_values;
    uint zero_cnt = 0;
    for (size_t i = 0; i < 3; i++) {
        emat2d submatrix;
        Computations::splitMatrix(A, submatrix, i);
        det = submatrix.determinant();
        if (det != 0.0) {
            // Remember the matrix.
            memcpy(m_values.data(), submatrix.data(), 4 * sizeof(double));

            switch (i) {
            case 0:
                // R is the free variable set the U vector.
                memcpy(ui_values.data(), A.data() + 4, 2 * sizeof(double));
                free_variable = 0;
                i = 4;
                break;
            case 1:
                // Z is the free variable set the U vector.
                memcpy(ui_values.data(), A.data() + 2, 2 * sizeof(double));
                free_variable = 1;
                i = 4;
                break;
            case 2:
                // Y is the free variable set the U vector.
                memcpy(ui_values.data(), A.data(), 2 * sizeof(double));
                free_variable = 2;
                i = 4;
                break;
            default:
                // This should never happen so return that we have 0 valid circles.
                return 0;
            }
        } else {
            zero_cnt++;
        }
    }

    if (zero_cnt > 2) {
        // Return that we have 0 valid circles.
        return 0;
    }

    // Remember everything and precompute certain values.
    // Get the values from the inverse matrix.
    double inv_det = 1.0 / det;
    double m_1 = m_values[3] * inv_det;
    double m_2 = -1.0 * m_values[1] * inv_det;
    double m_3 = -1.0 * m_values[2] * inv_det;
    double m_4 = m_values[0] * inv_det;

    // Get the values from the ui vector.
    double v_1 = ui_values[0];
    double v_2 = ui_values[1];

    // Compute the values that are used more than once.
    double m_1_sq = m_1 * m_1;
    double m_2_sq = m_2 * m_2;
    double m_3_sq = m_3 * m_3;
    double m_4_sq = m_4 * m_4;
    double v_1_sq = v_1 * v_1;
    double v_2_sq = v_2 * v_2;
    double m_1_m_3 = m_1 * m_3;
    double m_2_m_4 = m_2 * m_4;
    double rhs_0_sq = rhs(0) * rhs(0);
    double rhs_1_sq = rhs(1) * rhs(1);

    // Compute the a, b and c values for each of the aX² + bx + c terms. For each dependent variable we have one of
    // these terms.
    double a = m_1_sq * v_1_sq + m_3_sq * v_2_sq + 2.0 * m_1_m_3 * v_1 * v_2;
    double d = m_2_sq * v_1_sq + m_4_sq * v_2_sq + 2.0 * m_2_m_4 * v_1 * v_2;
    double b = (-m_1_sq * v_1 * rhs(0) - m_1_m_3 * v_1 * rhs(1) - m_1_m_3 * v_2 * rhs(0) - m_3_sq * v_2 * rhs(1)) * 2.0;
    double e = (-m_2_sq * v_1 * rhs(0) - m_2_m_4 * v_1 * rhs(1) - m_2_m_4 * v_2 * rhs(0) - m_4_sq * v_2 * rhs(1)) * 2.0;
    double c = m_1_sq * rhs_0_sq + m_3_sq * rhs_1_sq + 2.0 * m_1_m_3 * rhs(0) * rhs(1);
    double f = m_2_sq * rhs_0_sq + m_4_sq * rhs_1_sq + 2.0 * m_2_m_4 * rhs(0) * rhs(1);


    // Set the correct order for the computation of the polynom X² + Y² - R² = 0. The independent variable has the value
    // of 1 the the others are a and d.
    double m_a;
    switch (free_variable) {
    case 0:
        m_a = a + d - 1.0;
        break;
    case 1:
        m_a = a + 1.0 - d;
        break;
    case 2:
        m_a = 1.0 + a - d;
        break;
    default:
        // This should never happen so return that we have 0 valid circles.
        return 0;
    }

    // Set the a, b and c values for the "Mitternachts-Formel".
    double m_b = b + e;
    double m_c = c + f;

    // Compute the two solutions for the "Mitternachts-Formel".
    double inner = m_b * m_b - 4.0 * m_a * m_c;
    if (inner < 0.0) {
        // If the value under the square root is negative, there is no real solution.
        return 0;
    }
    double root = sqrt(inner);
    double denom = 1.0 / (2.0 * m_a);
    double solution_0 = (-m_b + root) * denom;
    double solution_1 = (-m_b - root) * denom;

    // Compute the new sphere based on the free variable and the solutions.
    double b_x_0;
    double b_y_0;
    double v_x_0;
    double v_y_0;
    double b_x_1;
    double b_y_1;
    double v_x_1;
    double v_y_1;
    uint results = 0;
    switch (free_variable) {
    case 0:
        // The R is the free variable, add all possible circles with radius > 0 to the result
        if (solution_0 > 0.0) {
            b_x_0 = rhs(0) - v_1 * solution_0;
            b_y_0 = rhs(1) - v_2 * solution_0;
            v_x_0 = (m_1 * b_x_0 + m_3 * b_y_0) + p_circles[circle_id].GetX();
            v_y_0 = (m_2 * b_x_0 + m_4 * b_y_0) + p_circles[circle_id].GetY();
            solution_0 -= p_circles[circle_id].GetZ();
            if (solution_0 > 0.0) {
                p_result_circles[results] = vec3d(v_x_0, v_y_0, solution_0);
                results++;
            }
        }
        if (solution_1 > 0.0) {
            b_x_1 = rhs(0) - v_1 * solution_1;
            b_y_1 = rhs(1) - v_2 * solution_1;
            v_x_1 = (m_1 * b_x_1 + m_3 * b_y_1) + p_circles[circle_id].GetX();
            v_y_1 = (m_2 * b_x_1 + m_4 * b_y_1) + p_circles[circle_id].GetY();
            solution_1 -= p_circles[circle_id].GetZ();
            if (solution_1 > 0.0) {
                p_result_circles[results] = vec3d(v_x_1, v_y_1, solution_1);
                results++;
            }
        }

        // If there are two results, take the one with smaller radius first.
        if (results > 1 && p_result_circles[0].GetZ() > p_result_circles[1].GetZ()) {
            std::swap(p_result_circles[0], p_result_circles[1]);
        }

        return results;

    case 1:
        // The Y is the free variable so choose the smallest not negative radius and return the new point.
        // Compute the first new sphere.
        b_x_0 = rhs(0) - v_1 * solution_0;
        b_y_0 = rhs(1) - v_2 * solution_0;
        v_x_0 = m_1 * b_x_0 + m_3 * b_y_0;
        v_y_0 = m_2 * b_x_0 + m_4 * b_y_0;

        // Compute the second new sphere.
        b_x_1 = rhs(0) - v_1 * solution_1;
        b_y_1 = rhs(1) - v_2 * solution_1;
        v_x_1 = m_1 * b_x_1 + m_3 * b_y_1;
        v_y_1 = m_2 * b_x_1 + m_4 * b_y_1;

        // Check the radii of both solutions and add the corresponding spheres to the result if they are not negative.
        if (v_y_0 > 0.0) {
            v_x_0 += p_circles[circle_id].GetX();
            solution_0 += p_circles[circle_id].GetY();
            v_y_0 -= p_circles[circle_id].GetZ();
            if (v_y_0 > 0.0) {
                p_result_circles[results] = vec3d(v_x_0, solution_0, v_y_0);
                results++;
            }
        }
        if (v_y_1 > 0.0) {
            v_x_1 += p_circles[circle_id].GetX();
            solution_1 += p_circles[circle_id].GetY();
            v_y_1 -= p_circles[circle_id].GetZ();
            if (v_y_1 > 0.0) {
                p_result_circles[results] = vec3d(v_x_1, solution_1, v_y_1);
                results++;
            }
        }

        // If there are two results, take the one with smaller radius first.
        if (results > 1 && p_result_circles[0].GetZ() > p_result_circles[1].GetZ()) {
            std::swap(p_result_circles[0], p_result_circles[1]);
        }

        return results;

    case 2:
        // The x is the free variable so choose the not negative radius and return the new point.
        // Compute the first new sphere.
        b_x_0 = rhs(0) - v_1 * solution_0;
        b_y_0 = rhs(1) - v_2 * solution_0;
        v_x_0 = m_1 * b_x_0 + m_3 * b_y_0;
        v_y_0 = m_2 * b_x_0 + m_4 * b_y_0;

        // Compute the second new sphere.
        b_x_1 = rhs(0) - v_1 * solution_1;
        b_y_1 = rhs(1) - v_2 * solution_1;
        v_x_1 = m_1 * b_x_1 + m_3 * b_y_1;
        v_y_1 = m_2 * b_x_1 + m_4 * b_y_1;

        // Check the radii of both solutions and add the corresponding spheres to the result if they are not negative.
        if (v_y_0 > 0.0) {
            solution_0 += p_circles[circle_id].GetX();
            v_x_0 += p_circles[circle_id].GetY();
            v_y_0 -= p_circles[circle_id].GetZ();
            if (v_y_0 > 0.0) {
                p_result_circles[results] = vec3d(solution_0, v_x_0, v_y_0);
                results++;
            }
        }
        if (v_y_1 > 0.0) {
            solution_1 += p_circles[circle_id].GetX();
            v_x_1 += p_circles[circle_id].GetY();
            v_y_1 -= p_circles[circle_id].GetZ();
            if (v_y_1 > 0.0) {
                p_result_circles[results] = vec3d(solution_1, v_x_1, v_y_1);
                results++;
            }
        }

        // If there are two results, take the one with smaller radius first.
        if (results > 1 && p_result_circles[0].GetZ() > p_result_circles[1].GetZ()) {
            std::swap(p_result_circles[0], p_result_circles[1]);
        }

        return results;

    default:
        // This should never happen so return that we have 0 valid circles.
        return 0;
    }

    // This should never happen so return that we have 0 valid circles.
    return 0;
}

/*
 * Computations::ComputeVoronoiSphere
 */
uint Computations::ComputeVoronoiSphere(const std::array<vec4d, 4>& p_spheres, std::array<vec4d, 2>& p_result_spheres) {
    // Look for the smallest radius.
    double radius = p_spheres[0].GetW();
    size_t sphere_id = 0;
    for (size_t i = 1; i < p_spheres.size(); i++) {
        if (p_spheres[i].GetW() < radius) {
            radius = p_spheres[i].GetW();
            sphere_id = i;
        }
    }

    // Create linear system matrix and compute the right hand side.
    mat3x4d A;
    evec3d rhs = evec3d(0.0, 0.0, 0.0);
    size_t row = 0;
    for (size_t i = 0; i < p_spheres.size(); i++) {
        if (i == sphere_id)
            continue;
        for (size_t j = 0; j < 4; j++) {
            // Compute the value.
            double val = p_spheres[i].PeekComponents()[j] - p_spheres[sphere_id].PeekComponents()[j];

            // Add it to the matrix.
            A(row, j) = 2.0f * val;

            // Add it to the right hand side.
            if (j != 3) {
                rhs[row] += (val * val);

            } else {
                rhs[row] -= (val * val);
            }
        }
        row++;
    }

    // Look for the first determinate that is non zero.
    double det;
    uint free_variable;
    std::array<double, 9> m_values;
    std::array<double, 3> ui_values;
    uint zeroCount = 0;
    for (size_t i = 0; i < 4; i++) {
        emat3d submatrix;
        Computations::splitMatrix(A, submatrix, i);
        det = submatrix.determinant();
        if (det != 0.0) {
            // Remember the matrix.
            memcpy(m_values.data(), submatrix.data(), 9 * sizeof(double));

            switch (i) {
            case 0:
                // R is the free variable set the U vector.
                memcpy(ui_values.data(), A.data() + 9, 3 * sizeof(double));
                free_variable = 0;
                i = 5;
                break;
            case 1:
                // Z is the free variable set the U vector.
                memcpy(ui_values.data(), A.data() + 6, 3 * sizeof(double));
                free_variable = 1;
                i = 5;
                break;
            case 2:
                // Y is the free variable set the U vector.
                memcpy(ui_values.data(), A.data() + 3, 3 * sizeof(double));
                free_variable = 2;
                i = 5;
                break;
            case 3:
                // X is the free variable set the U vector.
                memcpy(ui_values.data(), A.data(), 3 * sizeof(double));
                free_variable = 3;
                i = 5;
                break;
            default:
                // This should never happen so return that we have 0 valid spheres.
                return 0;
            }

        } else {
            zeroCount++;
        }
    }

    if (zeroCount > 3) {
        // Return that we have 0 valid spheres.
        return 0;
    }

    // Remember everything and precompute certain values.
    // Get the values from the inverse matrix.
    double inv_det = 1.0 / det;
    double m_1 = (m_values[4] * m_values[8] - m_values[5] * m_values[7]) * inv_det;
    double m_4 = (m_values[6] * m_values[5] - m_values[3] * m_values[8]) * inv_det;
    double m_7 = (m_values[3] * m_values[7] - m_values[6] * m_values[4]) * inv_det;
    double m_2 = (m_values[7] * m_values[2] - m_values[1] * m_values[8]) * inv_det;
    double m_5 = (m_values[0] * m_values[8] - m_values[6] * m_values[2]) * inv_det;
    double m_8 = (m_values[1] * m_values[6] - m_values[0] * m_values[7]) * inv_det;
    double m_3 = (m_values[1] * m_values[5] - m_values[2] * m_values[4]) * inv_det;
    double m_6 = (m_values[2] * m_values[3] - m_values[0] * m_values[5]) * inv_det;
    double m_9 = (m_values[0] * m_values[4] - m_values[1] * m_values[3]) * inv_det;

    // Get the values from the ui vector.
    double v_1 = ui_values[0];
    double v_2 = ui_values[1];
    double v_3 = ui_values[2];

    // Compute the values that are used more than once.
    double m_1_sq = m_1 * m_1;
    double m_2_sq = m_2 * m_2;
    double m_3_sq = m_3 * m_3;
    double m_4_sq = m_4 * m_4;
    double m_5_sq = m_5 * m_5;
    double m_6_sq = m_6 * m_6;
    double m_7_sq = m_7 * m_7;
    double m_8_sq = m_8 * m_8;
    double m_9_sq = m_9 * m_9;
    double r_1_sq = v_1 * v_1;
    double r_2_sq = v_2 * v_2;
    double r_3_sq = v_3 * v_3;
    double b_0_sq = rhs(0) * rhs(0);
    double b_1_sq = rhs(1) * rhs(1);
    double b_2_sq = rhs(2) * rhs(2);
    double m_1_m_4 = m_1 * m_4;
    double m_1_m_7 = m_1 * m_7;
    double m_2_m_5 = m_2 * m_5;
    double m_2_m_8 = m_2 * m_8;
    double m_3_m_6 = m_3 * m_6;
    double m_3_m_9 = m_3 * m_9;
    double m_4_m_7 = m_4 * m_7;
    double m_5_m_8 = m_5 * m_8;
    double m_6_m_9 = m_6 * m_9;
    double v_1_v_2 = v_1 * v_2;
    double v_1_v_3 = v_1 * v_3;
    double v_2_v_3 = v_2 * v_3;
    double v_1_b_0 = v_1 * rhs(0);
    double v_1_b_1 = v_1 * rhs(1);
    double v_1_b_2 = v_1 * rhs(2);
    double v_2_b_0 = v_2 * rhs(0);
    double v_2_b_1 = v_2 * rhs(1);
    double v_2_b_2 = v_2 * rhs(2);
    double v_3_b_0 = v_3 * rhs(0);
    double v_3_b_1 = v_3 * rhs(1);
    double v_3_b_2 = v_3 * rhs(2);
    double b_0_b_1 = rhs(0) * rhs(1);
    double b_0_b_2 = rhs(0) * rhs(2);
    double b_2_b_1 = rhs(2) * rhs(1);

    // Compute the a, b and c values for each of the aX² + bx + c terms. For each dependent variable we have one of
    // these terms.
    double a = m_1_sq * r_1_sq + m_4_sq * r_2_sq + m_7_sq * r_3_sq +
               2.0 * (m_1_m_4 * v_1_v_2 + m_1_m_7 * v_1_v_3 + m_4_m_7 * v_2_v_3);
    double b = (-m_1_sq * v_1_b_0 - m_1_m_4 * v_1_b_1 - m_1_m_4 * v_2_b_0 - m_1_m_7 * v_1_b_2 - m_1_m_7 * v_3_b_0 -
                   m_4_sq * v_2_b_1 - m_4_m_7 * v_2_b_2 - m_4_m_7 * v_3_b_1 - m_7_sq * v_3_b_2) *
               2.0;
    double c = m_1_sq * b_0_sq + m_4_sq * b_1_sq + m_7_sq * b_2_sq +
               2.0 * (m_1_m_7 * b_0_b_2 + m_4_m_7 * b_2_b_1 + m_1_m_4 * b_0_b_1);

    double d = m_2_sq * r_1_sq + m_5_sq * r_2_sq + m_8_sq * r_3_sq +
               2.0 * (m_2_m_5 * v_1_v_2 + m_2_m_8 * v_1_v_3 + m_5_m_8 * v_2_v_3);
    double e = (-m_2_sq * v_1_b_0 - m_2_m_5 * v_1_b_1 - m_2_m_5 * v_2_b_0 - m_2_m_8 * v_1_b_2 - m_2_m_8 * v_3_b_0 -
                   m_5_sq * v_2_b_1 - m_5_m_8 * v_2_b_2 - m_5_m_8 * v_3_b_1 - m_8_sq * v_3_b_2) *
               2.0;
    double f = m_2_sq * b_0_sq + m_5_sq * b_1_sq + m_8_sq * b_2_sq +
               2.0 * (m_2_m_8 * b_0_b_2 + m_5_m_8 * b_2_b_1 + m_2_m_5 * b_0_b_1);

    double g = m_3_sq * r_1_sq + m_6_sq * r_2_sq + m_9_sq * r_3_sq +
               2.0 * (m_3_m_6 * v_1_v_2 + m_3_m_9 * v_1_v_3 + m_6_m_9 * v_2_v_3);
    double h = (-m_3_sq * v_1_b_0 - m_3_m_6 * v_1_b_1 - m_3_m_6 * v_2_b_0 - m_3_m_9 * v_1_b_2 - m_3_m_9 * v_3_b_0 -
                   m_6_sq * v_2_b_1 - m_6_m_9 * v_2_b_2 - m_6_m_9 * v_3_b_1 - m_9_sq * v_3_b_2) *
               2.0;
    double i = m_3_sq * b_0_sq + m_6_sq * b_1_sq + m_9_sq * b_2_sq +
               2.0 * (m_3_m_9 * b_0_b_2 + m_6_m_9 * b_2_b_1 + m_3_m_6 * b_0_b_1);

    // Set the correct order for the computation of the polynom X² + Y² + Z² - R² = 0. The independent variable has the
    // value of 1 the the others are a, d and g.
    double m_a;
    switch (free_variable) {
    case 0:
        m_a = a + d + g - 1.0;
        break;
    case 1:
        m_a = a + d + 1.0 - g;
        break;
    case 2:
        m_a = a + 1.0 + d - g;
        break;
    case 3:
        m_a = 1.0 + a + d - g;
        break;
    default:
        // This should never happen so return that we have 0 valid spheres.
        return 0;
    }

    // Set the b and c values for the "Mitternachts-Formel".
    double m_b = b + e + h;
    double m_c = c + f + i;

    // Compute the two solutions for the "Mitternachtsformel".
    double inner = m_b * m_b - 4.0 * m_a * m_c;
    if (inner < 0.0) {
        // If the value under the square root is negative, there is no real solution.
        return 0;
    }
    double root = sqrt(inner);
    double denom = 2.0 * m_a;
    double solution_0 = (-m_b + root) / denom;
    double solution_1 = (-m_b - root) / denom;

    // Compute the new sphere based on the free variable and the solutions.
    double b_x_0;
    double b_y_0;
    double b_z_0;
    double v_x_0;
    double v_y_0;
    double v_z_0;
    double b_x_1;
    double b_y_1;
    double b_z_1;
    double v_x_1;
    double v_y_1;
    double v_z_1;
    uint results = 0;
    switch (free_variable) {
    case 0:
        // The R is the free variable so choose the not negative radius and return the new point.
        // Check the radii of both solutions and add the corresponding spheres to the result if they are not negative.
        if (solution_0 > 0.0) {
            // Compute the new point.
            b_x_0 = rhs(0) - v_1 * solution_0;
            b_y_0 = rhs(1) - v_2 * solution_0;
            b_z_0 = rhs(2) - v_3 * solution_0;
            v_x_0 = (m_1 * b_x_0 + m_4 * b_y_0 + m_7 * b_z_0) + p_spheres[sphere_id][0];
            v_y_0 = (m_2 * b_x_0 + m_5 * b_y_0 + m_8 * b_z_0) + p_spheres[sphere_id][1];
            v_z_0 = (m_3 * b_x_0 + m_6 * b_y_0 + m_9 * b_z_0) + p_spheres[sphere_id][2];
            solution_0 -= p_spheres[sphere_id][3];
            if (solution_0 > 0.0) {
                p_result_spheres[results] = vec4d(v_x_0, v_y_0, v_z_0, solution_0);
                results++;
            }
        }
        if (solution_1 > 0.0) {
            b_x_1 = rhs(0) - v_1 * solution_1;
            b_y_1 = rhs(1) - v_2 * solution_1;
            b_z_1 = rhs(2) - v_3 * solution_1;
            v_x_1 = (m_1 * b_x_1 + m_4 * b_y_1 + m_7 * b_z_1) + p_spheres[sphere_id][0];
            v_y_1 = (m_2 * b_x_1 + m_5 * b_y_1 + m_8 * b_z_1) + p_spheres[sphere_id][1];
            v_z_1 = (m_3 * b_x_1 + m_6 * b_y_1 + m_9 * b_z_1) + p_spheres[sphere_id][2];
            solution_1 -= p_spheres[sphere_id][3];
            if (solution_1 > 0.0) {
                p_result_spheres[results] = vec4d(v_x_1, v_y_1, v_z_1, solution_1);
                results++;
            }
        }

        // If there are two results, take the one with smaller radius first.
        if (results > 1 && p_result_spheres[0].GetW() > p_result_spheres[1].GetW()) {
            std::swap(p_result_spheres[0], p_result_spheres[1]);
        }

        return results;

    case 1:
        // The Z is the free variable so choose the smallest not negative radius and return the new point.
        // Compute the first new sphere.
        b_x_0 = rhs(0) - v_1 * solution_0;
        b_y_0 = rhs(1) - v_2 * solution_0;
        b_z_0 = rhs(2) - v_3 * solution_0;
        v_x_0 = m_1 * b_x_0 + m_4 * b_y_0 + m_7 * b_z_0;
        v_y_0 = m_2 * b_x_0 + m_5 * b_y_0 + m_8 * b_z_0;
        v_z_0 = m_3 * b_x_0 + m_6 * b_y_0 + m_9 * b_z_0;

        // Compute the second new sphere.
        b_x_1 = rhs(0) - v_1 * solution_1;
        b_y_1 = rhs(1) - v_2 * solution_1;
        b_z_1 = rhs(2) - v_3 * solution_1;
        v_x_1 = m_1 * b_x_1 + m_4 * b_y_1 + m_7 * b_z_1;
        v_y_1 = m_2 * b_x_1 + m_5 * b_y_1 + m_8 * b_z_1;
        v_z_1 = m_3 * b_x_1 + m_6 * b_y_1 + m_9 * b_z_1;

        // Check the radii of both solutions and add the corresponding spheres to the result if they are not negative.
        if (v_z_0 > 0.0) {
            v_x_0 += p_spheres[sphere_id][0];
            v_y_0 += p_spheres[sphere_id][1];
            solution_0 += p_spheres[sphere_id][2];
            v_z_0 -= p_spheres[sphere_id][3];
            if (v_z_0 > 0.0) {
                p_result_spheres[results] = vec4d(v_x_0, v_y_0, solution_0, v_z_0);
                results++;
            }
        }
        if (v_z_1 > 0.0) {
            v_x_1 += p_spheres[sphere_id][0];
            v_y_1 += p_spheres[sphere_id][1];
            solution_1 += p_spheres[sphere_id][2];
            v_z_1 -= p_spheres[sphere_id][3];
            if (v_z_1 > 0.0) {
                p_result_spheres[results] = vec4d(v_x_1, v_y_1, solution_1, v_z_1);
                results++;
            }
        }

        // If there are two results, take the one with smaller radius first.
        if (results > 1 && p_result_spheres[0].GetW() > p_result_spheres[1].GetW()) {
            std::swap(p_result_spheres[0], p_result_spheres[1]);
        }

        return results;

    case 2:
        // The Y is the free variable so choose the not negative radius and return the new point.
        // Compute the first new sphere.
        b_x_0 = rhs(0) - v_1 * solution_0;
        b_y_0 = rhs(1) - v_2 * solution_0;
        b_z_0 = rhs(2) - v_3 * solution_0;
        v_x_0 = m_1 * b_x_0 + m_4 * b_y_0 + m_7 * b_z_0;
        v_y_0 = m_2 * b_x_0 + m_5 * b_y_0 + m_8 * b_z_0;
        v_z_0 = m_3 * b_x_0 + m_6 * b_y_0 + m_9 * b_z_0;

        // Compute the second new sphere.
        b_x_1 = rhs(0) - v_1 * solution_1;
        b_y_1 = rhs(1) - v_2 * solution_1;
        b_z_1 = rhs(2) - v_3 * solution_1;
        v_x_1 = m_1 * b_x_1 + m_4 * b_y_1 + m_7 * b_z_1;
        v_y_1 = m_2 * b_x_1 + m_5 * b_y_1 + m_8 * b_z_1;
        v_z_1 = m_3 * b_x_1 + m_6 * b_y_1 + m_9 * b_z_1;

        // Check the radii of both solutions and add the corresponding spheres to the result if they are not negative.
        if (v_z_0 > 0.0) {
            v_x_0 += p_spheres[sphere_id][0];
            solution_0 += p_spheres[sphere_id][1];
            v_y_0 += p_spheres[sphere_id][2];
            v_z_0 -= p_spheres[sphere_id][3];
            if (v_z_0 > 0.0) {
                p_result_spheres[results] = vec4d(v_x_0, solution_0, v_y_0, v_z_0);
                results++;
            }
        }
        if (v_z_1 > 0.0) {
            v_x_1 += p_spheres[sphere_id][0];
            solution_1 += p_spheres[sphere_id][1];
            v_y_1 += p_spheres[sphere_id][2];
            v_z_1 -= p_spheres[sphere_id][3];
            if (v_z_1 > 0.0) {
                p_result_spheres[results] = vec4d(v_x_1, solution_1, v_y_1, v_z_1);
                results++;
            }
        }

        // If there are two results, take the one with smaller radius first.
        if (results > 1 && p_result_spheres[0].GetW() > p_result_spheres[1].GetW()) {
            std::swap(p_result_spheres[0], p_result_spheres[1]);
        }

        return results;

    case 3:
        // The X is the free variable so choose the not negative radius and return the new point.
        // Compute the first new sphere.
        b_x_0 = rhs(0) - v_1 * solution_0;
        b_y_0 = rhs(1) - v_2 * solution_0;
        b_z_0 = rhs(2) - v_3 * solution_0;
        v_x_0 = m_1 * b_x_0 + m_4 * b_y_0 + m_7 * b_z_0;
        v_y_0 = m_2 * b_x_0 + m_5 * b_y_0 + m_8 * b_z_0;
        v_z_0 = m_3 * b_x_0 + m_6 * b_y_0 + m_9 * b_z_0;

        // Compute the second new sphere.
        b_x_1 = rhs(0) - v_1 * solution_1;
        b_y_1 = rhs(1) - v_2 * solution_1;
        b_z_1 = rhs(2) - v_3 * solution_1;
        v_x_1 = m_1 * b_x_1 + m_4 * b_y_1 + m_7 * b_z_1;
        v_y_1 = m_2 * b_x_1 + m_5 * b_y_1 + m_8 * b_z_1;
        v_z_1 = m_3 * b_x_1 + m_6 * b_y_1 + m_9 * b_z_1;

        // Check the radii of both solutions and add the corresponding spheres to the result if they are not negative.
        if (v_z_0 > 0.0) {
            solution_0 += p_spheres[sphere_id][0];
            v_x_0 += p_spheres[sphere_id][1];
            v_y_0 += p_spheres[sphere_id][2];
            v_z_0 -= p_spheres[sphere_id][3];
            if (v_z_0 > 0.0) {
                p_result_spheres[results] = vec4d(solution_0, v_x_0, v_y_0, v_z_0);
                results++;
            }
        }
        if (v_z_1 > 0.0) {
            solution_1 += p_spheres[sphere_id][0];
            v_x_1 += p_spheres[sphere_id][1];
            v_y_1 += p_spheres[sphere_id][2];
            v_z_1 -= p_spheres[sphere_id][3];
            if (v_z_1 > 0.0) {
                p_result_spheres[results] = vec4d(solution_1, v_x_1, v_y_1, v_z_1);
                results++;
            }
        }

        // If there are two results, take the one with smaller radius first.
        if (results > 1 && p_result_spheres[0].GetW() > p_result_spheres[1].GetW()) {
            std::swap(p_result_spheres[0], p_result_spheres[1]);
        }

        return results;

    default:
        // This should never happen so return that we have 0 valid spheres.
        return 0;
    }

    // This should never happen so return that we have 0 valid spheres.
    return 0;
}

/*
 * Computations::ComputeVoronoiSphereR
 */
uint Computations::ComputeVoronoiSphereR(
    const std::array<vec4d, 4>& p_spheres, std::array<vec4d, 2>& p_result_spheres) {
    // Look for the smallest radius.
    double radius = p_spheres[0].GetW();
    size_t sphere_id = 0;
    for (size_t i = 1; i < p_spheres.size(); i++) {
        if (p_spheres[i].GetW() < radius) {
            radius = p_spheres[i].GetW();
            sphere_id = i;
        }
    }

    // Create linear system matrix and compute the right hand side.
    std::array<double, 12> A;
    std::array<double, 3> rhs{0.0, 0.0, 0.0};
    size_t row = 0;
    for (size_t i = 0; i < p_spheres.size(); i++) {
        if (i == sphere_id)
            continue;
        for (size_t j = 0; j < 4; j++) {
            // Compute the value.
            double val = p_spheres[i].PeekComponents()[j] - p_spheres[sphere_id].PeekComponents()[j];

            // Add it to the matrix.
            A[j * 3 + row] = 2.0f * val;

            // Add it to the right hand side.
            if (j != 3) {
                rhs[row] += (val * val);

            } else {
                rhs[row] -= (val * val);
            }
        }
        row++;
    }

    // Set the radius to the free variable and compute the determinate of the submatrix of A.
    double det = A[0] * A[4] * A[8] + A[3] * A[7] * A[2] + A[6] * A[1] * A[5] - A[2] * A[4] * A[6] -
                 A[5] * A[7] * A[0] - A[8] * A[1] * A[3];
    double inv_det = 1.0 / det;

    // Remember everything and precompute certain values.
    // Get the values from the inverse matrix.
    std::array<double, 9> m_values;
    memcpy(m_values.data(), A.data(), 9 * sizeof(double));
    double m_1 = (m_values[4] * m_values[8] - m_values[5] * m_values[7]) * inv_det;
    double m_4 = (m_values[6] * m_values[5] - m_values[3] * m_values[8]) * inv_det;
    double m_7 = (m_values[3] * m_values[7] - m_values[6] * m_values[4]) * inv_det;
    double m_2 = (m_values[7] * m_values[2] - m_values[1] * m_values[8]) * inv_det;
    double m_5 = (m_values[0] * m_values[8] - m_values[6] * m_values[2]) * inv_det;
    double m_8 = (m_values[1] * m_values[6] - m_values[0] * m_values[7]) * inv_det;
    double m_3 = (m_values[1] * m_values[5] - m_values[2] * m_values[4]) * inv_det;
    double m_6 = (m_values[2] * m_values[3] - m_values[0] * m_values[5]) * inv_det;
    double m_9 = (m_values[0] * m_values[4] - m_values[1] * m_values[3]) * inv_det;

    // Compute the values that are used more than once based on the inverse matrix.
    double m_1_sq = m_1 * m_1;
    double m_2_sq = m_2 * m_2;
    double m_3_sq = m_3 * m_3;
    double m_4_sq = m_4 * m_4;
    double m_5_sq = m_5 * m_5;
    double m_6_sq = m_6 * m_6;
    double m_7_sq = m_7 * m_7;
    double m_8_sq = m_8 * m_8;
    double m_9_sq = m_9 * m_9;
    double m_1_m_4 = m_1 * m_4;
    double m_1_m_7 = m_1 * m_7;
    double m_2_m_5 = m_2 * m_5;
    double m_2_m_8 = m_2 * m_8;
    double m_3_m_6 = m_3 * m_6;
    double m_3_m_9 = m_3 * m_9;
    double m_4_m_7 = m_4 * m_7;
    double m_5_m_8 = m_5 * m_8;
    double m_6_m_9 = m_6 * m_9;

    // Get the values from the ui vector.
    std::array<double, 3> ui_values;
    memcpy(ui_values.data(), A.data() + 9, 3 * sizeof(double));
    double v_1 = ui_values[0];
    double v_2 = ui_values[1];
    double v_3 = ui_values[2];

    // Compute the values that are used more than once based on the ui vector.
    double r_1_sq = v_1 * v_1;
    double r_2_sq = v_2 * v_2;
    double r_3_sq = v_3 * v_3;
    double v_1_v_2 = v_1 * v_2;
    double v_1_v_3 = v_1 * v_3;
    double v_2_v_3 = v_2 * v_3;
    double v_1_b_0 = v_1 * rhs[0];
    double v_1_b_1 = v_1 * rhs[1];
    double v_1_b_2 = v_1 * rhs[2];
    double v_2_b_0 = v_2 * rhs[0];
    double v_2_b_1 = v_2 * rhs[1];
    double v_2_b_2 = v_2 * rhs[2];
    double v_3_b_0 = v_3 * rhs[0];
    double v_3_b_1 = v_3 * rhs[1];
    double v_3_b_2 = v_3 * rhs[2];
    double b_0_b_1 = rhs[0] * rhs[1];
    double b_0_b_2 = rhs[0] * rhs[2];
    double b_2_b_1 = rhs[2] * rhs[1];
    double b_0_sq = rhs[0] * rhs[0];
    double b_1_sq = rhs[1] * rhs[1];
    double b_2_sq = rhs[2] * rhs[2];

    // Compute the a, b and c values for each of the aX² + bx + c terms. For each dependent variable we have one of
    // these terms.
    double a = m_1_sq * r_1_sq + m_4_sq * r_2_sq + m_7_sq * r_3_sq +
               2.0 * (m_1_m_4 * v_1_v_2 + m_1_m_7 * v_1_v_3 + m_4_m_7 * v_2_v_3);
    double d = m_2_sq * r_1_sq + m_5_sq * r_2_sq + m_8_sq * r_3_sq +
               2.0 * (m_2_m_5 * v_1_v_2 + m_2_m_8 * v_1_v_3 + m_5_m_8 * v_2_v_3);
    double g = m_3_sq * r_1_sq + m_6_sq * r_2_sq + m_9_sq * r_3_sq +
               2.0 * (m_3_m_6 * v_1_v_2 + m_3_m_9 * v_1_v_3 + m_6_m_9 * v_2_v_3);

    double b = (-m_1_sq * v_1_b_0 - m_1_m_4 * v_1_b_1 - m_1_m_4 * v_2_b_0 - m_1_m_7 * v_1_b_2 - m_1_m_7 * v_3_b_0 -
                   m_4_sq * v_2_b_1 - m_4_m_7 * v_2_b_2 - m_4_m_7 * v_3_b_1 - m_7_sq * v_3_b_2) *
               2.0;
    double e = (-m_2_sq * v_1_b_0 - m_2_m_5 * v_1_b_1 - m_2_m_5 * v_2_b_0 - m_2_m_8 * v_1_b_2 - m_2_m_8 * v_3_b_0 -
                   m_5_sq * v_2_b_1 - m_5_m_8 * v_2_b_2 - m_5_m_8 * v_3_b_1 - m_8_sq * v_3_b_2) *
               2.0;
    double h = (-m_3_sq * v_1_b_0 - m_3_m_6 * v_1_b_1 - m_3_m_6 * v_2_b_0 - m_3_m_9 * v_1_b_2 - m_3_m_9 * v_3_b_0 -
                   m_6_sq * v_2_b_1 - m_6_m_9 * v_2_b_2 - m_6_m_9 * v_3_b_1 - m_9_sq * v_3_b_2) *
               2.0;

    double c = m_1_sq * b_0_sq + m_4_sq * b_1_sq + m_7_sq * b_2_sq +
               2.0 * (m_1_m_7 * b_0_b_2 + m_4_m_7 * b_2_b_1 + m_1_m_4 * b_0_b_1);
    double f = m_2_sq * b_0_sq + m_5_sq * b_1_sq + m_8_sq * b_2_sq +
               2.0 * (m_2_m_8 * b_0_b_2 + m_5_m_8 * b_2_b_1 + m_2_m_5 * b_0_b_1);
    double i = m_3_sq * b_0_sq + m_6_sq * b_1_sq + m_9_sq * b_2_sq +
               2.0 * (m_3_m_9 * b_0_b_2 + m_6_m_9 * b_2_b_1 + m_3_m_6 * b_0_b_1);

    // Set the correct order for the computation of the polynom X² + Y² + Z² - R² = 0. The independent variable has the
    // value of 1 the the others are a, d and g.
    double m_a = a + d + g - 1.0;

    // Set the b and c values for the "Mitternachts-Formel".
    double m_b = b + e + h;
    double m_c = c + f + i;

    // Compute the two solutions for the "Mitternachtsformel".
    double inner = m_b * m_b - 4.0 * m_a * m_c;
    if (inner < 0.0) {
        // If the value under the square root is negative, there is no real solution.
        return 0;
    }
    double root = sqrt(inner);
    double denom = 1.0 / (2.0 * m_a);
    double solution_0 = (-m_b + root) * denom;
    double solution_1 = (-m_b - root) * denom;

    // Compute the new sphere based on the free variable and the solutions.
    // The R is the free variable so choose the not negative radius and return the new point.
    // Check the radii of both solutions and add the corresponding spheres to the result if they are not negative.
    uint results = 0;
    if (solution_0 > 0.0) {
        // Compute the new point.
        double b_x_0 = rhs[0] - v_1 * solution_0;
        double b_y_0 = rhs[1] - v_2 * solution_0;
        double b_z_0 = rhs[2] - v_3 * solution_0;
        double v_x_0 = (m_1 * b_x_0 + m_4 * b_y_0 + m_7 * b_z_0) + p_spheres[sphere_id].GetX();
        double v_y_0 = (m_2 * b_x_0 + m_5 * b_y_0 + m_8 * b_z_0) + p_spheres[sphere_id].GetY();
        double v_z_0 = (m_3 * b_x_0 + m_6 * b_y_0 + m_9 * b_z_0) + p_spheres[sphere_id].GetZ();
        solution_0 -= p_spheres[sphere_id].GetW();
        p_result_spheres[results].Set(v_x_0, v_y_0, v_z_0, solution_0);
        results++;
    }
    if (solution_1 > 0.0) {
        double b_x_1 = rhs[0] - v_1 * solution_1;
        double b_y_1 = rhs[1] - v_2 * solution_1;
        double b_z_1 = rhs[2] - v_3 * solution_1;
        double v_x_1 = (m_1 * b_x_1 + m_4 * b_y_1 + m_7 * b_z_1) + p_spheres[sphere_id].GetX();
        double v_y_1 = (m_2 * b_x_1 + m_5 * b_y_1 + m_8 * b_z_1) + p_spheres[sphere_id].GetY();
        double v_z_1 = (m_3 * b_x_1 + m_6 * b_y_1 + m_9 * b_z_1) + p_spheres[sphere_id].GetZ();
        solution_1 -= p_spheres[sphere_id].GetW();
        p_result_spheres[results].Set(v_x_1, v_y_1, v_z_1, solution_1);
        results++;
    }

    // If there are two results, take the one with smaller radius first.
    if (results > 1 && p_result_spheres[0].GetW() > p_result_spheres[1].GetW()) {
        std::swap(p_result_spheres[0], p_result_spheres[1]);
    }

    return results;
}

/*
 * Computations::CreatePlane
 */
vislib::math::Plane<double> Computations::CreatePlane(
    const vec4d& p_first_sphere, const vec4d& p_second_sphere, const vec4d& p_third_sphere) {
    vislib::math::Plane<double> retval(vislib::math::Point<double, 3>(p_first_sphere.PeekComponents()),
        vislib::math::Point<double, 3>(p_second_sphere.PeekComponents()),
        vislib::math::Point<double, 3>(p_third_sphere.PeekComponents()));
    return retval;
}

/*
 * Computations::incircleBaryCoords
 */
size_t Computations::incircleBaryCoords(
    const std::array<vec4d, 4>& p_spheres, std::array<vec4d, 2>& p_solutions, std::array<vec3d, 2>& p_circles) {
    // Check if we have three spheres.
    ASSERT(p_spheres.size() >= 3);

    // Compute the vectors between the three spheres to create a triangle.
    // The resulting vectors do not need a radius so set it to zero.
    auto dvec = p_spheres[1] - p_spheres[0];
    dvec.SetW(0.0);
    auto r1vec = p_spheres[2] - p_spheres[0];
    r1vec.SetW(0.0);
    auto r2vec = p_spheres[2] - p_spheres[1];
    r2vec.SetW(0.0);

    // Compute the length of all triangle sides.
    double d = dvec.Length();
    double r1 = r1vec.Length();
    double r2 = r2vec.Length();

    // Create the 2D representation of the gate spheres.
    vec3d p1(0.0, 0.0, p_spheres[0].W());
    vec3d p2(d, 0.0, p_spheres[1].W());
    double xcoord = (d * d - r2 * r2 + r1 * r1) / (2 * d);
    double ycoord = sqrt(r1 * r1 - xcoord * xcoord);
    vec3d p3(xcoord, ycoord, p_spheres[2].W());

    // Compute the cirlce that is tangent to all three gate circles.
    std::array<vec3d, 3> query{p1, p2, p3};
    auto queryResult_cnt = Computations::ComputeVoronoiCircle(query, p_circles);

    // Compute the barycentric coordinates for every gate center.
    for (size_t i = 0; i < queryResult_cnt; i++) {
        vec2d pos1(p1.PeekComponents());
        vec2d pos2(p2.PeekComponents());
        vec2d pos3(p3.PeekComponents());
        vec2d finalPos(p_circles[i].PeekComponents());

        auto v0 = pos2 - pos1;
        auto v1 = pos3 - pos1;
        auto v2 = finalPos - pos1;
        double d00 = v0.Dot(v0);
        double d01 = v0.Dot(v1);
        double d11 = v1.Dot(v1);
        double d20 = v2.Dot(v0);
        double d21 = v2.Dot(v1);
        double denom = d00 * d11 - d01 * d01;
        vec4d barycentric;
        barycentric.SetY((d11 * d20 - d01 * d21) / denom);
        barycentric.SetZ((d00 * d21 - d01 * d20) / denom);
        barycentric.SetX(1.0f - barycentric.GetY() - barycentric.GetZ());
        barycentric.SetW(p_circles[i].GetZ());
        p_solutions[i] = barycentric;
    }
    return queryResult_cnt;
}

/*
 * Computations::LiesInsideConvexHull
 */
bool Computations::LiesInsideConvexHull(
    const std::vector<vec3f>& p_atoms, const vec4d point_to_test, std::vector<vec3f>& p_directions) {
    // Intialise the convex hull test.
    vec3f testPoint;
    testPoint.Set(static_cast<float>(point_to_test.GetX()), static_cast<float>(point_to_test.GetY()),
        static_cast<float>(point_to_test.GetZ()));
    vec3f vectorSum(0.0, 0.0, 0.0);

    // Compute the sum of vectors from the point to all atoms and normalise it.
    for (uint atomID = 0; atomID < p_atoms.size(); atomID++) {
        Computations::subtractVector(p_atoms[atomID], testPoint, p_directions[atomID]);
        vectorSum += p_directions[atomID];
    }
    Computations::normaliseVector(vectorSum);

    // Check if the dot product between a vector and the normalised sum of all
    // vectors is negative.
    for (uint atomID = 0; atomID < p_atoms.size(); atomID++) {
        Computations::normaliseVector(p_directions[atomID]);
        if (vectorSum.Dot(p_directions[atomID]) < 0.0) {
            return true;
        }
    }

    return false;
}

/*
 * Computations::splitMatrix
 */
void Computations::splitMatrix(const mat3x4d& p_full_mat, emat3d& p_part_mat, const size_t p_id) {
    // Create the submatrix.
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            p_part_mat(row, col) = p_full_mat(row, Computations::access_columns_sphere[p_id * 3 + col]);
        }
    }
}

/*
 * Computations::splitMatrix
 */
void Computations::splitMatrix(const mat2x3d& p_full_mat, emat2d& p_part_mat, const size_t p_id) {
    // Create the submatrix.
    for (int col = 0; col < 2; col++) {
        for (int row = 0; row < 2; row++) {
            p_part_mat(row, col) = p_full_mat(row, Computations::access_columns_circle[p_id * 2 + col]);
        }
    }
}
