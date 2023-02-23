/*
 * Computations.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "Types.h"
#include "kxsort.h"

#include "protein_calls/MolecularDataCall.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Vector.h"

#include <Eigen/Dense>

// Eigen typedefs.
typedef Eigen::Vector2d evec2d;
typedef Eigen::Vector3d evec3d;
typedef Eigen::Vector4d evec4d;
typedef Eigen::Matrix2d emat2d;
typedef Eigen::Matrix3d emat3d;
typedef Eigen::Matrix<double, 2, 2> mat2x2d;
typedef Eigen::Matrix<double, 2, 3> mat2x3d;
typedef Eigen::Matrix<double, 3, 2> mat3x2d;
typedef Eigen::Matrix<double, 3, 3> mat3x3d;
typedef Eigen::Matrix<double, 3, 4> mat3x4d;
typedef Eigen::Matrix<double, 4, 3> mat4x3d;

namespace megamol {
namespace molecularmaps {

class Computations {
public:
    /** Dtor. */
    virtual ~Computations(void);

    /**
     * Computes the angular distance between the voronoi vertex belonging to a gate and another vertex.
     * See the Kim et al. paper.
     *
     * @param p_start_pivot the edge from the pivot to the start vertex
     * @param p_pivot the pivot point for the test
     * @param vertToTest the voronoi vertex we want to know the distance for
     *
     * @return The angle between the voroVert and the vertToTest.
     */
    static double AngularDistance(const vec3d& p_start_pivot, const vec3d& p_pivot, const vec4d& p_vertToTest);

    /** Ctor. */
    Computations(void);

    /**
     * Compute the gate center by compiting the barycentric coordinates of the center circle(s) of
     * the three gate vertices. If there is no incircle found the function returns false and the gate
     * center can be ignored. The final gate center(s) is/are the real middle sphere(s) that is/are
     * computed based on the barycentric coordinates.
     *
     * @param gateVector the current gate vector
     * @param p_gate_center will contain the gate center(s)
     * @param p_bary_solutions will contain the possible solutions for the incircle
     * @param p_circles will contain the circles that are used to compute the incircle
     *
     * @return true if there is at least one center, false otherwise
     */
    static size_t ComputeGateCenter(const std::array<vec4d, 4>& p_gate_vector, std::array<vec4d, 2>& p_gate_center,
        std::array<vec4d, 2>& p_bary_solutions, std::array<vec3d, 2>& p_circles);

    /**
     * Compute the pivot point based on the Kim et al. paper. The pivot is eqaul to
     * the center of the gate sphere with the smallest radius.
     *
     * @param p_gate_spheres the three gate spheres
     *
     * @return the pivot point
     */
    static vec3d ComputePivot(const std::array<vec4d, 4>& p_gate_spheres);

    /**
     * Computs the circle that is tangential to the three input circles. If no such
     * circle is found an empty vec3d is returned.
     *
     * @param p_circles the centerpoint and the radius for each of the three circles
     * @param p_result_circles will contain the resulting spheres, the solution with the smallest
     * radius is the first element in the vector
     *
     * @return the number of solutions found
     */
    static size_t ComputeVoronoiCircle(const std::array<vec3d, 3>& p_circles, std::array<vec3d, 2>& p_result_circles);

    /**
     * Computs the sphere that is tangential to the four input spheres. If no such
     * sphere is found an empty vector is returned. Looks for the independent variable.
     *
     * @param p_spheres the centerpoint and the radius for each of the four spheres
     * @param p_result_spheres will contain the resulting spheres, the solution with the smallest
     * radius is the first element in the vector
     *
     * @return the number of solutions found
     */
    static uint ComputeVoronoiSphere(const std::array<vec4d, 4>& p_spheres, std::array<vec4d, 2>& p_result_spheres);

    /**
     * Computs the sphere that is tangential to the four input spheres. If no such
     * sphere is found an empty vector is returned. Always takes the radius R as the independent
     * variable.
     *
     * @param p_spheres the centerpoint and the radius for each of the four spheres
     * @param p_result_spheres will contain the resulting spheres, the solution with the smallest
     * radius is the first element in the vector
     *
     * @return the number of solutions found
     */
    static uint ComputeVoronoiSphereR(const std::array<vec4d, 4>& p_spheres, std::array<vec4d, 2>& p_result_spheres);

    /**
     * Creates a plane from three given spheres by removing the radius to get the three
     * points. Then a vislib::math::Plane is created from the points.
     *
     * @param p_first_sphere the first sphere that is on the plane
     * @param p_second_sphere the second sphere that is on the plane
     * @param p_third_sphere the third sphere that is on the plane
     */
    static vislib::math::Plane<double> CreatePlane(
        const vec4d& p_first_sphere, const vec4d& p_second_sphere, const vec4d& p_third_sphere);

    /**
     * Floors the double value by casting it to an int. Should be much faster than std::floor
     * which also checks for overflows that we do not have!
     *
     * @param p_value the double value
     *
     * @return the floored integer value
     */
    static inline int Floor(const double p_value) {
        int retval = static_cast<int>(p_value);
        return retval - (retval > p_value);
    }

    /**
     * Tests, whether a given point lies inside the convex hull of a molecule.
     *
     * @param p_atoms the position of the atoms
     * @param point_to_test The vertex that should be tested against the convex hull.
     * @param p_directions the vectors from the test point to all atoms
     *
     * @return True, if point_to_test lies inside the convex hull. False otherwise.
     */
    static bool LiesInsideConvexHull(
        const std::vector<vec3f>& p_atoms, const vec4d point_to_test, std::vector<vec3f>& p_directions);

    /**
     * Uses SSE to normalisze the double vector.
     *
     * @param p_vec the 3D vector.
     *
     * @return the normalised vector.
     */
    static inline void NormaliseVector(vec3d& p_vec) {
        // Initialise the sse vector.
        __m128 sse_vector = _mm_setr_ps(
            static_cast<float>(p_vec.GetX()), static_cast<float>(p_vec.GetY()), static_cast<float>(p_vec.GetZ()), 0.0f);

        // Normalise the vector and return the result.
        __m128 inverse_norm = _mm_rsqrt_ps(_mm_dp_ps(sse_vector, sse_vector, 0x77));
        __m128 sse_vector_norm = _mm_mul_ps(sse_vector, inverse_norm);
        p_vec.SetX(static_cast<double>(sse_vector_norm.m128_f32[0]));
        p_vec.SetY(static_cast<double>(sse_vector_norm.m128_f32[1]));
        p_vec.SetZ(static_cast<double>(sse_vector_norm.m128_f32[2]));
    }

    /**
     * Use radix sort to sort the closest atom pairs.
     *
     * @param p_vec the pairs of closest atoms
     */
    static inline void RadixSort(std::vector<closestPair>& p_vec) {
        kx::radix_sort(p_vec.begin(), p_vec.end(), RadixTraitsPair());
    }

private:
    /**
     * Uses SSE to add up two float vectors.
     *
     * @param p_lhs the vector left of the add symbol
     * @param p_rhs the vector right of the add symbol
     * @param p_retval will contain the result
     */
    static inline void addVector(const vec3f& p_lhs, const vec3f& p_rhs, vec3f& p_retval) {
        // Initialise the sse vectors.
        __m128 sse_lhs = _mm_setr_ps(p_lhs.GetX(), p_lhs.GetY(), p_lhs.GetZ(), 0.0f);
        __m128 sse_rhs = _mm_setr_ps(p_rhs.GetX(), p_rhs.GetY(), p_rhs.GetZ(), 0.0f);

        // Compute the sum and return the result.
        __m128 sse_add = _mm_add_ps(sse_lhs, sse_rhs);
        p_retval.SetX(sse_add.m128_f32[0]);
        p_retval.SetY(sse_add.m128_f32[1]);
        p_retval.SetZ(sse_add.m128_f32[2]);
    }

    /**
     * Uses SSE to compute the dot product of two normalised float vectors.
     *
     * @param p_lhs the vector left of the dot product symbol
     * @param p_rhs the vector right of the dot product symbol
     * @param p_retval the result of the dot product
     */
    static inline void dotProduct(const vec3f& p_lhs, const vec3f& p_rhs, float& p_retval) {
        // Initialise the sse vectors.
        __m128 sse_lhs = _mm_setr_ps(p_lhs.GetX(), p_lhs.GetY(), p_lhs.GetZ(), 0.0f);
        __m128 sse_rhs = _mm_setr_ps(p_rhs.GetX(), p_rhs.GetY(), p_rhs.GetZ(), 0.0f);

        // Compute the dot product and return the result.
        __m128 sse_dot = _mm_dp_ps(sse_lhs, sse_rhs, 0x77);
        p_retval = sse_dot.m128_f32[0];
    }

    /**
     * Computes the barycentric coordinates of the incircle of three given spheres
     * relative to the three gate vertex coordinates.
     *
     * @param p_spheres The three given spheres.
     * @param p_solutions will contain all possible solutions, an element always
     * contains all 3 barycentric coordinates followed by the circle radius.
     * @param p_circles will contain the circles that are used to compute the incircle
     *
     * @return the number of possible solutions
     */
    static size_t incircleBaryCoords(
        const std::array<vec4d, 4>& p_spheres, std::array<vec4d, 2>& p_solutions, std::array<vec3d, 2>& p_circles);

    /**
     * Uses SSE to normalisze the float vector.
     *
     * @param p_vec the 3D vector.
     *
     * @return the normalised vector.
     */
    static inline void normaliseVector(vec3f& p_vec) {
        // Initialise the sse vector.
        __m128 sse_vector = _mm_setr_ps(p_vec.GetX(), p_vec.GetY(), p_vec.GetZ(), 0.0f);

        // Normalise the vector and return the result.
        __m128 inverse_norm = _mm_rsqrt_ps(_mm_dp_ps(sse_vector, sse_vector, 0x77));
        __m128 sse_vector_norm = _mm_mul_ps(sse_vector, inverse_norm);
        p_vec.SetX(sse_vector_norm.m128_f32[0]);
        p_vec.SetY(sse_vector_norm.m128_f32[1]);
        p_vec.SetZ(sse_vector_norm.m128_f32[2]);
    }

    /**
     * Split the 3x4 matrix into a 3x3 matrix based on the given ID. The first 3x3 matrix will
     * contain the first 3 columns, the second will contain all columns except the thrid one,
     * the third will contain all columns except the second one and the fourth will contain all
     * columns except for the first one.
     *
     * @param p_full_mat the 3x4 input matrix
     * @param p_part_mat will contain the submatrix
     * @param p_id the ID of the submatrix that will be extracted, hast to be 0, 1, 2 or 3
     */
    static void splitMatrix(const mat3x4d& p_full_mat, emat3d& p_part_mat, const size_t p_id);

    /**
     * Split the 2x3 matrix into a 2x2 matrix based on the given ID. The first 2x2 matrix will
     * contain the first 2 columns, the second will contain the first and the last column and
     * the third will contain the last two columns.
     *
     * @param p_full_mat the 2x3 input matrix
     * @param p_part_mat will contain the submatrix
     * @param p_id the ID of the submatrix that will be extracted, hast to be 0, 1 or 2
     */
    static void splitMatrix(const mat2x3d& p_full_mat, emat2d& p_part_mat, const size_t p_id);

    /**
     * Uses SSE to subtract two float vectors.
     *
     * @param p_lhs the vector left of the minus symbol
     * @param p_rhs the vector right of the minus symbol
     * @param p_retval will contain the result
     */
    static inline void subtractVector(const vec3f& p_lhs, const vec3f& p_rhs, vec3f& p_retval) {
        // Initialise the sse vectors.
        __m128 sse_lhs = _mm_setr_ps(p_lhs.GetX(), p_lhs.GetY(), p_lhs.GetZ(), 0.0f);
        __m128 sse_rhs = _mm_setr_ps(p_rhs.GetX(), p_rhs.GetY(), p_rhs.GetZ(), 0.0f);

        // Compute the difference and return the result.
        __m128 sse_diff = _mm_sub_ps(sse_lhs, sse_rhs);
        p_retval.SetX(sse_diff.m128_f32[0]);
        p_retval.SetY(sse_diff.m128_f32[1]);
        p_retval.SetZ(sse_diff.m128_f32[2]);
    }

    /** The columns that are accessed in the 3x4 matrix when it is split into 3x3 matrices. */
    static std::vector<int> access_columns_circle;

    /** The columns that are accessed in the 3x4 matrix when it is split into 3x3 matrices. */
    static std::vector<int> access_columns_sphere;

    /** The constant value for PI / 2 */
    static double pi_half;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */
