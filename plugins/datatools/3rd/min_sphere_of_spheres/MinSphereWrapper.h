#pragma once

#pragma warning(push)
#pragma warning(disable : 4267)
#include "Min_sphere_of_spheres_d.h"
#pragma warning(pop)

#include <array>
#include <vector>

/// <summary>
/// Ball type definition needed for the min_sphere of spheres library
/// </summary>
struct Ball {
private:
    double c[3]; // position
    double r;    // radius
public:
    /// <summary>
    /// Constructor
    /// </summary>
    Ball() {}

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="c0">x-coordintate of center</param>
    /// <param name="c1">y-coordintate of center</param>
    /// <param name="c2">z-coordintate of center</param>
    /// <param name="r">ball radius</param>
    Ball(double c0, double c1, double c2, double r) {
        c[0] = c0;
        c[1] = c1;
        c[2] = c2;
        this->r = r;
    }

    /// <summary>
    /// Returns the radius of the ball
    /// </summary>
    /// <returns></returns>
    double radius() const { return r; }

    typedef const double* ConstIterator;

    /// <summary>
    /// Returns a pointer to the coordinates
    /// </summary>
    /// <returns>Pointer to the coordinates</returns>
    ConstIterator beginCenter() const { return c; }
};

/// <summary>
/// Configuration struct for the algorithm
/// </summary>
struct BallTraits {
    typedef ::Ball Sphere;
    static const int D = 3;                    // always in 3 dimension
    typedef CGAL::Default_algorithm Algorithm; // the default algorithm should work
    typedef CGAL::Tag_true Use_square_roots;
    typedef double FT; // double precion is used

    typedef Sphere::ConstIterator Cartesian_const_iterator;
    static Cartesian_const_iterator center_cartesian_begin(const Sphere& b) { return b.beginCenter(); }
    static double radius(const Sphere& b) { return b.radius(); }
};

/// <summary>
/// Computes the minimal enclosing sphere for given 3d sphere coordinates
/// </summary>
/// <param name="spheres">The sphere coordinates, ordered x1,y1,z1,r1,x2,y2,z2,r2,...</param>
/// <returns>The minimal enclosing sphere ordered x, y, z, r</returns>
template <typename T> std::array<T, 4> getMinSphere(const std::vector<T>& spheres) {
    std::array<T, 4> result;
    typedef CGAL::Min_sphere_of_spheres_d<BallTraits> Minsphere;

    std::vector<Ball> ballVec;
    for (size_t i = 0; i < spheres.size() / 4; i++) {
        double x = static_cast<double>(spheres[4 * i + 0]);
        double y = static_cast<double>(spheres[4 * i + 1]);
        double z = static_cast<double>(spheres[4 * i + 2]);
        double r = static_cast<double>(spheres[4 * i + 3]);
        ballVec.push_back(Ball(x, y, z, r));
    }

    Minsphere res(ballVec.begin(), ballVec.end());
    result[0] = static_cast<T>(res.center_cartesian_begin()[0]);
    result[1] = static_cast<T>(res.center_cartesian_begin()[1]);
    result[2] = static_cast<T>(res.center_cartesian_begin()[2]);
    result[3] = static_cast<T>(res.radius());
    return result;
}
