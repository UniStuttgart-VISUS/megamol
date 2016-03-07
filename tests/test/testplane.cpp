#include "testplane.h"

#include "testhelper.h"

#include "vislib/math/Plane.h"
#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"


/*
 * TestPlane0
 */
template<class T> void TestPlane0(void) {
    typedef vislib::math::Plane<T> PlaneType;
    typedef vislib::math::Point<T, 3> PointType;
    typedef vislib::math::Vector<T, 3> VectorType;

    {
        PlaneType plane(1, 0, 0, 5);

        {
            PointType p1(0, 0, 0);
            PointType p2(1, 0, 0);
            PointType i;
            PointType r(-5, 0, 0);

            ::AssertEqual("One intersection", plane.Intersect(i, p1, p2), PlaneType::ONE);
            ::AssertEqual("Intersection point", i, r);
        }

        {
            PointType p1(0, 0, 0);
            PointType p2(0, 1, 0);
            PointType i;

            ::AssertEqual("No intersection", plane.Intersect(i, p1, p2), PlaneType::NONE);
        }

        {
            PointType p1(-5, 0, 0);
            PointType p2(-5, 10, 0);
            PointType i;

            ::AssertEqual("In plane", plane.Intersect(i, p1, p2), PlaneType::ALL);
        }
    }

    {
        PlaneType plane(4, 4, -5, -12);

        {
            PointType p1(0, 0, 1);
            PointType p2(1, 0, 1);
            PointType i;
            PointType r(17.0 / 4, 0, 1);

            ::AssertEqual("One intersection", plane.Intersect(i, p1, p2), PlaneType::ONE);
            ::AssertEqual("Intersection point", i, r);
        }

        {
            PointType p1(3, 2, 1);
            PointType p2(1, 0, 1);
            PointType i;
            PointType r(21.0 / 8, 13.0 /8, 1);

            ::AssertEqual("One intersection", plane.Intersect(i, p1, p2), PlaneType::ONE);
            ::AssertEqual("Intersection point", i, r);
        }

        {
            PointType p1(3, 2, 4);
            PointType p2(1, 0, 2);
            PointType i;
            PointType r(7, 6, 8);

            ::AssertEqual("One intersection", plane.Intersect(i, p1, p2), PlaneType::ONE);
            ::AssertEqual("Intersection point", i, r);
        }
    }
}


/*
 * ::TestPlane
 */
void TestPlane(void) {
    ::TestPlane0<float>();
    ::TestPlane0<double>();
}
