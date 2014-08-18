/*
 * AbstractPlane.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPLANE_H_INCLUDED
#define VISLIB_ABSTRACTPLANE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/mathtypes.h"
#include "vislib/Point.h"
#include "vislib/types.h"
#include "vislib/Vector.h"


namespace vislib {
namespace math {


    /**
     * This is the abstract superclass for planes and shallow planes.
     */
    template<class T, class S> class AbstractPlane {

    public:

        /** Enumeration for halfspaces of the plane. */
        enum HalfSpace { 
            NEGATIVE_HALFSPACE = HALFSPACE_NEGATIVE,
            IN_PLANE           = HALFSPACE_IN_PLANE,
            POSITIVE_HALFSPACE = HALFSPACE_POSITIVE
        };

        /** Result for intersection tests. */
        enum IntersectionCount { 
            NONE = 0,                   // No intersection.
            ONE,                        // Exactly one intersection.
            ALL                         // All points intersect.
        };

        /** Dtor. */
        ~AbstractPlane(void);

        /**
         * Answer the parameter a in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter a.
         */
        inline const T& A(void) const {
            return this->parameters[IDX_A];
        }

        /**
         * Answer the parameter b in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter b.
         */
        inline const T& B(void) const {
            return this->parameters[IDX_B];
        }

        /**
         * Answer the parameter c in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter c.
         */
        inline const T& C(void) const {
            return this->parameters[IDX_C];
        }

        /**
         * Calculates the point of intersection of three plane
         *
         * @param plane2 The second plane
         * @param plane3 The third plane
         * @param outPoint Point variable to receive the result
         *
         * @return True if there is a single intersection point, which has
         *         been stored in 'outPoint'. If the intersection of the three
         *         planes is not a single point, false is returned and the
         *         value of 'outPoint' is undefined
         */
        template<class Sp1, class Sp2, class Sp3>
        bool CalcIntersectionPoint(const AbstractPlane<T, Sp1>& plane2,
            const AbstractPlane<T, Sp2>& plane3,
            AbstractPoint<T, 3, Sp3>& outPoint) const;

        /**
         * Answer whether 'point' is in the plane.
         *
         * @param point The point to be tested.
         *
         * @return true, if 'point' satisfies the plane equation, false 
         *         otherwise.
         */
        template<class Tp, class Sp>
        inline bool Contains(const AbstractPoint<Tp, 3, Sp>& point) const {
            return IsEqual<T>(this->Distance(point), static_cast<T>(0));
        }
        

        /**
         * Answer the parameter d in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter d.
         */
        inline const T& D(void) const {
            return this->parameters[IDX_D];
        }

        /**
         * Answer the distance of 'point' to the plane.
         *
         * @param point The point to compute the distance of.
         *
         * @return The distance of 'point' to the plane.
         */
        template<class Tp, class Sp>
        T Distance(const AbstractPoint<Tp, 3, Sp>& point) const;

        /**
         * Answer the parameter a in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter a.
         */
        inline const T& GetA(void) const {
            return this->parameters[IDX_A];
        }

        /**
         * Answer the parameter b in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter b.
         */
        inline const T& GetB(void) const {
            return this->parameters[IDX_B];
        }

        /**
         * Answer the parameter c in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter c.
         */
        inline const T& GetC(void) const {
            return this->parameters[IDX_C];
        }

        /**
         * Answer the parameter d in the plane equation ax + by + cz + d = 0.
         *
         * @param The parameter d.
         */
        inline const T& GetD(void) const {
            return this->parameters[IDX_D];
        }

        /**
         * Answer in which halfspace 'point' lies in respect to the plane.
         *
         * @param point The point to be tested.
         *
         * @return The halfspace the point lies in.
         */
        template<class Tp, class Sp>
        HalfSpace Halfspace(const AbstractPoint<Tp, 3, Sp>& point) const;

        //template<class Tp, class Sp1, class Sp2>
        //IntersectionCount Intersect(const Line3D<Tp, Sp1>& line, AbstractPoint3D<T, Sp2>& outWhere) const;

        /**
         * Answer the normal of the plane. The vector returned is guaranteed to
         * be normalised.
         *
         * @return The normal of the plane.
         */
        inline Vector<T, 3> Normal(void) const {
            Vector<T, 3> retval(this->parameters);
            retval.Normalise();
            return retval;
        }

        /**
         * Normalise the plane parameters.
         */
        void Normalise(void);

        /**
         * Answer a point in the plane.
         *
         * @return A point in the plane.
         */
        math::Point<T, 3> Point(void) const;

        /**
         * Answer three points in the plane.
         *
         * @param outP1 Receives the first point.
         * @param outP2 Receives the second point.
         * @param outP3 Receives the third point.
         */
        void Points(vislib::math::Point<T, 3>& outP1, 
            vislib::math::Point<T, 3>& outP2, 
            vislib::math::Point<T, 3>& outP3) const;

        /** 
         * Change the plane equation.
         *
         * @param a The parameter a in the equation ax + by + cz + d = 0.
         * @param b The parameter b in the equation ax + by + cz + d = 0.
         * @param c The parameter c in the equation ax + by + cz + d = 0.
         * @param d The parameter d in the equation ax + by + cz + d = 0.
         */
        inline void Set(const T a, const T b, const T c, const T d) {
            this->parameters[IDX_A] = a;
            this->parameters[IDX_B] = b;
            this->parameters[IDX_C] = c;
            this->parameters[IDX_D] = d;
        }

        /**
         * Change the plane equation using a point on the plane and its formal.
         *
         * @param point  A point on the plane.
         * @param normal The plane normal
         */
        template<class Tp1, class Sp1, class Tp2, class Sp2>
        void Set(const AbstractPoint<Tp1, 3, Sp1>& point, 
            const AbstractVector<Tp2, 3, Sp2>& normal);

        /** 
         * Set the parameter a in the plan equation ax + by + cz + d = 0.
         *
         * @param a The parameter a.
         */
        inline void SetA(const T& a) {
            this->parameters[IDX_A] = a;
        }

        /** 
         * Set the parameter b in the plan equation ax + by + cz + d = 0.
         *
         * @param b The parameter b.
         */
        inline void SetB(const T& b) {
            this->parameters[IDX_B] = b;
        }

        /** 
         * Set the parameter c in the plan equation ax + by + cz + d = 0.
         *
         * @param c The parameter c.
         */
        inline void SetC(const T& c) {
            this->parameters[IDX_C] = c;
        }

        /** 
         * Set the parameter d in the plan equation ax + by + cz + d = 0.
         *
         * @param d The parameter d.
         */
        inline void SetD(const T& d) {
            this->parameters[IDX_D] = d;
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractPlane& operator =(const AbstractPlane& rhs);

        /**
         * Assignment. This operator allows arbitrary plane to plane 
         * conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractPlane& operator =(const AbstractPlane<Tp, Sp>& rhs);

        /**
         * Test for equality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const AbstractPlane& rhs) const;

        /**
         * Test for equality. This operator allows comparing planes that
         * have been instantiated for different scalar types. The IsEqual<T>
         * function for the scalar type of the left hand side operand is used
         * as comparison operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        template<class Tp, class Sp>
        bool operator ==(const AbstractPlane<Tp, Sp>& rhs) const;

        /**
         * Test for inequality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractPlane& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for inequality. This operator allows comparing planes that
         * have been instantiated for different scalar types. The IsEqual<T>
         * function for the scalar type of the left hand side operand is used
         * as comparison operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        template<class Tp, class Sp>
        inline bool operator !=(const AbstractPlane<Tp, Sp>& rhs) const {
            return !(*this == rhs);
        }


    protected:

        /** The index of the parameter a. */
        static const UINT_PTR IDX_A;

        /** The index of the parameter b. */
        static const UINT_PTR IDX_B;

        /** The index of the parameter c. */
        static const UINT_PTR IDX_C;

        /** The index of the parameter d. */
        static const UINT_PTR IDX_D;

        /** 
         * Disallow instances of this class.
         */
        inline AbstractPlane(void) {}

        /**
         * Answer the normalised plane parameters.
         *
         * @param outA Receives the normalised a parameter.
         * @param outB Receives the normalised b parameter.
         * @param outC Receives the normalised c parameter.
         * @param outD Receives the normalised d parameter.
         */
        void normalise(T& outA, T& outB, T& outC, T& outD) const;

        /** The parameters defining the plane. */
        S parameters;

    };


    /*
     * AbstractPlane<T, S>::~AbstractPlane
     */
    template<class T, class S> AbstractPlane<T, S>::~AbstractPlane(void) {
    }


    /*
     * AbstractPlane<T, S>::CalcIntersectionPoint
     */
    template<class T, class S> 
    template<class Sp1, class Sp2, class Sp3>
    bool AbstractPlane<T, S>::CalcIntersectionPoint(
            const AbstractPlane<T, Sp1>& plane2,
            const AbstractPlane<T, Sp2>& plane3,
            AbstractPoint<T, 3, Sp3>& outPoint) const {
        // planes: ax + by + cz + d = 0
        T a1, b1, c1, d1;
        this->normalise(a1, b1, c1, d1);
        T a2, b2, c2, d2;
        plane2.normalise(a2, b2, c2, d2);
        T a3, b3, c3, d3;
        plane3.normalise(a3, b3, c3, d3);

        Vector<T, 3> n1(a1, b1, c1);
        Vector<T, 3> n2(a2, b2, c2);
        Vector<T, 3> n3(a3, b3, c3);

        ASSERT(n1.IsNormalised());
        ASSERT(n2.IsNormalised());
        ASSERT(n3.IsNormalised());

        T denom = n1.Dot(n2.Cross(n3));

        if (IsEqual(denom, static_cast<T>(0))) return false;

        Vector<T, 3> m1 = n2.Cross(n3);
        Vector<T, 3> m2 = n3.Cross(n1);
        Vector<T, 3> m3 = n1.Cross(n2);

        m1 *= -d1;
        m2 *= -d2;
        m3 *= -d3;
        m1 += m2;
        m1 += m3;
        m1 /= denom;
        outPoint.Set(m1.X(), m1.Y(), m1.Z());

        // TODO: This is a numerical nightmare! Replace with some robust implementation!!!
        // Do not use IsEqual with default EPSILON, since we have numeric issues here
#if defined(DEBUG) || defined(_DEBUG)
        d1 = Abs(this->Distance(outPoint));
        d2 = Abs(plane2.Distance(outPoint));
        d3 = Abs(plane3.Distance(outPoint));
        ASSERT(d1 < static_cast<T>(0.002f)); // epsilon is adjusted for MegaMol :-/
        ASSERT(d2 < static_cast<T>(0.002f));
        ASSERT(d3 < static_cast<T>(0.002f));
#endif

        return true;
    }


    /*
     * AbstractPlane<T, S>::Distance
     */
    template<class T, class S>
    template<class Tp, class Sp>
    T AbstractPlane<T, S>::Distance(
            const AbstractPoint<Tp, 3, Sp>& point) const {
        T a, b, c, d;           // Normalised plane parameters.
        
        this->normalise(a, b, c, d);
        return (a * static_cast<T>(point.X()) + b * static_cast<T>(point.Y())
            + c * static_cast<T>(point.Z()) + d);
    }


    /* 
     * AbstractPlane<T, S>::Halfspace
     */
    template<class T, class S>
    template<class Tp, class Sp>
    typename AbstractPlane<T, S>::HalfSpace AbstractPlane<T, S>::Halfspace(
            const AbstractPoint<Tp, 3, Sp>& point) const {
        T dist = this->Distance(point);

        if (IsEqual(dist, static_cast<T>(0))) {
            return IN_PLANE;

        } else if (dist > static_cast<T>(0)) {
            return NEGATIVE_HALFSPACE;

        } else if (dist < static_cast<T>(0)) {
            return POSITIVE_HALFSPACE;
    
        } else {
            ASSERT(false);      // Should never happen.
            return IN_PLANE;
        }        
    }


    ///*
    // * AbstractPlane<T, S>::Intersect
    // */
    //template<class T, class S>
    //template<class Tp, class Sp1, class Sp2>
    //typename AbstractPlane<T, S>::IntersectCount AbstractPlane<T, S>::Intersect(
    //        const Line3D<Tp, Sp1>& line, 
    //        AbstractPoint3D<T, Sp2>& outWhere) const {
    //    T dx0 = this->parameters[IDX_A] * x0.X() 
    //        + this->parameters[IDX_B] * x0.Y() 
    //        + this->parameters[IDX_C] * x0.Z();
    //    T dx1 = this->parameters[IDX_A] * x1.X() 
    //        + this->parameters[IDX_B] * x1.Y() 
    //        + this->parameters[IDX_C] * x1.Z();
    //
    //    if (IsEqual(dx0, -this->d) && IsEqual(dx1, -this->d)) {
    //        outWhere = x0;
    //        return ALL;

    //    } else if (IsEqual(dx0, dx1)) {
    //        return NONE;

    //    } else {
    //        outWhere = x0 + (((dx0 + this->d) / (dx0 - dx1)) * (x1 - x0));
    //        return ONE;
    //    }
    //}


    /*
     * AbstractPlane<T, S>::Normalise
     */
    template<class T, class S>
    void AbstractPlane<T, S>::Normalise(void) {
        T a, b, c, d;
        this->normalise(a, b, c, d);
        this->Set(a, b, c, d);
    }


    /*
     * AbstractPlane<T, S>::Point
     */
    template<class T, class S>
    math::Point<T, 3> AbstractPlane<T, S>::Point(void) const {
        T a, b, c, d;
        this->normalise(a, b, c, d);
        return vislib::math::Point<T, 3>(-d * a, -d * b, -d * c);
    }


    /*
     * AbstractPlane<T, S>::Points
     */
    template<class T, class S>
    void AbstractPlane<T, S>::Points(vislib::math::Point<T, 3>& outP1, 
            vislib::math::Point<T, 3>& outP2,
            vislib::math::Point<T, 3>& outP3) const {
        T a, b, c, d;
        this->normalise(a, b, c, d);
        
        outP1 = outP3 = vislib::math::Point<T, 3>(-d * a, -d * b, -d * c);
        outP2 = vislib::math::Point<T, 3>(outP1.X() - b, outP1.Y() + a, 
            outP1.Z());
        outP3 += (outP2 - outP1).Cross(this->Normal());   

        ASSERT(this->Contains(outP1));
        ASSERT(this->Contains(outP2));
        ASSERT(this->Contains(outP3));
    }


    /*
     * AbstractPlane<T, S>::Set
     */
    template<class T, class S>
    template<class Tp1, class Sp1, class Tp2, class Sp2>
    void AbstractPlane<T, S>::Set(const AbstractPoint<Tp1, 3, Sp1>& point, 
            const AbstractVector<Tp2, 3, Sp2>& normal) {
        this->parameters[IDX_A] = normal.X();
        this->parameters[IDX_B] = normal.Y();
        this->parameters[IDX_C] = normal.Z();
        this->parameters[IDX_D] = -1.0f * (this->parameters[IDX_A] * point.X()
            + this->parameters[IDX_B] * point.Y() 
            + this->parameters[IDX_C] * point.Z());
    }


    /*
     * AbstractPlane<T, S>::operator =
     */
    template<class T, class S>
    AbstractPlane<T, S>& AbstractPlane<T, S>::operator =(
            const AbstractPlane& rhs) {
        if (this != &rhs) {
            ::memcpy(this->parameters, rhs.parameters, 4 * sizeof(T));
        }

        return *this;
    }


    /*
     * AbstractPlane<T, S>::operator =
     */
    template<class T, class S>
    template<class Tp, class Sp>
    AbstractPlane<T, S>& AbstractPlane<T, S>::operator =(
            const AbstractPlane<Tp, Sp>& rhs) {
        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            this->parameters[IDX_A] = static_cast<T>(rhs.A());
            this->parameters[IDX_B] = static_cast<T>(rhs.B());
            this->parameters[IDX_C] = static_cast<T>(rhs.C());
            this->parameters[IDX_D] = static_cast<T>(rhs.D());
        }

        return *this;
    }


    /*
     * AbstractPlane<T, S>::operator ==
     */
    template<class T, class S>
    bool AbstractPlane<T, S>::operator ==(const AbstractPlane& rhs) const {
        return (IsEqual(this->parameters[IDX_A], rhs.parameters[IDX_A])
            && IsEqual(this->parameters[IDX_B], rhs.parameters[IDX_B])
            && IsEqual(this->parameters[IDX_C], rhs.parameters[IDX_C])
            && IsEqual(this->parameters[IDX_D], rhs.parameters[IDX_D]));
    }


    /*
     * AbstractPlane<T, S>::operator ==
     */
    template<class T, class S>
    template<class Tp, class Sp>
    bool AbstractPlane<T, S>::operator ==(
            const AbstractPlane<Tp, Sp>& rhs) const {
        return (IsEqual<T>(this->parameters[IDX_A], rhs.parameters[IDX_A])
            && IsEqual<T>(this->parameters[IDX_B], rhs.parameters[IDX_B])
            && IsEqual<T>(this->parameters[IDX_C], rhs.parameters[IDX_C])
            && IsEqual<T>(this->parameters[IDX_D], rhs.parameters[IDX_D]));
    }


    /*
     * vislib::math::AbstractPlane<T, S>::IDX_A
     */
    template<class T, class S> 
    const UINT_PTR AbstractPlane<T, S>::IDX_A = 0;


    /*
     * vislib::math::AbstractPlane<T, S>::IDX_B
     */
    template<class T, class S> 
    const UINT_PTR AbstractPlane<T, S>::IDX_B = 1;


    /*
     * vislib::math::AbstractPlane<T, S>::IDX_C
     */
    template<class T, class S> 
    const UINT_PTR AbstractPlane<T, S>::IDX_C = 2;


    /*
     * vislib::math::AbstractPlane<T, S>::IDX_D
     */
    template<class T, class S> 
    const UINT_PTR AbstractPlane<T, S>::IDX_D = 3;


    /*
     * AbstractPlane<T, S>::normalise
     */
    template<class T, class S> 
    void AbstractPlane<T, S>::normalise(T& outA, T& outB, T& outC, 
            T& outD) const {
        T len = Sqrt(Sqr(this->parameters[IDX_A]) 
            + Sqr(this->parameters[IDX_B]) + Sqr(this->parameters[IDX_C]));

        if (!IsEqual(len, static_cast<T>(0))) {
            outA = this->parameters[IDX_A] / len;
            outB = this->parameters[IDX_B] / len;
            outC = this->parameters[IDX_C] / len;
            outD = this->parameters[IDX_D] / len;
        } else {
            outA = outB = outC = outD = static_cast<T>(0);
        }        
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPLANE_H_INCLUDED */
