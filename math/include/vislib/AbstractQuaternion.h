/*
 * AbstractQuaternion.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTQUATERNION_H_INCLUDED
#define VISLIB_ABSTRACTQUATERNION_H_INCLUDED
#if _MSC_VER > 1000
#pragma once
#endif /* _MSC_VER > 1000 */


#include <cmath>

#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/types.h"
#include "vislib/Vector3D.h"


namespace vislib {
namespace math {


    /**
     * Objects of this class represent a quaternion. The quaterion is internally
     * stored in order x, y, z, w, which must be kept in mind when passing 
     * values in array form.
     *
     * This class implements the functionality of the quaterion. Acutally used
     * can only be the derived classes Quaterion and ShallowQuaterion, which use
     * different means for storing their data.
     *
     * The quaterion is a template, which allows instantiations for different
     * scalar types, but only floating point instantiations (float or double) 
     * make any sense. You should not instantiate a quaterion for other types.
     */
    template<class T, class S = T[4]> class AbstractQuaternion {

    public:

        /** Dtor. */
        ~AbstractQuaternion(void);

        /**
         * Answer the angle and the axis of the rotation that is 
         * represented by this quaternion. 
         *
         * @param outAngle Receives the angle. 
         * @param outAxis  Receives the vector representing the rotation
         *                 axis. The vector is guaranteed to be normalised.
         */
        template<class Sp>
        void AngleAndAxis(T& outAngle, 
            AbstractVector3D<T, Sp>& outAxis) const;

        /**
         * Answer the w-component of the quaternion.
         *
         * @return The w-component.
         */
        inline const T& GetW(void) const {
            return this->components[IDX_W];
        }

        /**
         * Answer the x-component of the quaternion.
         *
         * @return The x-component.
         */
        inline const T& GetX(void) const {
            return this->components[IDX_X];
        }

        /**
         * Answer the y-component of the quaternion.
         *
         * @return The y-component.
         */
        inline float GetY(void) const {
            return this->components[IDX_Y];
        }

        /**
         * Answer the z-component of the quaternion.
         *
         * @return The z-component.
         */
        inline float GetZ(void) const {
            return this->components[IDX_Z];
        }

        //Quaternion Inverse(void) const;

        /** 
         * Invert the quaternion.
         */
        void Invert(void);

        /**
         * Answer the norm of the quaternion.
         *
         * @return The norm of the quaternion.
         */
        inline T Norm(void) const {
            return Sqrt(Sqr(this->components[IDX_X])
                + Sqr(this->components[IDX_Y])
                + Sqr(this->components[IDX_Z]) 
                + Sqr(this->components[IDX_W]));
        }

        /**
         * Normalise the quaternion.
         *
         * @return The norm BEFORE the normalisation.
         */
        T Normalise(void);

        /**
         * Provide direct access to the components of the quaternion.
         *
         * @return A pointer to the actual components.
         */
        inline const T *PeekComponents(void) const {
            return this->components;
        }

        /**
         * Provide direct access to the components of the quaternion.
         *
         * @return A pointer to the actual components.
         */
        inline T *PeekComponents(void) {
            return this->components;
        }

        /**
         * Set the components of the quaternion.
         *
         * @param x The new x-component.
         * @param y The new y-component.
         * @param z The new z-component.
         * @param w The new w-component.
         */
        inline void Set(const T& x, const T& y, const T& z, const T& w) {
            this->components[IDX_X] = x;
            this->components[IDX_Y] = y;
            this->components[IDX_Z] = z;
            this->components[IDX_W] = w;
        }

        /**
         * Set the w-component of the quaternion.
         *
         * @param w The new value for the component.
         */
        inline void SetW(const T& w) {
            this->components[IDX_W] = w;
        }

        /**
         * Set the x-component of the quaternion.
         *
         * @param x The new value for the component.
         */
        inline void SetX(const T& x) {
            this->components[IDX_X] = x;
        }

        /**
         * Set the y-component of the quaternion.
         *
         * @param y The new value for the component.
         */
        inline void SetY(const T& y) {
            this->components[IDX_Y] = y;
        }

        /**
         * Set the z-component of the quaternion.
         *
         * @param z The new value for the component.
         */
        inline void SetZ(const T& z) {
            this->components[IDX_Z] = z;
        }

        /**
         * Answer the w-component of the quaternion.
         *
         * @return The w-component.
         */
        inline const T& W(void) const {
            return this->components[IDX_W];
        }

        /**
         * Answer the x-component of the quaternion.
         *
         * @return The x-component.
         */
        inline const T& X(void) const {
            return components[IDX_X];
        }

        /**
         * Answer the y-component of the quaternion.
         *
         * @return The y-component.
         */
        inline const T& Y(void) const {
            return this->components[IDX_Y];
        }

        /**
         * Answer the z-component of the quaternion.
         *
         * @return The z-component.
         */
        inline const T& Z(void) const {
            return this->components[IDX_Z];
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractQuaternion& operator =(const AbstractQuaternion& rhs);

        /**
         * Assignment. This operator allows arbitrary quaternion to
         * quaternion conversions.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        template<class Tp, class Sp>
        AbstractQuaternion& operator =(const AbstractQuaternion<Tp, Sp>& rhs);

        /**
         * Test for equality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const AbstractQuaternion& rhs) const;

        /**
         * Test for equality. This operator allows comparing quaternions that
         * have been instantiated for different scalar types. The IsEqual<T>
         * function for the scalar type of the left hand side operand is used
         * as comparison operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        template<class Tp, class Sp>
        bool operator ==(const AbstractQuaternion<Tp, Sp>& rhs) const;

        /**
         * Test for inequality. The IsEqual function is used for this.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const AbstractQuaternion& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for inequality. This operator allows comparing quaternions that
         * have been instantiated for different scalar types. The IsEqual<T>
         * function for the scalar type of the left hand side operand is used
         * as comparison operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        template<class Tp, class Sp>
        inline bool operator !=(const AbstractQuaternion<Tp, Sp>& rhs) const {
            return !(*this == rhs);
        }

        template<class Sp>
        AbstractQuaternion<T, T[4]> operator *(
            const AbstractQuaternion<T, Sp>& rhs) const;

        template<class Sp>
        Vector3D<T> operator *(const AbstractVector3D<T, Sp>& rhs) const;

        //operator Matrix4x4(void) const;

    protected:

        /** The index of the w component. */
        static const UINT_PTR IDX_W;

        /** The index of the x component. */
        static const UINT_PTR IDX_X;

        /** The index of the y component. */
        static const UINT_PTR IDX_Y;

        /** The index of the z component. */
        static const UINT_PTR IDX_Z;

        /**
         * Disallow instances of this class.
         */
        inline AbstractQuaternion(void) {
        }

        /** 
         * The components of the quaterion. These are stored in the following
         * order: x, y, z (the vector), w.
         */
        S components;
    };


/*
 * vislib::math::AbstractQuaternion<T, S>::AngleAndAxis
 */
template<class T, class S>
template<class Sp>
void AbstractQuaternion<T, S>::AngleAndAxis(T& outAngle, 
        AbstractVector3D<T, Sp>& outAxis) const {
    T d = Sqrt(Sqr(this->components[IDX_X]) + Sqr(this->components[IDX_Y])
        + Sqr(this->components[IDX_Z]));

    if (!IsEqual<T>(d, static_cast<T>(0))) {
        outAxis.SetX(this->components[IDX_X] / d);
        outAxis.SetY(this->components[IDX_Y] / d);
        outAxis.SetZ(this->components[IDX_Z] / d);

        // TODO: Not nice.
        outAngle = static_cast<T>(2.0 
            * ::acos(static_cast<double>(this->components[IDX_W])));

    } else {
        outAxis.SetX(static_cast<T>(0));
        outAxis.SetY(static_cast<T>(0));
        outAxis.SetZ(static_cast<T>(1));
        outAngle = 0.0f;
    } 

    ASSERT(outAxis.IsNormalised());
}


/*
 * vislib::math::AbstractQuaternion<T, S>::Invert
 */
template<class T, class S>
void AbstractQuaternion<T, S>::Invert(void) {
    T norm = this->Norm();

    if (!IsEqual<T>(norm, static_cast<T>(0))) {
        this->components[IDX_X] /= -norm;
        this->components[IDX_Y] /= -norm;
        this->components[IDX_Z] /= -norm;
        this->components[IDX_W] /= norm;

    } else {
        this->components[IDX_X] = this->components[IDX_Y] 
            = this->components[IDX_Z] = static_cast<T>(0);
        this->components[IDX_W] = static_cast<T>(1);
    }
}


/*
 * vislib::math::AbstractQuaternion<T, S>::Normalise
 */
template<class T, class S>
T AbstractQuaternion<T, S>::Normalise(void) {
    T norm = this->Norm();

    if (!IsEqual<T>(norm, static_cast<T>(0))) {
        this->components[IDX_X] /= norm;
        this->components[IDX_Y] /= norm;
        this->components[IDX_Z] /= norm;
        this->components[IDX_W] /= norm;

    } else {
        this->components[IDX_X] = this->components[IDX_Y] 
            = this->components[IDX_Z] = static_cast<T>(0);
        this->components[IDX_W] = static_cast<T>(1);
    }

    return norm;
}


/*
 * vislib::math::AbstractQuaternion<T, S>::operator =
 */
template<class T, class S>
AbstractQuaternion<T, S>& AbstractQuaternion<T, S>::operator =(
       const AbstractQuaternion& rhs) {
    if (this != &rhs) {
        ::memcpy(this->components, rhs.components, 4 * sizeof(T));
    }

    return *this;
}


/*
 * vislib::math::AbstractQuaternion<T, S>::operator =
 */
template<class T, class S>
template<class Tp, class Sp>
AbstractQuaternion<T, S>& AbstractQuaternion<T, S>::operator =(
       const AbstractQuaternion<Tp, Sp>& rhs) {
    if (static_cast<void *>(this) != static_cast<void *>(&rhs)) {
        this->components[IDX_X] = static_cast<T>(rhs.X());
        this->components[IDX_Y] = static_cast<T>(rhs.Y());
        this->components[IDX_Z] = static_cast<T>(rhs.Z());
        this->components[IDX_W] = static_cast<T>(rhs.W());
    }

    return *this;
}



/*
 * vislib::math::AbstractQuaternion<T, S>::operator ==
 */
template<class T, class S>
bool AbstractQuaternion<T, S>::operator ==(
        const AbstractQuaternion& rhs) const {
    return (IsEqual(this->components[IDX_X], rhs.components[IDX_X])
        && IsEqual(this->components[IDX_Y], rhs.components[IDX_Y])
        && IsEqual(this->components[IDX_Z], rhs.components[IDX_Z])
        && IsEqual(this->components[IDX_W], rhs.components[IDX_W]));
}


/*
 * vislib::math::AbstractQuaternion<T, S>::operator ==
 */
template<class T, class S>
template<class Tp, class Sp>
bool AbstractQuaternion<T, S>::operator ==(
        const AbstractQuaternion<Tp, Sp>& rhs) const {
    return (IsEqual<T>(this->components[IDX_X], rhs.X())
        && IsEqual<T>(this->components[IDX_Y], rhs.Y())
        && IsEqual<T>(this->components[IDX_Z], rhs.Z())
        && IsEqual<T>(this->components[IDX_W], rhs.W()));
}


/*
 * vislib::math::AbstractQuaternion<T, S>::operator *
 */
template<class T, class S>
template<class Sp>
AbstractQuaternion<T, T[4]> AbstractQuaternion<T, S>::operator *(
        const AbstractQuaternion<T, Sp>& rhs) const {
    return AbstractQuaternion<T, T[4]>(
        this->components[IDX_W] * rhs.X() 
        + rhs.W() * this->components[IDX_X]
        + this->components[IDX_Y] * rhs.Z() 
            - this->components[IDX_Z] * rhs.Y(),

        this->components[IDX_W] * rhs.Y() 
        + rhs.W() * this->components[IDX_Y] 
        + this->components[IDX_Z] * rhs.X() 
            - this->components[IDX_X] * rhs.Z(),

        this->components[IDX_W] * rhs.Z() 
        + rhs.W() * this->comp[IDX_Z] 
        + this->components[IDX_X] * rhs.Y() 
        - this->components[IDX_Y] * rhs.X(),

        this->components[IDX_W] * rhs.W() 
        - (this->components[IDX_X] * rhs.X()
        + this->components[IDX_Y] * rhs.Y() 
        + this->components[IDX_Z] * rhs.Z()));
    return retval;
}


/*
 * vislib::math::AbstractQuaternion<T, S>::operator *
 */
template<class T, class S>
template<class Sp>
Vector3D<T> AbstractQuaternion<T, S>::operator *(
        const AbstractVector3D<T, Sp>& rhs) const {
    Vector3D<T> u(this->components);
    return ((2.0f * ((u.Dot(rhs) * u) + (this->W() * u.Cross(rhs))))
        + ((Sqr(this->W()) - u.Dot(u)) * rhs));
}


//template<class T> Quaternion<T>::operator Matrix3x3(void) {
//    Quaternion q = this->Normalised();
//    return Matrix4x4(Sqr(q.comp[3]) + Sqr(q.comp[0]) - Sqr(q.comp[1]) 
//        - Sqr(q.comp[2]), 
//        2.0f * q.comp[0] * q.comp[1] - 2.0f * q.comp[3] 
//        * q.comp[2], 
//        2.0f * q.comp[3] * q.comp[1] + 2.0f * q.comp[0] 
//        * q.comp[2], 
//        0.0f,
//
//        2.0f * q.comp[3] * q.comp[2] + 2.0f * q.comp[0] * q.comp[1],
//        Sqr(q.comp[3]) - Sqr(q.comp[0]) + Sqr(q.comp[1]) - Sqr(q.comp[2]),
//        2.0f * q.comp[1] * q.comp[2] - 2.0f * q.comp[3] * q.comp[0],
//        0.0f,
//
//        2.0f * q.comp[0] * q.comp[2] - 2.0f * q.comp[3] * q.comp[1],
//        2.0f * q.comp[3] * q.comp[0] - 2.0f * q.comp[1] * q.comp[2],
//        Sqr(q.comp[3]) - Sqr(q.comp[0]) - Sqr(q.comp[1]) + Sqr(q.comp[2]),
//        0.0f,
//
//        0.0f, 0.0f, 0.0f, 1.0f);
//}


/*
 * vislib::math::AbstractQuaternion<T, S>::IDX_W
 */
template<class T, class S> const UINT_PTR AbstractQuaternion<T, S>::IDX_W = 3;


/*
 * vislib::math::AbstractQuaternion<T, S>::IDX_X
 */
template<class T, class S> const UINT_PTR AbstractQuaternion<T, S>::IDX_X = 0;


/*
 * vislib::math::AbstractQuaternion<T, S>::IDX_Y
 */
template<class T, class S> const UINT_PTR AbstractQuaternion<T, S>::IDX_Y = 1;


/*
 * vislib::math::AbstractQuaternion<T, S>::IDX_Z
 */
template<class T, class S> const UINT_PTR AbstractQuaternion<T, S>::IDX_Z = 2;

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ABSTRACTQUATERNION_H_INCLUDED */
