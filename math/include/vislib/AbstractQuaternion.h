/*
 * AbstractQuaternion.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTQUATERNION_H_INCLUDED
#define VISLIB_ABSTRACTQUATERNION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <cmath>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/mathfunctions.h"
#include "vislib/types.h"
#include "vislib/Vector.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif /* !M_PI */

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
         * @param outAngle Receives the angle in radians. 
         * @param outAxis  Receives the vector representing the rotation
         *                 axis. The vector is guaranteed to be normalised.
         */
        void AngleAndAxis(T& outAngle, Vector<T, 3>& outAxis) const;

        /**
         * Conjugate the quaternion.
         */
        void Conjugate(void);

        /**
         * Answer the i-component (= x-component) of the quaternion.
         *
         * @return The i-component.
         */
        inline const T& GetI(void) const {
            return this->components[IDX_X];
        }

        /**
         * Answer the j-component (= y-component) of the quaternion.
         *
         * @return The j-component.
         */
        inline const T& GetJ(void) const {
            return this->components[IDX_Y];
        }

        /**
         * Answer the k-component (= z-component) of the quaternion.
         *
         * @return The k-component.
         */
        inline const T& GetK(void) const {
            return this->components[IDX_Z];
        }

        /**
         * Answer the r-component (= w-component) of the quaternion.
         *
         * @return The r-component.
         */
        inline const T& GetR(void) const {
            return this->components[IDX_W];
        }

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
        inline const T& GetY(void) const {
            return this->components[IDX_Y];
        }

        /**
         * Answer the z-component of the quaternion.
         *
         * @return The z-component.
         */
        inline const T& GetZ(void) const {
            return this->components[IDX_Z];
        }

        /**
         * Answer the i-component (= x-component) of the quaternion.
         *
         * @return The i-component.
         */
        inline const T& I(void) const {
            return this->components[IDX_X];
        }

        /**
         * Answer the j-component (= y-component) of the quaternion.
         *
         * @return The j-component.
         */
        inline const T& J(void) const {
            return this->components[IDX_Y];
        }

        /**
         * Answer the k-component (= z-component) of the quaternion.
         *
         * @return The k-component.
         */
        inline const T& K(void) const {
            return this->components[IDX_Z];
        }

        //Quaternion Inverse(void) const;

        /**
         * Interpolates between 'this' and 'rhs' linearly based on
         * '0 <= t <= 1'.
         *
         * @param rhs The second point to interpolate to (t=1)
         * @param t The interpolation value (0..1)
         *
         * @return The interpolation result
         */
        template<class Sp, class Tp>
        inline AbstractQuaternion<T, T[4]>
        Interpolate(const AbstractQuaternion<T, Sp>& rhs, Tp t) const {
            AbstractQuaternion<T, T[4]> rv;
            rv.Slerp(t, *this, rhs);
            return rv;
        }

        /** 
         * Invert the quaternion.
         */
        void Invert(void);

        /**
         * Answer whether the quaternion is pure imaginary.
         *
         * This operation uses an epsilon compare for T.
         *
         * @return true If the real part is zero, false otherwise.
         */
        inline bool IsPure(void) const {
            return IsEqual<T>(this->components[IDX_W], static_cast<T>(0));
        }

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
         * Answer the r-component (= w-component) of the quaternion.
         *
         * @return The r-component.
         */
        inline const T& R(void) const {
            return this->components[IDX_W];
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
         * Set the components of the quaternion in such a way that is represents
         * the specified rotation.
         *
         * @param angle The rotation angle in radians.
         * @param axis  The vector specifying the rotation axis.
         */
        template<class Tp, class Sp>
        void Set(const T& angle, const AbstractVector<Tp, 3, Sp>& axis);

        /**
         * Tries to set the quarternion from rotation matrix components.
         *
         * @param m11 Matrix component row 1, column 1.
         * @param m12 Matrix component row 1, column 2.
         * @param m13 Matrix component row 1, column 3.
         * @param m21 Matrix component row 2, column 1.
         * @param m22 Matrix component row 2, column 2.
         * @param m23 Matrix component row 2, column 3.
         * @param m31 Matrix component row 3, column 1.
         * @param m32 Matrix component row 3, column 2.
         * @param m33 Matrix component row 3, column 3.
         *
         * @throw IllegalParamException if the matrix components do not seem
         *                              to form a rotation-only matrix.
         */
        void SetFromRotationMatrix(const T& m11, const T& m12, const T& m13,
                const T& m21, const T& m22, const T& m23,
                const T& m31, const T& m32, const T& m33);

        /**
         * Set the i-component (= x-component) of the quaternion.
         *
         * @param i The new value for the component.
         */
        inline void SetI(const T& i) {
            this->components[IDX_X] = i;
        }

        /**
         * Set the j-component (= y-component) of the quaternion.
         *
         * @param j The new value for the component.
         */
        inline void SetJ(const T& j) {
            this->components[IDX_Y] = j;
        }

        /**
         * Set the k-component (= z-component) of the quaternion.
         *
         * @param k The new value for the component.
         */
        inline void SetK(const T& k) {
            this->components[IDX_Z] = k;
        }

        /**
         * Set the r-component (= w-component) of the quaternion.
         *
         * @param r The new value for the component.
         */
        inline void SetR(const T& r) {
            this->components[IDX_W] = r;
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
         * Sets the components of the quaternion by performing a slerp 
         * interpolation between 'a' and 'b' using 'alpha' as interpolation
         * parameter [0, 1]. If 'alpha' is zero 'a' is used; if 'alpha is one
         * 'b' is used.
         *
         * @param alpha The interpolation parameter [0, 1]
         * @param a The first interpolation value, used if 'alpha' is zero.
         * @param b The second interpolation value, used if 'alpha' is one.
         */
        template<class Sp1, class Sp2>
        void Slerp(float alpha, const AbstractQuaternion<T, Sp1>& a, 
            const AbstractQuaternion<T, Sp2>& b);

        /**
         * Make this quaternion the square of itself.
         */
        void Square(void);

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

        /**
         * Multiplication of two quaternions.
         *
         * @param rhs The right hand side operand.
         *
         * @return The result of *this * rhs.
         */
        template<class Sp>
        AbstractQuaternion<T, T[4]> operator *(
            const AbstractQuaternion<T, Sp>& rhs) const;

        /**
         * Multiplies a quaternion and a vector. The result is the vector 'rhs'
         * rotated by the rotation that is defined through this quaternion.
         *
         * Note: This multiplication does not compute the quaternion 
         * multiplication *this * (rhs, 0), but the actual vector transformation
         * that is equivalent to *this * (rhs, 0) * *this->Conjugate().
         *
         * @param rhs The right hand side operand.
         *
         * @return The resulting vector.
         */
        template<class Sp>
        Vector<T, 3> operator *(const AbstractVector<T, 3, Sp>& rhs) const;

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
        inline AbstractQuaternion(void) {}

        /**
         * Create a new quaternion.
         *
         * WARNING: Do not call this ctor but on deep storage instantiations!
         *
         * @param x The new x-component.
         * @param y The new y-component.
         * @param z The new z-component.
         * @param w The new w-component.
         */
        inline AbstractQuaternion(const T& x, const T& y, const T& z, 
                const T& w) {
            BECAUSE_I_KNOW(sizeof(this->components) == 4 * sizeof(T));
            this->components[IDX_X] = x;
            this->components[IDX_Y] = y;
            this->components[IDX_Z] = z;
            this->components[IDX_W] = w;
        }

        /** 
         * The components of the quaterion. These are stored in the following
         * order: x, y, z (the vector), w.
         */
        S components;
    };


    /*
     * vislib::math::AbstractQuaternion<T, S>::~AbstractQuaternion
     */
    template<class T, class S>
    AbstractQuaternion<T, S>::~AbstractQuaternion(void) {
    }


    /*
     * vislib::math::AbstractQuaternion<T, S>::AngleAndAxis
     */
    template<class T, class S>
    void AbstractQuaternion<T, S>::AngleAndAxis(T& outAngle, 
            Vector<T, 3>& outAxis) const {
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
     * vislib::math::AbstractQuaternion<T, S>::Conjugate
     */
    template<class T, class S> void AbstractQuaternion<T, S>::Conjugate(void) {
        this->components[IDX_X] *= static_cast<T>(-1);
        this->components[IDX_Y] *= static_cast<T>(-1);
        this->components[IDX_Z] *= static_cast<T>(-1);
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
     * vislib::math::AbstractQuaternion<T, S>::Set
     */
    template<class T, class S>
    template<class Tp, class Sp>
    void AbstractQuaternion<T, S>::Set(const T& angle, 
            const AbstractVector<Tp, 3, Sp>& axis) {
        Vector<Tp, 3> ax(axis);
        T len = ax.Normalise();
        double halfAngle = 0.5 * static_cast<double>(angle);

        if (!IsEqual(len, static_cast<T>(0))){
            len = static_cast<T>(::sin(halfAngle) / len);
            this->components[IDX_X] = ax.X() * len;
            this->components[IDX_Y] = ax.Y() * len;
            this->components[IDX_Z] = ax.Z() * len;
            this->components[IDX_W] = static_cast<T>(::cos(halfAngle));

        } else {
            this->components[IDX_X] = this->components[IDX_Y] 
                = this->components[IDX_Z] = static_cast<T>(0);
            this->components[IDX_W] = static_cast<T>(1);
        }
    }


    /*
     * AbstractQuaternion<T, S>::SetFromRotationMatrix
     */
    template<class T, class S>
    void AbstractQuaternion<T, S>::SetFromRotationMatrix(const T& m11,
            const T& m12, const T& m13, const T& m21, const T& m22,
            const T& m23, const T& m31, const T& m32, const T& m33) {
        Vector<T, 3> xi(m11, m21, m31);
        Vector<T, 3> yi(m12, m22, m32);
        Vector<T, 3> zi(m13, m23, m33);

        xi.Normalise(); // we could throw here, but let's be nice.
        yi.Normalise();
        zi.Normalise();

        Vector<T, 3> xo(static_cast<T>(1), static_cast<T>(0),
            static_cast<T>(0));
        Vector<T, 3> yo(static_cast<T>(0), static_cast<T>(1),
            static_cast<T>(0));
        Vector<T, 3> zo(static_cast<T>(0), static_cast<T>(0),
            static_cast<T>(1));

        AbstractQuaternion<T, T[4]> q1;
        if (xi == xo) {
            // rot 0°
            q1.Set(static_cast<T>(0), static_cast<T>(0),
                static_cast<T>(0), static_cast<T>(1));

        } else if (xi == -xo) {
            // rot 180°
            q1.Set(static_cast<T>(M_PI), yo);

        } else {
            // rot something
            T angle = ::acos(xo.Dot(xi));
            Vector<T, 3> axis = xo.Cross(xi);
            axis.Normalise();
            q1.Set(angle, axis);

            Vector<T, 3> xb = q1 * xo;
            if (xb != xi) {
                q1.Set(-angle, axis);
                xb = q1 * xo;
            }
            if (xb != xi) {
                throw IllegalParamException("Matrix is not rotation-only",
                    __FILE__, __LINE__);
            }

        }

        Vector<T, 3> yb = q1 * yo;
        if (yi == yb) {
            // rot 0°
            if ((q1 * zo) != zi) {
                throw IllegalParamException("Matrix is not rotation-only",
                    __FILE__, __LINE__);
            }

        } else if (yi == -yb) {
            // rot 180°
            AbstractQuaternion<T, T[4]> q2;
            q2.Set(static_cast<T>(M_PI), xi);
            q1 = q2 * q1;

        } else {
            // rot something
            AbstractQuaternion<T, T[4]> q2;
            T angle = ::acos(yb.Dot(yi));
            Vector<T, 3> axis = yb.Cross(yi);
            axis.Normalise();
            q2.Set(angle, axis);

            Vector<T, 3> yc = q2 * yb;
            if (yc != yi) {
                q2.Set(-angle, axis);
                yc = q2 * yb;
            }
            if (yc != yi) {
                throw IllegalParamException("Matrix is not rotation-only",
                    __FILE__, __LINE__);
            }
            q1 = q2 * q1;

        }

        Vector<T, 3> xb = q1 * xo;
        yb = q1 * yo;
        Vector<T, 3> zb = q1 * zo;
        if ((xb != xi) || (yb != yi) || (zb!= zi)) {
            throw IllegalParamException("Matrix is not rotation-only",
                __FILE__, __LINE__);
        }

        *this = q1;
    }


    /*
     * vislib::math::AbstractQuaternion<T, S>::Slerp
     */
    template<class T, class S>
    template<class Sp1, class Sp2>
    void AbstractQuaternion<T, S>::Slerp(float alpha, 
            const AbstractQuaternion<T, Sp1>& a, 
            const AbstractQuaternion<T, Sp2>& b) {

        if (alpha < FLOAT_EPSILON) {
            *this = a;
        } else if (alpha > (1.0f - FLOAT_EPSILON)) {
            *this = b;
        } else {
            bool flipT;
            float slerpFrom, slerpTo, omega, sinOmega;
            float cosOmega = a.X() * b.X() + a.Y() * b.Y() + a.Z() * b.Z() 
                + a.W() * b.W();

            if ((flipT = (cosOmega < 0.0f))) {
                cosOmega = -cosOmega;
            }

            if ((1.0f - cosOmega) > 0.0001f) {
                omega = acosf(cosOmega);
                sinOmega = sinf(omega);
                slerpFrom = sinf((1.0f - alpha) * omega) / sinOmega;
                slerpTo = sinf(alpha * omega) / sinOmega;
                if (flipT) {
                    slerpTo = -slerpTo;
                }
            } else {
                slerpFrom = 1.0f - alpha;
                slerpTo = alpha;
            }

            for (unsigned int i = 0; i < 4; i++) {
                this->components[i] = slerpFrom * a.components[i]
                    + slerpTo * b.components[i];
            }
        }
    }


    /*
     * vislib::math::AbstractQuaternion<T, S>::Square
     */
    template<class T, class S> void AbstractQuaternion<T, S>::Square(void) {
        T tmp = 2 * this->components[IDX_W];
        this->components[IDX_W] = Sqr(this->components[IDX_W])
            -(Sqr(this->components[IDX_X] + Sqr(this->components[IDX_Y]) 
            + Sqr(this->components[IDX_Z])));
        this->components[IDX_X] *= tmp;
        this->components[IDX_Y] *= tmp;
        this->components[IDX_Z] *= tmp;
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
        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
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
            + rhs.W() * this->components[IDX_Z] 
            + this->components[IDX_X] * rhs.Y() 
            - this->components[IDX_Y] * rhs.X(),

            this->components[IDX_W] * rhs.W() 
            - (this->components[IDX_X] * rhs.X()
            + this->components[IDX_Y] * rhs.Y() 
            + this->components[IDX_Z] * rhs.Z()));
    }


    /*
     * vislib::math::AbstractQuaternion<T, S>::operator *
     */
    template<class T, class S>
    template<class Sp>
    Vector<T, 3> AbstractQuaternion<T, S>::operator *(
            const AbstractVector<T, 3, Sp>& rhs) const {
        Vector<T, 3> u(this->components);
        return ((static_cast<T>(2.0) * ((u.Dot(rhs) * u) 
            + (this->W() * u.Cross(rhs))))
            + ((Sqr(this->W()) - u.Dot(u)) * rhs));
    }


    /*
     * vislib::math::AbstractQuaternion<T, S>::IDX_W
     */
    template<class T, class S> 
    const UINT_PTR AbstractQuaternion<T, S>::IDX_W = 3;


    /*
     * vislib::math::AbstractQuaternion<T, S>::IDX_X
     */
    template<class T, class S> 
    const UINT_PTR AbstractQuaternion<T, S>::IDX_X = 0;


    /*
     * vislib::math::AbstractQuaternion<T, S>::IDX_Y
     */
    template<class T, class S> 
    const UINT_PTR AbstractQuaternion<T, S>::IDX_Y = 1;


    /*
     * vislib::math::AbstractQuaternion<T, S>::IDX_Z
     */
    template<class T, class S> 
    const UINT_PTR AbstractQuaternion<T, S>::IDX_Z = 2;

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTQUATERNION_H_INCLUDED */
