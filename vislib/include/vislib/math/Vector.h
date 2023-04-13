/*
 * Vector.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/math/AbstractVector.h"


namespace vislib::math {

/**
 * This is the implementation of an AbstractVector that uses its own memory
 * in a statically allocated array of dimension D. Usually, you want to use
 * this vector class or derived classes.
 *
 * See documentation of AbstractVector for further information about the
 * vector classes.
 */
template<class T, unsigned int D>
class Vector : public AbstractVector<T, D, T[D]> {

public:
    using ValueT = T;

    /**
     * Create a null vector.
     */
    Vector();

    /**
     * Create a new vector initialised with 'components'. 'components' must
     * not be a NULL pointer.
     *
     * @param components The initial vector components.
     */
    explicit inline Vector(const T* components) : Super() {
        ASSERT(components != NULL);
        ::memcpy(this->components, components, D * sizeof(T));
    }

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline Vector(const Vector& rhs) : Super() {
        ::memcpy(this->components, rhs.components, D * sizeof(T));
    }

    /**
     * Create a copy of 'rhs'. This ctor allows for arbitrary vector to
     * vector conversions.
     *
     * @param rhs The vector to be cloned.
     */
    template<class Tp, unsigned int Dp, class Sp>
    Vector(const AbstractVector<Tp, Dp, Sp>& rhs);

    /** Dtor. */
    ~Vector();

    /**
     * Assignment.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this
     */
    inline Vector& operator=(const Vector& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /**
     * Assigment for arbitrary vectors. A valid static_cast between T and Tp
     * is a precondition for instantiating this template.
     *
     * This operation does <b>not</b> create aliases.
     *
     * If the two operands have different dimensions, the behaviour is as
     * follows: If the left hand side operand has lower dimension, the
     * highest (Dp - D) dimensions are discarded. If the left hand side
     * operand has higher dimension, the missing dimensions are filled with
     * zero components.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this
     */
    template<class Tp, unsigned int Dp, class Sp>
    inline Vector& operator=(const AbstractVector<Tp, Dp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

private:
    /** A typedef for the super class. */
    typedef AbstractVector<T, D, T[D]> Super;
};


/*
 * vislib::math::Vector<T, D>::Vector
 */
template<class T, unsigned int D>
Vector<T, D>::Vector() : Super() {
    for (unsigned int d = 0; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, D>::Vector
 */
template<class T, unsigned int D>
template<class Tp, unsigned int Dp, class Sp>
Vector<T, D>::Vector(const AbstractVector<Tp, Dp, Sp>& rhs) : Super() {
    for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
        this->components[d] = static_cast<T>(rhs[d]);
    }
    for (unsigned int d = Dp; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, D>::~Vector
 */
template<class T, unsigned int D>
Vector<T, D>::~Vector() {}


/**
 * Partial template specialisation for two-dimensional vectors. This class
 * provides an additional constructor with single components.
 */
template<class T>
class Vector<T, 2> : public AbstractVector<T, 2, T[2]> {

public:
    using ValueT = T;

    /** Behaves like primary class template. */
    Vector();

    /** Behaves like primary class template. */
    explicit inline Vector(const T* components) : Super() {
        ASSERT(components != NULL);
        ::memcpy(this->components, components, D * sizeof(T));
    }

    /**
     * Create a new vector.
     *
     * @param x The x-component.
     * @param y The y-component.
     */
    inline Vector(const T& x, const T& y) : Super() {
        this->components[0] = x;
        this->components[1] = y;
    }

    /**
     * Create a new vector.
     *
     * @param val Value for all components.
     */
    inline Vector(const T& val) : Super() {
        this->components[0] = val;
        this->components[1] = val;
    }

    /** Behaves like primary class template. */
    inline Vector(const Vector& rhs) : Super() {
        ::memcpy(this->components, rhs.components, D * sizeof(T));
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    Vector(const AbstractVector<Tp, Dp, Sp>& rhs);

    /** Behaves like primary class template. */
    ~Vector();

    /** Behaves like primary class template. */
    inline Vector& operator=(const Vector& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    inline Vector& operator=(const AbstractVector<Tp, Dp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

private:
    /** The dimension of the vector. */
    static const unsigned int D;

    /** A typedef for the super class. */
    typedef AbstractVector<T, 2, T[2]> Super;
};


/*
 * vislib::math::Vector<T, 2>::Vector
 */
template<class T>
Vector<T, 2>::Vector() : Super() {
    for (unsigned int d = 0; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, 2>::Vector
 */
template<class T>
template<class Tp, unsigned int Dp, class Sp>
Vector<T, 2>::Vector(const AbstractVector<Tp, Dp, Sp>& rhs) : Super() {
    for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
        this->components[d] = static_cast<T>(rhs[d]);
    }
    for (unsigned int d = Dp; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, 2>::~Vector
 */
template<class T>
Vector<T, 2>::~Vector() {}


/*
 * Vector<T, 2>::D
 */
template<class T>
const unsigned int Vector<T, 2>::D = 2;


/**
 * Partial template specialisation for three-dimensional vectors. This
 * class provides an additional constructor with single components.
 */
template<class T>
class Vector<T, 3> : public AbstractVector<T, 3, T[3]> {

public:
    using ValueT = T;

    /** Behaves like primary class template. */
    Vector();

    /** Behaves like primary class template. */
    explicit inline Vector(const T* components) : Super() {
        ASSERT(components != NULL);
        ::memcpy(this->components, components, D * sizeof(T));
    }

    /**
     * Create a new vector.
     *
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     */
    inline Vector(const T& x, const T& y, const T& z) : Super() {
        this->components[0] = x;
        this->components[1] = y;
        this->components[2] = z;
    }

    /**
     * Create a new vector.
     *
     * @param val Value for all components.
     */
    inline Vector(const T& val) : Super() {
        this->components[0] = val;
        this->components[1] = val;
        this->components[2] = val;
    }

    /** Behaves like primary class template. */
    inline Vector(const Vector& rhs) : Super() {
        ::memcpy(this->components, rhs.components, D * sizeof(T));
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    Vector(const AbstractVector<Tp, Dp, Sp>& rhs);

    /** Behaves like primary class template. */
    ~Vector();

    /** Behaves like primary class template. */
    inline Vector& operator=(const Vector& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    inline Vector& operator=(const AbstractVector<Tp, Dp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

private:
    /** The dimension of the vector. */
    static const unsigned int D;

    /** A typedef for the super class. */
    typedef AbstractVector<T, 3, T[3]> Super;
};


/*
 * vislib::math::Vector<T, 3>::Vector
 */
template<class T>
Vector<T, 3>::Vector() : Super() {
    for (unsigned int d = 0; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, 3>::Vector
 */
template<class T>
template<class Tp, unsigned int Dp, class Sp>
Vector<T, 3>::Vector(const AbstractVector<Tp, Dp, Sp>& rhs) : Super() {
    for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
        this->components[d] = static_cast<T>(rhs[d]);
    }
    for (unsigned int d = Dp; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, 3>::~Vector
 */
template<class T>
Vector<T, 3>::~Vector() {}


/*
 * Vector<T, 3>::D
 */
template<class T>
const unsigned int Vector<T, 3>::D = 3;


/**
 * Partial template specialisation for four-dimensional vectors. This
 * class provides an additional constructor with single components.
 */
template<class T>
class Vector<T, 4> : public AbstractVector<T, 4, T[4]> {

public:
    using ValueT = T;

    /** Behaves like primary class template. */
    Vector();

    /** Behaves like primary class template. */
    explicit inline Vector(const T* components) : Super() {
        ASSERT(components != NULL);
        ::memcpy(this->components, components, D * sizeof(T));
    }

    /**
     * Create a new vector.
     *
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     * @param w The w-component.
     */
    inline Vector(const T& x, const T& y, const T& z, const T& w) : Super() {
        this->components[0] = x;
        this->components[1] = y;
        this->components[2] = z;
        this->components[3] = w;
    }

    /**
     * Create a new vector.
     *
     * @param val Value for all components.
     */
    inline Vector(const T& val) : Super() {
        this->components[0] = val;
        this->components[1] = val;
        this->components[2] = val;
        this->components[3] = val;
    }

    /** Behaves like primary class template. */
    inline Vector(const Vector& rhs) : Super() {
        ::memcpy(this->components, rhs.components, D * sizeof(T));
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    Vector(const AbstractVector<Tp, Dp, Sp>& rhs);

    /** Behaves like primary class template. */
    ~Vector();

    /** Behaves like primary class template. */
    inline Vector& operator=(const Vector& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    inline Vector& operator=(const AbstractVector<Tp, Dp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

private:
    /** The dimension of the vector. */
    static const unsigned int D;

    /** A typedef for the super class. */
    typedef AbstractVector<T, 4, T[4]> Super;
};


/*
 * vislib::math::Vector<T, 4>::Vector
 */
template<class T>
Vector<T, 4>::Vector() : Super() {
    for (unsigned int d = 0; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, 4>::Vector
 */
template<class T>
template<class Tp, unsigned int Dp, class Sp>
Vector<T, 4>::Vector(const AbstractVector<Tp, Dp, Sp>& rhs) : Super() {
    for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
        this->components[d] = static_cast<T>(rhs[d]);
    }
    for (unsigned int d = Dp; d < D; d++) {
        this->components[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Vector<T, 4>::~Vector
 */
template<class T>
Vector<T, 4>::~Vector() {
    // intentionally empty
}


/*
 * Vector<T, 4>::D
 */
template<class T>
const unsigned int Vector<T, 4>::D = 4;

template<typename T, unsigned int D>
bool operator<(Vector<T, D> const& lhs, Vector<T, D> const& rhs) {
    for (int i = 0; i < D; ++i) {
        if (lhs[i] >= rhs[i]) {
            return false;
        }
    }
    return true;
}

template<typename T, unsigned int D>
bool operator>(Vector<T, D> const& lhs, Vector<T, D> const& rhs) {
    for (int i = 0; i < D; ++i) {
        if (lhs[i] <= rhs[i]) {
            return false;
        }
    }
    return true;
}

} // namespace vislib::math

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
