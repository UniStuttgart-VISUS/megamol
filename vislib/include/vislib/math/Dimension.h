/*
 * Dimension.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/math/AbstractDimension.h"


namespace vislib::math {

/**
 * This class represents extents in D-dimensional space.
 */
template<class T, unsigned int D>
class Dimension : public AbstractDimension<T, D, T[D]> {

public:
    /**
     * Create a zero sized dimension.
     */
    Dimension();

    /**
     * Create a dimension using the data from the 'dimension' array.
     *
     * @param dimension The initial value in an array. This pointer must not
     *                  be NULL.
     */
    inline Dimension(const T* dimension) {
        ASSERT(dimension != NULL);
        ::memcpy(this->dimension, dimension, D * sizeof(T));
    }

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline Dimension(const Dimension& rhs) {
        ::memcpy(this->dimension, rhs.dimension, D * sizeof(T));
    }

    /**
     * Allows arbitrary Dimension to dimension conversion. If the new
     * object has a smaller dimensionality, the values will be truncated,
     * if it has larger dimensionality, it will be padded with zeroes.
     *
     * @param rhs The object to copy.
     */
    template<class Tp, unsigned int Dp, class Sp>
    Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs);

    /** Dtor. */
    ~Dimension();

    /**
     * Assignment operator. This operator never creates an alias.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    inline Dimension& operator=(const Dimension& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /**
     * This operator allows for arbitrary dimension to dimension
     * assignments. If the left hand side operand has a smaller dimension,
     * the values will be truncated, if it has larger dimensionality, it
     * will be padded with zeroes.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    template<class Tp, unsigned int Dp, class Sp>
    inline Dimension& operator=(const AbstractDimension<Tp, Dp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

protected:
    /** A typedef for the super class. */
    typedef AbstractDimension<T, D, T[D]> Super;
};


/*
 * vislib::math::Dimension<T, D>::Dimension
 */
template<class T, unsigned int D>
Dimension<T, D>::Dimension() {
    for (unsigned int d = 0; d < D; d++) {
        this->dimension[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Dimension<T, D>::Dimension
 */
template<class T, unsigned int D>
template<class Tp, unsigned int Dp, class Sp>
Dimension<T, D>::Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs) {
    for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
        this->dimension[d] = static_cast<T>(rhs[d]);
    }
    for (unsigned int d = Dp; d < D; d++) {
        this->dimension[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Dimension<T, D>::~Dimension
 */
template<class T, unsigned int D>
Dimension<T, D>::~Dimension() {}


/**
 * Partial template specialisation for the two-dimensional case. This
 * class provides an additional constructor for creating a dimension by
 * specifying the components separately.
 */
template<class T>
class Dimension<T, 2> : public AbstractDimension<T, 2, T[2]> {

public:
    /** Behaves like primary class template. */
    Dimension();

    /** Behaves like primary class template. */
    inline Dimension(const T* dimension) {
        ASSERT(dimension != NULL);
        ::memcpy(this->dimension, dimension, D * sizeof(T));
    }

    /**
     * Create a new dimension.
     *
     * @param width  The width.
     * @param height The height.
     */
    inline Dimension(const T& width, const T& height) {
        this->dimension[0] = width;
        this->dimension[1] = height;
    }

    /** Behaves like primary class template. */
    inline Dimension(const Dimension& rhs) {
        ::memcpy(this->dimension, rhs.dimension, D * sizeof(T));
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs);

    /** Behaves like primary class template. */
    ~Dimension();

    /** Behaves like primary class template. */
    inline Dimension& operator=(const Dimension& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    inline Dimension& operator=(const AbstractDimension<Tp, Dp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

protected:
    /** The dimensionality. */
    static const unsigned int D;

    /** A typedef for the super class. */
    typedef AbstractDimension<T, 2, T[2]> Super;
};


/*
 * vislib::math::Dimension<T, 2>::Dimension
 */
template<class T>
Dimension<T, 2>::Dimension() {
    for (unsigned int d = 0; d < D; d++) {
        this->dimension[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Dimension<T, 2>::Dimension
 */
template<class T>
template<class Tp, unsigned int Dp, class Sp>
Dimension<T, 2>::Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs) {
    for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
        this->dimension[d] = static_cast<T>(rhs[d]);
    }
    for (unsigned int d = Dp; d < D; d++) {
        this->dimension[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Dimension<T, 2>::~Dimension
 */
template<class T>
Dimension<T, 2>::~Dimension() {}


/*
 * vislib::math::Dimension<T, 2>::D
 */
template<class T>
const unsigned int Dimension<T, 2>::D = 2;


/**
 * Partial template specialisation for the three-dimensional case. This
 * class provides an additional constructor for creating a dimension by
 * specifying the components separately.
 */
template<class T>
class Dimension<T, 3> : public AbstractDimension<T, 3, T[3]> {

public:
    /** Behaves like primary class template. */
    Dimension();

    /** Behaves like primary class template. */
    inline Dimension(const T* dimension) {
        ASSERT(dimension != NULL);
        ::memcpy(this->dimension, dimension, D * sizeof(T));
    }

    /**
     * Create a new dimension.
     *
     * @param width  The width.
     * @param height The height.
     * @param depth  The depth.
     */
    inline Dimension(const T& width, const T& height, const T& depth) {
        this->dimension[0] = width;
        this->dimension[1] = height;
        this->dimension[2] = depth;
    }

    /** Behaves like primary class template. */
    inline Dimension(const Dimension& rhs) {
        ::memcpy(this->dimension, rhs.dimension, D * sizeof(T));
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs);

    /** Behaves like primary class template. */
    ~Dimension();

    /** Behaves like primary class template. */
    inline Dimension& operator=(const Dimension& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /** Behaves like primary class template. */
    template<class Tp, unsigned int Dp, class Sp>
    inline Dimension& operator=(const AbstractDimension<Tp, Dp, Sp>& rhs) {
        Super::operator=(rhs);
        return *this;
    }

protected:
    /** The dimensionality. */
    static const unsigned int D;

    /** A typedef for the super class. */
    typedef AbstractDimension<T, 3, T[3]> Super;
};


/*
 * vislib::math::Dimension<T, 3>::Dimension
 */
template<class T>
Dimension<T, 3>::Dimension() {
    for (unsigned int d = 0; d < D; d++) {
        this->dimension[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Dimension<T, 3>::Dimension
 */
template<class T>
template<class Tp, unsigned int Dp, class Sp>
Dimension<T, 3>::Dimension(const AbstractDimension<Tp, Dp, Sp>& rhs) {
    for (unsigned int d = 0; (d < D) && (d < Dp); d++) {
        this->dimension[d] = static_cast<T>(rhs[d]);
    }
    for (unsigned int d = Dp; d < D; d++) {
        this->dimension[d] = static_cast<T>(0);
    }
}


/*
 * vislib::math::Dimension<T, 3>::~Dimension
 */
template<class T>
Dimension<T, 3>::~Dimension() {}


/*
 * vislib::math::Dimension<T, 3>::D
 */
template<class T>
const unsigned int Dimension<T, 3>::D = 3;

} // namespace vislib::math

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
