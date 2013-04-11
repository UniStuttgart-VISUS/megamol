/*
 * AbstractPolynom.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPOLYNOM_H_INCLUDED
#define VISLIB_ABSTRACTPOLYNOM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractPolynomImpl.h"
#include "vislib/UnsupportedOperationException.h"


namespace vislib {
namespace math {


    /**
     * All polynom implementations must inherit from this class. Do not
     * inherit directly from AbstractPolynomImpl as the abstraction layer of
     * AbstractPolynom ensures that the implementation can work correctly and
     * instantiate derived classes.
     *
     * The one-dimensional polynom is defined by its coefficients a_0 ... a_d
     * as:
     *  f(x) := a_d * x^d + a_{d-1} * x^{d-1} + ... + a_1 * x + a_0
     *
     * T scalar type
     * D Degree of the polynom
     * S Coefficient storage
     */
    template<class T, unsigned int D, class S>
    class AbstractPolynom
        : public AbstractPolynomImpl<T, D, S, AbstractPolynom> {
    public:

        /** Dtor. */
        ~AbstractPolynom(void);

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. If the output array
         * as not enough space to store all found roots only up to 'size'
         * roots will be stored. The order of the roots is undefined.
         *
         * A polynom of degree D has a maximum number of D roots.
         *
         * @param outRoots Pointer to the array to receive the found roots
         * @param size The size of the array to receive the found roots in
         *             number of elements.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T *outRoots, unsigned int size) const {
            if (size == 0) return 0;
            switch (this->EffectiveDegree()) {
                case 0: return 0;
                case 1: return this->findRootsDeg1(this->coefficients[0],
                            this->coefficients[1], outRoots, size);
                case 2: return this->findRootsDeg2(this->coefficients[0],
                            this->coefficients[1], this->coefficients[2],
                            outRoots, size);
                case 3: return this->findRootsDeg3(this->coefficients[0],
                            this->coefficients[1], this->coefficients[2],
                            this->coefficients[3], outRoots, size);
                case 4: return this->findRootsDeg4(this->coefficients[0],
                            this->coefficients[1], this->coefficients[2],
                            this->coefficients[3], this->coefficients[4],
                            outRoots, size);
                default: break;
            }

            // TODO: Implement numeric root finding (bairstow?)

            throw UnsupportedOperationException("FindRoots",
                __FILE__, __LINE__);

            return 0;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         *
         * @throw IllegalParamException if 'rhs' has an effective degree larger
         *        than D.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPolynom<T, D, S>& operator=(
                const AbstractPolynom<Tp, Dp, Sp>& rhs) {
            Super::operator=(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractPolynomImpl<T, D, S, vislib::math::AbstractPolynom>
            Super;

        /** Ctor. */
        inline AbstractPolynom(void) : Super() { }

    private:

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1,
            template<class Tf2, unsigned int Df2, class Sf2> class Cf>
            friend class AbstractPolynomImpl;

    };


    /*
     * AbstractPolynom<T, D, S>::~AbstractPolynom
     */
    template<class T, unsigned int D, class S>
    AbstractPolynom<T, D, S>::~AbstractPolynom(void) {
        // intentionally empty
    }


    /**
     * Partial template specialisation for polynoms of degree 1.
     */
    template<class T, class S>
    class AbstractPolynom<T, 1, S> :
        public AbstractPolynomImpl<T, 1, S, AbstractPolynom> {
    public:

        /** Dtor. */
        ~AbstractPolynom(void);

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. If the output array
         * as not enough space to store all found roots only up to 'size'
         * roots will be stored. The order of the roots is undefined.
         *
         * A polynom of degree D has a maximum number of D roots.
         *
         * @param outRoots Pointer to the array to receive the found roots
         * @param size The size of the array to receive the found roots in
         *             number of elements.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T *outRoots, unsigned int size) const {
            if (size == 0) return 0;
            if (IsEqual(this->coefficients[1], static_cast<T>(0))) return 0;
            return findRootsDeg1(this->coefficients[0], this->coefficients[1],
                outRoots, size);
        }

        /**
         * Finds the root of the polynom.
         *
         * A polynom of degree 1 has a maximum number of 1 root.
         *
         * @param outRoot1 Variable to receive the root.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T& outRoot) const {
            return this->FindRoots(&outRoot, 1);
        }

        /**
         * Finds the root of the polynom.
         *
         * A polynom of degree 1 has a maximum number of 1 root.
         *
         * @param outRoot Variable to receive the root.
         *
         * @return True if there is a root and it has been written to outRoot.
         */
        inline bool FindRoot(T& outRoot) const {
            return (this->FindRoots(&outRoot, 1) == 1);
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         *
         * @throw IllegalParamException if 'rhs' has an effective degree larger
         *        than D.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPolynom<T, 1, S>& operator=(
                const AbstractPolynom<Tp, Dp, Sp>& rhs) {
            Super::operator=(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractPolynomImpl<T, 1, S, vislib::math::AbstractPolynom>
            Super;

        /** Ctor. */
        inline AbstractPolynom(void) : Super() { }

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1,
            template<class Tf2, unsigned int Df2, class Sf2> class Cf>
            friend class AbstractPolynomImpl;

    };


    /*
     * AbstractPolynom<T, 1, S>::~AbstractPolynom
     */
    template<class T, class S>
    AbstractPolynom<T, 1, S>::~AbstractPolynom(void) {
        // intentionally empty
    }


    /**
     * Partial template specialisation for polynoms of degree 2.
     */
    template<class T, class S>
    class AbstractPolynom<T, 2, S> :
        public AbstractPolynomImpl<T, 2, S, AbstractPolynom> {
    public:

        /** Dtor. */
        ~AbstractPolynom(void);

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. If the output array
         * as not enough space to store all found roots only up to 'size'
         * roots will be stored. The order of the roots is undefined.
         *
         * A polynom of degree D has a maximum number of D roots.
         *
         * @param outRoots Pointer to the array to receive the found roots
         * @param size The size of the array to receive the found roots in
         *             number of elements.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T *outRoots, unsigned int size) const {
            if (size == 0) return 0;
            if (IsEqual(this->coefficients[2], static_cast<T>(0))) {
                if (IsEqual(this->coefficients[1], static_cast<T>(0))) {
                    return 0;
                }
                return findRootsDeg1(this->coefficients[0],
                    this->coefficients[1], outRoots, size);
            }
            return findRootsDeg2(this->coefficients[0], this->coefficients[1],
                this->coefficients[2], outRoots, size);
        }

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. The order of the roots
         * is undefined.
         *
         * A polynom of degree 2 has a maximum number of 2 roots.
         *
         * @param outRoot1 Variable to receive the first root.
         * @param outRoot2 Variable to receive the second root.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T& outRoot1, T& outRoot2) const {
            T r[2];
            unsigned int rv = this->FindRoots(r, 2);
            if (r > 0) outRoot1 = r[0];
            if (r > 1) outRoot2 = r[1];
            return rv;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         *
         * @throw IllegalParamException if 'rhs' has an effective degree larger
         *        than D.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPolynom<T, 2, S>& operator=(
                const AbstractPolynom<Tp, Dp, Sp>& rhs) {
            Super::operator=(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractPolynomImpl<T, 2, S, vislib::math::AbstractPolynom>
            Super;

        /** Ctor. */
        inline AbstractPolynom(void) : Super() { }

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1,
            template<class Tf2, unsigned int Df2, class Sf2> class Cf>
            friend class AbstractPolynomImpl;

    };


    /*
     * AbstractPolynom<T, 2, S>::~AbstractPolynom
     */
    template<class T, class S>
    AbstractPolynom<T, 2, S>::~AbstractPolynom(void) {
        // intentionally empty
    }


    /**
     * Partial template specialisation for polynoms of degree 3.
     */
    template<class T, class S>
    class AbstractPolynom<T, 3, S> :
        public AbstractPolynomImpl<T, 3, S, AbstractPolynom> {
    public:

        /** Dtor. */
        ~AbstractPolynom(void);

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. If the output array
         * as not enough space to store all found roots only up to 'size'
         * roots will be stored. The order of the roots is undefined.
         *
         * A polynom of degree D has a maximum number of D roots.
         *
         * @param outRoots Pointer to the array to receive the found roots
         * @param size The size of the array to receive the found roots in
         *             number of elements.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T *outRoots, unsigned int size) const {
            if (size == 0) return 0;
            if (IsEqual(this->coefficients[3], static_cast<T>(0))) {
                if (IsEqual(this->coefficients[2], static_cast<T>(0))) {
                    if (IsEqual(this->coefficients[1], static_cast<T>(0))) {
                        return 0;
                    }
                    return this->findRootsDeg1(this->coefficients[0],
                        this->coefficients[1], outRoots, size);
                }
                return this->findRootsDeg2(this->coefficients[0],
                    this->coefficients[1], this->coefficients[2], outRoots,
                    size);
            }
            return this->findRootsDeg3(this->coefficients[0],
                this->coefficients[1], this->coefficients[2],
                this->coefficients[3], outRoots, size);
        }

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. The order of the roots
         * is undefined.
         *
         * A polynom of degree 3 has a maximum number of 3 roots.
         *
         * @param outRoot1 Variable to receive the first root.
         * @param outRoot2 Variable to receive the second root.
         * @param outRoot3 Variable to receive the third root.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T& outRoot1, T& outRoot2, T& outRoot3)
                const {
            T r[3];
            unsigned int rv = this->FindRoots(r, 3);
            if (r > 0) outRoot1 = r[0];
            if (r > 1) outRoot2 = r[1];
            if (r > 2) outRoot3 = r[2];
            return rv;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         *
         * @throw IllegalParamException if 'rhs' has an effective degree larger
         *        than D.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPolynom<T, 3, S>& operator=(
                const AbstractPolynom<Tp, Dp, Sp>& rhs) {
            Super::operator=(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractPolynomImpl<T, 3, S, vislib::math::AbstractPolynom>
            Super;

        /** Ctor. */
        inline AbstractPolynom(void) : Super() { }

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1,
            template<class Tf2, unsigned int Df2, class Sf2> class Cf>
            friend class AbstractPolynomImpl;

    };


    /*
     * AbstractPolynom<T, 3, S>::~AbstractPolynom
     */
    template<class T, class S>
    AbstractPolynom<T, 3, S>::~AbstractPolynom(void) {
        // intentionally empty
    }


    /**
     * Partial template specialisation for polynoms of degree 4.
     */
    template<class T, class S>
    class AbstractPolynom<T, 4, S> :
        public AbstractPolynomImpl<T, 4, S, AbstractPolynom> {
    public:

        /** Dtor. */
        ~AbstractPolynom(void);

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. If the output array
         * as not enough space to store all found roots only up to 'size'
         * roots will be stored. The order of the roots is undefined.
         *
         * A polynom of degree D has a maximum number of D roots.
         *
         * @param outRoots Pointer to the array to receive the found roots
         * @param size The size of the array to receive the found roots in
         *             number of elements.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T *outRoots, unsigned int size) const {
            if (size == 0) return 0;
            if (IsEqual(this->coefficients[4], static_cast<T>(0))) {
                if (IsEqual(this->coefficients[3], static_cast<T>(0))) {
                    if (IsEqual(this->coefficients[2], static_cast<T>(0))) {
                        if (IsEqual(this->coefficients[1],
                                static_cast<T>(0))) {
                            return 0;
                        }
                        return this->findRootsDeg1(this->coefficients[0],
                            this->coefficients[1], outRoots, size);
                    }
                    return this->findRootsDeg2(this->coefficients[0],
                        this->coefficients[1], this->coefficients[2],
                        outRoots, size);
                }
                return this->findRootsDeg3(this->coefficients[0],
                    this->coefficients[1], this->coefficients[2],
                    this->coefficients[3], outRoots, size);
            }
            return this->findRootsDeg4(this->coefficients[0],
                this->coefficients[1], this->coefficients[2],
                this->coefficients[3], this->coefficients[4], outRoots, size);
        }

        /**
         * Finds the roots of the polynom. If roots only touch the x-axis they
         * will be present more than once in the output. The order of the roots
         * is undefined.
         *
         * A polynom of degree 4 has a maximum number of 4 roots.
         *
         * @param outRoot1 Variable to receive the first root.
         * @param outRoot2 Variable to receive the second root.
         * @param outRoot3 Variable to receive the third root.
         * @param outRoot4 Variable to receive the fourth root.
         *
         * @return The number of valid roots in the out parameter.
         */
        inline unsigned int FindRoots(T& outRoot1, T& outRoot2, T& outRoot3,
                T& outRoot4) const {
            T r[4];
            unsigned int rv = this->findRoots<4>(r, 4);
            if (r > 0) outRoot1 = r[0];
            if (r > 1) outRoot2 = r[1];
            if (r > 2) outRoot3 = r[2];
            if (r > 3) outRoot4 = r[3];
            return rv;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         *
         * @throw IllegalParamException if 'rhs' has an effective degree larger
         *        than D.
         */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractPolynom<T, 4, S>& operator=(
                const AbstractPolynom<Tp, Dp, Sp>& rhs) {
            Super::operator=(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractPolynomImpl<T, 4, S, vislib::math::AbstractPolynom>
            Super;

        /** Ctor. */
        inline AbstractPolynom(void) : Super() { }

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1,
            template<class Tf2, unsigned int Df2, class Sf2> class Cf>
            friend class AbstractPolynomImpl;

    };


    /*
     * AbstractPolynom<T, 4, S>::~AbstractPolynom
     */
    template<class T, class S>
    AbstractPolynom<T, 4, S>::~AbstractPolynom(void) {
        // intentionally empty
    }


} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPOLYNOM_H_INCLUDED */

