/*
 * AbstractDimension.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTDIMENSION_H_INCLUDED
#define VISLIB_ABSTRACTDIMENSION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "AbstractDimensionImpl.h"


namespace vislib {
namespace math {

    /**
     * This class represents extents in D-dimensional space. 
     */
    template<class T, unsigned int D, class S> class AbstractDimension 
            : public AbstractDimensionImpl<T, D, S, AbstractDimension> {

    public:

        /** Dtor. */
        ~AbstractDimension(void);

        /**
         * Assignment operator. This operator never creates an alias.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline AbstractDimension& operator =(const AbstractDimension& rhs) {
            Super::operator =(rhs);
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
        inline AbstractDimension& operator =(
                const AbstractDimension<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef of super class. */
        typedef AbstractDimensionImpl<T, D, S, vislib::math::AbstractDimension> 
            Super;

        /**
         * Disallow instances of this class.
         */
        inline AbstractDimension(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractDimensionImpl;
    };


    /*
     * vislib::math::AbstractDimension<T, D, S>::~AbstractDimension
     */
    template<class T, unsigned int D, class S>
    AbstractDimension<T, D, S>::~AbstractDimension(void) {
    }


    /**
     * This is the partial template specialisation for two-dimensional
     * dimensions. It provides additional named accessors for the components
     * of the dimension.
     */
    template<class T, class S> class AbstractDimension<T, 2, S> 
            : public AbstractDimensionImpl<T, 2, S, AbstractDimension> {

    public:

        /** Behaves like primary class template. */
        ~AbstractDimension(void);

        /**
         * Answer the height of the dimension.
         *
         * @return The height.
         */
        inline const T& GetHeight(void) const {
            return this->dimension[1];
        }

        /**
         * Answer the width of the dimension.
         *
         * @return The width.
         */
        inline const T& GetWidth(void) const {
            return this->dimension[0];
        }

        /**
         * Answer the height of the dimension.
         *
         * @return The height.
         */
        inline const T& Height(void) const {
            return this->dimension[1];
        }

        /**
         * Set the components of the dimension.
         *
         * @param width  The new width.
         * @param height The new height.
         */
        inline void Set(const T& width, const T& height) {
            this->dimension[0] = width;
            this->dimension[1] = height;
        }

        /**
         * Change the height.
         *
         * @param height The new height
         */
        inline void SetHeight(const T& height) {
            this->dimension[1] = height;
        }

        /**
         * Change the width.
         *
         * @param height The new width
         */
        inline void SetWidth(const T& width) {
            this->dimension[0] = width;
        }

        /**
         * Answer the width of the dimension.
         *
         * @return The width.
         */
        inline const T& Width(void) const {
            return this->dimension[0];
        }

        /** Behaves like primary class template. */
        inline AbstractDimension& operator =(const AbstractDimension& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractDimension& operator =(
                const AbstractDimension<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef of super class. */
        typedef AbstractDimensionImpl<T, 2, S, vislib::math::AbstractDimension> 
            Super;

        /**
         * Disallow instances of this class.
         */
        inline AbstractDimension(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractDimensionImpl;
    };


    /*
     * vislib::math::AbstractDimension<T, 2, S>::~AbstractDimension
     */
    template<class T, class S>
    AbstractDimension<T, 2, S>::~AbstractDimension(void) {
    }


    /**
     * This is the partial template specialisation for three-dimensional
     * dimensions. It provides additional named accessors for the components
     * of the dimension.
     */
    template<class T, class S> class AbstractDimension<T, 3, S> 
            : public AbstractDimensionImpl<T, 3, S, AbstractDimension> {

    public:

        /** Behaves like primary class template. */
        ~AbstractDimension(void);

        /**
         * Answer the depth of the dimension.
         *
         * @return The depth.
         */
        inline const T& Depth(void) const {
            return this->dimension[2];
        }

        /**
         * Answer the depth of the dimension.
         *
         * @return The depth.
         */
        inline const T& GetDepth(void) const {
            return this->dimension[2];
        }

        /**
         * Answer the height of the dimension.
         *
         * @return The height.
         */
        inline const T& GetHeight(void) const {
            return this->dimension[1];
        }

        /**
         * Answer the width of the dimension.
         *
         * @return The width.
         */
        inline const T& GetWidth(void) const {
            return this->dimension[0];
        }

        /**
         * Answer the height of the dimension.
         *
         * @return The height.
         */
        inline const T& Height(void) const {
            return this->dimension[1];
        }

        /**
         * Set the components of the dimension.
         *
         * @param width  The new width.
         * @param height The new height.
         * @param depth  The new depth.
         */
        inline void Set(const T& width, const T& height, const T& depth) {
            this->dimension[0] = width;
            this->dimension[1] = height;
            this->dimension[2] = depth;
        }

        /**
         * Change the depth.
         *
         * @param depth The new depth
         */
        inline void SetDepth(const T& depth) {
            this->dimension[2] = depth;
        }

        /**
         * Change the height.
         *
         * @param height The new height
         */
        inline void SetHeight(const T& height) {
            this->dimension[1] = height;
        }

        /**
         * Change the width.
         *
         * @param height The new width
         */
        inline void SetWidth(const T& width) {
            this->dimension[0] = width;
        }

        /**
         * Answer the width of the dimension.
         *
         * @return The width.
         */
        inline const T& Width(void) const {
            return this->dimension[0];
        }

        /** Behaves like primary class template. */
        inline AbstractDimension& operator =(const AbstractDimension& rhs) {
            Super::operator =(rhs);
            return *this;
        }

        /** Behaves like primary class template. */
        template<class Tp, unsigned int Dp, class Sp>
        inline AbstractDimension& operator =(
                const AbstractDimension<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** Typedef of super class. */
        typedef AbstractDimensionImpl<T, 3, S, vislib::math::AbstractDimension> 
            Super;

        /**
         * Disallow instances of this class.
         */
        inline AbstractDimension(void) : Super() {}

        /* Allow instances created by the implementation class. */
        template<class Tf1, unsigned int Df1, class Sf1, 
            template<class Tf2, unsigned int Df2, class Sf2> class Cf> 
            friend class AbstractDimensionImpl;
    };


    /*
     * vislib::math::AbstractDimension<T, 3, S>::~AbstractDimension
     */
    template<class T, class S>
    AbstractDimension<T, 3, S>::~AbstractDimension(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTDIMENSION_H_INCLUDED */
