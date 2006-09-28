/*
 * AbstractDimension3D.h  28.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTDIMENSION3D_H_INCLUDED
#define VISLIB_ABSTRACTDIMENSION3D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* _MSC_VER > 1000 */


#include "vislib/AbstractDimension.h"


namespace vislib {
namespace math {

    /**
     * This class represents extents in two-dimensional space.
     *
     * @author Christoph Mueller
     */
    template<class T, class S> class AbstractDimension3D 
            : public AbstractDimension<T, 3, S> {

    public:

        /** Dtor. */
        ~AbstractDimension3D(void);

        /**
         * Answer the depth component.
         *
         * @return The depth.
         */
        inline const T& Depth(void) const {
            return this->dimension[2];
        }

        /**
         * Answer the depth component.
         *
         * @return The depth.
         */
        inline const T& GetDepth(void) const {
            return this->dimension[2];
        }

        /**
         * Answer the height component.
         *
         * @return The height.
         */
        inline const T& GetHeight(void) const {
            return this->dimension[1];
        }

        /**
         * Answer the width component.
         *
         * @return The width.
         */
        inline const T& GetWidth(void) const {
            return this->dimension[0];
        }

        /**
         * Answer the height component.
         *
         * @return The height.
         */
        inline const T& Height(void) const {
            return this->dimension[1];
        }

        /**
         * Answer the width component.
         *
         * @return The width.
         */
        inline const T& Width(void) const {
            return this->dimension[0];
        }

        /**
         * Change the dimension.
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
         * Set the depth.
         *
         * @param depth The new depth.
         */
        inline void SetDepth(const T& depth) {
            this->dimension[2] = depth;
        }

        /**
         * Set the height.
         *
         * @param height The new height.
         */
        inline void SetHeight(const T& height) {
            this->dimension[1] = height;
        }

        /**
         * Set the width.
         *
         * @param width The new width.
         */
        inline void SetWidth(const T& width) {
            this->dimension[0] = width;
        }

        /**
         * Assignment operator. This operator never creates an alias.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline AbstractDimension3D& operator =(const AbstractDimension3D& rhs) {
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
        AbstractDimension3D& operator =(
                const AbstractDimension<Tp, Dp, Sp>& rhs) {
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /** A typedef for the super class. */
        typedef AbstractDimension<T, 3, S> Super;

        /**
         * Disallow instances of this class.
         */
        inline AbstractDimension3D(void) : Super() {}
    };


    /*
     * vislib::math::AbstractDimension3D<T, S>::~AbstractDimension3D
     */
    template<class T, class S> 
    AbstractDimension3D<T, S>::~AbstractDimension3D(void) {
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_ABSTRACTDIMENSION3D_H_INCLUDED */
