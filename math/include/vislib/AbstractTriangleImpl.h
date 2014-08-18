/*
 * AbstractTriangleImpl.h
 *
 * Copyright (C) 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTTRIANGLEIMPL_H_INCLUDED
#define VISLIB_ABSTRACTTRIANGLEIMPL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"


namespace vislib {
namespace math {


    /**
     * Implementation dump for AbstractTriangle.
     */
	template <class T, class S, 
			  template<class T, class S> class C>
			  class AbstractTriangleImpl {

    public:

        /** Dtor. */
        ~AbstractTriangleImpl(void);

        /**
         * Answer the area covered by the triangle.
         *
         * @return The area covered by the triangle.
         */
        template <class Tp> inline Tp Area(void) const {
            return (this->vertices[0] - this->vertices[1])
                .Cross(this->vertices[0] - this->vertices[2]).Length()
                / static_cast<Tp>(2.0);
        }

		/**
		 * Answer the circumference of the triangle.
		 *
		 * @return the circumference
		 */
		template <class Tp> inline Tp Circumference(void) const {
			return (this->vertices[0] - this->vertices[2]).Length()
				+ (this->vertices[1] - this->vertices[2]).Length()
				+ (this->vertices[0] - this->vertices[1]).Length();
		}

		/**
		 * Answer how many vertices this and rhs have in common.
		 *
		 * @param rhs the other triangle
		 *
		 * @return the number of common vertices
		 */
		template <class Tp, class Sp>
		unsigned int CountCommonVertices(const C<Tp, Sp>& rhs) const {
			return (((this->vertices[0] == rhs.vertices[0])
				|| (this->vertices[0] == rhs.vertices[1])
				|| (this->vertices[0] == rhs.vertices[2])) ? 1U : 0U)
				+ (((this->vertices[1] == rhs.vertices[0])
				|| (this->vertices[1] == rhs.vertices[1])
				|| (this->vertices[1] == rhs.vertices[2])) ? 1U : 0U)
				+ (((this->vertices[2] == rhs.vertices[0])
				|| (this->vertices[2] == rhs.vertices[1])
				|| (this->vertices[2] == rhs.vertices[2])) ? 1U : 0U);
		}

		/**
		 * Answer whether this and rhs have at least one edge in common.
		 *
		 * @param rhs the other triangle
		 *
		 * @return true if this and rhs have one edge in common
		 */
		template <class Tp, class Sp>
		bool HasCommonEdge(const C<Tp, Sp>& rhs) const {
			return this->CountCommonVertices(rhs) >= 2U;
		}

		/**
		 * Answer the normal of the triangle.
		 *
		 * @param outNormal Vector variable to receive the triangle normal
		 *
		 * @return reference to outNormal
		 */
		template <class Tp> inline Tp& Normal(Tp& outNormal) const {
			outNormal = (this->vertices[0] - this->vertices[1])
				.Cross(this->vertices[0] - this->vertices[2]);
			outNormal.Normalise();
			return outNormal;
		}

        /**
         * Directly access the internal pointer holding the vertices.
         * The object remains owner of the memory returned.
         *
         * @return The coordinates in an array.
         */
        inline T *PeekCoordinates(void) {
            return this->vertices;
        }

        /**
         * Directly access the internal pointer holding the vertices.
         * The object remains owner of the memory returned.
         *
         * @return The coordinates in an array.
         */
        inline const T *PeekCoordinates(void) const {
            return this->vertices;
        }

        /**
         * Directly access the 'i'th vertex of the triangle.
         *
         * @param i The index of the coordinate within [0, D[.
         *
         * @return A reference to the i-th vertex.
         *
         * @throws OutOfRangeException, if 'i' is not within [0, D[.
         */
        T& operator [](const int i);

        /**
         * Answer the vertices of the triangle.
         *
         * @param i The index of the coordinate within [0, D[.
         *
         * @return the i-th vertex
         *
         * @throws OutOfRangeException, if 'i' is not within [0, D[.
         */
        const T& operator [](const int i) const;

        /**
         * Assignment operator.
         *
         * This operation does <b>not</b> create aliases.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractTriangleImpl<T, S, C>& operator =(const C<T, S>& rhs);

        template<class Tp, class Sp>
        AbstractTriangleImpl<T, S, C>& operator =(const C<Tp, Sp>& rhs);

		/**
		 * Test for equality
		 *
		 * @param rhs the right hand side operand
		 * 
		 * @return true if this is rhs
		 */
		template <class Tp, class Sp>
		inline bool operator ==(const C<Tp, Sp>& rhs) const {
			return this->CountCommonVertices(rhs) == 3U;
		}

		/**
		 * Test for inequality
		 *
		 * @param rhs the right hand side operand
		 * 
		 * @return true if this is not rhs
		 */
		template <class Tp, class Sp>
		inline bool operator !=(const C<Tp, Sp>& rhs) const {
			return this->CountCommonVertices(rhs) != 3U;
		}

    protected:

        /**
         * Disallow instances of this class. This ctor does nothing!
         */
        inline AbstractTriangleImpl(void) {};

		// TODO copy const!, assign

		/** the vertices of the triangle */
		S vertices;

    private:


    };


    /*
     * vislib::math::AbstractTriangleImpl<T, S, C>::~AbstractTriangleImpl
     */
    template<class T, class S, template<class T, class S> class C>
	AbstractTriangleImpl<T, S, C>::~AbstractTriangleImpl(void) {
	}


	/*
     * vislib::math::AbstractTriangleImpl<T, S, C>::operator []
     */
	template<class T, class S, template<class T, class S> class C>
	T& AbstractTriangleImpl<T, S, C>::operator [](const int i) {
		if ((i >= 0) && (i < 3)) {
			return this->vertices[i];
		} else {
			throw OutOfRangeException(i, 0, 2, __FILE__, __LINE__);
		}
	}


	/*
     * vislib::math::AbstractTriangleImpl<T, S, C>::operator []
     */
	template<class T, class S, template<class T, class S> class C>
	const T& AbstractTriangleImpl<T, S, C>::operator [](const int i) const {
		if ((i >= 0) && (i < 3)) {
			return this->vertices[i];
		} else {
			throw OutOfRangeException(i, 0, 2, __FILE__, __LINE__);
		}
	}


    /*
     * vislib::math::AbstractTriangleImpl<T, S, C>::operator =
     */
    template<class T, class S, template<class T, class S> class C> 
    AbstractTriangleImpl<T, S, C>& AbstractTriangleImpl<T, S, C>::operator =(
        const C<T, S>& rhs) {
            if (this != &rhs) {
                this->vertices[0] = rhs.vertices[0];
                this->vertices[1] = rhs.vertices[1];
                this->vertices[2] = rhs.vertices[2];
            }

            return *this;
    }

    /*
     * vislib::math::AbstractTriangleImpl<T, D, S>::operator =
     */
    template<class T, class S, template<class T, class S> class C> 
    template<class Tp, class Sp>
    AbstractTriangleImpl<T, S, C>& AbstractTriangleImpl<T, S, C>::operator =(
           const C<Tp, Sp>& rhs) {

        if (static_cast<void *>(this) != static_cast<const void *>(&rhs)) {
            for (unsigned int d = 0; (d < 3); d++) {
                this->vertices[d] = rhs[d];
            }
        }

        return *this;
    }
    
} /* end namespace math */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTTRIANGLEIMPL_H_INCLUDED */

