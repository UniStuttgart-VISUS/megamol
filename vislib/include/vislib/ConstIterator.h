/*
 * ConstIterator.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CONSTITERATOR_H_INCLUDED
#define VISLIB_CONSTITERATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {


    /**
     * Class of constant container iterators. Use the non-const iterator class
     * as template parameter I. This iterator class must have an assignment
     * operator, a default ctor, a copy ctor.
     */
    template<class I> class ConstIterator {
    public:

        /**
         * Default ctor.
         */
        ConstIterator(void);

        /**
         * Ctor from non-const iterator.
         *
         * @param iter The non-const iterator.
         */
        ConstIterator(const I& iter);

        /**
         * Copy ctor.
         *
         * @param src The object to clone from.
         */
        ConstIterator(const ConstIterator<I>& src);

        /** Dtor. */
        virtual ~ConstIterator(void);

        /**
         * Answer whether there is a next element to iterator to.
         *
         * @return true if there is a next element, false otherwise.
         */
        virtual bool HasNext(void) const;

        /**
         * Iterates to the next element and returns this element.
         *
         * @return The next element, which becomes the current element after
         *         calling this methode.
         */
        virtual const typename I::Type& Next(void) const;

        /**
         * Assignmnet operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return reference to this.
         */
        virtual ConstIterator<I>& operator=(const ConstIterator<I>& rhs);

    private:

        /** the non-const iterator */
        mutable I iter;

    };


    /*
     * ConstIterator<I>::ConstIterator
     */
    template<class I>
    ConstIterator<I>::ConstIterator(void) : iter() {
        // intentionally empty
    }


    /*
     * ConstIterator<I>::ConstIterator
     */
    template<class I>
    ConstIterator<I>::ConstIterator(const I& iter) : iter(iter) {
        // intentionally empty
    }


    /*
     * ConstIterator<I>::ConstIterator
     */
    template<class I>
    ConstIterator<I>::ConstIterator(const ConstIterator<I>& src)
            : iter(src.iter) {
        // intentionally empty
    }


    /*
     * ConstIterator<I>::~ConstIterator
     */
    template<class I> ConstIterator<I>::~ConstIterator(void) {
        // intentionally empty
    }


    /*
     * ConstIterator<I>::HasNext
     */
    template<class I> bool ConstIterator<I>::HasNext(void) const {
        return this->iter.HasNext();
    }


    /*
     * ConstIterator<I>::Next
     */
    template<class I>
    const typename I::Type& ConstIterator<I>::Next(void) const {
        return const_cast<I&>(this->iter).Next();
    }


    /*
     * ConstIterator<I>::operator=
     */
    template<class I>
    ConstIterator<I>& ConstIterator<I>::operator=(
            const ConstIterator<I>& rhs) {
        this->iter = rhs.iter;
        return *this;
    }


} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CONSTITERATOR_H_INCLUDED */
