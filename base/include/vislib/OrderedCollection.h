/*
 * OrderedCollection.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ORDEREDCOLLECTION_H_INCLUDED
#define VISLIB_ORDEREDCOLLECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Collection.h"
#include "vislib/NullLockable.h"


namespace vislib {


    /**
     * OrderedCollection is a special collection which is assumed to have 
     * elements in a specific order. Therefore, it is possible to access the 
     * first and the last element and append and prepend new elements. The 
     * ordered collection can also be sorted.
     * The template parameter L specifies a Lockable class which is used for
     * synchronisation in a multi-thread environment. If 'NullLockable' is
     * used, the collections must be considdered to be not threadsafe.
     *
     * TODO: Remove default use of 'NullLockable' as soon as all collections
     *       have been fixed.
     */
    template<class T, class L = NullLockable> class OrderedCollection
        : public Collection<T, L> {

    public:

        /** Dtor. */
        virtual ~OrderedCollection(void);

        /**
         * Add 'element' as last element in the collection.
         *
         * @param element The element to be added.
         */
        virtual void Append(const T& element) = 0;

        /**
         * Answer the first element in the collection.
         *
         * @return A reference to the first element.
         *
         * @throws OutOfRangeException or
         *         NoSuchElementException If the collection is empty.
         */
        virtual const T& First(void) const = 0;

        /**
         * Answer the first element in the collection.
         *
         * @return A reference to the first element.
         *
         * @throws OutOfRangeException or
         *         NoSuchElementException If the collection is empty.
         */
        virtual T& First(void) = 0;

        /**
         * Answer the last element in the collection.
         *
         * @return A reference to the last element.
         *
         * @throws OutOfRangeException or
         *         NoSuchElementException If the collection is empty.
         */
        virtual const T& Last(void) const = 0;

        /**
         * Answer the last element in the collection.
         *
         * @return A reference to the last element.
         *
         * @throws OutOfRangeException or
         *         NoSuchElementException If the collection is empty.
         */
        virtual T& Last(void) = 0;

        /**
         * Add 'element' as first element in the collection.
         *
         * @param element The element to be added.
         */
        virtual void Prepend(const T& element) = 0;

        /**
         * Remove the first occurrence of an element that is equal to 'element' 
         * from the collection.
         *
         * @param element The element to be removed.
         */
        virtual void Remove(const T& element) = 0;

        /**
         * Remove the first element from the collection. If the collection
         * is empty, this method has no effect.
         */
        virtual void RemoveFirst(void) = 0;

        /**
         * Remove the last element from the collection. If the collection is
         * empty, this method has no effect.
         */
        virtual void RemoveLast(void) = 0;

        /**
         * Sorts the elements in the collection based on the results of the 
         * 'comparator' function:
         *   = 0 if lhs == rhs
         *   < 0 if lhs < rhs
         *   > 0 if lhs > rhs
         *
         * @param comparator The compare function defining the sort order.
         */
        virtual void Sort(int (*comparator)(const T& lhs, const T& rhs)) = 0;

    protected:

        /** Ctor. */
        inline OrderedCollection(void) : Collection<T, L>() {}

    };
    

    /*
     * vislib::OrderedCollection<T, L>::~OrderedCollection
     */
    template<class T, class L>
    OrderedCollection<T, L>::~OrderedCollection(void) {
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ORDEREDCOLLECTION_H_INCLUDED */
