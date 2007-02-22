/*
 * OrderedCollection.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ORDEREDCOLLECTION_H_INCLUDED
#define VISLIB_ORDEREDCOLLECTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Collection.h"


namespace vislib {


    /**
     * OrderedCollection is a special collection which is assumed to have 
     * elements in a specific order. Therefore, it is possible to access the 
     * first and the last element and append and prepend new elements. The 
     * ordered collection can also be sorted.
     */
    template<class T> class OrderedCollection : public Collection<T> {

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
         * Sort the collection.
         */
        //template<class C>
        //virtual void Sort(void) = 0;

    protected:

        /** Ctor. */
        inline OrderedCollection(void) : Collection<T>() {}

    };
    

    /*
     * vislib::OrderedCollection<T>::~OrderedCollection
     */
    template<class T> OrderedCollection<T>::~OrderedCollection(void) {
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ORDEREDCOLLECTION_H_INCLUDED */
