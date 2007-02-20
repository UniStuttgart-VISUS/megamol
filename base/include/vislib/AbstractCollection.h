/*
 * AbstractCollection.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCOLLECTION_H_INCLUDED
#define VISLIB_ABSTRACTCOLLECTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {


    /**
     * This is the abstract superclass of collections in the vislib.
     */
    template<class T> class AbstractCollection {

    public:

        // TODO Append, Prepend?
        // Add, AddRange
        // ContainsRange
        // GetIterator
        // ToArray
        // Retain

        // Count
        // Item  
        // Add
        // Clear 
        // Contains 
        // Equals
        // GetEnumerator  
        // GetHashCode 
        // Remove 
        // ToString

        // Synchronisation????
        // Sort

        // SortedCollection mit First/Last/Append/Prepend/...



        /** Dtor. */
        virtual ~AbstractCollection(void);

        /** Remove all elements from the collection. */
        virtual void Clear(void) = 0;

        /**
         * Answer whether 'element' is in the collection.
         *
         * @param element The element to be tested.
         *
         * @return true, if 'element' is at least once in the collection, false
         *         otherwise.
         */
        inline bool Contains(const T& element) const {
            return (this->Find(element) != NULL);
        }

        /**
         * Answer the number of items in the collection.
         *
         * @return Number of items in the collection.
         */
        virtual SIZE_T Count(void) const = 0;

        /**
         * Answer a pointer to the first copy of 'element' in the collection. 
         * If no element equal to 'element' is found, a NULL pointer is 
         * returned.
         *
         * @param element The element to be tested.
         *
         * @return A pointer to the local copy of 'element' or NULL, if no such
         *         element is found.
         */
        virtual const T *Find(const T& element) const = 0;

        /**
         * Answer a pointer to the first copy of 'element' in the collection. 
         * If no element equal to 'element' is found, a NULL pointer is 
         * returned.
         *
         * @param element The element to be tested.
         *
         * @return A pointer to the local copy of 'element' or NULL, if no such
         *         element is found.
         */
        virtual T *Find(const T& element) = 0;

        /**
         * Answer the first element in the collection.
         *
         * @return A reference to the first element.
         *
         * @throws OutOfRangeException, if the collection is empty.
         */
        //virtual const T& First(void) const = 0;

        /**
         * Answer the first element in the collection.
         *
         * @return A reference to the first element.
         *
         * @throws OutOfRangeException, if the collection is empty.
         */
        //virtual T& First(void) = 0;

        /**
         * Answer whether there is no element in the collection.
         *
         * @return true, if the collection is empty, false otherwise.
         */
        virtual bool IsEmpty(void) const = 0;

        /**
         * Answer the last element in the collection.
         *
         * @return A reference to the last element.
         *
         * @throws OutOfRangeException, if the collection is empty.
         */
        //virtual const T& Last(void) const = 0;

        /**
         * Answer the last element in the collection.
         *
         * @return A reference to the last element.
         *
         * @throws OutOfRangeException, if the collection is empty.
         */
        //virtual T& Last(void) = 0;

        /**
         * Remove all elements that are equal to 'element' from the collection.
         *
         * @param element The element to be removed.
         */
        virtual void Remove(const T& element) = 0;

    protected:

        /** Ctor. */
        AbstractCollection(void);

    };


    /*
     * vislib::AbstractCollection<T>::~AbstractCollection
     */
    template<class T>AbstractCollection<T>::~AbstractCollection(void) {
    }


    /*
     * vislib::AbstractCollection<T>::AbstractCollection
     */
    template<class T> AbstractCollection<T>::AbstractCollection(void) {
    }
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCOLLECTION_H_INCLUDED */
