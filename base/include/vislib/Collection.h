/*
 * Collection.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_COLLECTION_H_INCLUDED
#define VISLIB_COLLECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/memutils.h"
#include "vislib/NullLockable.h"
#include "vislib/types.h"



namespace vislib {


    /**
     * This is the abstract superclass of collections in the vislib.
     * The template parameter L specifies a Lockable class which is used for
     * synchronisation in a multi-thread environment. If 'NullLockable' is
     * used, the collections must be considdered to be not threadsafe.
     *
     * TODO: Remove default use of 'NullLockable' as soon as all collections
     *       have been fixed.
     */
    template<class T, class L = NullLockable> class Collection : public L {

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

        // inline const_cast-crowbar for equal const/non-const accessors?

        /** Dtor. */
        virtual ~Collection(void);

        /**
         * Add 'element' to the collection. 
         *
         * @param elemen The element to be added.
         */
        virtual void Add(const T& element) = 0;

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
        virtual inline bool Contains(const T& element) const {
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
         * Answer whether there is no element in the collection.
         *
         * @return true, if the collection is empty, false otherwise.
         */
        virtual bool IsEmpty(void) const = 0;

        /**
         * Acquires the lock of the collection.
         *
         * Implementation note: This method is required to emulate a mutable 
         * lock. We do not use a mutable member to avoid additional memory
         * used in non-synchronised collections.
         */
        VISLIB_FORCEINLINE void Lock(void) const {
            const_cast<L *>(static_cast<const L *>(this))->Lock();
        }

        /**
         * Remove all elements that are equal to 'element' from the collection.
         *
         * @param element The element to be removed.
         */
        virtual void RemoveAll(const T& element) = 0;

        /**
         * Releases the lock of the collection
         *
         * Implementation note: This method is required to emulate a mutable 
         * lock. We do not use a mutable member to avoid additional memory
         * used in non-synchronised collections.
         */
        VISLIB_FORCEINLINE void Unlock(void) const {
            const_cast<L *>(static_cast<const L *>(this))->Unlock();
        }

    protected:

        /** Ctor. */
        inline Collection(void) : L() {}

    };


    /*
     * vislib::Collection<T, L>::~Collection
     */
    template<class T, class L> Collection<T, L>::~Collection(void) {
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_COLLECTION_H_INCLUDED */
