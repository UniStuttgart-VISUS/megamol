/*
 * AssociativeCollection.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASSOCIATIVECOLLECTION_H_INCLUDED
#define VISLIB_ASSOCIATIVECOLLECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SingleLinkedList.h"


namespace vislib {


    /**
     * Abstract base class for associative collections (maps from keys of type
     * K to values of type V).
     */
    template<class K, class V> class AssociativeCollection {
    public:

        /** Dtor. */
        virtual ~AssociativeCollection(void) {
        }

        /**
         * Sets the value for the key 'key' to 'value'. If there was no entry
         * with the given key, it will be created.
         *
         * @param key The 'key' of the entry pair to be set.
         * @param value The 'value' to set the entry to.
         */
        virtual void Set(const K &key, const V &value) = 0;

        /**
         * Clears the whole map by removing all entries.
         */
        virtual void Clear(void) = 0;

        /**
         * Checks whether a given key is present in the map.
         *
         * @param key The key to search for.
         *
         * @return 'true' if the key 'key' is present in the map, 'false' 
         *         otherwise.
         */
        virtual bool Contains(const K &key) const = 0;

        /**
         * Answers the number of entries in the map.
         *
         * @return The number of entries in the map.
         */
        virtual SIZE_T Count(void) const = 0;

        /**
         * Finds all keys which are associated to a given value. The order of
         * the keys is not defined. This call might be very slow!
         *
         * @param value The value to search the key for.
         *
         * @return A single linked list of all keys associated with this value.
         */
        virtual SingleLinkedList<K> FindKeys(const V &value) const = 0;

        /**
         * Finds a value for the specified key. If the key is not present in
         * the map NULL is returned.
         *
         * @param key The key to receive the value.
         *
         * @return The value associated with the key or NULL if the key is not
         *         present in the map.
         */
        virtual const V * FindValue(const K &key) const = 0;

        /**
         * Finds a value for the specified key. If the key is not present in
         * the map NULL is returned.
         *
         * @param key The key to receive the value.
         *
         * @return The value associated with the key or NULL if the key is not
         *         present in the map.
         */
        virtual V * FindValue(const K &key) = 0;

        /**
         * Answers whether the map is empty or not.
         *
         * @return 'true' if the map is empty, 'false' otherwise.
         */
        virtual bool IsEmpty(void) const = 0;

        /**
         * Removes the given key from the map.
         *
         * @param key The key to be removed from the map.
         */
        virtual void Remove(const K &key) = 0;

        /**
         * The array operator will return the value of an existing key or will
         * create a new value if the key is not present in the map. This new
         * value will be associated with the key.
         *
         * @param key The key to receive the value to.
         *
         * @return The value associated with the given key.
         */
        virtual inline V & operator[](const K &key) {
            V * v = this->FindValue(key);
            if (v == NULL) {
                this->Set(key, V());
                v = this->FindValue(key);
            }
            return *v;
        }

    protected:

        /** Ctor. */
        AssociativeCollection(void) {
        }

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASSOCIATIVECOLLECTION_H_INCLUDED */

