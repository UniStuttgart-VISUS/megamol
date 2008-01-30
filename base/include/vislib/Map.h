/*
 * Map.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MAP_H_INCLUDED
#define VISLIB_MAP_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "vislib/AssociativeCollection.h"
#include "vislib/SingleLinkedList.h"


namespace vislib {


    /**
     * This map implementation of the associative collection uses 
     * 'OrderedCollection' objects to manage the data in the map and is 
     * therefore slow. You should expect that all operations are O(n)!
     */
    template<class K, class V> class Map : public AssociativeCollection<K, V> {
    public:

        /** Ctor. */
        Map(void);

        /** Dtor. */
        virtual ~Map(void);

        /**
         * Sets the value for the key 'key' to 'value'. If there was no entry
         * with the given key, it will be created.
         *
         * @param key The 'key' of the entry pair to be set.
         * @param value The 'value' to set the entry to.
         */
        virtual void Set(const K &key, const V &value);

        /**
         * Clears the whole map by removing all entries.
         */
        virtual void Clear(void);

        /**
         * Checks whether a given key is present in the map.
         *
         * @param key The key to search for.
         *
         * @return 'true' if the key 'key' is present in the map, 'false' 
         *         otherwise.
         */
        virtual bool Contains(const K &key) const;

        /**
         * Answers the number of entries in the map.
         *
         * @return The number of entries in the map.
         */
        virtual SIZE_T Count(void) const;

        /**
         * Finds all keys which are associated to a given value. The order of
         * the keys is not defined. This call might be very slow!
         *
         * @param value The value to search the key for.
         *
         * @return A single linked list of all keys associated with this value.
         */
        virtual SingleLinkedList<K> FindKeys(const V &value) const;

        /**
         * Finds a value for the specified key. If the key is not present in
         * the map NULL is returned.
         *
         * @param key The key to receive the value.
         *
         * @return The value associated with the key or NULL if the key is not
         *         present in the map.
         */
        virtual const V * FindValue(const K &key) const;

        /**
         * Finds a value for the specified key. If the key is not present in
         * the map NULL is returned.
         *
         * @param key The key to receive the value.
         *
         * @return The value associated with the key or NULL if the key is not
         *         present in the map.
         */
        virtual V * FindValue(const K &key);

        /**
         * Answers whether the map is empty or not.
         *
         * @return 'true' if the map is empty, 'false' otherwise.
         */
        virtual bool IsEmpty(void) const;

        /**
         * Removes the given key from the map.
         *
         * @param key The key to be removed from the map.
         */
        virtual void Remove(const K &key);

    private:

        /** array of keys */
        vislib::Array<K> keys;

        /** array of values */
        vislib::Array<V> values;

    };


    /*
     * vislib::Map::Map
     */
    template<class K, class V> Map<K, V>::Map(void)
            : AssociativeCollection<K, V>(), keys(), values() {
    }


    /*
     * vislib::Map::~Map
     */
    template<class K, class V> Map<K, V>::~Map(void) {
        this->Clear();
    }


    /*
     * vislib::Map::Set
     */
    template<class K, class V> 
    void Map<K, V>::Set(const K &key, const V &value) {
        INT_PTR idx = this->keys.IndexOf(key);
        if (idx == Array<K>::INVALID_POS) {
            this->keys.Append(key);
            this->values.Append(value);
        } else {
            this->values[static_cast<unsigned int>(idx)] = value;
        }
    }


    /*
     * vislib::Map::Clear
     */
    template<class K, class V> void Map<K, V>::Clear(void) {
        this->keys.Clear();
        this->values.Clear();
    }


    /*
     * vislib::Map::Contains
     */
    template<class K, class V> bool Map<K, V>::Contains(const K &key) const {
        return this->keys.Contains(key);
    }


    /*
     * vislib::Map::Count
     */
    template<class K, class V> SIZE_T Map<K, V>::Count(void) const {
        return this->keys.Count();
    }


    /*
     * vislib::Map::FindKeys
     */
    template<class K, class V> 
    SingleLinkedList<K> Map<K, V>::FindKeys(const V &value) const {
        SingleLinkedList<K> retval;
        unsigned int len = static_cast<unsigned int>(this->values.Count());
        for (unsigned int i = 0; i < len; i++) {
            if (this->values[i] == value) {
                retval.Append(this->keys[i]);
            }
        }
        return retval;
    }


    /*
     * vislib::Map::FindValue
     */
    template<class K, class V>
    const V * Map<K, V>::FindValue(const K &key) const {
        INT_PTR idx = this->keys.IndexOf(key);
        if (idx == Array<K>::INVALID_POS) {
            return NULL;
        } else {
            return &this->values[static_cast<unsigned int>(idx)];
        }
    }


    /*
     * vislib::Map::FindValue
     */
    template<class K, class V> V * Map<K, V>::FindValue(const K &key) {
        INT_PTR idx = this->keys.IndexOf(key);
        if (idx == Array<K>::INVALID_POS) {
            return NULL;
        } else {
            return &this->values[static_cast<unsigned int>(idx)];
        }
    }


    /*
     * vislib::Map::IsEmpty
     */
    template<class K, class V> bool Map<K, V>::IsEmpty(void) const {
        return this->keys.IsEmpty();
    }


    /*
     * vislib::Map::Remove
     */
    template<class K, class V> void Map<K, V>::Remove(const K &key) {
        INT_PTR idx = this->keys.IndexOf(key);
        if (idx != Array<K>::INVALID_POS) {
            this->values.Erase(idx);
            this->keys.Erase(idx);
        }
    }


} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MAP_H_INCLUDED */

