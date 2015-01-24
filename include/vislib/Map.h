/*
 * Map.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MAP_H_INCLUDED
#define VISLIB_MAP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "vislib/AssociativeCollection.h"
#include "vislib/ConstIterator.h"
#include "vislib/Iterator.h"
#include "vislib/SingleLinkedList.h"


namespace vislib {

    /**
     * This map implementation of the associative collection uses 
     * 'OrderedCollection' objects to manage the data in the map and is 
     * therefore slow. You should expect that all operations are O(n)!
     */
    template<class K, class V> class Map : public AssociativeCollection<K, V> {
    public:

        /** Helper class for pair iteration */
        class ElementPair {
        public:
            friend class ::vislib::Map<K, V>;
            friend class ::vislib::Map<K, V>::Iterator;

            /**
             * Copy ctor.
             *
             * @param src The object to clone from.
             */
            ElementPair(const ElementPair& src) {
                *this = src;
            }

            /**
             * Dtor.
             */
            ~ElementPair(void) {
                // DO NOT DELETE THE ARRAYS!
                this->key = NULL;
                this->value = NULL;
            }

            /**
             * Gets the key of this element pair.
             *
             * Note: there is no way to change the key using an iterator,
             * because this could corrupt the internal data structure of the
             * map.
             *
             * @return The key of this element pair.
             */
            const K& Key(void) const {
                return *this->key;
            }

            /**
             * Gets the value of this element pair for read write access.
             *
             * @return The value of this element pair for read write access.
             */
            V& Value(void) {
                return *this->value;
            }

            /**
             * Gets the value of this element pair.
             *
             * @return The value of this element pair.
             */
            const V& Value(void) const {
                return *this->value;
            }

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand.
             *
             * @return A reference to 'this' object.
             */
            ElementPair& operator=(const ElementPair& rhs) {
                this->key = rhs.key;
                this->value = rhs.value;
                return *this;
            }

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'rhs' and 'this' are equal, 'false' otherwise.
             */
            bool operator==(const ElementPair& rhs) const {
                return (this->key == rhs.key)
                    && (this->value == rhs.value);
            }

        private:

            /**
             * Private ctor.
             *
             * @param key The key.
             * @param value The value.
             */
            ElementPair(K *key, V *value)
                    : key(key), value(value) {
                // intentionally empty
            }

            /** the key */
            K *key;

            /** the value */
            V *value;

        };

        /**
         * Iterator class
         */
        class Iterator : public ::vislib::Iterator<ElementPair> {
        public:
            friend class Map<K, V>;

            /** default ctor */
            Iterator(void) : owner(NULL), idx(0), retval(NULL, NULL) {
                // intentionally empty
            }

            /**
             * copy ctor for assignment
             *
             * @param rhs The source object to clone from.
             */
            Iterator(const typename Map<K, V>::Iterator& rhs)
                    : retval(NULL, NULL) {
                *this = rhs;
            }

            /** Dtor. */
            virtual ~Iterator(void) {
                this->owner = NULL; // DO NOT DELETE
            }

            /** Behaves like Iterator<T>::HasNext */
            virtual bool HasNext(void) const {
                if (this->owner == NULL) return false;
                return this->owner->keys.Count() > this->idx;
            }

            /** 
             * Behaves like Iterator<T>::Next 
             *
             * @throw IllegalStateException if there is no next element
             */
            virtual ElementPair& Next(void) {
                if (!this->HasNext()) {
                    throw IllegalStateException("There is no next element",
                        __FILE__, __LINE__);
                }
                this->retval.key = &this->owner->keys[this->idx];
                this->retval.value = &this->owner->values[this->idx];
                this->idx++;
                return this->retval;
            }

            /**
             * assignment operator
             *
             * @param rhs The right hand side operand.
             *
             * @return A reference to 'this'.
             */
            Iterator& operator=(const typename Map<K, V>::Iterator& rhs) {
                this->owner = rhs.owner;
                this->idx = rhs.idx;
                return *this;
            }

        private:

            /**
             * Ctor.
             *
             * @param owner The owning map object.
             */
            Iterator(Map<K, V> &owner) : owner(&owner), idx(0),
                    retval(NULL, NULL) {
                // intentionally empty
            }

            /** The owning map object */
            Map<K, V>* owner;

            /** The next index */
            SIZE_T idx;

            /** The returned value */
            ElementPair retval;

        };

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
         * Gets a const iterator for all entries in the map.
         *
         * @return The const iterator to all entries in the map.
         */
        virtual ConstIterator<Iterator> GetConstIterator(void) const {
            return ConstIterator<Iterator>(Iterator(*const_cast<Map*>(this)));
        }

        /**
         * Gets an iterator for all entries in the map.
         *
         * @return The iterator to all entries in the map.
         */
        virtual Iterator GetIterator(void) {
            return Iterator(*this);
        }

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
            this->values[static_cast<SIZE_T>(idx)] = value;
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
        SIZE_T len = static_cast<SIZE_T>(this->values.Count());
        for (SIZE_T i = 0; i < len; i++) {
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
            return &this->values[static_cast<SIZE_T>(idx)];
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
            return &this->values[static_cast<SIZE_T>(idx)];
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

