/*
 * Array.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ARRAY_H_INCLUDED
#define VISLIB_ARRAY_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <stdexcept>

#include "vislib/assert.h"
#include "vislib/Iterator.h"
#include "vislib/OrderedCollection.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/memutils.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * A dynamically growing array of type T.
     *
     * The array grows dynamically, if elements are appended that do not fit 
     * into the current capacity. It is never shrinked except on user request.
     *
     * The array used typeless memory that can be reallocated.
     *
     * Note that the array will call the dtor of its elements before the memory
     * of a elements is released, just like the array delete operator.
     *
     * No memory is allocated, if the capacity is forced to zero.
     */
    template <class T> class Array : public OrderedCollection<T> {

    public:

        /** The default initial capacity. */
        static const SIZE_T DEFAULT_CAPACITY;

        /** This constant signals an invalid index position. */
        static const INT_PTR INVALID_POS;

        /** 
         * Create an array with the specified initial capacity.
         *
         * @param capacity The initial capacity of the array.
         */
        Array(const SIZE_T capacity = DEFAULT_CAPACITY);

        /**
         * Create a new array with the specified initial capacity and
         * use 'element' as default value for all elements.
         *
         * @param capacity The initial capacity of the array.
         * @param element  The default value to set.
         */
        Array(const SIZE_T capacity, const T& element);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        Array(const Array& rhs);

        /** Dtor. */
        virtual ~Array(void);

        /** 
         * Appends an element to the end of the array. If necessary, the 
         * capacity is increased.
         *
         * @param element The item to be appended.
         */
        virtual inline void Add(const T& element) {
            this->Append(element);
        }

        /** 
         * Appends an element to the end of the array. If necessary, the 
         * capacity is increased.
         *
         * @param element The item to be appended.
         */
        virtual void Append(const T& element);

        /**
         * Reserves memory for at least 'capacity' elements in the array. If
         * 'capacity' is less than or equal to the current capacity of the 
         * array, this method has no effect.
         *
         * @param capacity The minimum number of elements that should be 
         *                 allocated.
         *
         * @throws std::bad_alloc If there was insufficient memory for 
         *                        allocating the array.
         */
        void AssertCapacity(const SIZE_T capacity);

        /**
         * Erase all elements from the array.
         */
        virtual void Clear(void);

        /**
         * Answer the capacity of the array.
         *
         * @return The number of elements that are currently allocated.
         */
        inline SIZE_T Capacity(void) const {
            return this->capacity;
        }

        /**
         * Answer whether 'element' is in the array.
         *
         * @param element The element to be tested.
         *
         * @return true, if 'element' is at least once in the array, false 
         *         otherwise.
         */
        virtual inline bool Contains(const T& element) const {
            return (this->IndexOf(element) >= 0);
        }

        /**
         * Answer the number of items in the array. Note that the result is not
         * the capacity of the array which is currently allocated.
         *
         * @return Number of items in the array.
         */
        virtual inline SIZE_T Count(void) const {
            return this->count;
        }

        /**
         * Erase the element at position 'idx' from the array. If 'idx' is out
         * of range, this method has no effect.
         *
         * @param idx The index of the element to be removed.
         */
        void Erase(const SIZE_T idx);

        /**
         * Erase all 'cnt' elements beginning at 'beginIdx'. If elements out of
         * range would be affected, these are simply ignored.
         *
         * @param beginIdx The index of the first element to be removed.
         * @param cnt      The number of elements to be removed.
         */
        void Erase(const SIZE_T beginIdx, const SIZE_T cnt);

        /**
         * Answer a pointer to the first copy of 'element' in the array. If no
         * element equal to 'element' is found, a NULL pointer is returned.
         *
         * @param element The element to be tested.
         *
         * @return A pointer to the local copy of 'element' or NULL, if no such
         *         element is found.
         */
        virtual inline const T *Find(const T& element) const {
            INT_PTR idx = this->IndexOf(element);
            return (idx >= 0) ? (this->elements + idx) : NULL;
        }

        /**
         * Answer a pointer to the first copy of 'element' in the array. If no
         * element equal to 'element' is found, a NULL pointer is returned.
         *
         * @param element The element to be tested.
         *
         * @return A pointer to the local copy of 'element' or NULL, if no such
         *         element is found.
         */
        virtual T *Find(const T& element) {
            INT_PTR idx = this->IndexOf(element);
            return (idx >= 0) ? (this->elements + idx) : NULL;
        }

        /**
         * Answer the first element in the array.
         *
         * @return A reference to the first element in the array.
         *
         * @throws OutOfRangeException, if the array is empty.
         */
        virtual inline const T& First(void) const {
            return (*this)[0];
        }

        /**
         * Answer the first element in the array.
         *
         * @return A reference to the first element in the array.
         *
         * @throws OutOfRangeException, if the array is empty.
         */
        virtual inline T& First(void) {
            return (*this)[0];
        }

        /**
         * Answer the index of the first occurrence of 'element' in the array
         * after 'beginAt'.
         *
         * @param element The element to be searched.
         * @param beginAt The first index to be checked. Defaults to 0.
         *
         * @return The index of the first occurrence of 'element' in the array,
         *         or INVALID_POS if the element is not in the array.
         */
        INT_PTR IndexOf(const T& element, const SIZE_T beginAt = 0) const;

        /**
         * Answer whether there is no element in the array. Note that a return
         * value of true does not mean that no memory is allocated.
         *
         * @return true, if there is no element in the array, false otherwise.
         */
        virtual inline bool IsEmpty(void) const {
            return (this->count == 0);
        }

        /**
         * Answer the last element in the array.
         *
         * @return A reference to the last element in the array.
         *
         * @throws OutOfRangeException, if the array is empty.
         */
        virtual inline const T& Last(void) const {
            // This implementation is not nice, but should work at it overflows.
            return (*this)[this->count - 1];
        }


        /**
         * Answer the last element in the array.
         *
         * @return A reference to the last element in the array.
         *
         * @throws OutOfRangeException, if the array is empty.
         */
        virtual inline T& Last(void) {
            // This implementation is not nice, but should work at it overflows.
            return (*this)[this->count - 1];
        }

        /**
         * Add 'element' as first element of the array.
         *
         * @param element The element to be added.
         */
        virtual void Prepend(const T& element);

        /**
         * Remove all elements that are equal to 'element' from the array.
         *
         * @param element The element to be removed.
         */
        virtual void Remove(const T& element);

        /**
         * Remove the first element from the collection. 
         */
        virtual void RemoveFirst(void) {
            this->Erase(0);
        }

        /**
         * Remove the last element from the collection.
         */
        virtual void RemoveLast(void) {
            this->Erase(this->count - 1);
        }

        /**
         * Resize the array to have exactly 'capacity' elements. If 'capacity'
         * is less than the current number of elements in the array, all 
         * elements at and behind the index 'capacity' are erased.
         *
         * @param capacity The new size of the array.
         */
        void Resize(const SIZE_T capacity);

        /**
         * Make the array assume that it has 'count' valid elements. If 'count'
         * is less than the current number of elements, this has the same effect
         * as erasing all elements beginning at index 'count' until the end. If
         * 'count' is larger than the current element count, the requested 
         * number of elements is allocated. Note that the new elements might 
         * have undefined content. The method only guarantees the default ctor
         * to be called.
         *
         * @param count The new number of elements in the array.
         */
        void SetCount(const SIZE_T count);

        /**
         * Trim the capacity of the array to match the current number of 
         * elements. This has the same effect as calling Resize(Count()).
         */
        inline void Trim(void) {
            this->Resize(this->count);
        }

        /**
         * Access the 'idx'th element in the array.
         *
         * @param idx The index of the element to access. This must be a value
         *            within [0, this->Count()[.
         *

         * @return A reference to the 'idx'th element.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()[.
         */
        T& operator [](const SIZE_T idx);

        /**
         * Access the 'idx'th element in the array.
         *
         * @param idx The index of the element to access. This must be a value
         *            within [0, this->Count()[.
         *
         * @return A reference to the 'idx'th element.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()[.
         */
        const T& operator [](const SIZE_T idx) const;

        /**
         * Access the 'idx'th element in the array.
         *
         * @param idx The index of the element to access. This must be a value
         *            within [0, this->Count()[.
         *
         * @return A reference to the 'idx'th element.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()[.
         */
        inline T& operator [](const INT idx) {
            return (*this)[static_cast<SIZE_T>(idx)];
        }

        /**
         * Access the 'idx'th element in the array.
         *
         * @param idx The index of the element to access. This must be a value
         *            within [0, this->Count()[.
         *
         * @return A reference to the 'idx'th element.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()[.
         */
        const T& operator [](const INT idx) const {
            return (*this)[static_cast<SIZE_T>(idx)];
        }

        /**
         * Access the 'idx'th element in the array.
         *
         * @param idx The index of the element to access. This must be a value
         *            within [0, this->Count()[.
         *
         * @return A reference to the 'idx'th element.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()[.
         */
        inline T& operator [](const UINT idx) {
            return (*this)[static_cast<SIZE_T>(idx)];
        }

        /**
         * Access the 'idx'th element in the array.
         *
         * @param idx The index of the element to access. This must be a value
         *            within [0, this->Count()[.
         *
         * @return A reference to the 'idx'th element.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()[.
         */
        const T& operator [](const UINT idx) const {
            return (*this)[static_cast<SIZE_T>(idx)];
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand operand.
         *
         * @return *this.
         */
        Array& operator =(const Array& rhs);

    private:

        /** The number of actually allocated elements in 'elements'. */
        SIZE_T capacity;

        /** The number of used elements in 'elements'. */
        SIZE_T count;

        /** The actual array. */
        T *elements;

    };


    /*
     * vislib::Array<T>::DEFAULT_CAPACITY
     */
    template<class T>
    const SIZE_T Array<T>::DEFAULT_CAPACITY = 8;


    /*
     * vislib::Array<T>::INVALID_POS
     */
    template<class T>
    const INT_PTR Array<T>::INVALID_POS = -1;


    /*
     * vislib::Array<T>::Array
     */
    template<class T>
    Array<T>::Array(const SIZE_T capacity) 
            : OrderedCollection<T>(), capacity(0), count(0), elements(NULL) {
        this->AssertCapacity(capacity);
    }


    /*
     * vislib::Array<T>::Array
     */
    template<class T>
    Array<T>::Array(const SIZE_T capacity, const T& element) 
            : OrderedCollection<T>(), capacity(0), count(capacity), 
            elements(NULL) {
        this->AssertCapacity(capacity);
        for (SIZE_T i = 0; i < this->count; i++) {
            this->elements[i] = element;
        }
    }


    /*
     * vislib::Array<T>::Array
     */
    template<class T>
    Array<T>::Array(const Array& rhs) 
            : OrderedCollection<T>(), capacity(0), count(0), elements(NULL) {
        *this = rhs;
    }


    /*
     * vislib::Array<T>::~Array
     */
    template<class T>
    Array<T>::~Array(void) {
        this->Resize(0);
        ASSERT(this->elements == NULL);
    }


    /*
     * vislib::Array<T>::Append
     */
    template<class T>
    void Array<T>::Append(const T& element) {
        this->AssertCapacity(this->count + 1);
        this->elements[this->count] = element;
        this->count++;
    }


    /*
     * vislib::Array<T>::AssertCapacity
     */
    template<class T>
    void Array<T>::AssertCapacity(const SIZE_T capacity) {
        if (this->capacity < capacity) {
            // TODO: Could allocate more than required here.
            this->Resize(capacity);
        }
    }


    /*
     * vislib::Array<T>::Clear
     */
    template<class T>
    void Array<T>::Clear(void) {
        this->count = 0;
    }


    /*
     * vislib::Array<T>::Erase
     */
    template<class T>
    void Array<T>::Erase(const SIZE_T idx) {
        if (idx < this->count) {
            for (SIZE_T i = idx + 1; i < this->count; i++) {
                this->elements[i - 1] = this->elements[i];
            }

            this->count--;
            this->elements[this->count].~T();   // Call dtor on erased element.
        }
    }


    /*
     * vislib::Array<T>::Erase
     */
    template<class T>
    void Array<T>::Erase(const SIZE_T beginIdx, const SIZE_T cnt) {
        SIZE_T cntRemoved = cnt;
        SIZE_T range = cnt;

        if (beginIdx < this->count) {
            if (beginIdx + range >= this->count) {
                cntRemoved = range = this->count - beginIdx;
            }
            ASSERT(beginIdx + range <= this->count);

            for (SIZE_T i = beginIdx + range; i < this->count; i += range) {
                if (i + range >= this->count) {
                    range = this->count - i;
                }
                ::memcpy(this->elements + (i - range - 1), this->elements + i, 
                    range * sizeof(T));
            }

            // Call dtors on erased elements. 
            for (SIZE_T i = this->count - cntRemoved; i < this->count; i++) {
                this->elements[i].~T();
            }

            this->count -= cntRemoved;
        }
    }


    /*
     * vislib::Array<T>::IndexOf
     */
    template<class T>
    INT_PTR Array<T>::IndexOf(const T& element, const SIZE_T beginAt) const {
        for (SIZE_T i = 0; i < this->count; i++) {
            if (this->elements[i] == element) {
                return i;
            }
        }
        /* Nothing found. */

        return INVALID_POS;
    }


    /*
     * vislib::Array<T>::Prepend
     */
    template<class T>
    void Array<T>::Prepend(const T& element) {
        // TODO: This implementation is extremely inefficient. Could use single
        // memcpy when reallocating.
        this->AssertCapacity(this->count + 1);

        for (SIZE_T i = this->count; i > 0; i--) {
            this->elements[i] = this->elements[i - 1];
        }

        this->elements[0] = element;
        this->count++;
    }


    /*
     * vislib::Array<T>::Remove
     */
    template<class T>
    void Array<T>::Remove(const T& element) {

        for (SIZE_T i = 0; i < this->count; i++) {
            if (this->elements[i] == element) {
                for (SIZE_T j = i + 1; j < this->count; j++) {
                    this->elements[j - 1] = this->elements[j];
                }

                this->count--;
                this->elements[this->count].~T();   // Call dtor on erased elm.
                i--;                                // Next element moved fwd.
            }
        }
    }


    /*
     * vislib::Array<T>::Resize
     */
    template<class T>
    void Array<T>::Resize(const SIZE_T capacity) {
        void *newPtr = NULL;

        /*
         * Erase elements, if new capacity is smaller than old one. Ensure that 
         * the dtor of the elements is called, as we use typeless memory for the
         * array.
         */
        if (capacity < this->capacity) {
            for (SIZE_T i = capacity; i < this->count; i++) {
                this->elements[i].~T();
            }

            if (capacity < this->count) {
                /* Count cannot exceed capacity. */
                this->count = capacity;
            }
        }

        /* Allocate the new capacity. */
        this->capacity = capacity;
        if (this->capacity == 0) {
            /* Array is empty now, make 'this->elements' a NULL pointer. */
            SAFE_FREE(this->elements);

        } else {
            /* Reallocate elements. */
            if ((newPtr = ::realloc(this->elements, this->capacity * sizeof(T)))
                    != NULL) {
                this->elements = static_cast<T *>(newPtr);

                for (SIZE_T i = this->count; i < this->capacity; i++) {
                    new (this->elements + i) T;
                }
            } else {
                SAFE_FREE(this->elements);
                throw std::bad_alloc();
            }
        }
    }


    /*
     * vislib::Array<T>::SetCount
     */
    template<class T> void Array<T>::SetCount(const SIZE_T count) {
        if (count < this->count) {
            this->Erase(count, this->count - count);
        } else {
            this->AssertCapacity(count);
            this->count = count;
        }
    }


    /*
     * vislib::Array<T>::operator []
     */
    template<class T>
    T& Array<T>::operator [](const SIZE_T idx) {
        if (static_cast<SIZE_T>(idx) < this->count) {
            return this->elements[idx];
        } else {
            throw OutOfRangeException(static_cast<INT>(idx), 0, 
                static_cast<INT>(this->count - 1), __FILE__, __LINE__);
        }
    }


    /*
     * vislib::Array<T>::operator []
     */
    template<class T>
    const T& Array<T>::operator [](const SIZE_T idx) const {
        if (static_cast<SIZE_T>(idx) < this->count) {
            return this->elements[idx];
        } else {
            throw OutOfRangeException(static_cast<INT>(idx), 0, 
                static_cast<INT>(this->count - 1), __FILE__, __LINE__);
        }
    }


    /*
     * vislib::Array<T>::operator =
     */
    template<class T>
    Array<T>& Array<T>::operator =(const Array& rhs) {
        if (this != &rhs) {
            this->Resize(rhs.capacity);
            this->count = rhs.count;

            for (SIZE_T i = 0; i < rhs.count; i++) {
                this->elements[i] = rhs.elements[i];
            }
        }

        return *this;
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ARRAY_H_INCLUDED */
