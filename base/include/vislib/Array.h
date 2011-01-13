/*
 * Array.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ARRAY_H_INCLUDED
#define VISLIB_ARRAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <cstdlib>
#else /* _WIN32 */
#include "vislib/Stack.h"
#endif /* _WIN32 */

#if !defined(WIN32) || !defined(INCLUDED_FROM_ARRAY_CPP) /* avoid warning LNK4221 */
#include <stdexcept>
#endif /* (!defined(WIN32)) || !defined(INCLUDED_FROM_ARRAY_CPP) */

#include "vislib/ArrayElementDftCtor.h"
#include "vislib/assert.h"
#include "vislib/Iterator.h"
#include "vislib/OrderedCollection.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/memutils.h"
#include "vislib/NullLockable.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * A dynamically growing array of type T.
     *
     * The array grows dynamically by the defined increment, if elements are
     * appended that do not fit into the current capacity. It is never
     * shrinked except on user request.
     *
     * The array used typeless memory that can be reallocated.
     *
     * Note that the array will call the dtor of its elements before the memory
     * of a elements is released, just like the array delete operator. The
     * construction/destruction function C is used for this operation.
     *
     * Class L is a synchronisation functor that can be used to construct a
     * thread-safe array. By default, the NullLockable is used, i. e. the array
     * is not thread-safe.
     *
     * Note, that thread-safety is not guaranteed for any external iterator 
     * object that might be retrieved from the array. Note, that also 
     * references elements retrieved by accessors are not protected when being
     * used after the accessor returns. To ensure thread-safety over a longer
     * period of time, use the Lock() and Unlock methods to synchronise the
     * whole array explicitly.
     *
     * No memory is allocated, if the capacity is forced to zero.
     *
     * Implementation note: The construction/destruction policy of the array is 
     * that all elements within the current capacity must have been constructed.
     * In order to prevent memory leaks in the derived PtrArray class, elements
     * that are erased are immediately dtored and the free element(s) is/are
     * reconstructed using the default ctor after that.
     */
    template <class T, class L = NullLockable,
            class C = ArrayElementDftCtor<T> > class Array
            : public OrderedCollection<T, L> {

    public:

        /** The default initial capacity */
        static const SIZE_T DEFAULT_CAPACITY;

        /** The default initial capacity increment */
        static const SIZE_T DEFAULT_CAPACITY_INCREMENT;

        /** This constant signals an invalid index position. */
        static const INT_PTR INVALID_POS;

        /** 
         * Create an array with the specified initial capacity.
         *
         * @param capacity  The initial capacity of the array.
         * @param increment The initial capacity increment of the array.
         */
        Array(const SIZE_T capacity = DEFAULT_CAPACITY, const SIZE_T increment
            = DEFAULT_CAPACITY_INCREMENT);

        /**
         * Create a new array with the specified initial count and capacity
         * and use 'element' as default value for all these elements.
         *
         * @param count     The initial capacity of the array and number of
         *                  elements.
         * @param element   The default value to set.
         * @param increment The initial capacity increment of the array.
         */
        Array(const SIZE_T count, const T& element, const SIZE_T increment
            = DEFAULT_CAPACITY_INCREMENT);

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
         * capacity is increased by the capacity increment.
         *
         * @param element The item to be appended.
         */
        virtual inline void Add(const T& element) {
            this->Append(element);
        }

        /** 
         * Appends an element to the end of the array. If necessary, the 
         * capacity is increased by the capacity increment.
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
         * Erase all elements from the array.
         *
         * @param doTrim If true, the array is trimmed to have zero size, 
         *               otherwise the memory remains allocated.
         */
        inline void Clear(const bool doTrim) {
            this->Lock();
            this->Clear();
            if (doTrim) {
                this->Trim();
            }
            this->Unlock();
        }

        /**
         * Answer the capacity of the array.
         *
         * @return The number of elements that are currently allocated.
         */
        SIZE_T Capacity(void) const;

        /**
         * Answer the capacity increment of the array.
         *
         * @return The capacity increment
         */
        inline SIZE_T CapacityIncrement(void) const;

        /**
         * Answer whether 'element' is in the array.
         *
         * @param element The element to be tested.
         *
         * @return true, if 'element' is at least once in the array, false 
         *         otherwise.
         */
        virtual bool Contains(const T& element) const;

        /**
         * Answer the number of items in the array. Note that the result is not
         * the capacity of the array which is currently allocated.
         *
         * @return Number of items in the array.
         */
        virtual SIZE_T Count(void) const;

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
         * Exposes the internal data pointer.
         *
         * Using this method is inherently unsafe as the pointer is only 
         * guaranteed to be valid until an arbitrary method is called on the
         * array. This is analogous to String::PeekBuffer.
         *
         * YOU SHOULD NEVER MODIFY THIS POINTER, ESPECIALLY NEVER RELEASE THE
         * MEMORY!
         *
         * The data pointer might be NULL.
         *
         * @return The internal data pointer.
         */
        inline const T *PeekElements(void) const {
            return this->elements;
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
        virtual const T *Find(const T& element) const ;

        /**
         * Answer a pointer to the first copy of 'element' in the array. If no
         * element equal to 'element' is found, a NULL pointer is returned.
         *
         * @param element The element to be tested.
         *
         * @return A pointer to the local copy of 'element' or NULL, if no such
         *         element is found.
         */
        virtual T *Find(const T& element);

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
         * Insert 'element' at position 'idx' in the array. All elements behind
         * 'idx' are shifted one element right. 'idx' must be a valid index in 
         * the array or the index directly following the end, i. e. Count(). In
         * the latter case, the method behaves like Append().
         *
         * @param idx     The position to insert the element at.
         * @param element The element to add.
         *
         * @throws OutOfRangeException If 'idx' is not within 
         *                             [0, this->Count()].
         */
        virtual void Insert(const SIZE_T idx, const T& element);

        /**
         * Answer whether there is no element in the array. Note that a return
         * value of true does not mean that no memory is allocated.
         *
         * @return true, if there is no element in the array, false otherwise.
         */
        virtual bool IsEmpty(void) const;

        /**
         * Answer the last element in the array.
         *
         * @return A reference to the last element in the array.
         *
         * @throws OutOfRangeException, if the array is empty.
         */
        virtual const T& Last(void) const;

        /**
         * Answer the last element in the array.
         *
         * @return A reference to the last element in the array.
         *
         * @throws OutOfRangeException, if the array is empty.
         */
        virtual T& Last(void);

        /**
         * Add 'element' as first element of the array. If necessary, the 
         * capacity is increased by the capacity increment.
         *
         * @param element The element to be added.
         */
        virtual void Prepend(const T& element);

        /**
         * Remove the first occurrence of an element that is equal to 'element' 
         * from the collection.
         *
         * @param element The element to be removed.
         */
        virtual void Remove(const T& element);

        /**
         * Remove all elements that are equal to 'element' from the array.
         *
         * @param element The element to be removed.
         */
        virtual void RemoveAll(const T& element);

        /**
         * Erase the element at position 'idx' from the array. If 'idx' is out
         * of range, this method has no effect.
         *
         * @param idx The index of the element to be removed.
         */
        inline void RemoveAt(const SIZE_T idx) {
            this->Erase(idx);
        }

        /**
         * Remove the first element from the collection. 
         */
        virtual void RemoveFirst(void);

        /**
         * Remove the last element from the collection.
         */
        virtual void RemoveLast(void);

        /**
         * Resize the array to have exactly 'capacity' elements. If 'capacity'
         * is less than the current number of elements in the array, all 
         * elements at and behind the index 'capacity' are erased.
         *
         * @param capacity The new size of the array.
         */
        void Resize(const SIZE_T capacity);

        /**
         * Set the capacity increment for the array.
         *
         * @param capacityIncrement the new increment
         */
        void SetCapacityIncrement(const SIZE_T capacityIncrement);

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
         * Sorts the elements in the collection based on the results of the 
         * 'comparator' function:
         *   = 0 if lhs == rhs
         *   < 0 if lhs < rhs
         *   > 0 if lhs > rhs
         *
         * @param comparator The compare function defining the sort order.
         */
        virtual void Sort(int (*comparator)(const T& lhs, const T& rhs));

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

#if (defined _WIN32) || (defined _LIN64)
        // this define is correct!
        //  used on all windows platforms 
        //  and on 64 bit linux!

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

#endif /* (defined _WIN32) || (defined _LIN64) */

        /**
         * Assignment.
         *
         * @param rhs The right hand operand.
         *
         * @return *this.
         */
        Array& operator =(const Array& rhs);

        /**
         * Compare operator. Two arrays are equal if the elements in both 
         * lists are equal and in same order. Runtime complexity: O(n)
         *
         * @param rhs The right hand side operand
         *
         * @return if the lists are considered equal
         */
        bool operator ==(const Array& rhs) const;

        /**
         * Test for inequality. Two arrays are equal if the elements in both 
         * lists are equal and in same order. Runtime complexity: O(n)
         *
         * @param rhs The right hand side operand
         *
         * @return if the lists are considered not equal
         */
        inline bool operator !=(const Array& rhs) const {
            return !(*this == rhs);
        }

    private:

#ifdef _WIN32
        /**
         * Helper function used by the quicksort algorithm.
         *
         * @param context The context of the comparison
         * @param lhs The left hand side operand
         * @param rhs The right hand side operand
         *
         * @return The compare value used by 'qsort_s'
         */
        static int qsortHelper(void * context, const void * lhs, 
            const void * rhs);
#endif /* _WIN32 */

        /** The number of actually allocated elements in 'elements' */
        SIZE_T capacity;

        /** The capacity increment */
        SIZE_T capacityIncrement;

        /** The number of used elements in 'elements' */
        SIZE_T count;

        /** The actual array (PtrArray must have access) */
        T *elements;

    };


    /*
     * vislib::Array<T, L, C>::DEFAULT_CAPACITY
     */
    template<class T, class L, class C>
    const SIZE_T Array<T, L, C>::DEFAULT_CAPACITY = 8;


    /*
     * vislib::Array<T, L, C>::DEFAULT_CAPACITY_INCREMENT
     */
    template<class T, class L, class C>
    const SIZE_T Array<T, L, C>::DEFAULT_CAPACITY_INCREMENT = 1;


    /*
     * vislib::Array<T, L, C>::INVALID_POS
     */
    template<class T, class L, class C>
    const INT_PTR Array<T, L, C>::INVALID_POS = -1;


    /*
     * vislib::Array<T, L, C>::Array
     */
    template<class T, class L, class C>
    Array<T, L, C>::Array(const SIZE_T capacity,
            const SIZE_T capacityIncrement)	: OrderedCollection<T, L>(),
            capacity(0), capacityIncrement(capacityIncrement), count(0),
            elements(NULL) {
        ASSERT(capacityIncrement > 0);
        this->AssertCapacity(capacity);
    }


    /*
     * vislib::Array<T, L, C>::Array
     */
    template<class T, class L, class C>
    Array<T, L, C>::Array(const SIZE_T count, const T& element,
            const SIZE_T capacityIncrement) : OrderedCollection<T, L>(),
            capacity(0), capacityIncrement(capacityIncrement), count(count),
            elements(NULL) {
        ASSERT(capacityIncrement > 0);
        this->Lock();
        this->AssertCapacity(count);
        for (SIZE_T i = 0; i < this->count; i++) {
            this->elements[i] = element;
        }
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Array
     */
    template<class T, class L, class C> Array<T, L, C>::Array(const Array& rhs)
            : OrderedCollection<T, L>(), capacity(0), capacityIncrement(1),
            count(0), elements(NULL) {
        *this = rhs;
    }


    /*
     * vislib::Array<T, L, C>::~Array
     */
    template<class T, class L, class C> Array<T, L, C>::~Array(void) {
        this->Resize(0);
        ASSERT(this->elements == NULL);
    }


    /*
     * vislib::Array<T, L, C>::Append
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Append(const T& element) {
        this->Lock();
        this->AssertCapacity(this->count + 1);
        this->elements[this->count] = element;
        this->count++;
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::AssertCapacity
     */
    template<class T, class L, class C>
    void Array<T, L, C>::AssertCapacity(const SIZE_T capacity) {
        this->Lock();
        if (this->capacity < capacity) {
            if (capacity - this->capacity < this->capacityIncrement) {
                this->Resize(this->capacity + this->capacityIncrement);
            } else {
                this->Resize(capacity);
            }
        }
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Clear
     */
    template<class T, class L, class C> void Array<T, L, C>::Clear(void) {
        this->Lock();
        this->count = 0;
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Capacity
     */
    template<class T, class L, class C>
    SIZE_T Array<T, L, C>::Capacity(void) const {
        this->Lock();
        SIZE_T retval = this->capacity;
        this->Unlock();
        return retval;
    }


    /*
     * vislib::Array<T, L, C>::CapacityIncrement
     */
    template<class T, class L, class C>
    SIZE_T Array<T, L, C>::CapacityIncrement(void) const {
        this->Lock();
        SIZE_T retval = this->capacityIncrement;
        this->Unlock();
        return retval;
    }


    /*
     * vislib::Array<T, L, C>::Contains
     */
    template<class T, class L, class C>
    bool Array<T, L, C>::Contains(const T& element) const {
        this->Lock();
        INT_PTR idx = this->IndexOf(element);
        this->Unlock();
        return (idx >= 0);
    }


    /*
     * vislib::Array<T, L, C>::Count
     */
    template<class T, class L, class C>
    SIZE_T Array<T, L, C>::Count(void) const {
        this->Lock();
        SIZE_T retval = this->count;
        this->Unlock();
        return retval;
    }


    /*
     * vislib::Array<T, L, C>::Erase
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Erase(const SIZE_T idx) {
        this->Lock();
        if (idx < this->count) {
            /* Destruct element to erase. */
            C::Dtor(this->elements + idx);

            /* Move elements forward. */
            for (SIZE_T i = idx + 1; i < this->count; i++) {
                this->elements[i - 1] = this->elements[i];
            }
            this->count--;

            /* Element after valid range must now be reconstructed. */
            C::Ctor(this->elements + this->count);
        }
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Erase
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Erase(const SIZE_T beginIdx, const SIZE_T cnt) {
        SIZE_T cntRemoved = cnt;
        SIZE_T range = cnt;
        SIZE_T cntMove = range;

        this->Lock();

        if (beginIdx < this->count) {

            /* Sanity check. */
            if (beginIdx + range >= this->count) {
                cntRemoved = range = cntMove = this->count - beginIdx;
            }
            ASSERT(beginIdx + range <= this->count);

            /* Dtor element range to erase. */
            for (SIZE_T i = beginIdx; i < beginIdx + range; i++) {
                C::Dtor(this->elements + i);
            }

            /* Fill empty range. */
            for (SIZE_T i = beginIdx + range; i < this->count; i += range) {
                if (i + range >= this->count) {
                    cntMove = this->count - i;
                }
                ::memcpy(this->elements + (i - range), this->elements + i, 
                    cntMove * sizeof(T));
            }
            this->count -= cntRemoved;

            /* 'cntRemoved' elements after valid range must be reconstructed. */
            for (SIZE_T i = this->count; i < this->count + cntRemoved; i++) {
                C::Ctor(this->elements + i);
            }
        }

        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Find
     */
    template<class T, class L, class C>
    const T * Array<T, L, C>::Find(const T& element) const {
        // TODO: Larger critical section? Would not be safe anyway.
        INT_PTR idx = this->IndexOf(element);
        return (idx >= 0) ? (this->elements + idx) : NULL;
    }


    /*
     * vislib::Array<T, L, C>::Find
     */
    template<class T, class L, class C>
    T * Array<T, L, C>::Find(const T& element) {
        // TODO: Larger critical section? Would not be safe anyway.
        INT_PTR idx = this->IndexOf(element);
        return (idx >= 0) ? (this->elements + idx) : NULL;
    }


    /*
     * vislib::Array<T, L, C>::IndexOf
     */
    template<class T, class L, class C>
    INT_PTR Array<T, L, C>::IndexOf(const T& element,
            const SIZE_T beginAt) const {
        this->Lock();
        for (SIZE_T i = 0; i < this->count; i++) {
            if (this->elements[i] == element) {
                this->Unlock();
                return i;
            }
        }
        /* Nothing found. */

        this->Unlock();
        return INVALID_POS;
    }


    /*
     * vislib::Array<T, L, C>::Insert
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Insert(const SIZE_T idx, const T& element) {
        this->Lock();

        if (static_cast<SIZE_T>(idx) <= this->count) {
            this->AssertCapacity(this->count + 1);

            for (SIZE_T i = this->count; i > idx; i--) {
                this->elements[i] = this->elements[i - 1];
            }
            
            this->elements[idx] = element;
            this->count++;

        } else {
            this->Unlock();
            throw OutOfRangeException(static_cast<INT>(idx), 0, 
                static_cast<INT>(this->count), __FILE__, __LINE__);
        }

        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::IsEmpty
     */
    template<class T, class L, class C>
    bool Array<T, L, C>::IsEmpty(void) const {
        this->Lock();
        bool retval = (this->count == 0);
        this->Unlock();
        return retval;
    }


    /*
     * vislib::Array<T, L, C>::Last
     */
    template<class T, class L, class C>
    const T& Array<T, L, C>::Last(void) const {
        // This implementation is not nice, but should work as it overflows.
        // TODO: Larger critical section? Would not be safe anyway.
        return (*this)[this->count - 1];
    }


    /*
     * vislib::Array<T, L, C>::Last
     */
    template<class T, class L, class C> T& Array<T, L, C>::Last(void) {
        // This implementation is not nice, but should work as it overflows.
        // TODO: Larger critical section? Would not be safe anyway.
        return (*this)[this->count - 1];
    }


    /*
     * vislib::Array<T, L, C>::Prepend
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Prepend(const T& element) {
        // TODO: This implementation is extremely inefficient. Could use single
        // memcpy when reallocating.

        this->Lock();
        this->AssertCapacity(this->count + 1);

        for (SIZE_T i = this->count; i > 0; i--) {
            this->elements[i] = this->elements[i - 1];
        }

        this->elements[0] = element;
        this->count++;
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Remove
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Remove(const T& element) {

        this->Lock();

        for (SIZE_T i = 0; i < this->count; i++) {
            if (this->elements[i] == element) {
                /* Dtor element to remove. */
                C::Dtor(this->elements + i);

                /* Move elements forward. */
                for (SIZE_T j = i + 1; j < this->count; j++) {
                    this->elements[j - 1] = this->elements[j];
                }
                this->count--;

                /* Reconstruct invalid element at end. */
                C::Ctor(this->elements + this->count);

                break;          // Remove first element only.
            }
        }

        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::RemoveAll
     */
    template<class T, class L, class C>
    void Array<T, L, C>::RemoveAll(const T& element) {

        this->Lock();

        for (SIZE_T i = 0; i < this->count; i++) {
            if (this->elements[i] == element) {
                /* Dtor element to remove. */
                C::Dtor(this->elements + i);

                /* Move elements forward. */
                for (SIZE_T j = i + 1; j < this->count; j++) {
                    this->elements[j - 1] = this->elements[j];
                }
                this->count--;

                /* Reconstruct invalid element at end. */
                C::Ctor(this->elements + this->count);

                i--;            // One index was lost, so next moved forward.
            }
        }

        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::RemoveFirst
     */
    template<class T, class L, class C> void Array<T, L, C>::RemoveFirst(void) {
        this->Erase(0);
    }


    /*
     * vislib::Array<T, L, C>::RemoveLast
     */
    template<class T, class L, class C> void Array<T, L, C>::RemoveLast(void) {
        this->Lock();
        this->Erase(this->count - 1);
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Resize
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Resize(const SIZE_T capacity) {
        void *newPtr = NULL;
        this->Lock();
        SIZE_T oldCapacity = this->capacity;

        /*
         * Erase elements, if new capacity is smaller than old one. Ensure that 
         * the dtor of the elements is called, as we use typeless memory for the
         * array.
         */
        if (capacity < this->capacity) {
            for (SIZE_T i = capacity; i < this->capacity; i++) {
                C::Dtor(this->elements + i);
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

        } else if (oldCapacity != this->capacity) {
            /* Reallocate elements. */
            if ((newPtr = ::realloc(this->elements, this->capacity * sizeof(T)))
                    != NULL) {
                this->elements = static_cast<T *>(newPtr);

                for (SIZE_T i = oldCapacity; i < this->capacity; i++) {
                    C::Ctor(this->elements + i);
                }
            } else {
                for (SIZE_T i = 0; i < oldCapacity; i++) {
                    C::Dtor(this->elements + i);
                }
                SAFE_FREE(this->elements);
                this->Unlock();
                throw std::bad_alloc();
            }
        }

        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::SetCount
     */
    template<class T, class L, class C>
    void Array<T, L, C>::SetCount(const SIZE_T count) {
        this->Lock();
        if (count < this->count) {
            this->Erase(count, this->count - count);
        } else {
            this->AssertCapacity(count);
            this->count = count;
        }
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::SetCapacityIncrement
     */
    template<class T, class L, class C>
    void Array<T, L, C>::SetCapacityIncrement(const SIZE_T capacityIncrement) {
        this->Lock();
        this->capacityIncrement = capacityIncrement;
        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::Sort
     */
    template<class T, class L, class C>
    void Array<T, L, C>::Sort(int (*comparator)(const T& lhs, const T& rhs)) {
        this->Lock();

#ifdef _WIN32
        ::qsort_s(this->elements, this->count, sizeof(T), qsortHelper, 
            reinterpret_cast<void *>(comparator));

#else /* _WIN32 */
        // qsort by hand, because of frowsty ANSI api (based on Sedgewick)
        INT64 l, r, i, j;
        T tmp;
        Stack<UINT64> stack;

        l = 0;
        r = this->count - 1;

        stack.Push(l); 
        stack.Push(r);

        do {
            if (r > l) {
                // i = partition(l, r);
                T value = this->elements[r];
                i = l - 1;
                j = r;
                do {
                    do { 
                        i = i + 1;
                    } while (comparator(this->elements[i], value) < 0);
                    do { 
                        j = j - 1;
                    } while (comparator(this->elements[j], value) > 0);
                    tmp = this->elements[i];
                    this->elements[i] = this->elements[j];
                    this->elements[j] = tmp;
                } while (j > i);
                this->elements[j] = this->elements[i];
                this->elements[i] = this->elements[r];
                this->elements[r] = tmp;

                // recursion (unfold using the stack)
                if ((i - l) > (r - i)) {
                    stack.Push(l);
                    stack.Push(i - 1);
                    l = i + 1;
                } else {
                    stack.Push(i + 1);
                    stack.Push(r);
                    r = i - 1;
                }
            } else {
                r = stack.Pop();
                l = stack.Pop();
            }
        } while (!stack.IsEmpty());

#endif /* _WIN32 */

        this->Unlock();
    }


    /*
     * vislib::Array<T, L, C>::operator []
     */
    template<class T, class L, class C>
    T& Array<T, L, C>::operator [](const SIZE_T idx) {
        this->Lock();
        if (static_cast<SIZE_T>(idx) < this->count) {
            T& retval = this->elements[idx];
            this->Unlock();
            return retval;
        } else {
            this->Unlock();
            throw OutOfRangeException(static_cast<INT>(idx), 0, 
                static_cast<INT>(this->count - 1), __FILE__, __LINE__);
        }
    }


    /*
     * vislib::Array<T, L, C>::operator []
     */
    template<class T, class L, class C>
    const T& Array<T, L, C>::operator [](const SIZE_T idx) const {
        this->Lock();
        if (static_cast<SIZE_T>(idx) < this->count) {
            const T& retval = this->elements[idx];
            this->Unlock();
            return retval;
        } else {
            this->Unlock();
            throw OutOfRangeException(static_cast<INT>(idx), 0, 
                static_cast<INT>(this->count - 1), __FILE__, __LINE__);
        }
    }


    /*
     * vislib::Array<T, L, C>::operator =
     */
    template<class T, class L, class C>
    Array<T, L, C>& Array<T, L, C>::operator =(const Array& rhs) {
        if (this != &rhs) {
            this->Lock();
            this->Resize(rhs.capacity);
            this->count = rhs.count;
            this->capacityIncrement = rhs.capacityIncrement;

            for (SIZE_T i = 0; i < rhs.count; i++) {
                this->elements[i] = rhs.elements[i];
            }
            this->Unlock();
        }

        return *this;
    }


    /*
     * vislib::Array<T, L, C>::operator ==
     */
    template<class T, class L, class C>
    bool Array<T, L, C>::operator ==(const Array& rhs) const {
        if (this == &rhs) {
            return true;
        }
        this->Lock();
        if (this->count != rhs.count) {
            this->Unlock();
            return false;
        }
        for (SIZE_T i = 0; i < this->count; i++) {
            if (!(this->elements[i] == rhs.elements[i])) {
                this->Unlock();
                return false;
            }
        }
        this->Unlock();
        return true;
    }


#ifdef _WIN32
    /* 
     * vislib::Array<T, L, C>::qsortHelper
     */
    template<class T, class L, class C> 
    int Array<T, L, C>::qsortHelper(void * context, const void * lhs, 
            const void * rhs) {

        int (*comparator)(const T& lhs, const T& rhs) 
            = reinterpret_cast<int (*)(const T& lhs, const T& rhs)>(context);

        return comparator(*static_cast<const T*>(lhs), 
            *static_cast<const T*>(rhs));
    }
#endif /* _WIN32 */

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ARRAY_H_INCLUDED */
