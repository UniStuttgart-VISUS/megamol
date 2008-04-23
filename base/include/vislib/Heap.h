/*
 * Heap.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_HEAP_H_INCLUDED
#define VISLIB_HEAP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/utils.h"


namespace vislib {


    /**
     * This class implements a binary heap, which can be used as a priority
     * queue.
     *
     * The heap accepts multiple entries with the same key, i. e. the same
     * priporty.
     *
     * The return order of elements having the same key is undefined as
     * heaps are not stable.
     *
     * The template class T must support the method
     *
     * const KeyType& Key(void) const 
     *
     * with KeyType having an operator <
     *
     * You can use vislib::Pair for instantiating a heap using the first element
     * as key.
     *
     * Rationale: We decided to implement a binary heap because it has no memory
     * overhead (if implemented using an array). Binomial heaps and Fibonacci 
     * heaps are faster, but must be implemented using double-linked lists.
     */
    template<class T> class Heap {

    public:

        /** 
         * Create a new heap with the specified initial capacity allocated for
         * elements.
         * 
         * It is important to specify a meaningful capacity, e. g. the maximum
         * number of elements expected to be queued at one time in a priority
         * queue.
         *
         * @param capacity The initial capacity of the underlying array. This
         *                 amount of memory is directly allocated. If the heap,
         *                 however, needs more memory, it will dynamically 
         *                 reallocate the array.
         */
        Heap(const SIZE_T capacity = Array<T>::DEFAULT_CAPACITY);

        /** Dtor. */
        ~Heap(void);

        /**
         * Add a new element to the heap.
         *
         * This method has logarithmic runtime complexity.
         */
        inline void Add(const T& element) {
            this->elements.Append(element);
            BECAUSE_I_KNOW(!this->elements.IsEmpty());
            this->siftUp(this->elements.Count() - 1);
        }

        /**
         * Reserves memory for at least 'capacity' elements in the heap. If
         * 'capacity' is less than or equal to the current capacity of the 
         * array, this method has no effect.
         *
         * @param capacity The minimum number of elements that should be 
         *                 allocated.
         *
         * @throws std::bad_alloc If there was insufficient memory for 
         *                        allocating the array.
         */
        inline void AssertCapacity(const SIZE_T capacity) {
            this->elements.AssertCapacity(capacity);
        }

        /**
         * Answer the number of entries allocated for the heap.
         *
         * @return The current capacity of the heap.
         */
        inline SIZE_T Capacity(void) const {
            return this->elements.Capacity();
        }

        /**
         * Remove all elements from the heap.
         */
        inline void Clear(void) {
            this->elements.Clear();
        }

        // TODO: Contains using template for key?

        /**
         * Answer the number of elements in the heap.
         *
         * @return The number of elements in the heap.
         */
        inline SIZE_T Count(void) const {
            return this->elements.Count();
        }

        // TODO: Find using template for key?

        /**
         * Answer the first element in the heap, i. e. its root.
         *
         * This method has constant runtime complexity.
         *
         * @return The root element of the heap.
         *
         * @throws OutOfRangeException, if the heap is empty.
         */
        inline const T& First(void) const {
            return this->elements[0];
        }

        /**
         * Answer whether the heap is empty. 
         * 
         * Note, that even if the heap is empty, memory might be allocated. Use
         * Capacity to determine the current capacity of the heap.
         *
         * @return true, if no element is in the heap, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return this->elements.IsEmpty();
        }

        /**
         * Remove the root of the heap.
         *
         * This method has logarithmic runtime complexity.
         */
        void RemoveFirst(void);

        /**
         * Trim the capacity of the heap to match the current number of 
         * elements.
         */
        inline void Trim(void) {
            this->elements.Trim();
        }

    private:

        /** 
         * Enforces the heap property after removing an element at 'idx'.
         *
         * @param idx The index of the element removed. This should be the first
         *            element.
         */
        void siftDown(SIZE_T idx);

        /**
         * Enforces the heap property after inserting an element at 'idx'.
         *
         * @param idx The index of the inserted element. This should be the last
         *            element.
         */
        void siftUp(SIZE_T idx);

        /** The array of heap elements. */
        Array<T> elements;

    };


    /*
     * vislib::Heap<T>::Heap
     */
    template<class T> 
    Heap<T>::Heap(const SIZE_T capacity) : elements(capacity) {
    }


    /*
     * vislib::Heap<T>::~Heap
     */
    template<class T> Heap<T>::~Heap(void) {
    }


    /*
     * vislib::Heap<T>::RemoveFirst
     */
    template<class T> void Heap<T>::RemoveFirst(void) {
        if (this->elements.Count() > 1) {
            this->elements[0] = this->elements.Last();
            this->elements.RemoveLast();
            this->siftDown(0);
        } else {
            this->elements.RemoveFirst();
        }
    }


    /*
     * vislib::Heap<T>::siftDown
     */
    template<class T> void Heap<T>::siftDown(SIZE_T idx) {
        SIZE_T nextIdx = 0;

        while (idx < this->elements.Count() / 2) {
            nextIdx = 2 * idx + 1;

            if (nextIdx + 1 < this->elements.Count()) {
                /* Node at 'idx' has two children, check for smaller one. */
                
                if (this->elements[nextIdx + 1].Key() 
                        < this->elements[nextIdx].Key()) {
                    nextIdx++;
                }
            }
            ASSERT(idx >= 0);
            ASSERT(idx < this->elements.Count());
            ASSERT(nextIdx >= 0);
            ASSERT(nextIdx < this->elements.Count());

            if (this->elements[idx].Key() < this->elements[nextIdx].Key()) {
                /* Heap property fulfilled. */
                break;

            } else {
                /* Bubble down. */
                Swap(this->elements[idx], this->elements[nextIdx]);
                idx = nextIdx;
            }
        }
    }


    /*
     * vislib::Heap<T>::siftUp
     */
    template<class T> void Heap<T>::siftUp(SIZE_T idx) {
        SIZE_T nextIdx = 0;

        while (idx > 0) {
            nextIdx = (idx - 1) / 2;
            ASSERT(idx >= 0);
            ASSERT(idx < this->elements.Count());
            ASSERT(nextIdx >= 0);
            ASSERT(nextIdx < this->elements.Count());

            if (this->elements[nextIdx].Key() < this->elements[idx].Key()) {
                /* Heap property fulfilled. */
                break;

            } else {
                /* Bubble up. */
                Swap(this->elements[idx], this->elements[nextIdx]);
                idx = nextIdx;
            } 
        }
    }
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_HEAP_H_INCLUDED */

