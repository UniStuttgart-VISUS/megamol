/*
 * SmartPtr.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SMARTPTR_H_INCLUDED
#define VISLIB_SMARTPTR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/SingleAllocator.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * This is a smart pointer class that implements reference counting.
     *
     * All memory that a smart pointer can point to must have been allocated on 
     * the heap using the a method compatible with the Allocator A. The user is
     * responsible for allocating the memory that is assigned to a SmartPtr with
     * a compatible method, i. e. with new when using the default 
     * SingleAllocator and with new[] when using the array allocator.
     *
     * The template parameter T specifies the static type of the memory 
     * designated by the pointer. 
     *
     * The template parameter A is the allocator that specifies the method for
     * freeing the memory. It must have a Deallocate method that calls the dtor
     * of the object(s) deallocated. The allocator must support the same static
     * type T as the SmartPtr. The valor of A defaults to SingleAllocator<T>,
     * i. e. the memory is assumed to be allocated using "new T".
     */
    template <class T, class A = SingleAllocator<T> > class SmartPtr {

    public:

        /**
         * Create a new smart pointer pointing to 'ptr'.
         *
         * @param ptr The object that the smart pointer should point to.
         */
        SmartPtr(T *ptr = NULL);

        /**
         * Clone 'rhs'.
         *
         * This operation increments the reference count on the object 
         * designated by 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline SmartPtr(const SmartPtr& rhs) : ptr(NULL) {
            *this = rhs;
        }

        /** Dtor. */
        ~SmartPtr(void);

        /**
         * Returns a (non-smart) pointer to the object this pointer points to
         * dynamically casted to the class specified as template parameter.
         *
         * WARNING: Do not manipulate the returned pointer (e. g.: Do not 
         * call 'delete' with this pointer).
         *
         * @return A (non-smart) pointer to the object.
         */
        template<class Tp> Tp* DynamicCast(void) {
            return (this->ptr == NULL) ? NULL 
                : dynamic_cast<Tp*>(this->ptr->obj);
        }

        /**
         * Returns a (non-smart) pointer to the object this pointer points to
         * dynamically casted to the class specified as template parameter.
         *
         * WARNING: Do not manipulate the returned pointer (e. g.: Do not 
         * call 'delete' with this pointer).
         *
         * @return A (non-smart) pointer to the object.
         */
        template<class Tp> const Tp* DynamicCast(void) const {
            return (this->ptr == NULL) ? NULL 
                : dynamic_cast<Tp*>(this->ptr->obj);
        }

        /**
         * Answer, whether the smart pointer is a NULL pointer.
         *
         * @return true, if the pointer is a NULL pointer, false otherwise.
         */
        inline bool IsNull(void) const {
            ASSERT ((this->ptr == NULL) || (this->ptr->obj != NULL));
            return (this->ptr == NULL);
        }

        /**
         * Assignment operator.
         *
         * This operation makes decrements the reference count on the current
         * pointer and makes 'rhs' the new object with an reference count of 1.
         * If 'rhs' is NULL, the smart pointer is made the NULL pointer without
         * a special reference counting.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        SmartPtr& operator =(T *rhs);

        /**
         * Assignment operator.
         *
         * This operation increments the reference count on the object
         * designated by 'rhs', if 'rhs' is not this object.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        SmartPtr& operator =(const SmartPtr& rhs);

        /**
         * Answer, whether to smart pointers are equal, i. e. designate the
         * same object.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const SmartPtr& rhs) const {
            return (this->ptr == rhs.ptr);
        }

        /**
         * Answer, whether to smart pointers are not equal, i. e. designate 
         * different objects.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are noteequal, false otherwise.
         */
        inline bool operator !=(const SmartPtr& rhs) const {
            return (this->ptr != rhs.ptr);
        }

        /**
         * Dereferences the pointer. No measures against dereferencing a NULL
         * pointer are taken.
         *
         * @return A reference to the object designated by the pointer.
         */
        inline T& operator *(void) {
            return *(this->ptr->obj);
        }

        /**
         * Dereferences the pointer. No measures against dereferencing a NULL
         * pointer are taken.
         *
         * @return A reference to the object designated by the pointer.
         */
        inline const T& operator *(void) const {
            return *(this->ptr->obj);
        }

        /**
         * Member access. If the pointer is NULL, NULL is returned.
         *
         * @return A pointer to the object designated by the smart pointer.
         */
        inline T *operator ->(void) {
            return (this->ptr != NULL) ? this->ptr->obj : NULL;
        }

        /**
         * Member access. If the pointer is NULL, NULL is returned.
         *
         * @return A pointer to the object designated by the smart pointer.
         */
        inline const T *operator ->(void) const {
            return (this->ptr != NULL) ? this->ptr->obj : NULL;
        }

#if defined(DEBUG) || defined(_DEBUG)
        /**
         * Answer the current reference count. THIS METHOD IS ONLY AVAILABLE IN
         * DEBUG BUILDS!
         *
         * @return The reference count.
         */
        inline UINT _GetCnt(void) const {
            return (this->ptr != NULL) ? this->ptr->cnt : 0;
        }
#endif /* defined(DEBUG) || defined(_DEBUG) */

    private:

        /** This is a helper structure for doing the reference counting. */
        typedef struct CounterProxy_t {
            UINT cnt;           // The reference counter.
            T *obj;             // The actual object.

            /**
             * Initialise the object. The object is initialised to 'obj' and the
             * reference count for this object is set 1.
             *
             * @param obj The pointer that is wrapped by the smart pointer.
             */
            inline CounterProxy_t(T *obj) : cnt(1), obj(obj) {
                ASSERT(obj != NULL);
            }
        } CounterProxy;

        /** The reference counting helper. */
        CounterProxy *ptr;

    };


    /*
     * vislib::SmartPtr<T, A>::SmartPtr
     */
    template<class T, class A> SmartPtr<T, A>::SmartPtr(T *ptr) : ptr(NULL) {
        *this = ptr;
    }


    /*
     * vislib::SmartPtr<T, A>::~SmartPtr
     */
    template<class T, class A> SmartPtr<T, A>::~SmartPtr(void) {
        *this = NULL;
    }


    /*
     * vislib::SmartPtr<T, A>::operator =
     */
    template<class T, class A> 
    SmartPtr<T, A>& SmartPtr<T, A>::operator =(T *rhs) {

        /* Handle reference decrement on current object. */
        if ((this->ptr != NULL) && (--this->ptr->cnt == 0)) {
            A::Deallocate(this->ptr->obj);
            SAFE_DELETE(this->ptr);
        }

        if (rhs != NULL) {
            this->ptr = new CounterProxy(rhs);
        } else {
            this->ptr = NULL;
        }

        return *this;
    }


    /*
     * vislib::SmartPtr<T, A>::operator =
     */
    template<class T, class A> 
    SmartPtr<T, A>& SmartPtr<T, A>::operator =(const SmartPtr& rhs) {
        if (this != &rhs) {

            /* Handle reference decrement on current object. */
            if ((this->ptr != NULL) && (--this->ptr->cnt == 0)) {
                A::Deallocate(this->ptr->obj);
                SAFE_DELETE(this->ptr);
            }

            /* Handle reference increment on new object. */
            if ((this->ptr = rhs.ptr) != NULL) {
                this->ptr->cnt++;
            }
        }

        return *this;
    }
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SMARTPTR_H_INCLUDED */
