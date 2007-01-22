/*
 * SmartPtr.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SMARTPTR_H_INCLUDED
#define VISLIB_SMARTPTR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * This is a smart pointer class that implements reference counting.
     *
     * All memory that a smart pointer can point to must have been allocated on 
     * the heap using the global new operator, as the delete operator is used
     * for freeing the memory.
     */
    template <class T> class SmartPtr {

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
         * Answer, whether the smart pointer is a NULL pointer.
         *
         * @return true, if the pointer is a NULL pointer, false otherwise.
         */
        inline bool IsNull(void) const {
            return (this->ptr == NULL);
        }

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
     * vislib::SmartPtr<T>::SmartPtr
     */
    template<class T> SmartPtr<T>::SmartPtr(T *ptr) : ptr(NULL) {
        if (ptr != NULL) {
            this->ptr = new CounterProxy(ptr);
        }
    }


    /*
     * vislib::SmartPtr<T>::~SmartPtr
     */
    template<class T> SmartPtr<T>::~SmartPtr(void) {
        *this = NULL;
    }


    /*
     * vislib::SmartPtr<T>::operator =
     */
    template<class T> 
    SmartPtr<T>& SmartPtr<T>::operator =(T *rhs) {

        /* Handle reference decrement on current object. */
        if ((this->ptr != NULL) && (--this->ptr->cnt == 0)) {
            SAFE_DELETE(this->ptr->obj);
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
     * vislib::SmartPtr<T>::operator =
     */
    template<class T> 
    SmartPtr<T>& SmartPtr<T>::operator =(const SmartPtr& rhs) {
        if (this != &rhs) {

            /* Handle reference decrement on current object. */
            if ((this->ptr != NULL) && (--this->ptr->cnt == 0)) {
                SAFE_DELETE(this->ptr->obj);
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

#endif /* VISLIB_SMARTPTR_H_INCLUDED */
