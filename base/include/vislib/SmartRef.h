/*
 * SmartRef.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SMARTREF_H_INCLUDED
#define VISLIB_SMARTREF_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/ReferenceCounted.h"


namespace vislib {


    /**
     * The SmartRef class implements something like a smart pointer, but is
     * more lightweight (as it does not allocate proxy objects). On the other
     * side, SmartRefs are less safe than smart pointers as they are prone to
     * dangling references and manipluation and they require objects pointed to
     * to derive from ReferenceCounted.
     *
     * The SmartRef class takes care of incrementing and decrementing the 
     * reference count of designated objects as SmartRef objects are created,
     * manipulated and deleted. The designated objects are responsible for 
     * deleting themselves as the reference count reaches zero. ReferenceCounted
     * implements this behaviour.
     *
     * Any pointer of a ReferenceCounted object that is flat and no SmartRef
     * can be a "weak reference", i. e. a reference that is not counted. 
     * However, flat references are not necessarily weak references, as the user
     * can increment an decrement the reference counter manually.
     */
    template<class T> class SmartRef {

    public:

        /** 
         * Create a reference pointing to 'obj'.
         *
         * @param obj    The object to be referenced.
         * @param addRef Determines whether the ctor will increment the 
         *               reference count (default) or not.
         */
        SmartRef(T *obj = NULL, const bool addRef = true);

        /** 
         * Dtor. 
         *
         * If the object referenced is not NULL, the reference count will be
         * decremented.
         */
        ~SmartRef(void);

        /**
         * Answer, whether the smart reference is a NULL pointer.
         *
         * @return true, if the reference is a NULL pointer, false otherwise.
         */
        inline bool IsNull(void) const {
            return (this->obj == NULL);
        }

        /**
         * Assignment operator.
         *
         * This operation creates a strong reference, i. e. decrements the 
         * reference count of the old object and increments the reference count
         * of the new one.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        SmartRef& operator =(T *rhs);

        /**
         * Assignment operator.
         *
         * This operation creates a strong reference, i. e. decrements the 
         * reference count of the old object and increments the reference count
         * of the new one.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        SmartRef& operator =(const SmartRef& rhs);

        /**
         * Answer, whether two smart references are equal, i. e. designate the
         * same object.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const SmartRef& rhs) const {
            return (this->obj == rhs.obj);
        }

        /**
         * Answer, whether two smart references are not equal, i. e. designate 
         * different objects.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const SmartRef& rhs) const {
            return (this->obj != rhs.obj);
        }

        /**
         * Dereferences the smart reference. No measurements against 
         * dereferencing NULL pointers are taken.
         *
         * @return A reference to the object designated by the reference.
         */
        inline T& operator *(void) {
            return *(this->obj);
        }

        /**
         * Dereferences the smart reference. No measurements against 
         * dereferencing NULL pointers are taken.
         *
         * @return A reference to the object designated by the reference.
         */
        inline const T& operator *(void) const {
            return *(this->obj);
        }

        /**
         * Member access. 
         * 
         * The member access operation creates a weak reference to the 
         * referenced object, i. e. the reference count is not modified.
         *
         * @return A pointer to the object designated by the smart reference.
         */
        inline T *operator ->(void) {
            return this->obj;
        }

        /**
         * Member access. 
         * 
         * The member access operation creates a weak reference to the 
         * referenced object, i. e. the reference count is not modified.
         *
         * @return A pointer to the object designated by the smart reference.
         */
        inline const T *operator ->(void) const {
            return this->obj;
        }

    private:

        /** The referenced object. */
        T *obj;

    };


    /*
     * SmartRef<T>::SmartRef
     */
    template<class T> 
    SmartRef<T>::SmartRef(T *obj, const bool addRef) : obj(obj) {
        if (addRef && (this->obj != NULL)) {
            this->obj->AddRef();
        }
    }


    /*
     * SmartRef<T>::~SmartRef
     */
    template<class T> SmartRef<T>::~SmartRef(void) {
        if (this->obj != NULL) {
            this->obj->Release();
        }
    }


    /*
     * vislib::SmartRef<T>::operator =
     */
    template<class T> SmartRef<T>& SmartRef<T>::operator =(T *rhs) {
        if (this->obj != rhs) {
            if (rhs != NULL) {
                rhs->AddRef();
            }
            if (this->obj != NULL) {
                this->obj->Release();
            }
            this->obj = rhs;
        }
        return *this;
    }


    /*
     * vislib::SmartRef<T>::operator =
     */
    template<class T>
    SmartRef<T>& SmartRef<T>::operator =(const SmartRef& rhs) {
        if (this != &rhs) {
            if (rhs.obj != NULL) {
                rhs.obj->AddRef();
            }
            if (this->obj != NULL) {
                this->obj->Release();
            }
            this->obj = rhs;
        }
        return *this;
    }
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SMARTREF_H_INCLUDED */
