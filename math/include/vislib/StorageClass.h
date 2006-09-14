/*
 * StorageClass.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STORAGECLASS_H_INCLUDED
#define VISLIB_STORAGECLASS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/assert.h"


namespace vislib {
namespace math {

    /**
     * This class is part of the crowbar2-pattern which allows the creation of 
     * shallow and deep vectors/matrices/quaterions ...
     *
     * The template is instaniated for a value type T and a number of
     * components D.
     */
    template<class T, unsigned int C> class ShallowStorageClass {

    public:

        /**
         * Create a storage initialised with 'data', i. e. this storage is an 
         * alias of 'data'.
         *
         * @param data The initial data.
         */
        inline ShallowStorageClass(T *data) : data(data) {}

        /**
         * Clone 'rhs'. Note, that this ctor creates an alias of the data 
         * designated by 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline ShallowStorageClass(const ShallowStorageClass& rhs) 
            : data(rhs.data) {}

        /** Dtor. */
        inline ~ShallowStorageClass(void) {}

        /**
         * Assignment. Note, that this operation creates an alias of the data
         * designated by 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowStorageClass& operator =(
                    const ShallowStorageClass& rhs) {
            return (*this = rhs);
        }

        /**
         * Assignment from a T array. Note, that this operation creates an 
         * alias of the data designated by 'rhs'. 'rhs' must not be NULL.
         *
         * @param rhs The right hand side operand. This must not be NULL.
         *
         * @return *this.
         */
        ShallowStorageClass& operator =(T *rhs);

        /**
         * Access the 'i'th element in the vector. No range check is performed 
         * for 'i'.
         *
         * @param i The index of the element to be accessed, which must be 
         *          within [0, D - 1].
         *
         * @return The 'i'th element.
         */
        inline T& operator [](const int i) {
            return this->data[i];
        }

        /**
         * Access the 'i'th element in the vector. No range check is performed 
         * for 'i'.
         *
         * @param i The index of the element to be accessed, which must be 
         *          within [0, D - 1].
         *
         * @return The 'i'th element.
         */
        inline const T& operator [](const int i) const {
            return this->data[i];
        }

        /**
         * Provide direct access to the memory wrapped by the object.
         *
         * @return A pointer to the memory.
         */
        inline operator T *(void) {
            return this->data;
        }

        /**
         * Provide direct access to the memory wrapped by the object.
         *
         * @return A pointer to the memory.
         */
        inline operator const T *(void) const {
            return this->data;
        }

    private:

        /** 
         * Create a NULL pointer storage. Using this ctor is <b>inherently 
         * unsafe</b> and is therefore forbidden! 
         *
         * Declaring this ctor private prevents instantiation of unsafe ctors
         * in the classes using ShallowStorageClassT.
         */
        inline ShallowStorageClass(void) : data(NULL) {
            ASSERT(false);
        }

        /** Pointer to the actual data. */
        T *data;
    };


    /*
     * ShallowStorageClass<T, D>::operator =
     */
    template<class T, unsigned int D> 
    ShallowStorageClass<T, D>& ShallowStorageClass<T, D>::operator =(T *rhs) {
        ASSERT(rhs != NULL);

        if (this != &rhs) {
            this->data = rhs.data;
        }

        return *this;
    }


    /**
     * This class is part of the crowbar2-pattern as described above. See 
     * documentation of ShallowStorageClassT for more information.
     */
    template<class T, unsigned int D> class DeepStorageClass {

    public:
        
        /** 
         * Create a zeroed storage.
         */
        DeepStorageClass(void);

        /**
         * Create a storage initialised with T.
         *
         * @param data The initial data. This must not be NULL.
         */
        inline DeepStorageClass(const T *data) {
            ASSERT(data != NULL);
            ::memcpy(this->data, data, D * sizeof(T));
        }

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline DeepStorageClass(const DeepStorageClass& rhs) {
            ::memcpy(this->data, data, D * sizeof(T));
        }

        /** Dtor. */
        inline ~DeepStorageClass(void) {}

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline DeepStorageClass& operator =(const DeepStorageClass& rhs) {
            return (*this = rhs.data);
        }

        /**
         * Assignment from a T array. Creates a deep copy of the data designated
         * by 'rhs'. 'rhs' must not be NULL.
         *
         * @param rhs The right hand side operand. This must not be NULL.
         *
         * @return *this.
         */
        DeepStorageClass& operator =(const T *rhs);

        /**
         * Access the 'i'th element in the vector. No range check is performed 
         * for 'i'.
         *
         * @param i The index of the element to be accessed, which must be 
         *          within [0, D - 1].
         *
         * @return The 'i'th element.
         */
        inline T& operator [](const int i) {
            return this->data[i];
        }

        /**
         * Access the 'i'th element in the vector. No range check is performed 
         * for 'i'.
         *
         * @param i The index of the element to be accessed, which must be 
         *          within [0, D - 1].
         *
         * @return The 'i'th element.
         */
        inline const T& operator [](const int i) const {
            return this->data[i];
        }

        /**
         * Provide direct access to the memory wrapped by the object.
         *
         * @return A pointer to the memory.
         */
        inline operator T *(void) {
            return this->data;
        }

        /**
         * Provide direct access to the memory wrapped by the object.
         *
         * @return A pointer to the memory.
         */
        inline operator const T *(void) const {
            return this->data;
        }

    private:

        /** The actual data. */
        T data[D];
    };


    /*
     * DeepStorageClass<T, D>::DeepStorageClass
     */
    template<class T, unsigned int D> 
    DeepStorageClass<T, D>::DeepStorageClass(void) {
        for (unsigned int d = 0; d < D; d++) {
            this->data[d] = static_cast<T>(0);
        }
    }


    /*
     * DeepStorageClassT<T, D>::operator =
     */
    template<class T, unsigned int D> 
    DeepStorageClass<T, D>& DeepStorageClass<T, D>::operator =(const T *rhs) {
        ASSERT(rhs != NULL);

        if (this != &rhs) {
            ::memcpy(this->data, rhs, D * sizeof(T));
        }

        return *this;
    }

} /* end namespace math */
} /* end namespace vislib */

#endif /* VISLIB_STORAGECLASST_H_INCLUDED */
