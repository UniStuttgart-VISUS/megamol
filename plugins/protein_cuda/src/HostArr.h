//
// HostArr.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 14, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_HOSTARR_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_HOSTARR_H_INCLUDED

namespace megamol {
namespace protein_cuda {

template<class T>
class HostArr {

public:

    /** Ctor */
    HostArr() : size(0),  count(0), pt(NULL) {}

    /** Dtor */
    ~HostArr() {
        this->Release();
    }

    /**
     * Answers the number of elements in the array.
     *
     * @return The number of elements
     */
    inline size_t GetCount() const {
        return this->count;
    }

    /**
     * Answers the number of elements in the array.
     *
     * @return The number of elements
     */
    inline size_t GetSize() {
        return this->size;
    }

    /**
     * Returns the actual (non const) device pointer.
     *
     * @return The pointer to the device memory.
     */
    inline T *Peek() {
        return this->pt;
    }

    /**
     * Returns the actual (non const) device pointer.
     *
     * @return The pointer to the device memory.
     */
    inline const T *PeekConst() const {
        return this->pt;
    }

    /**
     * Deallocates the memory of the array and inits size with zero.
     */
    inline void Release() {
        if (this->pt != NULL) {
            //delete[] this->pt;
            free(this->pt);
        }
        this->size = 0;
        this->pt = NULL;
        this->count = 0;
    }

    /**
     * Inits each byte in the array with a given value.
     *
     * @param c The byte value
     */
    void Set(char c) {
        memset((char*)(this->pt), c, this->GetSize()*sizeof(T));
    }

    /**
     * If necessary (re)allocates the device memory to meet the desired new size
     * of the array.
     *
     * @param sizeNew The desired amount of elements.
     */
    inline void Validate(size_t sizeNew) {
        if ((this->pt == NULL) || (sizeNew > this->size)) {
            this->Release();
            //this->pt = new T[sizeNew];
            this->pt = (T*)malloc(sizeNew*sizeof(T));
            this->size = sizeNew;
        }
        this->count = sizeNew;
    }

private:

    /// The amount of allocated memory in sizeof(T)
    size_t size;

    /// The number of elements
    size_t count;

    /// The pointer to the device memory
    T *pt;
};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_HOSTARR_H_INCLUDED
