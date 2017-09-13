/*
 * UniGrid3D.h
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#ifndef MMPROTEINCUDAPLUGIN_UNIGRID3D_H
#define MMPROTEINCUDAPLUGIN_UNIGRID3D_H

#include "vislib/math/Vector.h"
#include "vislib/OutOfRangeException.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"

namespace megamol {
namespace protein_cuda {

/**
 * Class representing a 3D uniform grid containing data of type T.
 */
template <class T>
class UniGrid3D {

public:

    /**
     * Ctor.
     */
    UniGrid3D();

    /**
     * Ctor.
     * TODO
     */
    UniGrid3D(vislib::math::Vector<unsigned int, 3> gridDim,
            vislib::math::Vector<float, 3> gridOrg,
            float gridStepSize);

    /** Dtor. */
    ~UniGrid3D();

    cudaError MemCpyFromDevice(const T *dvPt) {
        cudaMemcpy(this->data, dvPt,
                this->gridDim.X()*this->gridDim.Y()*this->gridDim.Z()*sizeof(T),
                cudaMemcpyDeviceToHost);
        return cudaGetLastError();
    }

    /**
     * TODO
     */
    void MemCpyFromHost(const T *Pt) {
        memcpy(this->data, Pt, this->gridDim.X()*this->gridDim.Y()*this->gridDim.Z()*sizeof(T));
    }

    /**
     * Deallocate memory and reset vars
     */
    void Clear();

    /**
     * TODO
     */
    T GetAt(unsigned int x, unsigned int y, unsigned int z);

    /**
     * TODO
     */
    vislib::math::Vector<unsigned int, 3> GetGridDim() {
        return this->gridDim;
    }

    /**
     * TODO
     */
    vislib::math::Vector<float, 3> GetGridOrg() {
        return this->gridOrg;
    }

    /**
     * TODO
     */
    float GetGridStepSize() {
        return this->gridStepSize;
    }

    /**
     * TODO
     */
    void Init(vislib::math::Vector<unsigned int, 3> gridDim,
            vislib::math::Vector<float, 3> gridOrg,
            float gridStepSize);

    /**
     * TODO
     */
    const T *PeekBuffer() const;

    /**
     * TODO
     */
    void SetAt(unsigned int x, unsigned int y, unsigned int z, T val);

    /**
     * TODO
     */
    void SetAt(vislib::math::Vector<unsigned int, 3> pos, T val);

    /**
     * TODO
     */
    T SampleNearest(float x, float y, float z);

protected:

private:

    /** The dimension of the grid in all three directions */
    vislib::math::Vector<unsigned int, 3> gridDim;

    /** The origin of the grid in world coordinates */
    vislib::math::Vector<float, 3> gridOrg;

    /** The spacing (cell size) of the grid */
    float gridStepSize;

    /** Array haolding the actual data */
    T *data;
};


/*
 * UniGrid3D<T>::UniGrid3D
 */
template <class T>
UniGrid3D<T>::UniGrid3D() :
        gridDim(0, 0, 0), gridOrg(0.0f, 0.0f, 0.0f), gridStepSize(0.0f),
        data(NULL) {
}


/*
 * UniGrid3D<T>::UniGrid3D
 */
template <class T>
UniGrid3D<T>::UniGrid3D(vislib::math::Vector<unsigned int, 3> gridDim,
        vislib::math::Vector<float, 3> gridOrg,
        float gridStepSize) :
        gridDim(gridDim), gridOrg(gridOrg), gridStepSize(gridStepSize) {

    this->data = new T[this->gridDim.X()*this->gridDim.Y()*this->gridDim.Z()];
}


/*
 * UniGrid3D::~UniGrid3D
 */
template <class T>
UniGrid3D<T>::~UniGrid3D() {
    delete[] this->data;

}


/*
 * UniGrid3D<T>::Clear
 */
template <class T>
void UniGrid3D<T>::Clear() {
    this->gridDim.Set(0, 0, 0);
    this->gridOrg.Set(0.0f, 0.0f, 0.0f);
    this->gridStepSize = 0.0f;
    delete[] this->data;
    this->data = NULL;
}


/*
 * UniGrid3D<T>::GetAt
 */
template <class T>
T UniGrid3D<T>::GetAt(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int idx = this->gridDim.X()*(this->gridDim.Y()*z+y)+x;
    //ASSERT(idx < this->gridDim.X()*this->gridDim.Y()*this->gridDim.Z()); // TODO use vislib out of range exception
    return this->data[idx];
}


/*
 * UniGrid3D<T>::Init
 */
template <class T>
void UniGrid3D<T>::Init(vislib::math::Vector<unsigned int, 3> gridDim,
        vislib::math::Vector<float, 3> gridOrg,
        float gridStepSize) {

    // Clean up if necessary
    if(this->data != NULL) {
        delete[] this->data;
    }

    this->gridDim = gridDim;
    this->gridOrg = gridOrg;
    this->gridStepSize = gridStepSize;
    this->data = new T[this->gridDim.X()*this->gridDim.Y()*this->gridDim.Z()];
}


/*
 * UniGrid3D<T>::PeekBuffer
 */
template <class T>
const T *UniGrid3D<T>::PeekBuffer() const {
    return this->data;
}


/*
 * UniGrid3D<T>::SetAtSetAt
 */
template <class T>
void UniGrid3D<T>::SetAt(unsigned int x, unsigned int y, unsigned int z, T val) {
    this->data[this->gridDim.X()*(this->gridDim.Y()*z+y)+x] = val;
}


/*
 * UniGrid3D<T>::SetAtSetAt
 */
template <class T>
void UniGrid3D<T>::SetAt(vislib::math::Vector<unsigned int, 3> pos, T val) {
    this->data[this->gridDim.X()*(this->gridDim.Y()*pos.Z()+pos.Y())+pos.X()] = val;
}


/*
 * UniGrid3D<T>::SampleNearest
 */
template <class T>
T UniGrid3D<T>::SampleNearest(float x, float y, float z) {

    //printf("SAMPLENEAREST (x, y, z) = (%f %f %f)\n", x, y, z); // DEBUG
    float posXf = (x - this->gridOrg.X())/this->gridStepSize;
    float posYf = (y - this->gridOrg.Y())/this->gridStepSize;
    float posZf = (z - this->gridOrg.Z())/this->gridStepSize;

    //printf("SAMPLENEAREST (posXf, posYf, posZf) = (%f %f %f)\n", posXf, posYf, posZf); // DEBUG

    unsigned int posX, posY, posZ;
    if((posXf - static_cast<int>(posXf)) < 0.5) {
        posX = static_cast<unsigned int>(posXf);
    }
    else {
        posX = static_cast<unsigned int>(posXf) + 1;
    }
    if((posYf - static_cast<int>(posYf)) < 0.5) {
        posY = static_cast<unsigned int>(posYf);
    }
    else {
        posY = static_cast<unsigned int>(posYf) + 1;
    }
    if((posZf - static_cast<int>(posZf)) < 0.5) {
        posZ = static_cast<unsigned int>(posZf);
    }
    else {
        posZ = static_cast<unsigned int>(posZf) + 1;
    }

    //printf("SAMPLENEAREST (posX, posY, posZ) = (%u %u %u)\n", posX, posY, posZ); // DEBUG

    //printf("SAMPLENEAREST %f\n", this->GetAt(posX, posY, posZ)); // DEBUG

    return this->GetAt(posX, posY, posZ);
}

} /* end namespace protein_cuda */
} /* end namespace megamol */


#endif /* MMPROTEINCUDAPLUGIN_UNIGRID3D_H */
