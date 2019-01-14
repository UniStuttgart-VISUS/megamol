/*
 * CallSpheres.cpp
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "CallSpheres.h"

using namespace megamol;
using namespace megamol::megamol101;

/*
 * CallSpheres::CallForGetData
 */
const unsigned int CallSpheres::CallForGetData = 0;

/*
 * CallSpheres::CallForGetExtent
 */
const unsigned int CallSpheres::CallForGetExtent = 1;

/*
 * CallSpheres::CallSpheres
 */
CallSpheres::CallSpheres(void) : spheres(nullptr), colors(nullptr), count(0), colorsAvailable(false) {}

/*
 * CallSpheres::~CallSpheres
 */
CallSpheres::~CallSpheres(void) {
    spheres = nullptr;
    colors = nullptr;
    // The actual deletion of the data does not happen here.
    // This happens in the module that stores the data.
}

/*
 * CallSpheres::Count
 */
SIZE_T CallSpheres::Count(void) const { return this->count; }

/*
 * CallSpheres::GetColors
 */
float* CallSpheres::GetColors(void) const { return this->colors; }

/*
 * CallSpheres::GetSpheres
 */
const float* CallSpheres::GetSpheres(void) const { return this->spheres; }

/*
 * CallSpheres::HasColors
 */
bool CallSpheres::HasColors(void) const { return this->colorsAvailable; }

/*
 * CallSpheres::ResetColors
 */
void CallSpheres::ResetColors(void) {
    this->colors = nullptr;
    this->colorsAvailable = false;
}

/*
 * CallSpheres::SetColors
 */
void CallSpheres::SetColors(float* colors) {
    this->colors = colors;
    this->colorsAvailable = (colors != nullptr);
}

/*
 * CallSpheres::SetData
 */
void CallSpheres::SetData(SIZE_T count, const float* spheres, float* colors) {
    this->count = count;
    this->spheres = spheres;
    this->colors = colors;
    this->colorsAvailable = (colors != nullptr);
}

/*
 * CallSpheres::operator=
 */
CallSpheres& CallSpheres::operator=(const CallSpheres& rhs) {
    AbstractGetData3DCall::operator=(rhs);
    this->count = rhs.count;
    this->colorsAvailable = rhs.colorsAvailable;
    this->spheres = rhs.spheres;
    this->colors = rhs.colors;
    return *this;
}
