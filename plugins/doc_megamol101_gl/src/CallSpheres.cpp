/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#include "CallSpheres.h"

using namespace megamol;
using namespace megamol::megamol101_gl;

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
CallSpheres::CallSpheres() : count(0), colorsAvailable(false), spheres(nullptr), colors(nullptr) {}

/*
 * CallSpheres::~CallSpheres
 */
CallSpheres::~CallSpheres() {
    spheres = nullptr;
    colors = nullptr;
    // The actual deletion of the data does not happen here.
    // This happens in the module that stores the data.
}

/*
 * CallSpheres::Count
 */
std::size_t CallSpheres::Count() const {
    return this->count;
}

/*
 * CallSpheres::GetColors
 */
float* CallSpheres::GetColors() const {
    return this->colors;
}

/*
 * CallSpheres::GetSpheres
 */
const float* CallSpheres::GetSpheres() const {
    return this->spheres;
}

/*
 * CallSpheres::HasColors
 */
bool CallSpheres::HasColors() const {
    return this->colorsAvailable;
}

/*
 * CallSpheres::ResetColors
 */
void CallSpheres::ResetColors() {
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
void CallSpheres::SetData(std::size_t count, const float* spheres, float* colors) {
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
