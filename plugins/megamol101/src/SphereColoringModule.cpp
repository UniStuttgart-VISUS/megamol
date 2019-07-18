/*
 * SphereColoringModule.cpp
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "SphereColoringModule.h"
#include <cfloat>
#include "CallSpheres.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "vislib/math/ShallowVector.h"

using namespace megamol;
using namespace megamol::megamol101;

/*
 * SphereColoringModule::SphereColoringModule
 */
SphereColoringModule::SphereColoringModule(void)
    : core::Module()
    , inSlot("inData", "Input slot for sphere data.")
    , outSlot("outData", "Output slot for colored sphere data.")
    , minColorSlot("minColor", "The min color for the interpolation")
    , maxColorSlot("maxColor", "The max color for the interpolation")
    , singleColorSlot("useSingleColor", "Just use minColor")
    , isActiveSlot("isActive", "Is this module active?") {
    // TUTORIAL: A name and a description for each slot (CallerSlot, CalleeSlot, ParamSlot) has to be given in the
    // constructor initializer list

    // TUTORIAL: For each CalleeSlot the callback functions have to be set
    this->outSlot.SetCallback(CallSpheres::ClassName(), "GetData", &SphereColoringModule::getDataCallback);
    this->outSlot.SetCallback(CallSpheres::ClassName(), "GetExtent", &SphereColoringModule::getExtentCallback);
    this->MakeSlotAvailable(&this->outSlot);

    // TUTORIAL: For each CallerSlot all compatible calls have to be set
    this->inSlot.SetCompatibleCall<CallSpheresDescription>();
    this->MakeSlotAvailable(&this->inSlot);

    // TUTORIAL: For each ParamSlot a default value has to be set
    this->minColorSlot.SetParameter(new core::param::ColorParam(0.0f, 0.0f, 1.0f, 1.0f));
    this->MakeSlotAvailable(&this->minColorSlot);

    this->maxColorSlot.SetParameter(new core::param::ColorParam(1.0f, 0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->maxColorSlot);

    this->singleColorSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->singleColorSlot);

    this->isActiveSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->isActiveSlot);

    // TUTORIAL: Each slot that shall be visible in the GUI has to be made available by this->MakeSlotAvailable(...)

    // other default values
    lastHash = 0;
    hashOffset = 0;
    numColor = 0;
    colors = nullptr;
    isDirty = true;
}

/*
 * SphereColoringModule::SphereColoringModule
 */
SphereColoringModule::~SphereColoringModule(void) { this->Release(); }

/*
 * SphereColoringModule::areSlotsDirty
 */
bool SphereColoringModule::areSlotsDirty(void) {

    if (this->singleColorSlot.IsDirty()) return true;
    if (this->minColorSlot.IsDirty()) return true;
    if (this->maxColorSlot.IsDirty()) return true;
    if (this->isActiveSlot.IsDirty()) return true;

    return false;
}

/*
 * SphereColoringModule::create
 */
bool SphereColoringModule::create(void) { return true; }

/*
 * SphereColoringModule::getDataCallback
 */
bool SphereColoringModule::getDataCallback(core::Call& call) {
    CallSpheres* csOut = dynamic_cast<CallSpheres*>(&call);
    if (csOut == nullptr) return false;

    CallSpheres* csIn = this->inSlot.CallAs<CallSpheres>();
    if (csIn == nullptr) return false;

    if (!(*csIn)(CallSpheres::CallForGetData)) return false;

    csOut->operator=(*csIn); // deep copy

    if (isDirty) {
        modifyColors(csOut);
    }

    csOut->SetDataHash(csOut->DataHash() + hashOffset);

    return true;
}

/*
 * SphereColoringModule::getExtentCallback
 */
bool SphereColoringModule::getExtentCallback(core::Call& call) {
    CallSpheres* csOut = dynamic_cast<CallSpheres*>(&call);
    if (csOut == nullptr) return false;

    CallSpheres* csIn = this->inSlot.CallAs<CallSpheres>();
    if (csIn == nullptr) return false;

    if (!(*csIn)(CallSpheres::CallForGetExtent)) return false;

    bool slotsDirty = this->areSlotsDirty();
    if (lastHash != csIn->DataHash() || slotsDirty) {
        lastHash = csIn->DataHash();
        if (slotsDirty) hashOffset++;
        resetDirtySlots();
        isDirty = true;
    }

    csOut->operator=(*csIn); // deep copy
    csOut->SetDataHash(csOut->DataHash() + hashOffset);

    return true;
}

/*
 * SphereColoringModule::modifyColors
 */
void SphereColoringModule::modifyColors(CallSpheres* cs) {

    if (!this->isActiveSlot.Param<core::param::BoolParam>()->Value()) {
        if (this->colors != nullptr) {
            delete[] this->colors;
            this->colors = nullptr;
        }
        this->numColor = 0;
        return;
    }

    this->numColor = cs->Count();

    // does this module have to store the colors or this there a module
    // that stores them already?
    if (!cs->HasColors()) {
        // if this module has to store the color: allocate the needed space and set the pointer
        if (this->colors != nullptr) {
            delete[] this->colors;
            this->colors = nullptr;
        }
        this->colors = new float[this->numColor * 4];
        cs->SetColors(this->colors);
    }

    auto minColorP = this->minColorSlot.Param<core::param::ColorParam>()->Value();
    auto maxColorP = this->maxColorSlot.Param<core::param::ColorParam>()->Value();
    vislib::math::ShallowVector<float, 4> minColor(minColorP.data());
    vislib::math::ShallowVector<float, 4> maxColor(maxColorP.data());

    // this pointer might be to the array stored by this module or to an array stored by another one
    float* colPtr = cs->GetColors();

    if (this->singleColorSlot.Param<core::param::BoolParam>()->Value()) {
        // just use minColor for every node
        for (SIZE_T i = 0; i < numColor; i++) {
            auto cP = vislib::math::ShallowVector<float, 4>(&colPtr[i * 4]);
            cP = minColor;
        }
    } else {
        // compute the min and max radius
        float minRad = FLT_MAX;
        float maxRad = FLT_MIN;
        auto spherePtr = cs->GetSpheres();
        for (SIZE_T i = 0; i < numColor; i++) {
            auto r = spherePtr[i * 4 + 3];
            if (r < minRad) minRad = r;
            if (r > maxRad) maxRad = r;
        }

        // interpolate between the two colors using the radius as alpha
        for (SIZE_T i = 0; i < numColor; i++) {
            auto cP = vislib::math::ShallowVector<float, 4>(&colPtr[i * 4]);
            float alpha = (spherePtr[i * 4 + 3] - minRad) / (maxRad - minRad);
            cP = (1.0f - alpha) * minColor + alpha * maxColor;
        }
    }
}

/*
 * SphereColoringModule::release
 */
void SphereColoringModule::release(void) {
    if (this->colors != nullptr) {
        delete[] this->colors;
        this->colors = nullptr;
    }
}

/*
 * SphereColoringModule::resetDirtySlots
 */
void SphereColoringModule::resetDirtySlots(void) {
    this->singleColorSlot.ResetDirty();
    this->minColorSlot.ResetDirty();
    this->maxColorSlot.ResetDirty();
    this->isActiveSlot.ResetDirty();
}
