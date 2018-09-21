/*
 * AbstractMeshManipulator.cpp
 *
 * Copyright (C) 2018
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmstd_datatools/AbstractMeshManipulator.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::AbstractMeshManipulator::AbstractMeshManipulator
 */
datatools::AbstractMeshManipulator::AbstractMeshManipulator(const char *outSlotName, const char *inSlotName) : megamol::core::Module(),
        outDataSlot(outSlotName, "providing access to the manipulated data"),
        inDataSlot(inSlotName, "accessing the original data") {

    this->outDataSlot.SetCallback(geocalls::CallTriMeshData::ClassName(), "GetData", &AbstractMeshManipulator::getDataCallback);
    this->outDataSlot.SetCallback(
        geocalls::CallTriMeshData::ClassName(), "GetExtent", &AbstractMeshManipulator::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<geocalls::CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

}


/*
 * datatools::AbstractMeshManipulator::~AbstractMeshManipulator
 */
datatools::AbstractMeshManipulator::~AbstractMeshManipulator(void) {
}


/*
 * datatools::AbstractMeshManipulator::create
 */
bool datatools::AbstractMeshManipulator::create(void) {
    return true;
}


/*
 * datatools::AbstractMeshManipulator::release
 */
void datatools::AbstractMeshManipulator::release(void) {
}


/*
 * datatools::AbstractMeshManipulator::manipulateData
 */
bool datatools::AbstractMeshManipulator::manipulateData(
    geocalls::CallTriMeshData& outData, geocalls::CallTriMeshData& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractMeshManipulator::manipulateExtent
 */
bool datatools::AbstractMeshManipulator::manipulateExtent(
    geocalls::CallTriMeshData& outData, geocalls::CallTriMeshData& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


/*
 * datatools::AbstractMeshManipulator::getDataCallback
 */
bool datatools::AbstractMeshManipulator::getDataCallback(megamol::core::Call& c) {
 
    geocalls::CallTriMeshData* outMpdc = dynamic_cast<geocalls::CallTriMeshData*>(&c);
    if (outMpdc == NULL) return false;

    geocalls::CallTriMeshData* inMpdc = this->inDataSlot.CallAs<geocalls::CallTriMeshData>();
    if (inMpdc == NULL) return false;

    *inMpdc = *outMpdc; // to get the correct request time
    if (!(*inMpdc)(0)) return false;

    if (!this->manipulateData(*outMpdc, *inMpdc)) {
        inMpdc->Unlock();
        return false;
    }

    inMpdc->Unlock();

    return true;
}


/*
 * datatools::AbstractMeshManipulator::getExtentCallback
 */
bool datatools::AbstractMeshManipulator::getExtentCallback(megamol::core::Call& c) {

    geocalls::CallTriMeshData* outMpdc = dynamic_cast<geocalls::CallTriMeshData*>(&c);
    if (outMpdc == NULL) return false;

    geocalls::CallTriMeshData* inMpdc = this->inDataSlot.CallAs<geocalls::CallTriMeshData>();
    if (inMpdc == NULL) return false;

    *inMpdc = *outMpdc; // to get the correct request time
    if (!(*inMpdc)(1)) return false;

    if (!this->manipulateExtent(*outMpdc, *inMpdc)) {
        inMpdc->Unlock();
        return false;
    }

    inMpdc->Unlock();

    return true;
}
