/*
 * CPERAWDataSource.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CPERAWDataSource.h"

#include <chrono>
#include <fstream>
#include <vector>
#include <sstream>

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"

#include "vislib/sys/Log.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"


using namespace megamol;
using namespace megamol::stdplugin::datatools::io;


CPERAWDataSource::CPERAWDataSource(void) : core::Module(),
      filenameSlot("filename", "The path to the CPERAW file to load."),
      radiusSlot("radius", "the radius of the particles"),
      getData("getdata", "Slot to request data from this data source.") {

    this->getData.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(), core::moldyn::MultiParticleDataCall::FunctionName(0), &CPERAWDataSource::getDataCallback);
    this->getData.SetCallback(
        core::moldyn::MultiParticleDataCall::ClassName(), core::moldyn::MultiParticleDataCall::FunctionName(1), &CPERAWDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->filenameSlot << new core::param::FilePathParam("");
    //this->filenameSlot.SetUpdateCallback(&CPERAWDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filenameSlot);

    this->radiusSlot << new core::param::FloatParam(0.1f, 0.0001f, 10.0f);
    this->radiusSlot.SetUpdateCallback(&CPERAWDataSource::radiusChanged);
    this->MakeSlotAvailable(&this->radiusSlot);
}


CPERAWDataSource::~CPERAWDataSource(void) {
    this->Release();
}


bool CPERAWDataSource::create(void) {
    return true;
}


void CPERAWDataSource::release(void) {
}


bool CPERAWDataSource::assertData() {
    size_t len = 0;
    numPoints = 0;
    data.resize(0);

    const char* fname = this->filenameSlot.Param<core::param::FilePathParam>()->Value();

    FILE *f;
#ifdef WIN32
    auto const err = fopen_s(&f, fname, "rb");
#else
    auto const err = fopen(fname, "rb");
#endif
    if (err == 0) {
#ifdef WIN32
        struct _stat64 fs{};
        _fstat64(_fileno(f), &fs);
        len = fs.st_size;
#else
        struct stat fs{};
        fstat(fileno(f), &fs);
        len = fs.st_size;
#endif
        fclose(f);
    } else {
        vislib::sys::Log::DefaultLog.WriteError("Cannot get size of file %s", fname);
        return false;
    }

    if (len < headerLen) {
        vislib::sys::Log::DefaultLog.WriteError("File %s has illegal content: not enough information for both bounding boxes", fname);
        return false;
    }
    if (len == headerLen) {
        vislib::sys::Log::DefaultLog.WriteWarn("File %s is empty", fname);
        return true;
    }

    std::ifstream file(fname, std::ios::binary);
    size_t const payload = len - headerLen; // we have two float bounding boxes up front

    if (payload % pointStride != 0) {
        vislib::sys::Log::DefaultLog.WriteError("File %s has illegal content: payload is not a multiple of %u bytes: %llu", fname, pointStride, payload);
        return false;
    }

    data.resize(payload);

    for (int x = 0; x < 6; ++x) {
        file.read(reinterpret_cast<char *>(&globalBBox[x]), 4);
    }
    for (int x = 0; x < 6; ++x) {
        file.read(reinterpret_cast<char *>(&localBBox[x]), 4);
    }

    vislib::sys::Log::DefaultLog.WriteInfo("File %s global bbox: (%f, %f, %f - %f, %f, %f)", fname, 
        globalBBox[0], globalBBox[1], globalBBox[2], globalBBox[3], globalBBox[4], globalBBox[5]);
    vislib::sys::Log::DefaultLog.WriteInfo("File %s local bbox : (%f, %f, %f - %f, %f, %f)", fname,
        localBBox[0], localBBox[1], localBBox[2], localBBox[3], localBBox[4], localBBox[5]);

    // data.clear();

    file.read(data.data(), payload);
    numPoints = payload / pointStride;
    this->dataHash++;
    newFile = true;
    return true;
}


bool CPERAWDataSource::isDirty(void) {
    return this->filenameSlot.IsDirty() || this->radiusSlot.IsDirty();
}


void CPERAWDataSource::resetDirty(void) {
    this->filenameSlot.ResetDirty();
    this->radiusSlot.ResetDirty();
}


bool CPERAWDataSource::radiusChanged(core::param::ParamSlot &slot) {
    //this->dataHash++;
    return true;
}


bool CPERAWDataSource::getDataCallback(core::Call &c) {
    try {
        if (this->filenameSlot.IsDirty()) {
            this->assertData();
            this->filenameSlot.ResetDirty();
        }

        auto *mdc = dynamic_cast<core::moldyn::MultiParticleDataCall *>(&c);

        if (this->numPoints > 0) {
            if (newFile) {
                mdc->SetParticleListCount(1);
                mdc->SetFrameID(0);
                mdc->SetDataHash(this->dataHash);
                // TODO Unlocker
                auto &pl = mdc->AccessParticles(0);
                pl.SetGlobalRadius(this->radiusSlot.Param<core::param::FloatParam>()->Value());
                pl.SetBBox(vislib::math::Cuboid<float>(localBBox[0], localBBox[1], localBBox[2], localBBox[3], localBBox[4], localBBox[5]));
                pl.SetCount(numPoints);
                pl.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ, this->data.data() + 0, this->pointStride);
                pl.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGB, this->data.data() + 24, this->pointStride);
                newFile = false;
            } else {
                // just update radius, if necessary
                if (mdc->GetParticleListCount() == 1) {
                    mdc->AccessParticles(0).SetGlobalRadius(this->radiusSlot.Param<core::param::FloatParam>()->Value());
                    //mdc->SetDataHash(this->dataHash);
                }
            }
        } else {
            mdc->SetParticleListCount(0);
            return true;
        }
    } catch (...) {
        return false;
    }

    return true;
}


bool CPERAWDataSource::getExtentCallback(core::Call &c) {
    try {
        if (this->filenameSlot.IsDirty()) {
            this->assertData();
            this->filenameSlot.ResetDirty();
        }
        auto *mdc = dynamic_cast<core::moldyn::MultiParticleDataCall *>(&c);
        if (mdc != nullptr) {

            float const radius = this->radiusSlot.Param<core::param::FloatParam>()->Value();
            for (int x = 0; x < 3; x++) {
                globalCBox[x] = globalBBox[x] - radius;
            }
            for (int x = 3; x < 6; x++) {
                globalCBox[x] = globalBBox[x] + radius;
            }

            // TODO BBoxes
            mdc->SetFrameCount(1);
            mdc->AccessBoundingBoxes().Clear();
            mdc->AccessBoundingBoxes().SetObjectSpaceBBox(globalBBox[0], globalBBox[1], globalBBox[2], globalBBox[3], globalBBox[4], globalBBox[5]);
            mdc->AccessBoundingBoxes().SetObjectSpaceClipBox(globalCBox[0], globalCBox[1], globalCBox[2], globalCBox[3], globalCBox[4], globalCBox[5]);
            mdc->SetDataHash(this->dataHash);
        } else {
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}
