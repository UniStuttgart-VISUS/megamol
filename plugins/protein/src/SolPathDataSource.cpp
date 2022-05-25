/*
 * SolPathRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#define _USE_MATH_DEFINES 1
#include "SolPathDataSource.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/MemmappedFile.h"
#include <cfloat>
#include <climits>
#include <cmath>

#include "mmcore/BoundingBoxes_2.h"

using namespace megamol::core;
using namespace megamol::protein;


/*
 * SolPathDataSource::SolPathDataSource
 */
SolPathDataSource::SolPathDataSource(void)
        : core::Module()
        , getdataslot("getdata", "Publishes the data for other modules")
        , filenameslot("filename", "The path of the solpath file to load")
        , smoothSlot("smooth", "Flag whether or not to smooth the data")
        , smoothValueSlot("smoothValue", "Value for the smooth filter")
        , smoothExpSlot("smoothExp", "The smoothing filter function exponent")
        , speedOfSmoothedSlot(
              "speedOfSmoothed", "Flag whether or not to use the smoothed data for the speed calculation")
        , clusterOfSmoothedSlot("clusterOfSmoothed", "Flag to cluster the smoothed or unsmoothed data") {

    this->getdataslot.SetCallback(SolPathDataCall::ClassName(), "GetData", &SolPathDataSource::getData);
    this->getdataslot.SetCallback(SolPathDataCall::ClassName(), "GetExtent", &SolPathDataSource::getExtent);
    this->MakeSlotAvailable(&this->getdataslot);

    this->filenameslot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameslot);

    this->smoothSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->smoothSlot);

    this->smoothValueSlot << new param::FloatParam(10.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->smoothValueSlot);

    this->smoothExpSlot << new param::FloatParam(3.0f, 2.0f);
    this->MakeSlotAvailable(&this->smoothExpSlot);

    this->speedOfSmoothedSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->speedOfSmoothedSlot);

    this->clusterOfSmoothedSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->clusterOfSmoothedSlot);
}


/*
 * SolPathDataSource::~SolPathDataSource
 */
SolPathDataSource::~SolPathDataSource(void) {
    this->Release();
    this->clear();
}


/*
 * SolPathDataSource::create
 */
bool SolPathDataSource::create(void) {
    if (this->anyParamslotDirty()) {
        this->loadData();
    }
    return true;
}


/*
 * SolPathDataSource::release
 */
void SolPathDataSource::release(void) {
    this->clear();
}


/*
 * SolPathDataSource::getData
 */
bool SolPathDataSource::getData(megamol::core::Call& call) {
    SolPathDataCall* spdc = dynamic_cast<SolPathDataCall*>(&call);
    if (spdc == NULL)
        return false;

    if (this->anyParamslotDirty()) {
        this->loadData();
    }

    spdc->Set(static_cast<unsigned int>(this->pathlines.Count()), this->pathlines.PeekElements(), this->minTime,
        this->maxTime, this->minSpeed, this->maxSpeed);

    return true;
}


/*
 * SolPathDataSource::getExtent
 */
bool SolPathDataSource::getExtent(megamol::core::Call& call) {
    SolPathDataCall* spdc = dynamic_cast<SolPathDataCall*>(&call);
    if (spdc == NULL)
        return false;

    if (this->anyParamslotDirty()) {
        this->loadData();
    }

    spdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);

    return false;
}


/*
 * SolPathDataSource::clear
 */
void SolPathDataSource::clear(void) {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->minTime = 0.0f;
    this->maxTime = 0.0f;
    this->minSpeed = 0.0f;
    this->maxSpeed = 0.0f;
    this->vertices.Clear();
    this->pathlines.Clear();
}


/*
 * SolPathDataSource::loadData
 */
void SolPathDataSource::loadData(void) {
    using megamol::core::utility::log::Log;
    using vislib::sys::File;
    vislib::sys::MemmappedFile file;

    this->filenameslot.ResetDirty();
    this->smoothSlot.ResetDirty();
    this->smoothValueSlot.ResetDirty();
    this->smoothExpSlot.ResetDirty();
    this->speedOfSmoothedSlot.ResetDirty();
    this->clusterOfSmoothedSlot.ResetDirty();

    this->clear();

    if (file.Open(this->filenameslot.Param<param::FilePathParam>()->Value().native().c_str(), File::READ_ONLY,
            File::SHARE_READ, File::OPEN_ONLY) == false) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open data file %s",
            this->filenameslot.Param<param::FilePathParam>()->Value().generic_u8string().c_str());
        return;
    }

    vislib::StringA headerID;
    if (file.Read(headerID.AllocateBuffer(7), 7) != 7) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read data file %s",
            this->filenameslot.Param<param::FilePathParam>()->Value().generic_u8string().c_str());
        return;
    }

    vislib::SingleLinkedList<fileBlockInfo> fileStruct;
    fileBlockInfo* blockInfo = NULL;

    if (headerID.Equals("SolPath")) {
        unsigned int version;
        if (file.Read(&version, 4) != 4) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read data file %s",
                this->filenameslot.Param<param::FilePathParam>()->Value().generic_u8string().c_str());
            return;
        }
        if (version > 1) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Data file %s uses unsupported version %u",
                this->filenameslot.Param<param::FilePathParam>()->Value().generic_u8string().c_str(), version);
            return;
        }

        fileBlockInfo info;
        while (!file.IsEOF()) {
            if (file.Read(&info.id, 4) != 4)
                break;
            if (file.Read(&info.size, 8) != 8)
                break;
            info.start = file.Tell();
            file.Seek(info.size, vislib::sys::File::CURRENT);
            fileStruct.Add(info);
        }

    } else {
        // legacy file support
        fileBlockInfo info;
        info.id = 0;                // <= pathline data
        info.start = 0;             // <= start of file
        info.size = file.GetSize(); // <= whole file
        fileStruct.Add(info);
    }

    // search for pathline data block
    vislib::SingleLinkedList<fileBlockInfo>::Iterator iter = fileStruct.GetIterator();
    while (iter.HasNext()) {
        fileBlockInfo& info = iter.Next();
        if (info.id == 0) {
            blockInfo = &info;
            break;
        }
    }
    if (blockInfo == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "File %s does not contain a path line data block",
            this->filenameslot.Param<param::FilePathParam>()->Value().generic_u8string().c_str());
        return;
    }
    file.Seek(blockInfo->start, File::BEGIN);

    // load data
    unsigned int atomCount;
    file.Read(&atomCount, 4);

    this->pathlines.AssertCapacity(atomCount);

    for (SIZE_T a = 0; a < atomCount; a++) {
        unsigned int tmp, framesCount;
        SolPathDataCall::Vertex vertex;
        SolPathDataCall::Pathline path;
        vertex.speed = 0.0f; // will be set later
        vertex.clusterID = 0;
        vertex.time = -2.0f;
        path.length = 0;
        path.data = NULL; // will be set later

        file.Read(&tmp, 4);
        path.id = tmp;
        file.Read(&framesCount, 4);

        this->vertices.AssertCapacity(this->vertices.Count() + framesCount);

        for (SIZE_T e = 0; e < framesCount; e++) {
            file.Read(&tmp, 4);
            file.Read(&vertex.x, 12);

            if (tmp != static_cast<unsigned int>(vertex.time) + 1) {
                // start a new path
                if (path.length > 0) {
                    this->pathlines.Add(path);
                }
                path.length = 1;
            } else {
                // continue path
                path.length++;
            }
            vertex.time = static_cast<float>(tmp);

            this->vertices.Add(vertex);
        }

        if (path.length > 0) {
            this->pathlines.Add(path);
        }
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 1000, "Finished solpath file IO");

    // calculate pointers, bounding data, and speed values
    SIZE_T off = 0;
    for (SIZE_T p = 0; p < this->pathlines.Count(); p++) {
        off += this->pathlines[p].length;
    }
    if (off < this->vertices.Count()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Path data inconsistent: too many vertices");
    } else if (off > this->vertices.Count()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Path data inconsistent: too few vertices");
        this->clear();
        return;
    }

    if (off <= 0) {
        // data is empty!
        return;
    }

    off = 0;
    this->maxTime = 0.0f;
    this->minTime = FLT_MAX;
    this->maxSpeed = -FLT_MAX;
    this->minSpeed = FLT_MAX;
    this->bbox.Set(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (SIZE_T p = 0; p < this->pathlines.Count(); p++) {
        this->pathlines[p].data = this->vertices.PeekElements() + off;

        for (SIZE_T v = 0; v < this->pathlines[p].length; v++) {
            SolPathDataCall::Vertex& v1 = this->vertices[off + v];

            if (v > 0) {
                v1.speed = vislib::math::ShallowPoint<float, 3>(&v1.x).Distance(
                    vislib::math::ShallowPoint<float, 3>(&this->vertices[off + v - 1].x));
            }

            if (v1.time > this->maxTime)
                this->maxTime = v1.time;
            if (v1.time < this->minTime)
                this->minTime = v1.time;
            if (v1.speed > 0.01f) {
                if (v1.speed > this->maxSpeed)
                    this->maxSpeed = v1.speed;
                if (v1.speed < this->minSpeed)
                    this->minSpeed = v1.speed;
            }
            if (this->bbox.Left() > v1.x)
                this->bbox.SetLeft(v1.x);
            if (this->bbox.Right() < v1.x)
                this->bbox.SetRight(v1.x);
            if (this->bbox.Bottom() > v1.y)
                this->bbox.SetBottom(v1.y);
            if (this->bbox.Top() < v1.y)
                this->bbox.SetTop(v1.y);
            if (this->bbox.Back() > v1.z)
                this->bbox.SetBack(v1.z);
            if (this->bbox.Front() < v1.z)
                this->bbox.SetFront(v1.z);
        }
        if (this->pathlines[p].length > 2) {
            this->vertices[off].speed = this->vertices[off + 1].speed;
        }

        off += this->pathlines[p].length;
    }

    this->bbox.EnforcePositiveSize();

    if (!this->smoothSlot.Param<param::BoolParam>()->Value() ||
        !this->clusterOfSmoothedSlot.Param<param::BoolParam>()->Value()) {

        // TODO: calculate clusters here
    }

    if (this->smoothSlot.Param<param::BoolParam>()->Value()) {

        // smooth
        float smoothValue = this->smoothValueSlot.Param<param::FloatParam>()->Value();
        vislib::Array<float> filter(1 + static_cast<unsigned int>(::ceil(smoothValue)), 0.0f);
        filter[0] = 1.0f;
        if (smoothValue > 0.00001f) {
            float exp = this->smoothExpSlot.Param<param::FloatParam>()->Value();
            for (SIZE_T i = 1; i < filter.Count(); i++) {
                filter[i] = ::pow(::cos(static_cast<float>(M_PI) * static_cast<float>(i) / (1.0f + smoothValue)), exp);
                filter[i] *= filter[i];
            }
        }

        off = 0;
        for (SIZE_T p = 0; p < this->pathlines.Count(); p++) {
            for (SIZE_T v = 0; v < this->pathlines[p].length; v++) {
                float fac = 1.0f;
                vislib::math::ShallowVector<float, 3> pos(&this->vertices[off + v].x);
                for (SIZE_T f = 1; f < filter.Count(); f++) {
                    if (f <= v) {
                        pos += vislib::math::ShallowVector<float, 3>(&this->vertices[off + v - f].x) * filter[f];
                        fac += filter[f];
                    }
                    if (f + v < this->pathlines[p].length) {
                        pos += vislib::math::ShallowVector<float, 3>(&this->vertices[off + v + f].x) * filter[f];
                        fac += filter[f];
                    }
                }
                pos /= fac;
            }
            off += this->pathlines[p].length;
        }

        // Note: smoothing does change the positions, but will never leave the
        // original bounding box. Since everything is just an approximation we
        // keep the old bounding box.

        if (this->speedOfSmoothedSlot.Param<param::BoolParam>()->Value()) {
            // recalculate speed
            off = 0;
            this->maxSpeed = -FLT_MAX;
            this->minSpeed = FLT_MAX;
            for (SIZE_T p = 0; p < this->pathlines.Count(); p++) {
                for (SIZE_T v = 1; v < this->pathlines[p].length; v++) {
                    SolPathDataCall::Vertex& v1 = this->vertices[off + v];
                    v1.speed = vislib::math::ShallowPoint<float, 3>(&v1.x).Distance(
                        vislib::math::ShallowPoint<float, 3>(&this->vertices[off + v - 1].x));
                    if (v1.speed > 0.01f) {
                        if (v1.speed > this->maxSpeed)
                            this->maxSpeed = v1.speed;
                        if (v1.speed < this->minSpeed)
                            this->minSpeed = v1.speed;
                    }
                }
                if (this->pathlines[p].length > 2) {
                    this->vertices[off].speed = this->vertices[off + 1].speed;
                }
                off += this->pathlines[p].length;
            }
        }

        if (this->clusterOfSmoothedSlot.Param<param::BoolParam>()->Value()) {

            // TODO: calculate clusters here
        }
    }
}
