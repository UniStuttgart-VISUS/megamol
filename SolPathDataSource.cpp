/*
 * SolPathRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "SolPathDataSource.h"
#include "param/StringParam.h"
#include "vislib/Log.h"
#include "vislib/MemmappedFile.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include <climits>
#include <cfloat>

using namespace megamol::core;
using namespace megamol::protein;


/*
 * SolPathDataSource::SolPathDataSource
 */
SolPathDataSource::SolPathDataSource(void) : core::Module(),
        getdataslot("getdata", "Publishes the data for other modules"),
        filenameslot("filename", "The path of the solpath file to load") {

    this->getdataslot.SetCallback(SolPathDataCall::ClassName(), "GetData", &SolPathDataSource::getData);
    this->getdataslot.SetCallback(SolPathDataCall::ClassName(), "GetExtent", &SolPathDataSource::getExtent);
    this->MakeSlotAvailable(&this->getdataslot);

    this->filenameslot << new param::StringParam("");
    this->MakeSlotAvailable(&this->filenameslot);

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
    if (this->filenameslot.IsDirty()) {
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
bool SolPathDataSource::getData(megamol::core::Call &call) {
    SolPathDataCall *spdc = dynamic_cast<SolPathDataCall *>(&call);
    if (spdc == NULL) return false;

    if (this->filenameslot.IsDirty()) {
        this->loadData();
    }

    spdc->Set(this->pathlines.Count(), this->pathlines.PeekElements(),
        this->minTime, this->maxTime, this->minSpeed, this->maxSpeed);

    return true;
}


/*
 * SolPathDataSource::getExtent
 */
bool SolPathDataSource::getExtent(megamol::core::Call &call) {
    SolPathDataCall *spdc = dynamic_cast<SolPathDataCall *>(&call);
    if (spdc == NULL) return false;

    if (this->filenameslot.IsDirty()) {
        this->loadData();
    }

    megamol::core::BoundingBoxes &bboxs = spdc->AccessBoundingBoxes();
    bboxs.SetObjectSpaceBBox(this->bbox);
    bboxs.SetObjectSpaceClipBox(bboxs.ObjectSpaceBBox());
    bboxs.MakeScaledWorld(1.0f); // at least this is what i think

    return false;
}


/*
 * SolPathDataSource::clear
 */
void SolPathDataSource::clear(void) {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->minTime = 0;
    this->maxTime = 0;
    this->minSpeed = 0.0f;
    this->maxSpeed = 0.0f;
    this->vertices.Clear();
    this->pathlines.Clear();
}


/*
 * SolPathDataSource::loadData
 */
void SolPathDataSource::loadData(void) {
    using vislib::sys::File;
    using vislib::sys::Log;
    vislib::sys::MemmappedFile file;
    this->filenameslot.ResetDirty();

    this->clear();

    if (file.Open(this->filenameslot.Param<param::StringParam>()->Value(),
            File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY) == false) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to open data file %s", vislib::StringA(
            this->filenameslot.Param<param::StringParam>()->Value()).PeekBuffer());
        return;
    }

    vislib::StringA headerID;
    if (file.Read(headerID.AllocateBuffer(7), 7) != 7) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to read data file %s", vislib::StringA(
            this->filenameslot.Param<param::StringParam>()->Value()).PeekBuffer());
        return;
    }

    vislib::SingleLinkedList<fileBlockInfo> fileStruct;
    fileBlockInfo *blockInfo = NULL;

    if (headerID.Equals("SolPath")) {
        unsigned int version;
        if (file.Read(&version, 4) != 4) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to read data file %s", vislib::StringA(
                this->filenameslot.Param<param::StringParam>()->Value()).PeekBuffer());
            return;
        }
        if (version > 1) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Data file %s uses unsupported version %u", vislib::StringA(
                this->filenameslot.Param<param::StringParam>()->Value()).PeekBuffer(),
                version);
            return;
        }

        fileBlockInfo info;
        while (!file.IsEOF()) {
            if (file.Read(&info.id, 4) != 4) break;
            if (file.Read(&info.size, 8) != 8) break;
            info.start = file.Tell();
            file.Seek(info.size, vislib::sys::File::CURRENT);
            fileStruct.Add(info);
        }

    } else {
        // legacy file support
        fileBlockInfo info;
        info.id = 0; // <= pathline data
        info.start = 0; // <= start of file
        info.size = file.GetSize(); // <= whole file
        fileStruct.Add(info);
    }

    // search for pathline data block
    vislib::SingleLinkedList<fileBlockInfo>::Iterator iter
        = fileStruct.GetIterator();
    while (iter.HasNext()) {
        fileBlockInfo &info = iter.Next();
        if (info.id == 0) {
            blockInfo = &info;
            break;
        }
    }
    if (blockInfo == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "File %s does not contain a path line data block", vislib::StringA(
            this->filenameslot.Param<param::StringParam>()->Value()).PeekBuffer());
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
        vertex.time = UINT_MAX - 2;
        path.length = 0;
        path.data = NULL; // will be set later

        file.Read(&tmp, 4);
        path.id = tmp;
        file.Read(&framesCount, 4);

        this->vertices.AssertCapacity(this->vertices.Count() + framesCount);

        for (SIZE_T e = 0; e < framesCount; e++) {
            file.Read(&tmp, 4);
            file.Read(&vertex.x, 12);

            if (tmp != vertex.time + 1) {
                // start a new path
                if (path.length > 0) {
                    this->pathlines.Add(path);
                }
                path.length = 1;
            } else {
                // continue path
                path.length++;
            }
            vertex.time = tmp;

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
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Path data inconsistent: too many vertices");
    } else if (off > this->vertices.Count()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Path data inconsistent: too few vertices");
        this->clear();
        return;
    }

    if (off <= 0) {
        // data is empty!
        return;
    }

    off = 0;
    this->maxTime = 0;
    this->minTime = UINT_MAX;
    this->maxSpeed = -FLT_MAX;
    this->minSpeed = FLT_MAX;
    this->bbox.Set(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (SIZE_T p = 0; p < this->pathlines.Count(); p++) {
        this->pathlines[p].data = this->vertices.PeekElements() + off;

        for (SIZE_T v = 0; v < this->pathlines[p].length; v++) {
            SolPathDataCall::Vertex &v1 = this->vertices[off + v];

            v1.speed = 0.0f; // TODO: Speed calculus

            if (v1.time > this->maxTime) this->maxTime = v1.time;
            if (v1.time < this->minTime) this->minTime = v1.time;
            if (v1.speed > this->maxSpeed) this->maxSpeed = v1.speed;
            if (v1.speed < this->minSpeed) this->minSpeed = v1.speed;
            if (this->bbox.Left() > v1.x) this->bbox.SetLeft(v1.x);
            if (this->bbox.Right() < v1.x) this->bbox.SetRight(v1.x);
            if (this->bbox.Bottom() > v1.y) this->bbox.SetBottom(v1.y);
            if (this->bbox.Top() < v1.y) this->bbox.SetTop(v1.y);
            if (this->bbox.Back() > v1.z) this->bbox.SetBack(v1.z);
            if (this->bbox.Front() < v1.z) this->bbox.SetFront(v1.z);
        }

        off += this->pathlines[p].length;
    }

    this->bbox.EnforcePositiveSize();

}
