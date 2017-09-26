/*
 * VolumeCache.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "VolumeCache.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
//#include "mmcore/moldyn/DataFileSequence.h"
#include "vislib/sys/File.h"
#include "vislib/assert.h"
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * volume::VolumeCache::VolumeCache
 */
volume::VolumeCache::VolumeCache(void) : Module(),
        outDataSlot("outData", "Gets the data from the cache"),
        inDataSlot("inData", "Connects to the real data provider"),
        filenameSlot("filename", "The cache file"),
        useCacheSlot("useCache", "Use the cache or pipe through"),
        forceAndSaveSlot("forceSave", "Update the cache"),
        saveSlot("save", "Saves the cache"),
        dataHash(0), frameIdx(0), attr(), data(), bboxes(), frameCount() {

    this->res[0] = this->res[1] = this->res[2] = 1;

    this->outDataSlot.SetCallback(core::CallVolumeData::ClassName(), core::CallVolumeData::FunctionName(0), &VolumeCache::outDataCallback);
    this->outDataSlot.SetCallback(core::CallVolumeData::ClassName(), core::CallVolumeData::FunctionName(1), &VolumeCache::outExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<core::CallVolumeDataDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->useCacheSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->useCacheSlot);

    this->forceAndSaveSlot << new core::param::ButtonParam();
    this->MakeSlotAvailable(&this->forceAndSaveSlot);

    this->saveSlot << new core::param::ButtonParam();
    this->MakeSlotAvailable(&this->saveSlot);

}


/*
 * volume::VolumeCache::~VolumeCache
 */
volume::VolumeCache::~VolumeCache(void) {
    this->Release();
}



/*
 * volume::VolumeCache::create
 */
bool volume::VolumeCache::create(void) {
    // Intentionally empty
    return true;
}


/*
 * volume::VolumeCache::release
 */
void volume::VolumeCache::release(void) {
    // paranoia
    this->attr.Clear();
    this->data.EnforceSize(0);
}


/*
 * volume::VolumeCache::outDataCallback
 */
bool volume::VolumeCache::outDataCallback(core::Call& caller) {
    core::CallVolumeData *o = dynamic_cast<core::CallVolumeData*>(&caller);
    if (o == NULL) return false;
    core::CallVolumeData *i = this->inDataSlot.CallAs<core::CallVolumeData>();

    if (this->useCacheSlot.Param<core::param::BoolParam>()->Value()) {

        bool save = this->saveSlot.IsDirty();
        this->saveSlot.ResetDirty();

        if ((this->dataHash == 0) || (this->frameIdx != o->FrameID()) || this->forceAndSaveSlot.IsDirty()) {
            // no data
            bool load = vislib::sys::File::Exists(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
            //// TODO: PORRRQUEEEEEEEE
            //
            // TODO: This hack needs to be removed!
            //
            //const core::moldyn::DataFileSequence* test = dynamic_cast<const core::moldyn::DataFileSequence*>(o->PeekCallerSlot()->Parent());
            //if (test == NULL) {
            //    if (load && (this->dataHash != 0) && (this->frameIdx != o->FrameID())) {
            //        load = false; // most likely we want to rebuild the cache
            //    }
            //}
            if (load && this->forceAndSaveSlot.IsDirty() && (i != NULL)) {
                load = false; // no, we want to build and save instead
            }
            this->forceAndSaveSlot.ResetDirty();

            if (!load || !this->loadCache()) {
                // build cache!
                *i = *o;
                this->buildCache(i);
                save = true;
            }

        }
        // Do not check datahash of i-module, because this would trigger all calculations we want to avoid!
        if (this->dataHash == 0) return false; // something failed
        if (save) {
            this->saveCache();
        }

        // TODO: why is this different from the code for getextents, which sets only half of these!?!?
        o->SetDataHash(this->dataHash);
        o->AccessBoundingBoxes() = this->bboxes;
        o->SetFrameCount(this->frameCount);
        o->SetFrameID(this->frameIdx);
        o->SetSize(this->res[0], this->res[1], this->res[2]);
        o->SetAttributeCount(static_cast<unsigned int>(this->attr.Count()));
        for (unsigned int a = 0; a < static_cast<unsigned int>(this->attr.Count()); a++) {
            o->Attribute(a) = this->attr[a];
        }
        o->SetUnlocker(NULL);

        return true;

    } else {
        // byepass cache completely
        if (i == NULL) return false;

        *i = *o;
        if ((*i)(0)) {
            *o = *i;
            return true;
        }
        return false;
    }

    return false; // unreachable code
}


/*
 * volume::VolumeCache::outExtentCallback
 */
bool volume::VolumeCache::outExtentCallback(core::Call& caller) {
    core::CallVolumeData *o = dynamic_cast<core::CallVolumeData*>(&caller);
    if (o == NULL) return false;
    core::CallVolumeData *i = this->inDataSlot.CallAs<core::CallVolumeData>();

    if (this->useCacheSlot.Param<core::param::BoolParam>()->Value()) {

        bool save = this->saveSlot.IsDirty();
        this->saveSlot.ResetDirty();

        if ((this->dataHash == 0) || (this->frameIdx != o->FrameID()) || this->forceAndSaveSlot.IsDirty()) {
            // no data
            bool load = vislib::sys::File::Exists(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
            //// TODO: PORRRQUEEEEEEEE
            //
            // TODO: This hack needs to be removed!
            //
            //const core::moldyn::DataFileSequence* test = dynamic_cast<const core::moldyn::DataFileSequence*>(o->PeekCallerSlot()->Parent());
            //if (test == NULL) {
            //    if (load && (this->dataHash != 0) && (this->frameIdx != o->FrameID())) {
            //        load = false; // most likely we want to rebuild the cache
            //    }
            //}
            if (load && this->forceAndSaveSlot.IsDirty() && (i != NULL)) {
                load = false; // no, we want to build and save instead
            }
            this->forceAndSaveSlot.ResetDirty();

            if (!load || !this->loadCache()) {
                // build cache!
                *i = *o;
                this->buildCache(i);
                save = true;
            }

        }
        // Do not check datahash of i-module, because this would trigger all calculations we want to avoid!
        if (this->dataHash == 0) return false; // something failed
        if (save) {
            this->saveCache();
        }

        o->SetDataHash(this->dataHash);
        o->AccessBoundingBoxes() = this->bboxes;
        o->SetFrameCount(this->frameCount);
        o->SetUnlocker(NULL);

        return true;

    } else {
        // byepass cache completely
        if (i == NULL) return false;

        *i = *o;
        if ((*i)(1)) {
            *o = *i;
            return true;
        }
        return false;
    }

    return false; // unreachable code
}


/*
 * volume::VolumeCache::loadCache
 */
bool volume::VolumeCache::loadCache(void) {
    vislib::sys::File file;
    if (!file.Open(this->filenameSlot.Param<core::param::FilePathParam>()->Value(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        return false;
    }

    UINT64 ui64;
    unsigned int magicNumber = 10041940;
    unsigned int ui;
    unsigned char b;
    float f[6];

    if (file.Read(&ui, 4) != 4) return false;
    if (ui != magicNumber) return false;
    if (file.Read(&this->frameIdx, 4) != 4) return false;
    if (file.Read(&this->frameCount, 4) != 4) return false;
    this->bboxes.Clear();
    if (file.Read(&b, 1) != 1) return false;
    if (file.Read(&f, 6 * sizeof(float)) != 6 * sizeof(float)) return false;
    if (b != 0) this->bboxes.SetObjectSpaceBBox(f[0], f[1], f[2], f[3], f[4], f[5]); // minor hazard
    if (file.Read(&b, 1) != 1) return false;
    if (file.Read(&f, 6 * sizeof(float)) != 6 * sizeof(float)) return false;
    if (b != 0) this->bboxes.SetObjectSpaceClipBox(f[0], f[1], f[2], f[3], f[4], f[5]); // minor hazard
    if (file.Read(&b, 1) != 1) return false;
    if (file.Read(&f, 6 * sizeof(float)) != 6 * sizeof(float)) return false;
    if (b != 0) this->bboxes.SetWorldSpaceBBox(f[0], f[1], f[2], f[3], f[4], f[5]); // minor hazard
    if (file.Read(&b, 1) != 1) return false;
    if (file.Read(&f, 6 * sizeof(float)) != 6 * sizeof(float)) return false;
    if (b != 0) this->bboxes.SetWorldSpaceClipBox(f[0], f[1], f[2], f[3], f[4], f[5]); // minor hazard
    if (file.Read(&this->res, 3 * 4) != 3 * 4) return false;
    if (file.Read(&ui, 4) != 4) return false;
    this->attr.SetCount(ui);
    for (unsigned int a = 0; a < static_cast<unsigned int>(this->attr.Count()); a++) {
        if (file.Read(&ui, 4) != 4) return false;
        if (file.Read(const_cast<vislib::StringA&>(this->attr[a].Name()).AllocateBuffer(ui + 1), ui) != ui) return false;
        const_cast<char*>(this->attr[a].Name().PeekBuffer())[ui] = 0;
        if (file.Read(&b, 1) != 1) return false;
        this->attr[a].SetType(static_cast<core::CallVolumeData::DataType>(b));
        if (file.Read(&ui64, 8) != 8) return false;
        this->attr[a].SetData(reinterpret_cast<const void*>(ui64));
    }
    if (file.Read(&ui64, 8) != 8) return false;
    this->data.EnforceSize(static_cast<SIZE_T>(ui64));
    if (file.Read(this->data, static_cast<vislib::sys::File::FileSize>(ui64))
            != static_cast<vislib::sys::File::FileSize>(ui64)) return false;
    if (file.Read(&ui, 4) != 4) return false;
    if (ui != magicNumber) return false;
    for (unsigned int a = 0; a < static_cast<unsigned int>(this->attr.Count()); a++) {
        this->attr[a].SetData(this->data.As<unsigned char>() + reinterpret_cast<SIZE_T>(this->attr[a].Bytes()));
    }

    file.Close();
    this->dataHash++;

    return true;
}


/*
 * volume::VolumeCache::saveCache
 */
void volume::VolumeCache::saveCache(void) {
    vislib::sys::File file;
    if (!file.Open(this->filenameSlot.Param<core::param::FilePathParam>()->Value(),
            vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to write volume cache file");
        return;
    }

    UINT64 ui64;
    unsigned int magicNumber = 10041940;
    file.Write(&magicNumber, 4);
    file.Write(&frameIdx, 4);
    file.Write(&frameCount, 4);
    unsigned char b = this->bboxes.IsObjectSpaceBBoxValid() ? 1 : 0;
    file.Write(&b, 1);
    file.Write(this->bboxes.ObjectSpaceBBox().PeekBounds(), 6 * sizeof(float));
    b = this->bboxes.IsObjectSpaceClipBoxValid() ? 1 : 0;
    file.Write(&b, 1);
    file.Write(this->bboxes.ObjectSpaceClipBox().PeekBounds(), 6 * sizeof(float));
    b = this->bboxes.IsWorldSpaceBBoxValid() ? 1 : 0;
    file.Write(&b, 1);
    file.Write(this->bboxes.WorldSpaceBBox().PeekBounds(), 6 * sizeof(float));
    b = this->bboxes.IsWorldSpaceClipBoxValid() ? 1 : 0;
    file.Write(&b, 1);
    file.Write(this->bboxes.WorldSpaceClipBox().PeekBounds(), 6 * sizeof(float));
    file.Write(this->res, 3 * 4);
    unsigned int ui = static_cast<unsigned int>(this->attr.Count());
    file.Write(&ui, 4);
    for (unsigned int a = 0; a < static_cast<unsigned int>(this->attr.Count()); a++) {
        ui = this->attr[a].Name().Length();
        file.Write(&ui, 4);
        file.Write(this->attr[a].Name().PeekBuffer(), ui);
        b = static_cast<unsigned char>(this->attr[a].Type());
        file.Write(&b, 1);
        ui64 = this->attr[a].Bytes() - this->data.As<unsigned char>();
        file.Write(&ui64, 8);
    }
    ui64 = static_cast<UINT64>(this->data.GetSize());
    file.Write(&ui64, 8);
    file.Write(this->data, static_cast<vislib::sys::File::FileSize>(ui64));
    file.Write(&magicNumber, 4);

    file.Close();
}


/*
 * volume::VolumeCache::buildCache
 */
void volume::VolumeCache::buildCache(core::CallVolumeData* inDat) {
    ASSERT(inDat != NULL);

    if (!(*inDat)(1)) {
        this->dataHash = 0; // failed
        return;
    }

    this->bboxes = inDat->AccessBoundingBoxes();
    this->frameCount = inDat->FrameCount();

    if (!(*inDat)(0)) {
        this->dataHash = 0; // failed
        return;
    }

    this->frameIdx = inDat->FrameID();
    this->res[0] = inDat->XSize();
    this->res[1] = inDat->YSize();
    this->res[2] = inDat->ZSize();
    this->attr.SetCount(inDat->AttributeCount());
    SIZE_T voxelCnt = this->res[0] * this->res[1] * this->res[2];
    SIZE_T dataSize = 0;
    for (unsigned int a = 0; a < static_cast<unsigned int>(this->attr.Count()); a++) {
        this->attr[a] = inDat->Attribute(a);
        SIZE_T bpv = 0;
        switch (this->attr[a].Type()) {
        case core::CallVolumeData::TYPE_BYTE: bpv = 1; break;
        case core::CallVolumeData::TYPE_FLOAT: bpv = 4; break;
        case core::CallVolumeData::TYPE_DOUBLE: bpv = 8; break;
        default: ASSERT(false);
        }
        dataSize += voxelCnt * bpv;
    }
    this->data.EnforceSize(dataSize);
    dataSize = 0;
    for (unsigned int a = 0; a < static_cast<unsigned int>(this->attr.Count()); a++) {
        SIZE_T bpv = 0;
        switch (inDat->Attribute(a).Type()) {
        case core::CallVolumeData::TYPE_BYTE: bpv = 1; break;
        case core::CallVolumeData::TYPE_FLOAT: bpv = 4; break;
        case core::CallVolumeData::TYPE_DOUBLE: bpv = 8; break;
        default: ASSERT(false);
        }
        bpv *= voxelCnt;
        this->attr[a].SetData(this->data.At(dataSize));
        ASSERT(inDat->Attribute(a).RawData());
        ::memcpy(this->data.At(dataSize), inDat->Attribute(a).RawData(), bpv);
        dataSize += bpv;
    }

    this->dataHash++;
}
