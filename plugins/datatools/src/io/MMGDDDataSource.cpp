/*
 * MMGDDWriter.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "io/MMGDDDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/SystemInformation.h"
#include <cassert>

using namespace megamol;
using namespace megamol::datatools;


io::MMGDDDataSource::MMGDDDataSource(void)
        : core::view::AnimDataModule()
        , filename("filename", "The path to the MMPLD file to load.")
        , getData("getdata", "Slot to request data from this data source.")
        , file(nullptr)
        , frameIdx()
        , data_hash(0) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&MMGDDDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback("GraphDataCall", "GetData", &MMGDDDataSource::getDataCallback);
    this->getData.SetCallback("GraphDataCall", "GetExtent", &MMGDDDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}

io::MMGDDDataSource::~MMGDDDataSource(void) {
    Release();
}

core::view::AnimDataModule::Frame* io::MMGDDDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<io::MMGDDDataSource*>(this));
    return f;
}

bool io::MMGDDDataSource::create(void) {
    // intentionally empty
    return true;
}

void io::MMGDDDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
    using megamol::core::utility::log::Log;
    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == nullptr)
        return;
    if (this->file == nullptr) {
        f->Clear();
        return;
    }

    assert(idx < FrameCount());
    file->Seek(frameIdx[idx]);
    if (!f->LoadFrame(file, idx, frameIdx[idx + 1] - frameIdx[idx])) {
        // failed
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from MMGDD file\n", idx);
    }
}

void io::MMGDDDataSource::release(void) {
    this->resetFrameCache();
    if (file != nullptr) {
        vislib::sys::File* f = file;
        file = nullptr;
        f->Close();
        delete f;
    }
    frameIdx.clear();
}

bool io::MMGDDDataSource::filenameChanged(core::param::ParamSlot& slot) {
    using megamol::core::utility::log::Log;
    using vislib::sys::File;
    this->resetFrameCache();
    this->data_hash++;

    if (file == nullptr) {
        file = new vislib::sys::FastFile();
    } else {
        file->Close();
    }
    assert(filename.Param<core::param::FilePathParam>() != nullptr);
    if (!file->Open(filename.Param<core::param::FilePathParam>()->Value().native().c_str(), File::READ_ONLY,
            File::SHARE_READ, File::OPEN_ONLY)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open MMGDD-File \"%s\".",
            filename.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        SAFE_DELETE(file);
        this->setFrameCount(1);
        this->initFrameCache(1);
        return true;
    }

#define _ERROR_OUT(MSG)                              \
    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, MSG); \
    SAFE_DELETE(this->file);                         \
    this->setFrameCount(1);                          \
    this->initFrameCache(1);                         \
    return true;
#define _ASSERT_READFILE(BUFFER, BUFFERSIZE)                        \
    if (this->file->Read((BUFFER), (BUFFERSIZE)) != (BUFFERSIZE)) { \
        _ERROR_OUT("Unable to read MMPLD file header");             \
    }

    char magicid[6];
    _ASSERT_READFILE(magicid, 6);
    if (::memcmp(magicid, "MMGDD", 6) != 0) {
        _ERROR_OUT("MMGDD file header id wrong");
    }
    unsigned short ver;
    _ASSERT_READFILE(&ver, 2);
    if (ver != 100) {
        _ERROR_OUT("MMGDD file header version wrong");
    }

    uint32_t frmCnt = 0;
    _ASSERT_READFILE(&frmCnt, 4);
    if (frmCnt == 0) {
        _ERROR_OUT("MMGDD file does not contain any frame information");
    }

    frameIdx.resize(frmCnt + 1);
    _ASSERT_READFILE(frameIdx.data(), frameIdx.size() * 8);

    double memHere = static_cast<double>(vislib::sys::SystemInformation::AvailableMemorySize());
    memHere *= 0.25; // only use max 25% of the memory of this data
    Log::DefaultLog.WriteInfo("Memory available: %u MB\n", static_cast<uint32_t>(memHere / (1024.0 * 1024.0)));
    double memWant = static_cast<double>(frameIdx.back() - frameIdx.front());
    Log::DefaultLog.WriteInfo("Memory required: %u MB for %u frames total\n",
        static_cast<uint32_t>(memWant / (1024.0 * 1024.0)), static_cast<uint32_t>(frameIdx.size()));
    uint32_t cacheSize = static_cast<uint32_t>((memHere / memWant) * static_cast<double>(frameIdx.size()) + 0.5);
    Log::DefaultLog.WriteInfo("Cache set to %u frames\n", cacheSize);

    this->setFrameCount(frmCnt);
    this->initFrameCache(cacheSize);

#undef _ASSERT_READFILE
#undef _ERROR_OUT

    return true;
}

bool io::MMGDDDataSource::getDataCallback(core::Call& caller) {
    GraphDataCall* c2 = dynamic_cast<GraphDataCall*>(&caller);
    if (c2 == nullptr)
        return false;

    Frame* f = nullptr;
    if (c2 != nullptr) {
        f = dynamic_cast<Frame*>(this->requestLockedFrame(c2->FrameID(), true));
        if (f == nullptr)
            return false;
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());
        c2->SetDataHash(this->data_hash);
        f->SetData(*c2);
    }

    return true;
}

bool io::MMGDDDataSource::getExtentCallback(core::Call& caller) {
    GraphDataCall* c2 = dynamic_cast<GraphDataCall*>(&caller);

    if (c2 != nullptr) {
        c2->SetFrameCount(this->FrameCount());
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return false;
}
