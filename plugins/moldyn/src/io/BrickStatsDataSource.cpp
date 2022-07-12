/*
 * MMPLDDataSource.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "BrickStatsDataSource.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "moldyn/BrickStatsCall.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/sysfunctions.h"

using namespace megamol::core;
using namespace megamol::moldyn;

/*
 * moldyn::BrickStatsDataSource::BrickStatsDataSource
 */
BrickStatsDataSource::BrickStatsDataSource(void)
        : Module()
        , filename("filename", "The path to the stat file to load.")
        , getData("getdata", "Slot to request data from this data source.")
        , skipHeaderLine("skipheader", "Slot to switch skipping of header line in file.")
        , file(NULL)
        , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , data_hash(0)
        , info() {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&BrickStatsDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->skipHeaderLine << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->skipHeaderLine);

    this->getData.SetCallback(
        BrickStatsCall::ClassName(), BrickStatsCall::FunctionName(0), &BrickStatsDataSource::getDataCallback);
    this->getData.SetCallback(
        BrickStatsCall::ClassName(), BrickStatsCall::FunctionName(1), &BrickStatsDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->info.SetCapacityIncrement(10000);
}


/*
 * moldyn::MMPLDDataSource::~MMPLDDataSource
 */
BrickStatsDataSource::~BrickStatsDataSource(void) {
    this->Release();
}


/*
 * moldyn::MMPLDDataSource::create
 */
bool BrickStatsDataSource::create(void) {
    using megamol::core::utility::log::Log;
    if (BrickStatsCall::GetTypeSize() != sizeof(BrickStatsCall::BrickStatsType)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "BrickStatsCall::BrickStatsType has different size across object files!");
        return false;
    }
    return true;
}


/*
 * moldyn::MMPLDDataSource::release
 */
void BrickStatsDataSource::release(void) {
    if (this->file != NULL) {
        vislib::sys::File* f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
    this->info.Clear();
}


/*
 * moldyn::MMPLDDataSource::filenameChanged
 */
bool BrickStatsDataSource::filenameChanged(param::ParamSlot& slot) {
    using megamol::core::utility::log::Log;
    using vislib::sys::File;
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->clipbox = this->bbox;
    this->data_hash++;

    this->info.Clear();

    if (this->file == NULL) {
        this->file = new vislib::sys::FastFile();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<param::FilePathParam>()->Value().native().c_str(), File::READ_ONLY,
            File::SHARE_READ, File::OPEN_ONLY)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to open MMPLD-File \"%s\".",
            this->filename.Param<param::FilePathParam>()->Value().generic_u8string().c_str());

        SAFE_DELETE(this->file);

        return true;
    }

    UINT64 lineNum = -1;
    if (this->skipHeaderLine.Param<param::BoolParam>()->Value()) {
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file);
        lineNum++;
    }

    while (!this->file->IsEOF()) {
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file);
        lineNum++;

        auto arr = vislib::StringTokeniserA::Split(line, ',');
        if (arr.Count() != 14) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to parse line %lu: not enough fields (%u)", lineNum, arr.Count());
            continue;
        }
        try {
            UINT64 offset = vislib::CharTraitsA::ParseUInt64(arr[0]);
            UINT64 len = vislib::CharTraitsA::ParseUInt64(arr[1]);
            BrickStatsCall::BrickStatsType minX =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[2]));
            BrickStatsCall::BrickStatsType maxX =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[3]));
            BrickStatsCall::BrickStatsType meanX =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[4]));
            BrickStatsCall::BrickStatsType stddevX =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[5]));
            BrickStatsCall::BrickStatsType minY =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[6]));
            BrickStatsCall::BrickStatsType maxY =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[7]));
            BrickStatsCall::BrickStatsType meanY =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[8]));
            BrickStatsCall::BrickStatsType stddevY =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[9]));
            BrickStatsCall::BrickStatsType minZ =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[10]));
            BrickStatsCall::BrickStatsType maxZ =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[11]));
            BrickStatsCall::BrickStatsType meanZ =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[12]));
            BrickStatsCall::BrickStatsType stddevZ =
                static_cast<BrickStatsCall::BrickStatsType>(vislib::CharTraitsA::ParseDouble(arr[13]));

            this->info.Add(BrickStatsCall::BrickInfo(
                offset, len, minX, minY, minZ, maxX, maxY, maxZ, meanX, meanY, meanZ, stddevX, stddevY, stddevZ));
            //this->bbox.GrowToPoint(minX - stddevX, minY - stddevY, minZ - stddevZ);
            //this->bbox.GrowToPoint(maxX + stddevX, maxY + stddevY, maxZ + stddevZ);
            this->bbox.GrowToPoint(minX, minY, minZ);
            this->bbox.GrowToPoint(maxX, maxY, maxZ);
        } catch (vislib::FormatException) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("parse error in line %lu", lineNum);
        }
    }

    this->clipbox = this->bbox;

    /*

#define _ERROR_OUT(MSG) Log::DefaultLog.WriteError( MSG); \
        SAFE_DELETE(this->file); \
        this->setFrameCount(1); \
        this->initFrameCache(1); \
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f); \
        this->clipbox = this->bbox; \
        return true;
        #define _ASSERT_READFILE(BUFFER, BUFFERSIZE) if (this->file->Read((BUFFER), (BUFFERSIZE)) != (BUFFERSIZE)) { \
        _ERROR_OUT("Unable to read MMPLD file header"); \
        }

        char magicid[6];
        _ASSERT_READFILE(magicid, 6);
        if (::memcmp(magicid, "MMPLD", 6) != 0) {
        _ERROR_OUT("MMPLD file header id wrong");
        }
        unsigned short ver;
        _ASSERT_READFILE(&ver, 2);
        if (ver != 100 && ver != 101 && ver != 102) {
        _ERROR_OUT("MMPLD file header version wrong");
        }
        this->fileVersion = ver;

        UINT32 frmCnt = 0;
        _ASSERT_READFILE(&frmCnt, 4);
        if (frmCnt == 0) {
        _ERROR_OUT("MMPLD file does not contain any frame information");
        }

        float box[6];
        _ASSERT_READFILE(box, 4 * 6);
        this->bbox.Set(box[0], box[1], box[2], box[3], box[4], box[5]);
        _ASSERT_READFILE(box, 4 * 6);
        this->clipbox.Set(box[0], box[1], box[2], box[3], box[4], box[5]);

        delete[] this->frameIdx;
        this->frameIdx = new UINT64[frmCnt + 1];
        _ASSERT_READFILE(this->frameIdx, 8 * (frmCnt + 1));
        double size = 0.0;
        for (UINT32 i = 0; i < frmCnt; i++) {
        size += static_cast<double>(this->frameIdx[i + 1] - this->frameIdx[i]);
        }
        size /= static_cast<double>(frmCnt);
        size *= CACHE_FRAME_FACTOR;

        UINT64 mem = vislib::sys::SystemInformation::AvailableMemorySize();
        if (this->limitMemorySlot.Param<param::BoolParam>()->Value()) {
        mem = vislib::math::Min(mem,
        (UINT64)(this->limitMemorySizeSlot.Param<param::IntParam>()->Value())
        * (UINT64)(1024u * 1024u));
        }
        unsigned int cacheSize = static_cast<unsigned int>(mem / size);

        if (cacheSize > CACHE_SIZE_MAX) {
        cacheSize = CACHE_SIZE_MAX;
        }
        if (cacheSize < CACHE_SIZE_MIN) {
        vislib::StringA msg;
        msg.Format("Frame cache size forced to %i. Calculated size was %u.\n",
        CACHE_SIZE_MIN, cacheSize);
        megamol::core::utility::log::Log::DefaultLog.WriteWarn( msg);
        cacheSize = CACHE_SIZE_MIN;
        } else {
        vislib::StringA msg;
        msg.Format("Frame cache size set to %i.\n", cacheSize);
        megamol::core::utility::log::Log::DefaultLog.WriteInfo( msg);
        }

        this->setFrameCount(frmCnt);
        this->initFrameCache(cacheSize);

        #undef _ASSERT_READFILE
        #undef _ERROR_OUT
        */
    return true;
}


/*
 * moldyn::MMPLDDataSource::getDataCallback
 */
bool BrickStatsDataSource::getDataCallback(Call& caller) {
    BrickStatsCall* c2 = dynamic_cast<BrickStatsCall*>(&caller);
    if (c2 == NULL)
        return false;

    c2->SetDataHash(this->data_hash);
    c2->SetBricks(&this->info);
    //Frame *f = NULL;
    //if (c2 != NULL) {
    //    f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID(), c2->IsFrameForced()));
    //    if (f == NULL) return false;
    //    c2->SetUnlocker(new Unlocker(*f));
    //    c2->SetFrameID(f->FrameNumber());
    //    c2->SetDataHash(this->data_hash);
    //    f->SetData(*c2);
    //}

    return true;
}


/*
 * moldyn::MMPLDDataSource::getExtentCallback
 */
bool BrickStatsDataSource::getExtentCallback(Call& caller) {
    BrickStatsCall* c2 = dynamic_cast<BrickStatsCall*>(&caller);
    if (c2 == NULL)
        return false;

    c2->SetFrameCount(1);
    c2->AccessBoundingBoxes().Clear();
    c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
    c2->SetDataHash(this->data_hash);

    //if (c2 != NULL) {
    //    c2->SetFrameCount(this->FrameCount());
    //    c2->AccessBoundingBoxes().Clear();
    //    c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    //    c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
    //    c2->SetDataHash(this->data_hash);
    //    return true;
    //}

    return true;
}
