/*
 * MMGDDWriter.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "io/MMGDDWriter.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"

using namespace megamol;
using namespace megamol::datatools;


io::MMGDDWriter::MMGDDWriter(void)
        : AbstractDataWriter()
        , filenameSlot("filename", "The path to the MMGDD file to be written")
        , dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot.SetParameter(new core::param::FilePathParam(
        "", megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"mmgdd"}));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->dataSlot.SetCompatibleCall<GraphDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}

io::MMGDDWriter::~MMGDDWriter(void) {
    Release();
}

bool io::MMGDDWriter::create(void) {
    // intentionally empty
    return true;
}

void io::MMGDDWriter::release(void) {
    // intentionally empty
}

bool io::MMGDDWriter::run(void) {
    using megamol::core::utility::log::Log;
    vislib::TString filename(
        this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteError("No file name specified. Abort.");
        return false;
    }

    GraphDataCall* gdc = this->dataSlot.CallAs<GraphDataCall>();
    if (gdc == nullptr) {
        Log::DefaultLog.WriteError("No data source connected. Abort.");
        return false;
    }

    if (vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteWarn(
            "File %s already exists and will be overwritten.", vislib::StringA(filename).PeekBuffer());
    }

    gdc->SetFrameID(0);
    if (!(*gdc)(GraphDataCall::GET_EXTENT))
        return false;
    uint32_t frameCnt = gdc->FrameCount();
    gdc->Unlock();

    vislib::sys::FastFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
            vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteError(
            "Unable to create output file \"%s\". Abort.", vislib::StringA(filename).PeekBuffer());
        gdc->Unlock();
        return false;
    }

    Log::DefaultLog.WriteInfo("MMGDD Start Writing ...\n");

    file.Write("MMGDD", 6);
    uint16_t version = 100;
    file.Write(&version, 2);
    file.Write(&frameCnt, 4);

    uint64_t frameOffset = 0;
    for (uint32_t i = 0; i <= frameCnt; i++) {
        file.Write(&frameCnt, 8);
    }

    for (uint32_t i = 0; i < frameCnt; i++) {
        frameOffset = static_cast<UINT64>(file.Tell());
        file.Seek(12 + i * 8);
        file.Write(&frameOffset, 8);
        file.Seek(frameOffset);

        gdc->SetFrameID(i);
        unsigned char flags = 0;
        if (!(*gdc)(GraphDataCall::GET_DATA)) {
            Log::DefaultLog.WriteError("Unable to fetch data for frame %u. skipping.", i);
            file.Write(&flags, 1);
            continue;
        }
        if (gdc->FrameID() != i) {
            Log::DefaultLog.WriteWarn("Wrong frame data answered: %u requested, %u received", i, gdc->FrameID());
        }

#define __WITH_VALUE_SUMMARY 0

#if __WITH_VALUE_SUMMARY
        unsigned int minVal = 1000000, maxVal = 0;
        for (const auto& e : *gdc) {
            if (minVal > e.i1)
                minVal = e.i1;
            if (maxVal < e.i1)
                maxVal = e.i1;
            if (minVal > e.i2)
                minVal = e.i2;
            if (maxVal < e.i2)
                maxVal = e.i2;
        }
#endif
        Log::DefaultLog.WriteInfo("MMGDD frame #%u: %u edges"
#if __WITH_VALUE_SUMMARY
                                  ", indices [%u %u]"
#endif
                                  "\n",
            i, static_cast<unsigned int>(gdc->GetEdgeCount())
#if __WITH_VALUE_SUMMARY
                   ,
            minVal, maxVal
#endif
        );

        flags = gdc->IsDirected() ? 1 : 0;
        file.Write(&flags, 1);

        file.Write(gdc->GetEdgeData(), gdc->GetEdgeCount() * sizeof(GraphDataCall::edge));

        gdc->Unlock();
    }

    frameOffset = static_cast<UINT64>(file.Tell());
    file.Seek(12 + frameCnt * 8);
    file.Write(&frameOffset, 8);
    file.Seek(frameOffset);

    version = 0;
    file.Write(&version, 1);

    file.Close();

    Log::DefaultLog.WriteInfo("MMGDD Writing Completed...\n");

    return true;
}

bool io::MMGDDWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}
