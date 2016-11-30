/*
 * MMGDDWriter.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "io/MMGDDWriter.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/String.h"

using namespace megamol;
using namespace megamol::stdplugin::datatools;


io::MMGDDWriter::MMGDDWriter(void) : AbstractDataWriter(),
        filenameSlot("filename", "The path to the MMGDD file to be written"),
        dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
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
    using vislib::sys::Log;
    vislib::TString filename(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No file name specified. Abort.");
        return false;
    }

    GraphDataCall *gdc = this->dataSlot.CallAs<GraphDataCall>();
    if (gdc == nullptr) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No data source connected. Abort.");
        return false;
    }

    if (vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "File %s already exists and will be overwritten.", vislib::StringA(filename).PeekBuffer());
    }

    gdc->SetFrameID(0);
    if (!(*gdc)(GraphDataCall::GET_EXTENT)) return false;
    uint32_t frameCnt = gdc->FrameCount();
    gdc->Unlock();

    vislib::sys::FastFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create output file \"%s\". Abort.", vislib::StringA(filename).PeekBuffer());
        gdc->Unlock();
        return false;
    }

    file.Write("MMGDD", 6);
    uint16_t version = 100;
    file.Write(&version, 2);
    file.Write(&frameCnt, 4);

    uint64_t frameOffset = 0;
    for (uint32_t i = 0; i <= frameCnt; i++) {
        file.Write(&frameCnt, 8);
    }

    for (uint32_t i = 0; i <= frameCnt; i++) {
        frameOffset = static_cast<UINT64>(file.Tell());
        file.Seek(12 + i * 8);
        file.Write(&frameOffset, 8);
        file.Seek(frameOffset);

        gdc->SetFrameID(i);
        if (!(*gdc)(GraphDataCall::GET_DATA)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to fetch data for frame %u. skipping.", i);
            continue;
        }

        unsigned char flags = gdc->IsDirected() ? 1 : 0;
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

    return true;
}

bool io::MMGDDWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}
