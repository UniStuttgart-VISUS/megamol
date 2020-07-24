/*
 * MMFTDataWriter.cpp
 *
 * Copyright (C) 2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMFTDataWriter.h"

#include "mmcore/param/FilePathParam.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/String.h"

using namespace megamol::stdplugin::datatools;
using namespace megamol::stdplugin::datatools::table;
using namespace megamol;

MMFTDataWriter::MMFTDataWriter(void) : core::AbstractDataWriter(),
        filenameSlot("filename", "The path to the MMFT file to be written"),
        dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->dataSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}


MMFTDataWriter::~MMFTDataWriter(void) {
    this->Release();
}


bool MMFTDataWriter::create(void) {
    return true;
}


void MMFTDataWriter::release(void) {
}


bool MMFTDataWriter::run(void) {
    using megamol::core::utility::log::Log;
    vislib::TString filename(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteError("No file name specified. Abort.");
        return false;
    }

    TableDataCall *cftd = this->dataSlot.CallAs<TableDataCall>();
    if (cftd == NULL) {
        Log::DefaultLog.WriteError("No data source connected. Abort.");
        return false;
    }

    if (vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteWarn("File %s already exists and will be overwritten.", vislib::StringA(filename).PeekBuffer());
    }

    if (!(*cftd)(0)) {
        Log::DefaultLog.WriteError("Failed to get data. Abort.");
        return false;
    }

    vislib::sys::FastFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteError("Unable to create output file \"%s\". Abort.", vislib::StringA(filename).PeekBuffer());
        cftd->Unlock();
        return false;
    }

#define ASSERT_WRITEOUT(A, S) if (file.Write((A), (S)) != (S)) { \
        Log::DefaultLog.WriteError("Write error %d", __LINE__); \
        file.Close(); \
        cftd->Unlock(); \
        return false; \
    }

    vislib::StringA magicID("MMFTD");
    ASSERT_WRITEOUT(magicID.PeekBuffer(), 6);
    uint16_t version = 0;
    ASSERT_WRITEOUT(&version, 2);

    uint32_t colCnt = static_cast<uint32_t>(cftd->GetColumnsCount());
    ASSERT_WRITEOUT(&colCnt, 4);

    for (uint32_t c = 0; c < colCnt; ++c) {
        const TableDataCall::ColumnInfo& ci = cftd->GetColumnsInfos()[c];
        uint16_t nameLen = static_cast<uint16_t>(ci.Name().size());
        ASSERT_WRITEOUT(&nameLen, 2);
        ASSERT_WRITEOUT(ci.Name().data(), nameLen);
        uint8_t type = (ci.Type() == TableDataCall::ColumnType::CATEGORICAL) ? 1 : 0;
        ASSERT_WRITEOUT(&type, 1);
        float f = ci.MinimumValue();
        ASSERT_WRITEOUT(&f, 4);
        f = ci.MaximumValue();
        ASSERT_WRITEOUT(&f, 4);
    }

    uint64_t rowCnt = static_cast<uint64_t>(cftd->GetRowsCount());
    ASSERT_WRITEOUT(&rowCnt, 8);

    ASSERT_WRITEOUT(cftd->GetData(), rowCnt * colCnt * 4);

    return true;
}


bool MMFTDataWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}
