#include "stdafx.h"
#include "CSVDataWriter.h"

#include "mmcore/param/FilePathParam.h"

#include "vislib/sys/FastFile.h"
#include "vislib/sys/Log.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"


megamol::stdplugin::datatools::floattable::CSVDataWriter::CSVDataWriter(void)
    : megamol::core::AbstractDataWriter()
    , filenameSlot("filename", "Name of file to write")
    , dataSlot("dataIn", "data input") {
    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->dataSlot.SetCompatibleCall<CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}


megamol::stdplugin::datatools::floattable::CSVDataWriter::~CSVDataWriter(void) { this->Release(); }


bool megamol::stdplugin::datatools::floattable::CSVDataWriter::create(void) { return true; }


void megamol::stdplugin::datatools::floattable::CSVDataWriter::release(void) {}


bool megamol::stdplugin::datatools::floattable::CSVDataWriter::run(void) {
    using vislib::sys::Log;
    vislib::TString filename(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteError("No file name specified. Abort.");
        return false;
    }

    CallFloatTableData *cftd = this->dataSlot.CallAs<CallFloatTableData>();
    if (cftd == nullptr) {
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

    // write data
    uint32_t colCnt = static_cast<uint32_t>(cftd->GetColumnsCount());
    uint64_t rowCnt = static_cast<uint64_t>(cftd->GetRowsCount());

    for (uint32_t c = 0; c < colCnt; ++c) {
        const CallFloatTableData::ColumnInfo& ci = cftd->GetColumnsInfos()[c];
        auto buf = ci.Name();
        if (c < colCnt-1) {
            buf += ", ";
        } else {
            buf += "\n";
        }
        file.Write(buf.data(), buf.size());
    }

    auto const data = cftd->GetData();

    for (uint64_t r = 0; r < rowCnt; ++r) {
        std::string buf;
        for (uint32_t c = 0; c < colCnt; ++c) {
            buf += std::to_string(data[c + r * colCnt]);
            if (c < colCnt - 1) {
                buf += ", ";
            }
            else {
                buf += "\n";
            }
        }
        file.Write(buf.data(), buf.size());
    }

    return true;
}


bool megamol::stdplugin::datatools::floattable::CSVDataWriter::getCapabilities(
    megamol::core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}
