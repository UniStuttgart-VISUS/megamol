/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#include "MMFTDataWriter.h"

#include <filesystem>
#include <fstream>

#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::datatools;
using namespace megamol::datatools::table;
using namespace megamol;

MMFTDataWriter::MMFTDataWriter()
        : core::AbstractDataWriter()
        , filenameSlot("filename", "The path to the MMFT file to be written")
        , dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot << new core::param::FilePathParam(
        "", megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"mmft"});
    this->MakeSlotAvailable(&this->filenameSlot);

    this->dataSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}

MMFTDataWriter::~MMFTDataWriter() {
    this->Release();
}

bool MMFTDataWriter::create() {
    return true;
}

void MMFTDataWriter::release() {}

bool MMFTDataWriter::run() {
    using megamol::core::utility::log::Log;
    auto filename = this->filenameSlot.Param<core::param::FilePathParam>()->Value();
    if (filename.empty()) {
        Log::DefaultLog.WriteError("No file name specified. Abort.");
        return false;
    }

    TableDataCall* cftd = this->dataSlot.CallAs<TableDataCall>();
    if (cftd == nullptr) {
        Log::DefaultLog.WriteError("No data source connected. Abort.");
        return false;
    }

    if (!(*cftd)(0)) {
        Log::DefaultLog.WriteError("Failed to get data. Abort.");
        return false;
    }

    if (std::filesystem::exists(filename)) {
        Log::DefaultLog.WriteWarn("File %s already exists and will be overwritten.", filename.generic_string().c_str());
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        Log::DefaultLog.WriteError("Unable to create output file \"%s\". Abort.", filename.generic_string().c_str());
        cftd->Unlock();
        return false;
    }

    try {
        file.exceptions(std::ios::failbit);

        std::string magicID("MMFTD");
        file.write(magicID.data(), 6);

        uint16_t version = 0;
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint16_t));

        uint32_t colCnt = static_cast<uint32_t>(cftd->GetColumnsCount());
        file.write(reinterpret_cast<const char*>(&colCnt), sizeof(uint32_t));

        for (uint32_t c = 0; c < colCnt; ++c) {
            const TableDataCall::ColumnInfo& ci = cftd->GetColumnsInfos()[c];
            uint16_t nameLen = static_cast<uint16_t>(ci.Name().size());
            file.write(reinterpret_cast<const char*>(&nameLen), sizeof(uint16_t));
            file.write(ci.Name().data(), nameLen);

            uint8_t type = (ci.Type() == TableDataCall::ColumnType::CATEGORICAL) ? 1 : 0;
            file.write(reinterpret_cast<const char*>(&type), sizeof(uint8_t));

            float f = ci.MinimumValue();
            file.write(reinterpret_cast<const char*>(&f), sizeof(float));
            f = ci.MaximumValue();
            file.write(reinterpret_cast<const char*>(&f), sizeof(float));
        }

        uint64_t rowCnt = static_cast<uint64_t>(cftd->GetRowsCount());
        file.write(reinterpret_cast<const char*>(&rowCnt), sizeof(uint64_t));

        file.write(reinterpret_cast<const char*>(cftd->GetData()), rowCnt * colCnt * sizeof(float));

    } catch (...) {
        Log::DefaultLog.WriteError("Write error \"%s\".", filename.generic_string().c_str());
        cftd->Unlock();
        return false;
    }

    cftd->Unlock();

    return true;
}

bool MMFTDataWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}
