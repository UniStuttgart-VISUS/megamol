#include "CSVWriter.h"

#include <filesystem>
#include <fstream>
#include <locale>

#include "mmcore/param/FilePathParam.h"

#include "datatools/table/TableDataCall.h"


megamol::datatools::CSVWriter::CSVWriter() : _data_in_slot("inData", ""), _filename_slot("filename", "") {
    _data_in_slot.SetCompatibleCall<table::TableDataCallDescription>();
    MakeSlotAvailable(&_data_in_slot);

    _filename_slot << new core::param::FilePathParam(
        "out.csv", megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"csv"});
    MakeSlotAvailable(&_filename_slot);
}


megamol::datatools::CSVWriter::~CSVWriter() {
    this->Release();
}


bool megamol::datatools::CSVWriter::create() {
    return true;
}


void megamol::datatools::CSVWriter::release() {}


bool megamol::datatools::CSVWriter::run() {
    auto filename = _filename_slot.Param<core::param::FilePathParam>()->Value();
    if (filename.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CSVWriter]: No file name specified. Abort.");
        return false;
    }

    auto inCall = _data_in_slot.CallAs<table::TableDataCall>();
    if (inCall == NULL) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CSVWriter]: No data source connected. Abort.");
        return false;
    }

    if (std::filesystem::exists(filename)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[CSVWriter]: File %s already exists and will be overwritten.", filename.string());
    }

    if (!(*inCall)(1) || !(*inCall)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[CSVWriter]: Could not load input data. Abort.");
        return false;
    }

    auto col_count = inCall->GetColumnsCount();
    auto row_count = inCall->GetRowsCount();
    auto infos = inCall->GetColumnsInfos();
    auto data = inCall->GetData();

    std::ofstream file = std::ofstream(filename);

    file.imbue(std::locale::classic());

    for (size_t col = 0; col < col_count; ++col) {
        if (col < col_count - 1) {
            file << infos[col].Name() << ",";
        } else {
            file << infos[col].Name() << "\n";
        }
    }

    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            if (col < col_count - 1) {
                file << inCall->GetData(col, row) << ",";
            } else {
                file << inCall->GetData(col, row) << "\n";
            }
        }
    }

    file.close();

    return true;
}


bool megamol::datatools::CSVWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}
