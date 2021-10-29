/*
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "MMFTDataSource.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"

using namespace megamol::datatools::table;
using namespace megamol;

namespace {
template<typename T>
T read(std::istream& stream) {
    T value;
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream.good()) {
        throw std::runtime_error("Error reading from stream!");
    }
    return value;
}

template<typename T>
std::vector<T> read_vector(std::istream& stream, std::size_t size) {
    std::vector<T> vec(size);
    stream.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    if (!stream.good()) {
        throw std::runtime_error("Error reading from stream!");
    }
    return vec;
}

std::string read_string(std::istream& stream, std::size_t size, bool trim_null = true) {
    std::string str(size, '\0');
    stream.read(str.data(), size * sizeof(std::string::value_type));
    if (!stream.good()) {
        throw std::runtime_error("Error reading from stream!");
    }
    if (trim_null) {
        str.erase(std::find(str.begin(), str.end(), '\0'), str.end());
    }
    return str;
}

} // namespace

MMFTDataSource::MMFTDataSource()
        : core::Module()
        , getDataSlot_("getData", "Slot providing the data")
        , filenameSlot_("filename", "The file name")
        , reloadSlot_("reload", "Reload file")
        , dataHash_(0)
        , reload_(false)
        , columns_()
        , values_() {

    filenameSlot_ << new core::param::FilePathParam("");
    MakeSlotAvailable(&filenameSlot_);
    reloadSlot_ << new core::param::ButtonParam();
    reloadSlot_.SetUpdateCallback(this, &MMFTDataSource::reloadCallback);
    MakeSlotAvailable(&reloadSlot_);

    getDataSlot_.SetCallback(TableDataCall::ClassName(), "GetData", &MMFTDataSource::getDataCallback);
    getDataSlot_.SetCallback(TableDataCall::ClassName(), "GetHash", &MMFTDataSource::getHashCallback);
    MakeSlotAvailable(&getDataSlot_);
}

MMFTDataSource::~MMFTDataSource() {
    Release();
}

bool MMFTDataSource::create() {
    // nothing to do
    return true;
}

void MMFTDataSource::release() {
    columns_.clear();
    values_.clear();
}

void MMFTDataSource::assertData() {
    using namespace std::string_literals;

    if (!filenameSlot_.IsDirty() && !reload_) {
        return; // nothing to do
    }

    filenameSlot_.ResetDirty();
    reload_ = false;

    columns_.clear();
    values_.clear();

    auto filename = filenameSlot_.Param<core::param::FilePathParam>()->Value();
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to open file \"%s\". Abort.", filename.generic_u8string().c_str());
        return;
    }

    try {
        if (!(read_string(file, 6, false) == "MMFTD\0"s)) {
            throw std::runtime_error("Wrong file format magic ID!");
        }

        auto version = read<uint16_t>(file);
        if (version != 0) {
            throw std::runtime_error("Wrong file format version number");
        }

        auto colCount = read<uint32_t>(file);
        columns_.resize(colCount);

        for (uint32_t c = 0; c < colCount; ++c) {
            TableDataCall::ColumnInfo& ci = columns_[c];
            auto nameLen = read<uint16_t>(file);
            ci.SetName(read_string(file, nameLen));
            auto type = read<uint8_t>(file);
            ci.SetType((type == 1) ? TableDataCall::ColumnType::CATEGORICAL : TableDataCall::ColumnType::QUANTITATIVE);
            ci.SetMinimumValue(read<float>(file));
            ci.SetMaximumValue(read<float>(file));
        }

        auto rowCount = read<uint64_t>(file);

        values_ = read_vector<float>(file, rowCount * colCount);

        dataHash_++;

    } catch (std::exception& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(ex.what());
        columns_.clear();
        values_.clear();
        return;
    }
}

bool MMFTDataSource::getDataCallback(core::Call& caller) {
    TableDataCall* tfd = dynamic_cast<TableDataCall*>(&caller);
    if (tfd == nullptr) {
        return false;
    }

    assertData();

    tfd->SetDataHash(dataHash_);
    tfd->SetFrameCount(1);
    if (values_.empty()) {
        tfd->Set(0, 0, nullptr, nullptr);
    } else {
        assert((values_.size() % columns_.size()) == 0);
        tfd->Set(columns_.size(), values_.size() / columns_.size(), columns_.data(), values_.data());
    }
    tfd->SetUnlocker(nullptr);

    return true;
}

bool MMFTDataSource::getHashCallback(core::Call& caller) {
    TableDataCall* tfd = dynamic_cast<TableDataCall*>(&caller);
    if (tfd == nullptr) {
        return false;
    }

    assertData();

    tfd->SetFrameCount(1);
    tfd->SetDataHash(dataHash_);
    tfd->SetUnlocker(nullptr);

    return true;
}

bool MMFTDataSource::reloadCallback(core::param::ParamSlot& caller) {
    reload_ = true;
    return true;
}
