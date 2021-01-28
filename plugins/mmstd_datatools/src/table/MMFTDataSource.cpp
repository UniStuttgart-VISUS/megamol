/*
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "MMFTDataSource.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"

#include "vislib/String.h"
#include "vislib/sys/FastFile.h"

using namespace megamol::stdplugin::datatools;
using namespace megamol::stdplugin::datatools::table;
using namespace megamol;

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

void MMFTDataSource::assertData(void) {
    if (!filenameSlot_.IsDirty() && !reload_) {
        return; // nothing to do
    }

    filenameSlot_.ResetDirty();
    reload_ = false;

    columns_.clear();
    values_.clear();


    vislib::sys::FastFile file;
    if (!file.Open(filenameSlot_.Param<core::param::FilePathParam>()->Value(), vislib::sys::File::READ_ONLY,
            vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to open file \"%s\". Abort.",
            vislib::StringA(filenameSlot_.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        return;
    }

#define ABORT_ERROR(...)                                                      \
    {                                                                         \
        megamol::core::utility::log::Log::DefaultLog.WriteError(__VA_ARGS__); \
        file.Close();                                                         \
        columns_.clear();                                                     \
        values_.clear();                                                      \
        return;                                                               \
    }
#define ASSERT_READ(A, S)           \
    if (file.Read((A), (S)) != (S)) \
    ABORT_ERROR("Read error %d", __LINE__)

    vislib::StringA magicID;
    ASSERT_READ(magicID.AllocateBuffer(6), 6);
    if (!magicID.Equals("MMFTD"))
        ABORT_ERROR("Wrong file format magic ID");
    uint16_t version;
    ASSERT_READ(&version, 2);
    if (version != 0)
        ABORT_ERROR("Wrong file format version number");

    uint32_t colCnt;
    ASSERT_READ(&colCnt, 4);
    columns_.resize(colCnt);

    for (uint32_t c = 0; c < colCnt; ++c) {
        TableDataCall::ColumnInfo& ci = columns_[c];
        uint16_t nameLen;
        ASSERT_READ(&nameLen, 2);
        vislib::StringA name;
        ASSERT_READ(name.AllocateBuffer(nameLen), nameLen);
        ci.SetName(name.PeekBuffer());
        uint8_t type;
        ASSERT_READ(&type, 1);
        ci.SetType((type == 1) ? TableDataCall::ColumnType::CATEGORICAL : TableDataCall::ColumnType::QUANTITATIVE);
        float f;
        ASSERT_READ(&f, 4);
        ci.SetMinimumValue(f);
        ASSERT_READ(&f, 4);
        ci.SetMaximumValue(f);
    }

    uint64_t rowCnt;
    ASSERT_READ(&rowCnt, 8);

    values_.resize(static_cast<std::vector<float>::size_type>(rowCnt * colCnt));

    ASSERT_READ(values_.data(), rowCnt * colCnt * 4);

    dataHash_++;
}

bool MMFTDataSource::getDataCallback(core::Call& caller) {
    TableDataCall* tfd = dynamic_cast<TableDataCall*>(&caller);
    if (tfd == nullptr)
        return false;

    assertData();

    tfd->SetDataHash(dataHash_);
    tfd->SetFrameCount(1);
    if (values_.size() == 0) {
        tfd->Set(0, 0, nullptr, nullptr);
    } else {
        assert((values.size() % columns.size()) == 0);
        tfd->Set(columns_.size(), values_.size() / columns_.size(), columns_.data(), values_.data());
    }
    tfd->SetUnlocker(nullptr);

    return true;
}

bool MMFTDataSource::getHashCallback(core::Call& caller) {
    TableDataCall* tfd = dynamic_cast<TableDataCall*>(&caller);
    if (tfd == nullptr)
        return false;

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
