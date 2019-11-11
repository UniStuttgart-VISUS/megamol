/*
 * MMFTDataSource.cpp
 *
 * Copyright (C) 2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMFTDataSource.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/CoreInstance.h"

#include "vislib/sys/FastFile.h"
#include "vislib/String.h"

using namespace megamol::stdplugin::datatools;
using namespace megamol::stdplugin::datatools::table;
using namespace megamol;

MMFTDataSource::MMFTDataSource(void) : core::Module(),
        filenameSlot("filename", "The file name"),
        getDataSlot("getData", "Slot providing the data"),
        dataHash(0), columns(), values() {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback(TableDataCall::ClassName(), "GetData", &MMFTDataSource::getDataCallback);
    this->getDataSlot.SetCallback(TableDataCall::ClassName(), "GetHash", &MMFTDataSource::getHashCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}

MMFTDataSource::~MMFTDataSource(void) {
    this->Release();
}

bool MMFTDataSource::create(void) {
    // nothing to do
    return true;
}

void MMFTDataSource::release(void) {
    this->columns.clear();
    this->values.clear();
}

void MMFTDataSource::assertData(void) {
    if (!this->filenameSlot.IsDirty()) {
        return; // nothing to do
    }
    
    this->filenameSlot.ResetDirty();

    this->columns.clear();
    this->values.clear();


    vislib::sys::FastFile file;
    if (!file.Open(filenameSlot.Param<core::param::FilePathParam>()->Value(), vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to open file \"%s\". Abort.", vislib::StringA(filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        return;
    }

#define ABORT_ERROR(...) { \
        vislib::sys::Log::DefaultLog.WriteError(__VA_ARGS__); \
        file.Close(); \
        this->columns.clear(); \
        this->values.clear(); \
        return; \
    }
#define ASSERT_READ(A, S) if (file.Read((A), (S)) != (S)) ABORT_ERROR("Read error %d", __LINE__)

    vislib::StringA magicID;
    ASSERT_READ(magicID.AllocateBuffer(6), 6);
    if (!magicID.Equals("MMFTD")) ABORT_ERROR("Wrong file format magic ID");
    uint16_t version;
    ASSERT_READ(&version, 2);
    if (version != 0) ABORT_ERROR("Wrong file format version number");

    uint32_t colCnt;
    ASSERT_READ(&colCnt, 4);
    columns.resize(colCnt);

    for (uint32_t c = 0; c < colCnt; ++c) {
        TableDataCall::ColumnInfo& ci = columns[c];
        uint16_t nameLen;
        ASSERT_READ(&nameLen, 2);
        vislib::StringA name;
        ASSERT_READ(name.AllocateBuffer(nameLen), nameLen);
        ci.SetName(name.PeekBuffer());
        uint8_t type;
        ASSERT_READ(&type, 1);
        ci.SetType(
            (type == 1) ? TableDataCall::ColumnType::CATEGORICAL
            : TableDataCall::ColumnType::QUANTITATIVE);
        float f;
        ASSERT_READ(&f, 4);
        ci.SetMinimumValue(f);
        ASSERT_READ(&f, 4);
        ci.SetMaximumValue(f);
    }

    uint64_t rowCnt;
    ASSERT_READ(&rowCnt, 8);

    values.resize(static_cast<std::vector<float>::size_type>(rowCnt * colCnt));

    ASSERT_READ(values.data(), rowCnt * colCnt * 4);

    this->dataHash++;
}

bool MMFTDataSource::getDataCallback(core::Call& caller) {
    TableDataCall *tfd = dynamic_cast<TableDataCall*>(&caller);
    if (tfd == nullptr) return false;

    this->assertData();

    tfd->SetDataHash(this->dataHash);
    tfd->SetFrameCount(1);
    if (values.size() == 0) {
        tfd->Set(0, 0, nullptr, nullptr);
    } else {
        assert((values.size() % columns.size()) == 0);
        tfd->Set(columns.size(), values.size() / columns.size(), columns.data(), values.data());
    }
    tfd->SetUnlocker(nullptr);

    return true;
}

bool MMFTDataSource::getHashCallback(core::Call& caller) {
    TableDataCall *tfd = dynamic_cast<TableDataCall*>(&caller);
    if (tfd == nullptr) return false;

    this->assertData();

    tfd->SetFrameCount(1);
    tfd->SetDataHash(this->dataHash);
    tfd->SetUnlocker(nullptr);

    return true;
}
