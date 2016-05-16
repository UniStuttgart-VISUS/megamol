/*
 * MMFTDataSource.cpp
 *
 * Copyright (C) 2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "floattable/MMFTDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/CoreInstance.h"
#include "vislib/sys/FastFile.h"
#include "vislib/String.h"

using namespace megamol;
using namespace megamol::stdplugin;


datatools::floattable::MMFTDataSource::MMFTDataSource(void) : core::Module(),
        filenameSlot("filename", "The file name"),
        getDataSlot("getData", "Slot providing the data"),
        dataHash(0), columns(), values() {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback(CallFloatTableData::ClassName(), "GetData", &MMFTDataSource::getDataCallback);
    this->getDataSlot.SetCallback(CallFloatTableData::ClassName(), "GetHash", &MMFTDataSource::getHashCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}

datatools::floattable::MMFTDataSource::~MMFTDataSource(void) {
    this->Release();
}

bool datatools::floattable::MMFTDataSource::create(void) {
    // nothing to do
    return true;
}

void datatools::floattable::MMFTDataSource::release(void) {
    this->columns.clear();
    this->values.clear();
}

void datatools::floattable::MMFTDataSource::assertData(void) {
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
        CallFloatTableData::ColumnInfo& ci = columns[c];
        uint16_t nameLen;
        ASSERT_READ(&nameLen, 2);
        vislib::StringA name;
        ASSERT_READ(name.AllocateBuffer(nameLen), nameLen);
        ci.SetName(name.PeekBuffer());
        uint8_t type;
        ASSERT_READ(&type, 1);
        ci.SetType(
            (type == 1) ? CallFloatTableData::ColumnType::CATEGORICAL
            : CallFloatTableData::ColumnType::QUANTITATIVE);
        float f;
        ASSERT_READ(&f, 4);
        ci.SetMinimumValue(f);
        ASSERT_READ(&f, 4);
        ci.SetMaximumValue(f);
    }

    uint64_t rowCnt;
    ASSERT_READ(&rowCnt, 8);

    values.resize(rowCnt * colCnt);

    ASSERT_READ(values.data(), rowCnt * colCnt * 4);

    this->dataHash++;
}

bool datatools::floattable::MMFTDataSource::getDataCallback(core::Call& caller) {
    CallFloatTableData *tfd = dynamic_cast<CallFloatTableData*>(&caller);
    if (tfd == nullptr) return false;

    this->assertData();

    tfd->SetDataHash(this->dataHash);
    if (values.size() == 0) {
        tfd->Set(0, 0, nullptr, nullptr);
    } else {
        assert((values.size() % columns.size()) == 0);
        tfd->Set(columns.size(), values.size() / columns.size(), columns.data(), values.data());
    }
    tfd->SetUnlocker(nullptr);

    return true;
}

bool datatools::floattable::MMFTDataSource::getHashCallback(core::Call& caller) {
    CallFloatTableData *tfd = dynamic_cast<CallFloatTableData*>(&caller);
    if (tfd == nullptr) return false;

    this->assertData();

    tfd->SetDataHash(this->dataHash);
    tfd->SetUnlocker(nullptr);

    return true;
}
