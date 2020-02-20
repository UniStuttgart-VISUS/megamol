/*
 * TableJoin.cpp
 *
 * Copyright (C) 2016-2017 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TableJoin.h"

#include <limits>

using namespace megamol::stdplugin::datatools;
using namespace megamol::stdplugin::datatools::table;
using namespace megamol;

std::string TableJoin::ModuleName
    = std::string("TableJoin");

size_t hash_combine(size_t lhs, size_t rhs) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
}

TableJoin::TableJoin(void) : core::Module(),
    firstTableInSlot("firstTableIn", "First input"),
    secondTableInSlot("secondTableIn", "Second input"),
    dataOutSlot("dataOut", "Output"),
    frameID(-1),
    firstDataHash(std::numeric_limits<unsigned long>::max()), secondDataHash(std::numeric_limits<unsigned long>::max()) {
    this->firstTableInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->firstTableInSlot);

    this->secondTableInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->secondTableInSlot);

    this->dataOutSlot.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(0),
        &TableJoin::processData);
    this->dataOutSlot.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(1),
        &TableJoin::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);
}

TableJoin::~TableJoin(void) {
    this->Release();
}

bool TableJoin::create(void) {
    return true;
}

void TableJoin::release(void) {

}

bool TableJoin::processData(core::Call &c) {
    try {
        TableDataCall *outCall = dynamic_cast<TableDataCall *>(&c);
        if (outCall == NULL) return false;

        TableDataCall *firstInCall = this->firstTableInSlot.CallAs<TableDataCall>();
        if (firstInCall == NULL) return false;

        TableDataCall *secondInCall = this->secondTableInSlot.CallAs<TableDataCall>();
        if (secondInCall == NULL) return false;

        // call getHash before check of frame count
        if (!(*firstInCall)(1)) return false;
        if (!(*secondInCall)(1)) return false;

        // check time compatibility
        if (firstInCall->GetFrameCount() != secondInCall->GetFrameCount()) {
            vislib::sys::Log::DefaultLog.WriteError(_T("%hs: Cannot join tables. ")
                _T("They are required to have equal frame count\n"), ModuleName.c_str());
            return false;
        }

        // request frame data
        firstInCall->SetFrameID(outCall->GetFrameID());
        secondInCall->SetFrameID(outCall->GetFrameID());

        // issue calls
        if (!(*firstInCall)()) return false;
        if (!(*secondInCall)()) return false;

        if (this->firstDataHash != firstInCall->DataHash() || this->secondDataHash != secondInCall->DataHash()
            || this->frameID != firstInCall->GetFrameID() || this->frameID != secondInCall->GetFrameID()) {
            this->firstDataHash = firstInCall->DataHash();
            this->secondDataHash = secondInCall->DataHash();
            ASSERT(firstInCall->GetFrameID() == secondInCall->GetFrameID());
            this->frameID = firstInCall->GetFrameID();

            // retrieve data
            auto firstRowsCount = firstInCall->GetRowsCount();
            auto firstColumnCount = firstInCall->GetColumnsCount();
            auto firstColumnInfos = firstInCall->GetColumnsInfos();
            auto firstData = firstInCall->GetData();

            auto secondRowsCount = secondInCall->GetRowsCount();
            auto secondColumnCount = secondInCall->GetColumnsCount();
            auto secondColumnInfos = secondInCall->GetColumnsInfos();
            auto secondData = secondInCall->GetData();

            // concatenate
            this->rows_count = std::max(firstRowsCount, secondRowsCount);
            this->column_count = firstColumnCount + secondColumnCount;
            this->column_info.clear();
            this->column_info.reserve(this->column_count);
            memcpy(this->column_info.data(), firstColumnInfos,
                sizeof(TableDataCall::ColumnInfo)*firstColumnCount);
            memcpy(&(this->column_info.data()[firstColumnCount]), secondColumnInfos,
                sizeof(TableDataCall::ColumnInfo)*secondColumnCount);
            this->data.clear();
            this->data.resize(this->rows_count * this->column_count);

            this->concatenate(this->data.data(), this->rows_count, this->column_count,
				firstData, firstRowsCount, firstColumnCount,
				secondData, secondRowsCount, secondColumnCount);
        }

        outCall->SetFrameCount(firstInCall->GetFrameCount());
        outCall->SetFrameID(this->frameID);
        outCall->SetDataHash(hash_combine(this->firstDataHash, this->secondDataHash));
        outCall->Set(this->column_count, this->rows_count, this->column_info.data(), this->data.data());
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::processData\n"),
            ModuleName.c_str());
        return false;
    }

    return true;
}

void TableJoin::concatenate(
	float* const out, const size_t rowCount, const size_t columnCount,
	const float* const first, const size_t firstRowCount, const size_t firstColumnCount, 
    const float* const second,  const size_t secondRowCount, const size_t secondColumnCount) {
    assert(rowCount >= firstRowCount && rowCount >= secondRowCount && "Not enough rows");
    assert(columnCount >= firstColumnCount + secondColumnCount && "Not enough columns");
    for (size_t row = 0; row < firstRowCount; row++) {
        float* outR = &out[row * columnCount];
        memcpy(outR, &first[row * firstColumnCount], sizeof(float) * firstColumnCount);
    }
    for (size_t row = firstRowCount; row < rowCount; row++) {
        for (size_t col = 0; col < firstColumnCount; col++) {
            out[col + row * columnCount] = NAN;
        }
    }
    for (size_t row = 0; row < secondRowCount; row++) {
        float* outR = &out[firstColumnCount + row * columnCount];
        memcpy(outR, &second[row * secondColumnCount], sizeof(float) * secondColumnCount);
    }
    for (size_t row = secondRowCount; row < rowCount; row++) {
        for (size_t col = 0; col < secondColumnCount; col++) {
            out[col + firstColumnCount + row * columnCount] = NAN;
        }
    }
}

bool TableJoin::getExtent(core::Call &c) {
    try {
        TableDataCall *outCall = dynamic_cast<TableDataCall *>(&c);
        if (outCall == NULL) return false;

        TableDataCall *inCall = this->firstTableInSlot.CallAs<TableDataCall>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1)) return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(hash_combine(this->firstDataHash, this->secondDataHash));
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::getExtent\n"), ModuleName.c_str());
        return false;
    }

    return true;
}
