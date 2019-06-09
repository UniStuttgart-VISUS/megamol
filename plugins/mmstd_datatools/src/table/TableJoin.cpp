/*
 * FloatTableJoin.cpp
 *
 * Copyright (C) 2016-2017 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FloatTableJoin.h"
#include <limits>

std::string megamol::stdplugin::datatools::floattable::FloatTableJoin::ModuleName
    = std::string("FloatTableJoin");

size_t hash_combine(size_t lhs, size_t rhs) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
}

megamol::stdplugin::datatools::floattable::FloatTableJoin::FloatTableJoin(void) : core::Module(),
    firstFloatTableInSlot("firstFloatTableIn", "First input"),
    secondFloatTableInSlot("secondFloatTableIn", "Second input"),
    dataOutSlot("dataOut", "Output"),
    frameID(-1),
    firstDataHash(std::numeric_limits<unsigned long>::max()), secondDataHash(std::numeric_limits<unsigned long>::max()) {
    this->firstFloatTableInSlot.SetCompatibleCall<CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->firstFloatTableInSlot);

    this->secondFloatTableInSlot.SetCompatibleCall<CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->secondFloatTableInSlot);

    this->dataOutSlot.SetCallback(CallFloatTableData::ClassName(),
        CallFloatTableData::FunctionName(0),
        &FloatTableJoin::processData);
    this->dataOutSlot.SetCallback(CallFloatTableData::ClassName(),
        CallFloatTableData::FunctionName(1),
        &FloatTableJoin::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);
}

megamol::stdplugin::datatools::floattable::FloatTableJoin::~FloatTableJoin(void) {
    this->Release();
}

bool megamol::stdplugin::datatools::floattable::FloatTableJoin::create(void) {
    return true;
}

void megamol::stdplugin::datatools::floattable::FloatTableJoin::release(void) {

}

bool megamol::stdplugin::datatools::floattable::FloatTableJoin::processData(core::Call &c) {
    try {
        CallFloatTableData *outCall = dynamic_cast<CallFloatTableData *>(&c);
        if (outCall == NULL) return false;

        CallFloatTableData *firstInCall = this->firstFloatTableInSlot.CallAs<CallFloatTableData>();
        if (firstInCall == NULL) return false;

        CallFloatTableData *secondInCall = this->secondFloatTableInSlot.CallAs<CallFloatTableData>();
        if (secondInCall == NULL) return false;

        // check time compatibility
        if (firstInCall->GetFrameCount() != secondInCall->GetFrameCount()) {
            vislib::sys::Log::DefaultLog.WriteError(_T("%hs: Cannot join float tables. ")
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
                sizeof(CallFloatTableData::ColumnInfo)*firstColumnCount);
            memcpy(&(this->column_info.data()[firstColumnCount]), secondColumnInfos,
                sizeof(CallFloatTableData::ColumnInfo)*secondColumnCount);
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

void megamol::stdplugin::datatools::floattable::FloatTableJoin::concatenate(
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
        for (size_t col = 0; col < secondColumnCount; col++) {
            float* outR = &out[(col + firstColumnCount) + row * columnCount];
            memcpy(outR, &second[col + row * secondColumnCount], sizeof(float) * secondColumnCount);
        }
    }
    for (size_t row = secondRowCount; row < rowCount; row++) {
        for (size_t col = 0; col < secondColumnCount; col++) {
            out[(col + firstColumnCount) + row * columnCount] = NAN;
        }
    }
}

bool megamol::stdplugin::datatools::floattable::FloatTableJoin::getExtent(core::Call &c) {
    try {
        CallFloatTableData *outCall = dynamic_cast<CallFloatTableData *>(&c);
        if (outCall == NULL) return false;

        CallFloatTableData *inCall = this->firstFloatTableInSlot.CallAs<CallFloatTableData>();
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
