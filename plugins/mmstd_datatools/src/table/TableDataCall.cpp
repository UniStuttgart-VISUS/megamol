/*
 * CallFloatTableData.cpp
 *
 * Copyright (C) 2015-2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmstd_datatools/table/TableDataCall.h"

using namespace megamol::stdplugin::datatools;
using namespace megamol::stdplugin::datatools::table;
using namespace megamol;


CallFloatTableData::ColumnInfo::ColumnInfo() : name(), type(ColumnType::CATEGORICAL), minVal(0.0f), maxVal(0.0f) {
    // intentionally empty
}

CallFloatTableData::ColumnInfo::ColumnInfo(const ColumnInfo& src) : name(src.name), type(src.type), minVal(src.minVal), maxVal(src.maxVal) {
    // intentionally empty
}

CallFloatTableData::ColumnInfo::~ColumnInfo() {
    // intentionally empty
}

CallFloatTableData::ColumnInfo& CallFloatTableData::ColumnInfo::operator=(const ColumnInfo& rhs) {
    SetName(rhs.Name());
    SetType(rhs.Type());
    SetMinimumValue(rhs.MinimumValue());
    SetMaximumValue(rhs.MaximumValue());
    return *this;
}

bool CallFloatTableData::ColumnInfo::operator==(const ColumnInfo& rhs) const{
    return (name == rhs.name)
        && (type == rhs.type)
        && (minVal == rhs.minVal) // epsilon test is not required since this is for testing real identity of info objects
        && (maxVal == rhs.maxVal);
}


CallFloatTableData::CallFloatTableData(void) : core::AbstractGetDataCall(), columns_count(0), rows_count(0), columns(nullptr), data(nullptr), frameCount(0), frameID(0) {
    // intentionally empty
}

CallFloatTableData::~CallFloatTableData(void) {
    columns_count = 0; // paranoia
    rows_count = 0; // paranoia
    columns = nullptr; // do not delete, since we do not own the memory of the objects
    data = nullptr; // do not delete, since we do not own the memory of the objects
}
