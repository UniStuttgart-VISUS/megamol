/*
 * CallFloatTableData.cpp
 *
 * Copyright (C) 2015-2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmstd_datatools/CallFloatTableData.h"

using namespace megamol;
using namespace megamol::stdplugin;


datatools::CallFloatTableData::ColumnInfo::ColumnInfo() : name(), type(ColumnType::CATEGORICAL), minVal(0.0f), maxVal(0.0f) {
    // intentionally empty
}

datatools::CallFloatTableData::ColumnInfo::ColumnInfo(const ColumnInfo& src) : name(src.name), type(src.type), minVal(src.minVal), maxVal(src.maxVal) {
    // intentionally empty
}

datatools::CallFloatTableData::ColumnInfo::~ColumnInfo() {
    // intentionally empty
}

datatools::CallFloatTableData::ColumnInfo& datatools::CallFloatTableData::ColumnInfo::operator=(const ColumnInfo& rhs) {
    SetName(rhs.Name());
    SetType(rhs.Type());
    SetMinimumValue(rhs.MinimumValue());
    SetMaximumValue(rhs.MaximumValue());
    return *this;
}

bool datatools::CallFloatTableData::ColumnInfo::operator==(const ColumnInfo& rhs) const{
    return (name == rhs.name)
        && (type == rhs.type)
        && (minVal == rhs.minVal) // epsilon test is not required since this is for testing real identity of info objects
        && (maxVal == rhs.maxVal);
}


datatools::CallFloatTableData::CallFloatTableData(void) : core::AbstractGetDataCall(), columns_count(0), rows_count(0), columns(nullptr), data(nullptr) {
    // intentionally empty
}

datatools::CallFloatTableData::~CallFloatTableData(void) {
    columns_count = 0; // paranoia
    rows_count = 0; // paranoia
    columns = nullptr; // do not delete, since we do not own the memory of the objects
    data = nullptr; // do not delete, since we do not own the memory of the objects
}
