//
// VTKLegacyDataUnstructuredGrid.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 23, 2013
// Author     : scharnkn
//

#include "stdafx.h"

#include "mmcore/utility/log/Log.h"
#include "protein/VTKLegacyDataUnstructuredGrid.h"

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core::utility::log;


/*
 * VTKLegacyDataUnstructuredGrid::AddPointData
 */
void VTKLegacyDataUnstructuredGrid::AddPointData(
    const char* data, size_t nElements, size_t nComponents, DataType type, vislib::StringA name) {

    // Add new element to point data array
    this->pointData.SetCount(this->pointData.Count() + 1);
    this->pointData.Last().SetData(data, nElements, nComponents, type, name);
}


/*
 * VTKLegacyDataUnstructuredGrid::VTKLegacyDataUnstructuredGrid
 */
VTKLegacyDataUnstructuredGrid::VTKLegacyDataUnstructuredGrid()
        : AbstractVTKLegacyData()
        , points(NULL)
        , nPoints(0)
        , cells(NULL)
        , nCells(0)
        , cellTypes(NULL)
        , nCellData(0) {}


/*
 * VTKLegacyDataUnstructuredGrid::~VTKLegacyDataUnstructuredGrid
 */
VTKLegacyDataUnstructuredGrid::~VTKLegacyDataUnstructuredGrid() {
    if (this->points)
        delete[] this->points;
    if (this->cells)
        delete[] this->cells;
    if (this->cellTypes)
        delete[] this->cellTypes;
}


/*
 * VTKLegacyDataUnstructuredGrid::PeekPointDataByIndex
 */
const AbstractVTKLegacyData::AttributeArray* VTKLegacyDataUnstructuredGrid::PeekPointDataByIndex(size_t idx) const {
    if (idx >= this->pointData.Count()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Requested idx out of bound, returning NULL.");
        return NULL;
    } else {
        return &this->pointData[idx];
    }
}


/*
 * VTKLegacyDataUnstructuredGrid::PeekPointDataByName
 */
const AbstractVTKLegacyData::AttributeArray* VTKLegacyDataUnstructuredGrid::PeekPointDataByName(
    vislib::StringA name) const {
    // Check whether the id is in use
    bool isUsed = false;
    int idx = -1;
    for (unsigned int i = 0; i < this->pointData.Count(); ++i) {
        if (this->pointData[i].GetId() == name) {
            isUsed = true;
            idx = i;
            break;
        }
    }

    // If the id is not in use: return null
    if (!isUsed) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Requested id '%s' not in use, returning NULL.", name.PeekBuffer());
        return NULL;
    } else { // else: return the data array
        return &this->pointData[idx];
    }
}
